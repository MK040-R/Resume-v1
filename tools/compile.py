#!/usr/bin/env python3
"""
compile.py — Scan raw/ and build/update wiki/

Usage:
    python tools/compile.py              # incremental (default)
    python tools/compile.py --full       # force re-process all files
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add repo root to path so tools/ can import utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    SUPPORTED_EXTENSIONS,
    append_changelog,
    atomic_write,
    console,
    extract_json,
    extract_text,
    file_hash,
    get_anthropic_client,
    llm_call_with_retry,
    load_config,
    load_manifest,
    log_token_usage,
    resolve_paths,
    save_manifest,
    slugify,
)

# ── Prompts ───────────────────────────────────────────────────────────────────

SUMMARIZE_PROMPT = """\
You are a knowledge base assistant. Summarize the following source material.
Be concise but complete — capture all key ideas, arguments, technical details, and facts.
Output structured markdown with appropriate headings.
Do NOT add any preamble like "Here is a summary..." — just output the markdown content directly.

Source file: {filename}

---
{content}
"""

CONCEPT_EXTRACTION_PROMPT = """\
You are a knowledge base assistant. Analyze the following collection of source summaries.
Identify ALL distinct concepts that appear in these summaries.
Include concepts that appear in only one source — they may still be worth indexing.

For each concept return a JSON object with these exact fields:
  - name: human-readable concept name (title case)
  - slug: filesystem-safe slug (lowercase, hyphens only)
  - definition: one clear paragraph definition
  - key_points: array of 3-8 concise bullet strings
  - sources: array of source filenames (just the filename, no path) that reference this concept
  - related_concepts: array of slugs of other concepts in this list that are related

Return ONLY a valid JSON array of concept objects. No prose, no markdown fences.

Summaries:
---
{summaries}
"""

SHOULD_UPDATE_PROMPT = """\
You are a knowledge base assistant. Decide whether the existing concept article needs updating.

Existing article:
---
{existing}

New information from source "{source_file}":
---
{new_info}

Does the new information add facts, nuance, corrections, or new examples NOT already in the article?
Reply with exactly: YES or NO, followed by a single sentence explaining why.
"""

UPDATE_CONCEPT_PROMPT = """\
You are a knowledge base assistant. Update the following concept article to incorporate new information.
Preserve all existing content. Add new key points, update the Sources list, and update Related Concepts.
Keep the exact article format shown below. Update "Last updated" and "Source count".

Existing article:
---
{existing}

New information from source "{source_file}":
---
{new_info}

Return ONLY the updated markdown article. No preamble.
"""

CATEGORY_PROMPT = """\
You are a knowledge base assistant. Group the following concepts into logical categories.
Each category should contain 2 or more related concepts.

Concepts (slug → name):
{concept_list}

For each category return a JSON object:
  - name: human-readable category name
  - slug: filesystem-safe slug
  - description: one sentence description
  - concepts: array of concept slugs in this category

Return ONLY a valid JSON array. No prose, no markdown fences.
"""


# ── Article formatting ────────────────────────────────────────────────────────

def format_concept_article(concept: dict, today: str) -> str:
    key_points = "\n".join(f"- {p}" for p in concept.get("key_points", []))
    sources = "\n".join(
        f"- [[{s}]] - referenced in this source" for s in concept.get("sources", [])
    )
    related = "\n".join(
        f"- [[{r}]]" for r in concept.get("related_concepts", [])
    )
    source_count = len(concept.get("sources", []))

    return f"""# {concept['name']}

## Definition
{concept['definition']}

## Key Points
{key_points}

## Sources
{sources if sources else '(none yet)'}

## Related Concepts
{related if related else '(none identified yet)'}

## Derived Insights
(Query answers will be filed back here over time)

---
Last updated: {today}
Source count: {source_count}
"""


def format_category_article(category: dict, today: str) -> str:
    concepts_list = "\n".join(f"- [[{c}]]" for c in category.get("concepts", []))
    return f"""# {category['name']}

## Description
{category['description']}

## Concepts in this Category
{concepts_list}

---
Last updated: {today}
"""


def format_summary_article(filename: str, summary_text: str, today: str) -> str:
    return f"""# Summary: {filename}

> Auto-generated summary. Source: `raw/{filename}`
> Last updated: {today}

---

{summary_text}
"""


# ── Core compilation steps ────────────────────────────────────────────────────

def scan_raw_files(raw_dir: Path) -> list[Path]:
    """Return all supported files in raw/ (non-hidden)."""
    files = []
    for f in raw_dir.iterdir():
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return sorted(files)


def summarize_file(
    filepath: Path,
    wiki_dir: Path,
    client,
    model: str,
    max_tokens: int,
) -> tuple[str, str]:
    """
    Summarize one raw file → saves to wiki/summaries/filename.md
    Returns (summary_text, summary_path_str)
    """
    console.print(f"  [cyan]Summarizing[/cyan] {filepath.name} ...")

    content = extract_text(filepath, client)
    if not content.strip():
        content = "(file appears to be empty or unreadable)"

    prompt = SUMMARIZE_PROMPT.format(filename=filepath.name, content=content[:15000])

    response = llm_call_with_retry(
        client,
        _operation="summarize",
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    summary_text = response.content[0].text.strip()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    article = format_summary_article(filepath.name, summary_text, today)
    summary_path = wiki_dir / "summaries" / (filepath.stem + ".md")
    atomic_write(summary_path, article)

    log_token_usage(
        operation="summarize",
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        source_file=filepath.name,
    )

    return summary_text, str(summary_path)


def load_all_summaries(wiki_dir: Path) -> dict[str, str]:
    """Return {filename_stem: summary_text} for all summaries."""
    summaries_dir = wiki_dir / "summaries"
    result = {}
    if not summaries_dir.exists():
        return result
    for f in summaries_dir.glob("*.md"):
        result[f.stem] = f.read_text(encoding="utf-8")
    return result


def extract_concepts(
    summaries: dict[str, str],
    client,
    model: str,
    max_tokens: int,
) -> list[dict]:
    """Ask LLM to identify concepts across all summaries. Returns list of concept dicts."""
    console.print("  [cyan]Extracting concepts across summaries...[/cyan]")

    summaries_text = ""
    for stem, text in summaries.items():
        summaries_text += f"\n\n### {stem}\n{text[:3000]}"

    prompt = CONCEPT_EXTRACTION_PROMPT.format(summaries=summaries_text)

    response = llm_call_with_retry(
        client,
        _operation="concept_extraction",
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        concepts = extract_json(raw)
        if not isinstance(concepts, list):
            concepts = []
    except ValueError as e:
        console.print(f"[red]Failed to parse concepts JSON: {e}[/red]")
        concepts = []

    return concepts


def process_concept(
    concept: dict,
    wiki_dir: Path,
    new_summary_stems: set[str],
    summaries: dict[str, str],
    client,
    model: str,
    max_tokens: int,
    today: str,
) -> tuple[str, bool]:
    """
    Create or update a single concept article.
    Returns (changelog_entry, was_changed).
    """
    slug = slugify(concept.get("slug", concept.get("name", "unknown")))
    concept["slug"] = slug
    concept_path = wiki_dir / "concepts" / f"{slug}.md"

    # Find which of this concept's sources are new
    concept_sources = set(
        Path(s).stem if "." in s else s for s in concept.get("sources", [])
    )
    newly_relevant = concept_sources & new_summary_stems

    if not concept_path.exists():
        # New concept — create it
        article = format_concept_article(concept, today)
        atomic_write(concept_path, article)
        return f"Created concept article: [[{slug}]]", True

    if not newly_relevant:
        # Existing concept, no new relevant sources — skip
        return "", False

    # Existing concept + new sources — ask if update is needed
    existing = concept_path.read_text(encoding="utf-8")
    new_info_parts = []
    for stem in newly_relevant:
        if stem in summaries:
            new_info_parts.append(f"[{stem}]\n{summaries[stem][:2000]}")
    new_info = "\n\n---\n\n".join(new_info_parts)

    should_update_response = llm_call_with_retry(
        client,
        _operation="should_update_check",
        model=model,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": SHOULD_UPDATE_PROMPT.format(
                    existing=existing[:3000],
                    source_file=", ".join(newly_relevant),
                    new_info=new_info[:2000],
                ),
            }
        ],
    )
    decision = should_update_response.content[0].text.strip().upper()

    if not decision.startswith("YES"):
        return "", False

    # Update the concept article
    update_response = llm_call_with_retry(
        client,
        _operation="concept_update",
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": UPDATE_CONCEPT_PROMPT.format(
                    existing=existing,
                    source_file=", ".join(newly_relevant),
                    new_info=new_info,
                ),
            }
        ],
    )
    updated_article = update_response.content[0].text.strip()
    atomic_write(concept_path, updated_article)
    return f"Updated concept article: [[{slug}]]", True


def update_categories(
    wiki_dir: Path,
    concepts: list[dict],
    client,
    model: str,
    max_tokens: int,
    today: str,
) -> list[str]:
    """Ask LLM to group concepts into categories. Returns changelog entries."""
    if len(concepts) < 2:
        return []

    console.print("  [cyan]Organizing categories...[/cyan]")
    concept_list = "\n".join(
        f"  {c.get('slug', slugify(c['name']))} → {c['name']}" for c in concepts
    )

    response = llm_call_with_retry(
        client,
        _operation="categorize",
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": CATEGORY_PROMPT.format(concept_list=concept_list)}],
    )

    raw = response.content[0].text.strip()
    try:
        categories = extract_json(raw)
        if not isinstance(categories, list):
            categories = []
    except ValueError:
        categories = []

    changelog = []
    for cat in categories:
        slug = slugify(cat.get("slug", cat.get("name", "unknown")))
        cat_path = wiki_dir / "categories" / f"{slug}.md"
        article = format_category_article(cat, today)
        atomic_write(cat_path, article)
        changelog.append(f"Created/updated category: [[{slug}]]")

    return changelog


def regenerate_index(wiki_dir: Path, today: str) -> None:
    """Rebuild _index.md from current wiki state."""
    concepts_dir = wiki_dir / "concepts"
    categories_dir = wiki_dir / "categories"
    summaries_dir = wiki_dir / "summaries"

    concept_lines = []
    for f in sorted(concepts_dir.glob("*.md")):
        # Extract one-line description from definition section
        text = f.read_text(encoding="utf-8")
        desc = _extract_first_sentence(text, "## Definition")
        source_count = _extract_source_count(text)
        concept_lines.append(
            f"- [[{f.stem}]] — {desc} (sources: {source_count})"
        )

    category_lines = []
    for f in sorted(categories_dir.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        desc = _extract_first_sentence(text, "## Description")
        # Extract concept list
        concepts_in_cat = []
        in_section = False
        for line in text.splitlines():
            if line.startswith("## Concepts"):
                in_section = True
                continue
            if in_section and line.startswith("- [["):
                slug = line.split("[[")[1].split("]]")[0]
                concepts_in_cat.append(slug)
            elif in_section and line.startswith("##"):
                break
        covers = ", ".join(concepts_in_cat[:5])
        category_lines.append(f"- [[{f.stem}]] — {desc} (covers: {covers})")

    summary_lines = []
    for f in sorted(summaries_dir.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        # Get first non-empty, non-heading line after the divider
        desc = _extract_summary_desc(text)
        summary_lines.append(f"- [[{f.stem}]] — {desc}")

    n_sources = len(summary_lines)
    n_concepts = len(concept_lines)
    n_categories = len(category_lines)

    content = f"""# Knowledge Base Index

Last updated: {today} | Sources: {n_sources} | Concepts: {n_concepts} | Categories: {n_categories}

> Auto-maintained by `compile.py`. Do not edit manually.

## Concepts

{chr(10).join(concept_lines) if concept_lines else '(none yet)'}

## Categories

{chr(10).join(category_lines) if category_lines else '(none yet)'}

## Summaries

{chr(10).join(summary_lines) if summary_lines else '(none yet)'}
"""
    atomic_write(wiki_dir / "_index.md", content)


def _extract_first_sentence(text: str, section_header: str) -> str:
    """Extract the first sentence from a markdown section."""
    in_section = False
    for line in text.splitlines():
        if line.strip() == section_header:
            in_section = True
            continue
        if in_section:
            line = line.strip()
            if line and not line.startswith("#"):
                # Return up to first period or 80 chars
                sentence = line.split(".")[0]
                return sentence[:80]
            if line.startswith("##"):
                break
    return "(no description)"


def _extract_source_count(text: str) -> str:
    """Extract 'Source count: N' from article footer."""
    for line in text.splitlines():
        if line.startswith("Source count:"):
            return line.split(":", 1)[1].strip()
    return "?"


def _extract_summary_desc(text: str) -> str:
    """Extract a short description from a summary article."""
    past_divider = False
    for line in text.splitlines():
        if line.strip() == "---":
            past_divider = True
            continue
        if past_divider:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith(">"):
                return line[:80]
    return "(summary)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compile raw/ into wiki/")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-process all files even if already summarized",
    )
    args = parser.parse_args()

    config = load_config()
    paths = resolve_paths(config)
    raw_dir = paths["raw"]
    wiki_dir = paths["wiki"]
    model = config["llm"]["model"]
    max_tokens = config["llm"]["max_tokens"]
    incremental = config["compile"]["incremental"] and not args.full

    console.rule("[bold blue]Knowledge Base Compiler[/bold blue]")

    # Ensure directories exist
    for d in [raw_dir, wiki_dir / "concepts", wiki_dir / "categories", wiki_dir / "summaries", paths["outputs"]]:
        d.mkdir(parents=True, exist_ok=True)

    raw_files = scan_raw_files(raw_dir)
    if not raw_files:
        console.print("[yellow]No files found in raw/. Add some files and try again.[/yellow]")
        return

    console.print(f"Found [bold]{len(raw_files)}[/bold] file(s) in raw/")

    client = get_anthropic_client()
    manifest = load_manifest(raw_dir) if incremental else {"files": {}}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── Step 1: Summarize new/changed files ──────────────────────────────────
    new_summary_stems: set[str] = set()
    changelog_entries: list[str] = []

    console.print("\n[bold]Step 1: Summarizing source files[/bold]")
    for filepath in raw_files:
        current_hash = file_hash(filepath)
        existing_entry = manifest["files"].get(filepath.name, {})

        if incremental and existing_entry.get("hash") == current_hash:
            console.print(f"  [dim]Skipping {filepath.name} (unchanged)[/dim]")
            continue

        try:
            summary_text, summary_path = summarize_file(
                filepath, wiki_dir, client, model, max_tokens
            )
            manifest["files"][filepath.name] = {
                "hash": current_hash,
                "last_processed": datetime.now(timezone.utc).isoformat(),
                "summary_path": summary_path,
            }
            new_summary_stems.add(filepath.stem)
            changelog_entries.append(f"Summarized source: [[{filepath.stem}]]")
            save_manifest(raw_dir, manifest)
        except Exception as e:
            console.print(f"  [red]Error summarizing {filepath.name}: {e}[/red]")
            continue

    if not new_summary_stems:
        console.print("\n[green]All files already up to date. Nothing to recompile.[/green]")
        return

    console.print(f"\nSummarized [bold]{len(new_summary_stems)}[/bold] new/changed file(s)")

    # ── Step 2: Extract concepts ─────────────────────────────────────────────
    console.print("\n[bold]Step 2: Extracting concepts[/bold]")
    all_summaries = load_all_summaries(wiki_dir)
    concepts = extract_concepts(all_summaries, client, model, max_tokens)
    console.print(f"  Identified [bold]{len(concepts)}[/bold] concept(s)")

    # ── Step 3: Create/update concept articles ───────────────────────────────
    console.print("\n[bold]Step 3: Building concept articles[/bold]")
    all_concept_dicts = []
    for concept in concepts:
        try:
            entry, changed = process_concept(
                concept,
                wiki_dir,
                new_summary_stems,
                all_summaries,
                client,
                model,
                max_tokens,
                today,
            )
            all_concept_dicts.append(concept)
            if entry:
                changelog_entries.append(entry)
                console.print(f"  [green]✓[/green] {entry}")
            else:
                slug = slugify(concept.get("slug", concept.get("name", "?")))
                console.print(f"  [dim]~ Skipped (no new info): {slug}[/dim]")
        except Exception as e:
            console.print(f"  [red]Error processing concept '{concept.get('name', '?')}': {e}[/red]")

    # ── Step 4: Update categories ────────────────────────────────────────────
    console.print("\n[bold]Step 4: Organizing categories[/bold]")
    cat_changelog = update_categories(
        wiki_dir, all_concept_dicts, client, model, max_tokens, today
    )
    changelog_entries.extend(cat_changelog)
    for entry in cat_changelog:
        console.print(f"  [green]✓[/green] {entry}")

    # ── Step 5: Regenerate index ─────────────────────────────────────────────
    console.print("\n[bold]Step 5: Regenerating index[/bold]")
    regenerate_index(wiki_dir, today)
    changelog_entries.append("Regenerated _index.md")
    console.print("  [green]✓[/green] _index.md updated")

    # ── Step 6: Write changelog ──────────────────────────────────────────────
    append_changelog(wiki_dir, changelog_entries)

    # ── Done ─────────────────────────────────────────────────────────────────
    console.print(f"\n[bold green]Done![/bold green] Processed {len(new_summary_stems)} source(s), "
                  f"{len(all_concept_dicts)} concept(s), {len(cat_changelog)} category/categories.")
    console.print("Token usage logged to [cyan]usage.log[/cyan]")

    # Auto-lint if configured
    if config["compile"].get("auto_lint_after_compile"):
        console.print("\n[yellow]Running auto-lint...[/yellow]")
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).parent / "lint.py")], check=False)


if __name__ == "__main__":
    main()
