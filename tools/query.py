#!/usr/bin/env python3
"""
query.py — Answer questions against the wiki

Usage:
    python tools/query.py "What are the key themes across my sources?"
    python tools/query.py "How does attention work?" --no-writeback
    python tools/query.py "Explain BERT" --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    append_changelog,
    atomic_write,
    console,
    extract_json,
    get_anthropic_client,
    llm_call_with_retry,
    load_config,
    resolve_paths,
    slugify,
)

# ── Prompts ───────────────────────────────────────────────────────────────────

SELECT_ARTICLES_PROMPT = """\
You are a knowledge base assistant. A user has asked a question.
Given the knowledge base index below, identify which concept and category articles
are most relevant to answering the question.

Return a JSON array of article slugs (filenames without .md extension).
Include at most {top_k} slugs. Order by relevance, most relevant first.
Only include articles from the "Concepts" and "Categories" sections of the index.
Return ONLY the JSON array. No prose.

Question: {question}

Knowledge Base Index:
---
{index_content}
"""

ANSWER_PROMPT = """\
You are a knowledgeable assistant. Answer the following question using ONLY the
information provided in the knowledge base articles below.

Be comprehensive, well-structured, and cite specific sources using [[article-name]] notation.
Use markdown formatting with headings and bullet points as appropriate.

Question: {question}

Knowledge Base Articles:
---
{articles_content}
"""

DERIVE_INSIGHTS_PROMPT = """\
You are a knowledge base assistant. A user asked a question and received an answer.
Identify which concept articles should be enriched with derived insights from this Q&A.

For each relevant concept, provide:
  - concept_slug: the slug of the concept article to update
  - insight: a concise (2-4 sentence) derived insight to add under "Derived Insights"

Return ONLY a valid JSON array of objects with fields "concept_slug" and "insight".
If no concepts need enrichment, return an empty array [].

Question: {question}

Answer:
{answer}

Available concept slugs (from index):
{concept_slugs}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_index(wiki_dir: Path) -> str:
    index_path = wiki_dir / "_index.md"
    if not index_path.exists():
        return ""
    return index_path.read_text(encoding="utf-8")


def read_articles(wiki_dir: Path, slugs: list[str]) -> str:
    """Read and concatenate selected concept/category articles."""
    parts = []
    for slug in slugs:
        # Try concepts first, then categories
        for subdir in ("concepts", "categories"):
            path = wiki_dir / subdir / f"{slug}.md"
            if path.exists():
                content = path.read_text(encoding="utf-8")
                parts.append(f"### [{slug}]\n{content}")
                break
        else:
            console.print(f"  [yellow]Warning: article not found for slug '{slug}'[/yellow]")
    return "\n\n---\n\n".join(parts)


def get_concept_slugs(wiki_dir: Path) -> list[str]:
    """Return all concept slugs currently in wiki/concepts/."""
    concepts_dir = wiki_dir / "concepts"
    if not concepts_dir.exists():
        return []
    return [f.stem for f in concepts_dir.glob("*.md")]


def write_back_insights(
    wiki_dir: Path,
    insights: list[dict],
    question: str,
    today: str,
) -> list[str]:
    """
    Append derived insights to relevant concept articles.
    Returns list of changelog entries.
    """
    changelog = []
    for item in insights:
        slug = item.get("concept_slug", "").strip()
        insight = item.get("insight", "").strip()
        if not slug or not insight:
            continue

        concept_path = wiki_dir / "concepts" / f"{slug}.md"
        if not concept_path.exists():
            console.print(f"  [yellow]Concept '{slug}' not found for write-back[/yellow]")
            continue

        content = concept_path.read_text(encoding="utf-8")

        # Find the "## Derived Insights" section and append
        derived_marker = "## Derived Insights"
        separator_marker = "---\nLast updated:"

        new_entry = (
            f"\n**Q: {question[:100]}{'...' if len(question) > 100 else ''}**\n"
            f"{insight}\n"
            f"*(filed: {today})*\n"
        )

        if derived_marker in content:
            # Insert after the Derived Insights header
            idx = content.index(derived_marker) + len(derived_marker)
            # Find next section or end
            next_section = content.find("\n---\n", idx)
            if next_section == -1:
                next_section = len(content)
            insert_pos = idx
            updated = content[:insert_pos] + "\n" + new_entry + content[insert_pos:]
        else:
            # Append section before the footer divider
            if "---\n" in content:
                idx = content.rindex("---\n")
                updated = content[:idx] + f"## Derived Insights\n{new_entry}\n---\n" + content[idx + 4:]
            else:
                updated = content + f"\n## Derived Insights\n{new_entry}\n"

        # Update Last updated date
        import re
        updated = re.sub(r"Last updated: \d{4}-\d{2}-\d{2}", f"Last updated: {today}", updated)

        atomic_write(concept_path, updated)
        changelog.append(f"Filed derived insight into [[{slug}]]")
        console.print(f"  [green]✓[/green] Insight filed → [[{slug}]]")

    return changelog


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Q&A against the wiki")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max number of articles to read (default: 5)",
    )
    parser.add_argument(
        "--no-writeback",
        action="store_true",
        help="Skip filing derived insights back into wiki/",
    )
    args = parser.parse_args()

    config = load_config()
    paths = resolve_paths(config)
    wiki_dir = paths["wiki"]
    outputs_dir = paths["outputs"]
    model = config["llm"]["model"]
    max_tokens = config["llm"]["max_tokens"]

    outputs_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Knowledge Base Query[/bold blue]")
    console.print(f"Question: [bold]{args.question}[/bold]\n")

    # ── Step 1: Read index ───────────────────────────────────────────────────
    index_content = read_index(wiki_dir)
    if not index_content.strip() or "none yet" in index_content:
        console.print(
            "[red]Wiki index is empty. Run `python tools/compile.py` first.[/red]"
        )
        sys.exit(1)

    client = get_anthropic_client()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── Step 2: Select relevant articles ────────────────────────────────────
    console.print("[bold]Step 1:[/bold] Identifying relevant articles...")
    select_response = llm_call_with_retry(
        client,
        _operation="query_select_articles",
        model=model,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": SELECT_ARTICLES_PROMPT.format(
                    question=args.question,
                    top_k=args.top_k,
                    index_content=index_content,
                ),
            }
        ],
    )

    try:
        selected_slugs = extract_json(select_response.content[0].text.strip())
        if not isinstance(selected_slugs, list):
            selected_slugs = []
    except ValueError:
        selected_slugs = []

    if not selected_slugs:
        console.print("[yellow]No relevant articles identified. Searching all concepts...[/yellow]")
        selected_slugs = get_concept_slugs(wiki_dir)[:args.top_k]

    console.print(f"  Reading {len(selected_slugs)} article(s): {', '.join(selected_slugs)}")

    # ── Step 3: Read articles and generate answer ────────────────────────────
    articles_content = read_articles(wiki_dir, selected_slugs)
    if not articles_content:
        console.print("[red]Could not read any relevant articles from wiki/.[/red]")
        sys.exit(1)

    console.print("\n[bold]Step 2:[/bold] Generating answer...")
    answer_response = llm_call_with_retry(
        client,
        _operation="query_answer",
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    question=args.question,
                    articles_content=articles_content,
                ),
            }
        ],
    )
    answer = answer_response.content[0].text.strip()

    # ── Step 4: Save answer to outputs/ ─────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_filename = f"answer-{timestamp}.md"
    output_path = outputs_dir / output_filename

    output_content = f"""# Query Answer

**Question:** {args.question}
**Date:** {today}
**Articles consulted:** {', '.join(f'[[{s}]]' for s in selected_slugs)}

---

{answer}
"""
    atomic_write(output_path, output_content)
    console.print(f"\n  Answer saved → [cyan]{output_path.relative_to(Path.cwd())}[/cyan]")

    # ── Step 5: Write-back derived insights ─────────────────────────────────
    changelog_entries = []
    if not args.no_writeback:
        console.print("\n[bold]Step 3:[/bold] Filing derived insights back into wiki...")
        concept_slugs = get_concept_slugs(wiki_dir)
        insights_response = llm_call_with_retry(
            client,
            _operation="query_derive_insights",
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": DERIVE_INSIGHTS_PROMPT.format(
                        question=args.question,
                        answer=answer[:3000],
                        concept_slugs=", ".join(concept_slugs),
                    ),
                }
            ],
        )
        try:
            insights = extract_json(insights_response.content[0].text.strip())
            if not isinstance(insights, list):
                insights = []
        except ValueError:
            insights = []

        changelog_entries = write_back_insights(wiki_dir, insights, args.question, today)

    # ── Step 6: Changelog ────────────────────────────────────────────────────
    all_changes = [f"Answered query: '{args.question[:80]}'", f"Saved → {output_filename}"]
    all_changes.extend(changelog_entries)
    append_changelog(wiki_dir, all_changes)

    # ── Print answer ─────────────────────────────────────────────────────────
    console.print("\n")
    console.rule("[bold green]Answer[/bold green]")
    console.print(Markdown(answer))
    console.rule()


if __name__ == "__main__":
    main()
