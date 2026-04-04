#!/usr/bin/env python3
"""
lint.py — Health checks on the wiki knowledge base

Usage:
    python tools/lint.py                   # generate report only
    python tools/lint.py --fix             # auto-fix issues found
    python tools/lint.py --output path/to/report.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

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

LINT_ANALYSIS_PROMPT = """\
You are a knowledge base quality reviewer. Analyze the following concept articles
and identify issues. Be thorough and specific.

Check for:
1. TERMINOLOGY_INCONSISTENCY: Same concept referred to by different names across articles
2. MISSING_CONCEPT: A concept is mentioned/linked but has no dedicated article
3. MISSING_CONNECTION: Two articles are clearly related but don't link to each other
4. THIN_ARTICLE: An article has fewer than 3 key points or a very short definition
5. BROKEN_WIKILINK: A [[wikilink]] in one article points to a concept that doesn't exist

For each issue, return a JSON object with:
  - type: one of the types above
  - severity: "high", "medium", or "low"
  - affected_files: list of article slugs involved
  - description: clear description of the issue
  - suggested_fix: what should be done to fix it

Return ONLY a valid JSON array of issue objects. No prose.

Available concept slugs: {concept_slugs}

Articles:
---
{articles_content}
"""

FIX_TERMINOLOGY_PROMPT = """\
You are a knowledge base assistant. Fix a terminology inconsistency in the following article.
The issue: {description}

Current article:
---
{article_content}

Return ONLY the corrected markdown article. No preamble.
"""

FIX_MISSING_CONNECTION_PROMPT = """\
You are a knowledge base assistant. Add a missing connection to the following article.
The issue: {description}

Current article:
---
{article_content}

Add the missing link to the "Related Concepts" section and return the updated article.
Return ONLY the updated markdown article. No preamble.
"""

FIX_THIN_ARTICLE_PROMPT = """\
You are a knowledge base assistant. Expand a thin article with more key points.
The issue: {description}

Current article:
---
{article_content}

Add 3-5 additional key points to the "Key Points" section.
Update "Last updated" to today's date: {today}
Return ONLY the updated markdown article. No preamble.
"""

CREATE_MISSING_CONCEPT_PROMPT = """\
You are a knowledge base assistant. Create a new concept article for a concept that
is referenced in other articles but has no dedicated article.

Concept name: {concept_name}
Context (from other articles mentioning this concept):
{context}

Create a concept article following this EXACT format:

# {concept_name}

## Definition
[One paragraph definition]

## Key Points
- [Point 1]
- [Point 2]
- [Point 3]

## Sources
(Referenced from other articles — no direct source yet)

## Related Concepts
[List related concepts as [[slug]] links]

## Derived Insights
(Query answers will be filed back here over time)

---
Last updated: {today}
Source count: 0

Return ONLY the markdown article. No preamble.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_concept_articles(wiki_dir: Path) -> dict[str, str]:
    """Return {slug: content} for all concept articles."""
    concepts_dir = wiki_dir / "concepts"
    if not concepts_dir.exists():
        return {}
    return {f.stem: f.read_text(encoding="utf-8") for f in sorted(concepts_dir.glob("*.md"))}


def load_summary_articles(wiki_dir: Path) -> dict[str, str]:
    """Return {slug: content} for all summary articles."""
    summaries_dir = wiki_dir / "summaries"
    if not summaries_dir.exists():
        return {}
    return {f.stem: f.read_text(encoding="utf-8") for f in sorted(summaries_dir.glob("*.md"))}


def format_report(
    issues: list[dict],
    wiki_dir: Path,
    today: str,
    fixed_count: int = 0,
) -> str:
    """Format a lint report as markdown."""
    high = [i for i in issues if i.get("severity") == "high"]
    medium = [i for i in issues if i.get("severity") == "medium"]
    low = [i for i in issues if i.get("severity") == "low"]

    lines = [
        f"# Wiki Lint Report",
        f"",
        f"**Generated:** {today}  ",
        f"**Total issues:** {len(issues)}  ",
        f"**High severity:** {len(high)}  ",
        f"**Medium severity:** {len(medium)}  ",
        f"**Low severity:** {len(low)}  ",
    ]
    if fixed_count:
        lines.append(f"**Auto-fixed:** {fixed_count}  ")

    lines += ["", "---", ""]

    for severity, label in [("high", "High Severity"), ("medium", "Medium Severity"), ("low", "Low Severity")]:
        group = [i for i in issues if i.get("severity") == severity]
        if not group:
            continue
        lines.append(f"## {label} Issues\n")
        for issue in group:
            issue_type = issue.get("type", "UNKNOWN")
            desc = issue.get("description", "")
            files = issue.get("affected_files", [])
            fix = issue.get("suggested_fix", "")
            files_str = ", ".join(f"[[{f}]]" for f in files) if files else "N/A"
            lines += [
                f"### {issue_type}",
                f"**Affected:** {files_str}",
                f"**Issue:** {desc}",
                f"**Suggested fix:** {fix}",
                "",
            ]

    if not issues:
        lines += ["## No Issues Found", "", "The wiki is in good health! 🎉", ""]

    return "\n".join(lines)


def display_issues(issues: list[dict]) -> None:
    """Print issues in a rich table."""
    if not issues:
        console.print("[green]No issues found! Wiki is in good health.[/green]")
        return

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Severity", width=8)
    table.add_column("Type", width=25)
    table.add_column("Files", min_width=20)
    table.add_column("Description", min_width=40)

    severity_colors = {"high": "red", "medium": "yellow", "low": "blue"}
    for issue in issues:
        severity = issue.get("severity", "low")
        color = severity_colors.get(severity, "white")
        files = ", ".join(issue.get("affected_files", []))
        table.add_row(
            f"[{color}]{severity.upper()}[/{color}]",
            issue.get("type", "UNKNOWN"),
            files[:40],
            issue.get("description", "")[:80],
        )

    console.print(table)


def apply_fix(
    issue: dict,
    wiki_dir: Path,
    concepts: dict[str, str],
    client,
    model: str,
    max_tokens: int,
    today: str,
) -> tuple[bool, str]:
    """
    Attempt to auto-fix a single issue.
    Returns (success, changelog_entry).
    """
    issue_type = issue.get("type", "")
    affected = issue.get("affected_files", [])
    description = issue.get("description", "")

    if issue_type == "TERMINOLOGY_INCONSISTENCY" and affected:
        slug = affected[0]
        if slug not in concepts:
            return False, ""
        response = llm_call_with_retry(
            client,
            _operation="lint_fix_terminology",
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": FIX_TERMINOLOGY_PROMPT.format(
                    description=description,
                    article_content=concepts[slug],
                ),
            }],
        )
        updated = response.content[0].text.strip()
        atomic_write(wiki_dir / "concepts" / f"{slug}.md", updated)
        concepts[slug] = updated
        return True, f"Fixed terminology in [[{slug}]]"

    elif issue_type == "MISSING_CONNECTION" and len(affected) >= 2:
        slug = affected[0]
        if slug not in concepts:
            return False, ""
        response = llm_call_with_retry(
            client,
            _operation="lint_fix_connection",
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": FIX_MISSING_CONNECTION_PROMPT.format(
                    description=description,
                    article_content=concepts[slug],
                ),
            }],
        )
        updated = response.content[0].text.strip()
        atomic_write(wiki_dir / "concepts" / f"{slug}.md", updated)
        concepts[slug] = updated
        return True, f"Added missing connection in [[{slug}]]"

    elif issue_type == "THIN_ARTICLE" and affected:
        slug = affected[0]
        if slug not in concepts:
            return False, ""
        response = llm_call_with_retry(
            client,
            _operation="lint_fix_thin",
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": FIX_THIN_ARTICLE_PROMPT.format(
                    description=description,
                    article_content=concepts[slug],
                    today=today,
                ),
            }],
        )
        updated = response.content[0].text.strip()
        atomic_write(wiki_dir / "concepts" / f"{slug}.md", updated)
        concepts[slug] = updated
        return True, f"Expanded thin article [[{slug}]]"

    elif issue_type == "MISSING_CONCEPT" and affected:
        # The affected_files should contain the concept name to create
        concept_name = issue.get("description", "").split('"')[1] if '"' in issue.get("description", "") else affected[0]
        concept_name = concept_name.strip()
        slug = slugify(concept_name)

        # Gather context from other articles that mention this concept
        context_parts = []
        for s, content in concepts.items():
            if concept_name.lower() in content.lower() or slug in content.lower():
                context_parts.append(f"[{s}]: ...{_find_context(content, concept_name)}...")
                if len(context_parts) >= 3:
                    break
        context = "\n\n".join(context_parts) if context_parts else "(mentioned in other articles)"

        response = llm_call_with_retry(
            client,
            _operation="lint_create_concept",
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": CREATE_MISSING_CONCEPT_PROMPT.format(
                    concept_name=concept_name,
                    context=context,
                    today=today,
                ),
            }],
        )
        new_article = response.content[0].text.strip()
        new_path = wiki_dir / "concepts" / f"{slug}.md"
        atomic_write(new_path, new_article)
        concepts[slug] = new_article
        return True, f"Created missing concept article [[{slug}]]"

    elif issue_type == "BROKEN_WIKILINK" and affected:
        # Fix empty [[]] wikilinks by removing them with a simple regex
        import re
        fixed_any = False
        fixed_slugs = []
        for slug in affected:
            if slug not in concepts:
                continue
            content = concepts[slug]
            # Remove empty wikilinks [[]] or [[ ]]
            cleaned = re.sub(r'\[\[\s*\]\]', '', content)
            # Remove wikilinks pointing to non-existent concepts
            def remove_dead_link(m):
                target = m.group(1).strip()
                if target and target not in concepts:
                    return target  # replace [[missing]] with just the text
                return m.group(0)  # keep valid links as-is
            cleaned = re.sub(r'\[\[([^\]]*)\]\]', remove_dead_link, cleaned)
            if cleaned != content:
                atomic_write(wiki_dir / "concepts" / f"{slug}.md", cleaned)
                concepts[slug] = cleaned
                fixed_any = True
                fixed_slugs.append(slug)
        if fixed_any:
            return True, f"Fixed broken wikilinks in [[{']], [['.join(fixed_slugs)}]]"

    return False, ""


def _find_context(content: str, term: str, window: int = 150) -> str:
    """Find context around a term in text."""
    idx = content.lower().find(term.lower())
    if idx == -1:
        return content[:window]
    start = max(0, idx - 50)
    end = min(len(content), idx + window)
    return content[start:end]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Lint the wiki knowledge base")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues found")
    parser.add_argument(
        "--output",
        help="Custom output path for the lint report (default: outputs/lint-report-TIMESTAMP.md)",
    )
    args = parser.parse_args()

    config = load_config()
    paths = resolve_paths(config)
    wiki_dir = paths["wiki"]
    outputs_dir = paths["outputs"]
    model = config["llm"]["model"]
    max_tokens = config["llm"]["max_tokens"]

    outputs_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    console.rule("[bold blue]Wiki Linter[/bold blue]")

    concepts = load_concept_articles(wiki_dir)
    if not concepts:
        console.print("[yellow]No concept articles found. Run `python tools/compile.py` first.[/yellow]")
        return

    console.print(f"Analyzing [bold]{len(concepts)}[/bold] concept article(s)...\n")

    client = get_anthropic_client()

    # Prepare articles content for LLM (cap to avoid token overload)
    articles_parts = []
    for slug, content in concepts.items():
        articles_parts.append(f"### {slug}\n{content[:2000]}")
    articles_content = "\n\n---\n\n".join(articles_parts)

    concept_slugs = list(concepts.keys())

    # ── Run lint analysis ────────────────────────────────────────────────────
    console.print("[bold]Step 1:[/bold] Running LLM analysis...")
    response = llm_call_with_retry(
        client,
        _operation="lint_analysis",
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": LINT_ANALYSIS_PROMPT.format(
                concept_slugs=", ".join(concept_slugs),
                articles_content=articles_content,
            ),
        }],
    )

    try:
        issues = extract_json(response.content[0].text.strip())
        if not isinstance(issues, list):
            issues = []
    except ValueError as e:
        console.print(f"[red]Failed to parse lint results: {e}[/red]")
        issues = []

    console.print(f"Found [bold]{len(issues)}[/bold] issue(s)\n")
    display_issues(issues)

    # ── Auto-fix ─────────────────────────────────────────────────────────────
    fixed_count = 0
    changelog_entries = []

    if args.fix and issues:
        console.print(f"\n[bold]Step 2:[/bold] Auto-fixing {len(issues)} issue(s)...")
        for issue in issues:
            try:
                success, entry = apply_fix(
                    issue, wiki_dir, concepts, client, model, max_tokens, today
                )
                if success:
                    fixed_count += 1
                    changelog_entries.append(entry)
                    console.print(f"  [green]✓[/green] {entry}")
                else:
                    console.print(
                        f"  [dim]~ Could not auto-fix: {issue.get('type')}[/dim]"
                    )
            except Exception as e:
                console.print(f"  [red]Error fixing {issue.get('type')}: {e}[/red]")

        console.print(f"\nAuto-fixed [bold]{fixed_count}[/bold] issue(s)")

    # ── Write report ─────────────────────────────────────────────────────────
    report_content = format_report(issues, wiki_dir, today, fixed_count)
    report_path = Path(args.output) if args.output else outputs_dir / f"lint-report-{timestamp}.md"
    atomic_write(report_path, report_content)

    console.print(f"\nReport saved → [cyan]{report_path}[/cyan]")

    # ── Changelog ────────────────────────────────────────────────────────────
    all_changes = [f"Ran lint — found {len(issues)} issue(s), fixed {fixed_count}"]
    all_changes.extend(changelog_entries)
    append_changelog(wiki_dir, all_changes)

    # ── Summary ──────────────────────────────────────────────────────────────
    high_count = sum(1 for i in issues if i.get("severity") == "high")
    if high_count:
        console.print(f"\n[red]⚠ {high_count} high-severity issue(s) need attention[/red]")
    elif issues:
        console.print(f"\n[yellow]✓ No critical issues. {len(issues)} minor issue(s) found.[/yellow]")
    else:
        console.print("\n[green]✓ Wiki is in good health![/green]")


if __name__ == "__main__":
    main()
