#!/usr/bin/env python3
"""
charts.py — Generate Matplotlib visualizations from wiki content

The LLM reads relevant wiki articles, extracts numerical/comparative data,
and produces executable Python chart code which is then run to create a PNG.

Usage:
    python tools/charts.py "comparison of model sizes"
    python tools/charts.py "key concepts and their relationships" --type network
    python tools/charts.py "attention head counts across architectures" --type bar
    python tools/charts.py "timeline of NLP milestones" --type timeline
    python tools/charts.py "concept frequency in sources" --type heatmap
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

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
)

CHART_TYPES = ["bar", "line", "pie", "scatter", "heatmap", "network", "timeline", "auto"]

# ── Prompts ───────────────────────────────────────────────────────────────────

SELECT_CHART_ARTICLES_PROMPT = """\
You are a data visualization assistant. A user wants a chart about a topic.
Given the knowledge base index, identify which articles contain the most
relevant quantitative, comparative, or relational data for this visualization.

Return a JSON array of article slugs (without .md extension).
Include at most {top_k} slugs. Return ONLY the JSON array. No prose.

Chart topic: {topic}
Chart type requested: {chart_type}

Knowledge Base Index:
---
{index_content}
"""

GENERATE_CHART_PROMPT = """\
You are a data visualization expert. Analyze the provided knowledge base articles
and generate Python matplotlib code to create a chart about the given topic.

Chart topic: {topic}
Chart type preference: {chart_type}

Instructions:
1. Extract all numerical data, comparisons, rankings, timelines, or relationships from the articles
2. If no numerical data exists, create a conceptual visualization (e.g., concept relationship network, frequency chart based on mention counts)
3. Write complete, runnable Python code using matplotlib (and optionally numpy, networkx for network graphs)
4. The chart must be informative, well-labeled, and publication-quality
5. Save the chart to: {output_path}
6. Use a clean style: plt.style.use('seaborn-v0_8-whitegrid') or 'ggplot'
7. Include a descriptive title, axis labels, and legend where appropriate
8. Use color thoughtfully — avoid rainbow palettes
9. Figure size: (12, 7) inches, dpi=150

IMPORTANT:
- Import only: matplotlib, numpy, networkx (if needed) — these are standard
- End with plt.tight_layout() and plt.savefig('{output_path}', dpi=150, bbox_inches='tight')
- Do NOT call plt.show()
- Do NOT use any external data files — all data must be hardcoded from the articles

Knowledge Base Articles:
---
{articles_content}

Return ONLY the Python code. No markdown fences, no explanation.
Start directly with import statements.
"""

CHART_SUMMARY_PROMPT = """\
In 2-3 sentences, describe what this chart shows and why it is useful for
understanding the topic "{topic}". Be specific about the data visualized.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_index(wiki_dir: Path) -> str:
    index_path = wiki_dir / "_index.md"
    if not index_path.exists():
        return ""
    return index_path.read_text(encoding="utf-8")


def read_articles(wiki_dir: Path, slugs: list[str]) -> str:
    parts = []
    for slug in slugs:
        for subdir in ("concepts", "categories", "summaries"):
            path = wiki_dir / subdir / f"{slug}.md"
            if path.exists():
                content = path.read_text(encoding="utf-8")
                parts.append(f"### [{slug}]\n{content}")
                break
    return "\n\n---\n\n".join(parts)


def clean_code(raw: str) -> str:
    """Strip markdown fences if the LLM wrapped the code anyway."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        # Remove first and last fence lines
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        raw = "\n".join(lines[start:end])
    return raw.strip()


def run_chart_code(code: str, output_path: Path) -> tuple[bool, str]:
    """
    Execute the generated chart code in a subprocess.
    Returns (success, error_message).
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix="kb_chart_",
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False, result.stderr[-2000:]
        if not output_path.exists():
            return False, f"Chart file not created at {output_path}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Chart generation timed out (60s)"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def file_chart_to_wiki(
    wiki_dir: Path,
    chart_path: Path,
    chart_summary: str,
    topic: str,
    slugs: list[str],
    today: str,
) -> list[str]:
    """File a reference to the generated chart back into concept articles."""
    changelog = []
    entry = (
        f"\n**Chart generated:** `{chart_path.name}`  \n"
        f"{chart_summary}  \n"
        f"*(filed: {today})*\n"
    )
    for slug in slugs[:2]:
        concept_path = wiki_dir / "concepts" / f"{slug}.md"
        if not concept_path.exists():
            continue
        content = concept_path.read_text(encoding="utf-8")
        derived_marker = "## Derived Insights"
        if derived_marker in content:
            idx = content.index(derived_marker) + len(derived_marker)
            updated = content[:idx] + "\n" + entry + content[idx:]
        else:
            updated = content + f"\n## Derived Insights\n{entry}\n"

        import re
        updated = re.sub(r"Last updated: \d{4}-\d{2}-\d{2}", f"Last updated: {today}", updated)
        atomic_write(concept_path, updated)
        changelog.append(f"Filed chart reference into [[{slug}]]")

    return changelog


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Matplotlib charts from wiki")
    parser.add_argument("topic", help="What to visualize (data question or topic)")
    parser.add_argument(
        "--type",
        choices=CHART_TYPES,
        default="auto",
        help="Chart type (default: auto — LLM decides)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max articles to use as source (default: 5)",
    )
    parser.add_argument(
        "--no-writeback",
        action="store_true",
        help="Skip filing chart reference back into wiki",
    )
    parser.add_argument(
        "--save-code",
        action="store_true",
        help="Also save the generated Python code alongside the chart",
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
    slug_topic = args.topic.lower().replace(" ", "-")[:40]
    chart_filename = f"chart-{slug_topic}-{timestamp}.png"
    output_path = outputs_dir / chart_filename

    console.rule("[bold blue]Chart Generator[/bold blue]")
    console.print(f"Topic: [bold]{args.topic}[/bold]")
    console.print(f"Chart type: {args.type}\n")

    # ── Step 1: Read index ───────────────────────────────────────────────────
    index_content = read_index(wiki_dir)
    if not index_content.strip() or "none yet" in index_content:
        console.print("[red]Wiki index is empty. Run `python tools/compile.py` first.[/red]")
        sys.exit(1)

    client = get_anthropic_client()

    # ── Step 2: Select relevant articles ────────────────────────────────────
    console.print("[bold]Step 1:[/bold] Identifying relevant articles...")
    select_response = llm_call_with_retry(
        client,
        _operation="charts_select_articles",
        model=model,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": SELECT_CHART_ARTICLES_PROMPT.format(
                topic=args.topic,
                chart_type=args.type,
                top_k=args.top_k,
                index_content=index_content,
            ),
        }],
    )
    try:
        selected_slugs = extract_json(select_response.content[0].text.strip())
        if not isinstance(selected_slugs, list):
            selected_slugs = []
    except ValueError:
        selected_slugs = []

    if not selected_slugs:
        concepts_dir = wiki_dir / "concepts"
        selected_slugs = [f.stem for f in sorted(concepts_dir.glob("*.md"))][:args.top_k]

    console.print(f"  Using {len(selected_slugs)} article(s): {', '.join(selected_slugs)}")

    # ── Step 3: Read articles ────────────────────────────────────────────────
    articles_content = read_articles(wiki_dir, selected_slugs)
    if not articles_content:
        console.print("[red]No articles found in wiki/.[/red]")
        sys.exit(1)

    # ── Step 4: Generate chart code ──────────────────────────────────────────
    console.print("\n[bold]Step 2:[/bold] Generating chart code...")
    code_response = llm_call_with_retry(
        client,
        _operation="charts_generate_code",
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": GENERATE_CHART_PROMPT.format(
                topic=args.topic,
                chart_type=args.type,
                output_path=str(output_path),
                articles_content=articles_content,
            ),
        }],
    )
    chart_code = clean_code(code_response.content[0].text.strip())

    # Optionally save the code
    if args.save_code:
        code_path = outputs_dir / f"chart-{slug_topic}-{timestamp}.py"
        atomic_write(code_path, chart_code)
        console.print(f"  Code saved → [cyan]{code_path}[/cyan]")

    # ── Step 5: Execute the code ─────────────────────────────────────────────
    console.print("\n[bold]Step 3:[/bold] Rendering chart...")
    success, error = run_chart_code(chart_code, output_path)

    if not success:
        console.print(f"[red]Chart rendering failed:[/red]\n{error}")
        # Save the code for debugging even if --save-code wasn't set
        debug_path = outputs_dir / f"chart-{slug_topic}-{timestamp}-FAILED.py"
        atomic_write(debug_path, chart_code)
        console.print(f"[yellow]Failed code saved for debugging → {debug_path}[/yellow]")
        sys.exit(1)

    console.print(f"  [green]✓[/green] Chart saved → [cyan]{output_path}[/cyan]")

    # ── Step 6: Generate chart summary ───────────────────────────────────────
    console.print("\n[bold]Step 4:[/bold] Writing chart summary...")
    summary_response = llm_call_with_retry(
        client,
        _operation="charts_summarize",
        model=model,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": CHART_SUMMARY_PROMPT.format(topic=args.topic),
        }],
    )
    chart_summary = summary_response.content[0].text.strip()

    # Save a companion markdown file
    companion_content = (
        f"# Chart: {args.topic}\n\n"
        f"**Generated:** {today}  \n"
        f"**Type:** {args.type}  \n"
        f"**Sources:** {', '.join(f'[[{s}]]' for s in selected_slugs)}\n\n"
        f"![{args.topic}]({chart_filename})\n\n"
        f"## Summary\n{chart_summary}\n"
    )
    companion_path = outputs_dir / f"chart-{slug_topic}-{timestamp}.md"
    atomic_write(companion_path, companion_content)
    console.print(f"  Companion .md → [cyan]{companion_path}[/cyan]")

    # ── Step 7: Write-back ───────────────────────────────────────────────────
    changelog_entries = [
        f"Generated chart: {chart_filename}",
        f"Topic: '{args.topic}'",
    ]
    if not args.no_writeback:
        console.print("\n[bold]Step 5:[/bold] Filing chart reference back into wiki...")
        wb_entries = file_chart_to_wiki(
            wiki_dir, output_path, chart_summary, args.topic, selected_slugs, today
        )
        changelog_entries.extend(wb_entries)
        for e in wb_entries:
            console.print(f"  [green]✓[/green] {e}")

    append_changelog(wiki_dir, changelog_entries)

    # ── Done ─────────────────────────────────────────────────────────────────
    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"\n[dim]{chart_summary}[/dim]")


if __name__ == "__main__":
    main()
