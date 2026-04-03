#!/usr/bin/env python3
"""
slides.py — Generate Marp-format slide decks from wiki content

Marp (https://marp.app) renders markdown as presentation slides.
Output files open directly in VS Code with the Marp extension,
or convert via: npx @marp-team/marp-cli outputs/slides-*.md --pdf

Usage:
    python tools/slides.py "Transformer Architecture"
    python tools/slides.py "attention mechanism" --slides 12
    python tools/slides.py "BERT and fine-tuning" --theme default --no-writeback
    python tools/slides.py "NLP overview" --from-category nlp
"""

from __future__ import annotations

import argparse
import sys
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

# ── Prompts ───────────────────────────────────────────────────────────────────

SELECT_SLIDES_ARTICLES_PROMPT = """\
You are a presentation assistant. A user wants a slide deck on a topic.
Given the knowledge base index, identify which concept and category articles
contain the most relevant content for building this presentation.

Return a JSON array of article slugs (filenames without .md extension).
Include at most {top_k} slugs. Order by relevance, most relevant first.
Return ONLY the JSON array. No prose.

Topic: {topic}

Knowledge Base Index:
---
{index_content}
"""

GENERATE_SLIDES_PROMPT = """\
You are a presentation designer. Create a professional Marp slide deck on the given topic
using ONLY the knowledge base articles provided.

Requirements:
- Target exactly {num_slides} slides (±2 is acceptable)
- First slide: title + one-sentence subtitle
- Second slide: agenda/outline
- Content slides: one key idea per slide, use bullet points (max 5 bullets per slide)
- Last slide: "Key Takeaways" summary (3-5 bullets)
- Use [[concept-name]] notation when referencing wiki concepts
- Include speaker notes after each slide using HTML comment syntax: <!-- Note: ... -->

Marp formatting rules:
- Start with YAML front matter (already provided — do NOT add it again)
- Separate slides with ---
- Use # for slide titles
- Use **bold** for emphasis
- Use > blockquote for important quotes or definitions
- Use `code` for technical terms

Topic: {topic}

Knowledge Base Articles:
---
{articles_content}

Output ONLY the slide content (no front matter — it will be prepended automatically).
Start directly with the first slide title.
"""

MARP_FRONT_MATTER = """\
---
marp: true
theme: {theme}
paginate: true
backgroundColor: #fefefe
style: |
  section {{
    font-family: 'Segoe UI', sans-serif;
    font-size: 28px;
  }}
  h1 {{
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
  }}
  strong {{
    color: #2980b9;
  }}
  blockquote {{
    border-left: 4px solid #3498db;
    padding-left: 16px;
    color: #555;
    font-style: italic;
  }}
---
"""

MARP_THEMES = {"default", "gaia", "uncover"}


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


def file_back_to_wiki(
    wiki_dir: Path,
    slides_path: Path,
    topic: str,
    slugs: list[str],
    today: str,
) -> list[str]:
    """Add a reference to the generated slide deck in relevant concept articles."""
    changelog = []
    entry = (
        f"\n**Slide deck generated:** [[{slides_path.name}]]  \n"
        f"*Topic: {topic} — filed: {today}*\n"
    )
    for slug in slugs[:3]:  # Only top 3 most relevant concepts
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
        changelog.append(f"Filed slide deck reference into [[{slug}]]")

    return changelog


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Marp slide decks from wiki")
    parser.add_argument("topic", help="Topic or question for the slide deck")
    parser.add_argument(
        "--slides",
        type=int,
        default=10,
        help="Target number of slides (default: 10)",
    )
    parser.add_argument(
        "--theme",
        choices=list(MARP_THEMES),
        default="default",
        help="Marp theme (default: default)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Max articles to use as source (default: 6)",
    )
    parser.add_argument(
        "--no-writeback",
        action="store_true",
        help="Skip filing slide reference back into wiki",
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

    console.rule("[bold blue]Slide Deck Generator[/bold blue]")
    console.print(f"Topic: [bold]{args.topic}[/bold]")
    console.print(f"Target slides: {args.slides} | Theme: {args.theme}\n")

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
        _operation="slides_select_articles",
        model=model,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": SELECT_SLIDES_ARTICLES_PROMPT.format(
                topic=args.topic,
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
        console.print("[yellow]Could not auto-select articles. Using all concepts.[/yellow]")
        concepts_dir = wiki_dir / "concepts"
        selected_slugs = [f.stem for f in sorted(concepts_dir.glob("*.md"))][:args.top_k]

    console.print(f"  Using {len(selected_slugs)} article(s): {', '.join(selected_slugs)}")

    # ── Step 3: Read articles ────────────────────────────────────────────────
    articles_content = read_articles(wiki_dir, selected_slugs)
    if not articles_content:
        console.print("[red]No articles found. Check that wiki/ is populated.[/red]")
        sys.exit(1)

    # ── Step 4: Generate slides ──────────────────────────────────────────────
    console.print("\n[bold]Step 2:[/bold] Generating slide content...")
    slides_response = llm_call_with_retry(
        client,
        _operation="slides_generate",
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": GENERATE_SLIDES_PROMPT.format(
                topic=args.topic,
                num_slides=args.slides,
                articles_content=articles_content,
            ),
        }],
    )
    slides_content = slides_response.content[0].text.strip()

    # ── Step 5: Assemble final Marp document ─────────────────────────────────
    front_matter = MARP_FRONT_MATTER.format(theme=args.theme)
    metadata_comment = (
        f"<!-- Generated by knowledge base slides.py -->\n"
        f"<!-- Topic: {args.topic} -->\n"
        f"<!-- Date: {today} -->\n"
        f"<!-- Sources: {', '.join(selected_slugs)} -->\n\n"
    )
    full_deck = front_matter + metadata_comment + slides_content

    # ── Step 6: Save ─────────────────────────────────────────────────────────
    slug_topic = args.topic.lower().replace(" ", "-")[:40]
    output_filename = f"slides-{slug_topic}-{timestamp}.md"
    output_path = outputs_dir / output_filename
    atomic_write(output_path, full_deck)

    console.print(f"\n  Slides saved → [cyan]{output_path}[/cyan]")

    # ── Step 7: Write-back ───────────────────────────────────────────────────
    changelog_entries = [
        f"Generated slide deck: {output_filename}",
        f"Topic: '{args.topic}' | Sources: {', '.join(selected_slugs[:3])}",
    ]
    if not args.no_writeback:
        console.print("\n[bold]Step 3:[/bold] Filing slide reference back into wiki...")
        wb_entries = file_back_to_wiki(
            wiki_dir, output_path, args.topic, selected_slugs, today
        )
        changelog_entries.extend(wb_entries)
        for e in wb_entries:
            console.print(f"  [green]✓[/green] {e}")

    append_changelog(wiki_dir, changelog_entries)

    # ── Summary ──────────────────────────────────────────────────────────────
    slide_count = slides_content.count("\n---\n") + 1
    console.print(f"\n[bold green]Done![/bold green] ~{slide_count} slides generated.")
    console.print("\n[dim]To render:[/dim]")
    console.print(f"  [cyan]npx @marp-team/marp-cli {output_path} --pdf[/cyan]")
    console.print(f"  [cyan]npx @marp-team/marp-cli {output_path} --html[/cyan]")
    console.print("  Or open in VS Code with the Marp extension installed.")


if __name__ == "__main__":
    main()
