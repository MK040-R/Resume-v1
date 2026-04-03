#!/usr/bin/env python3
"""
search.py — Full-text search over wiki/

Importable API:
    from tools.search import search
    results = search("transformer attention", wiki_dir=Path("wiki"), top_k=5)

CLI usage:
    python tools/search.py "transformer attention"
    python tools/search.py "BERT fine-tuning" --top-k 10
    python tools/search.py "attention" --path wiki/concepts
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from utils import console, load_config, resolve_paths

from rich.table import Table
from rich.text import Text


# ── TF-IDF search implementation ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def build_index(files: list[Path]) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """
    Build a TF-IDF index from a list of markdown files.
    Returns:
      - tf_index: {filepath_str: {term: tf_score}}
      - contents: {filepath_str: raw_text}
    """
    tf_index: dict[str, dict[str, float]] = {}
    contents: dict[str, str] = {}

    for filepath in files:
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        contents[str(filepath)] = text
        tokens = tokenize(text)
        if not tokens:
            continue
        freq: dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        total = len(tokens)
        tf_index[str(filepath)] = {term: count / total for term, count in freq.items()}

    return tf_index, contents


def compute_idf(tf_index: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute IDF scores for all terms across all documents."""
    n_docs = len(tf_index)
    if n_docs == 0:
        return {}
    doc_freq: dict[str, int] = {}
    for tf_scores in tf_index.values():
        for term in tf_scores:
            doc_freq[term] = doc_freq.get(term, 0) + 1
    return {
        term: math.log((n_docs + 1) / (df + 1)) + 1
        for term, df in doc_freq.items()
    }


def score_document(
    query_terms: list[str],
    tf_scores: dict[str, float],
    idf_scores: dict[str, float],
) -> float:
    """Compute TF-IDF score for a document against query terms."""
    score = 0.0
    for term in query_terms:
        tf = tf_scores.get(term, 0.0)
        idf = idf_scores.get(term, 0.0)
        score += tf * idf
    return score


def extract_excerpt(text: str, query_terms: list[str], excerpt_len: int = 200) -> str:
    """
    Find the paragraph in the text most relevant to query_terms and return
    a short excerpt around the best match.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text[:excerpt_len]

    best_para = ""
    best_score = -1
    for para in paragraphs:
        para_tokens = set(tokenize(para))
        score = sum(1 for t in query_terms if t in para_tokens)
        if score > best_score:
            best_score = score
            best_para = para

    # Strip markdown headings from excerpt
    best_para = re.sub(r"^#{1,6}\s+", "", best_para, flags=re.MULTILINE)
    best_para = re.sub(r"\[\[([^\]]+)\]\]", r"\1", best_para)  # unwrap wikilinks

    if len(best_para) <= excerpt_len:
        return best_para
    # Find a good truncation point near the first query term hit
    for term in query_terms:
        idx = best_para.lower().find(term)
        if idx != -1:
            start = max(0, idx - 40)
            end = min(len(best_para), start + excerpt_len)
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(best_para) else ""
            return prefix + best_para[start:end] + suffix
    return best_para[:excerpt_len] + "..."


def get_title(text: str, filepath: Path) -> str:
    """Extract the first H1 heading from a markdown file."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return filepath.stem.replace("-", " ").title()


def collect_wiki_files(wiki_dir: Path, subpath: str | None = None) -> list[Path]:
    """Collect all .md files from wiki/ or a specific subdirectory."""
    search_root = wiki_dir if subpath is None else wiki_dir / subpath
    if not search_root.exists():
        return []
    return sorted(search_root.rglob("*.md"))


# ── Public API ────────────────────────────────────────────────────────────────

def search(
    query: str,
    wiki_dir: Path,
    top_k: int = 5,
    subpath: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search wiki/ for relevant articles.

    Args:
        query: Natural language or keyword query string
        wiki_dir: Path to the wiki/ directory
        top_k: Maximum number of results to return
        subpath: Optional subfolder within wiki/ to restrict search
                 (e.g., "concepts", "categories")

    Returns:
        List of dicts with keys: file, title, excerpt, score, slug
        Ordered by relevance (highest first).
    """
    files = collect_wiki_files(wiki_dir, subpath)
    if not files:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return []

    tf_index, contents = build_index(files)
    idf_scores = compute_idf(tf_index)

    scored = []
    for filepath in files:
        key = str(filepath)
        tf_scores = tf_index.get(key, {})
        score = score_document(query_terms, tf_scores, idf_scores)
        if score > 0:
            text = contents.get(key, "")
            scored.append({
                "file": str(filepath),
                "slug": filepath.stem,
                "title": get_title(text, filepath),
                "excerpt": extract_excerpt(text, query_terms),
                "score": round(score, 4),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Search the wiki knowledge base")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--path",
        default=None,
        help="Restrict search to a wiki subdirectory (e.g., concepts, categories, summaries)",
    )
    parser.add_argument(
        "--no-excerpt",
        action="store_true",
        help="Only show titles and scores, not excerpts",
    )
    args = parser.parse_args()

    config = load_config()
    paths = resolve_paths(config)
    wiki_dir = paths["wiki"]

    console.rule("[bold blue]Wiki Search[/bold blue]")
    console.print(f"Query: [bold]{args.query}[/bold]")
    if args.path:
        console.print(f"Scope: [dim]{args.path}[/dim]")
    console.print()

    results = search(
        query=args.query,
        wiki_dir=wiki_dir,
        top_k=args.top_k,
        subpath=args.path,
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="cyan", min_width=20)
    table.add_column("Slug", style="dim", min_width=15)
    table.add_column("Score", justify="right", style="green", width=8)
    if not args.no_excerpt:
        table.add_column("Excerpt", min_width=40)

    for i, result in enumerate(results, 1):
        row = [
            str(i),
            result["title"],
            result["slug"],
            str(result["score"]),
        ]
        if not args.no_excerpt:
            row.append(result["excerpt"])
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[dim]Found {len(results)} result(s)[/dim]")


if __name__ == "__main__":
    main()
