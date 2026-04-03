#!/usr/bin/env python3
"""
ingest.py — Add new files to raw/ and trigger incremental compile

Usage:
    python tools/ingest.py https://example.com/article
    python tools/ingest.py /path/to/paper.pdf
    python tools/ingest.py /path/to/notes.md
    python tools/ingest.py https://arxiv.org/abs/2310.06825 --no-compile
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import console, load_config, resolve_paths, SUPPORTED_EXTENSIONS


def is_url(s: str) -> bool:
    """Return True if the string looks like an HTTP/HTTPS URL."""
    return s.startswith("http://") or s.startswith("https://")


def url_to_slug(url: str) -> str:
    """Convert a URL to a reasonable filename slug."""
    parsed = urllib.parse.urlparse(url)
    # Combine host + path, strip query strings
    path = parsed.netloc + parsed.path
    # Remove common extensions
    path = re.sub(r"\.(html?|php|aspx?)$", "", path, flags=re.IGNORECASE)
    # Replace non-alphanumeric chars with hyphens
    slug = re.sub(r"[^\w]+", "-", path).strip("-").lower()
    # Truncate to 80 chars
    return slug[:80] or "page"


def fetch_url(url: str) -> str:
    """
    Fetch a URL and convert its HTML content to clean markdown.
    Returns the markdown string.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests not installed. Run: pip install requests")

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")

    try:
        import html2text
    except ImportError:
        raise ImportError("html2text not installed. Run: pip install html2text")

    console.print(f"  Fetching [cyan]{url}[/cyan] ...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "pdf" in content_type:
        raise ValueError(
            f"URL points to a PDF. Download it and use: "
            f"python tools/ingest.py /path/to/file.pdf"
        )

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    # Remove noisy elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "advertisement", "iframe", "noscript"]):
        tag.decompose()

    # Try to find main content area
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main|article", re.I))
        or soup.find(class_=re.compile(r"content|main|article|post", re.I))
        or soup.body
    )

    html_content = str(main) if main else response.text

    # Convert to markdown
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0  # Don't wrap lines
    h.ignore_emphasis = False
    markdown = h.handle(html_content)

    # Clean up excessive blank lines
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = markdown.strip()

    # Prepend metadata header
    header = (
        f"# {title}\n\n"
        f"> Source: {url}\n"
        f"> Ingested by knowledge base system\n\n"
        f"---\n\n"
    )
    return header + markdown


def ingest_file(filepath: Path, raw_dir: Path) -> Path:
    """Copy a local file into raw/. Returns the destination path."""
    ext = filepath.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {supported}"
        )
    dest = raw_dir / filepath.name
    if dest.exists():
        console.print(
            f"  [yellow]Warning: {filepath.name} already exists in raw/. Overwriting.[/yellow]"
        )
    shutil.copy2(str(filepath), str(dest))
    return dest


def ingest_url(url: str, raw_dir: Path) -> Path:
    """Fetch a URL and save its content as markdown in raw/. Returns the destination path."""
    markdown = fetch_url(url)
    slug = url_to_slug(url)
    dest = raw_dir / f"{slug}.md"
    if dest.exists():
        console.print(
            f"  [yellow]Warning: {dest.name} already exists in raw/. Overwriting.[/yellow]"
        )
    dest.write_text(markdown, encoding="utf-8")
    return dest


def trigger_compile(repo_root: Path, full: bool = False) -> bool:
    """Run compile.py incrementally. Returns True on success."""
    compile_script = repo_root / "tools" / "compile.py"
    cmd = [sys.executable, str(compile_script)]
    if full:
        cmd.append("--full")

    console.print("\n[bold]Triggering incremental compile...[/bold]")
    result = subprocess.run(cmd, cwd=str(repo_root))
    return result.returncode == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a file or URL into the knowledge base"
    )
    parser.add_argument(
        "source",
        help="File path or URL to ingest",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip running compile.py after ingesting",
    )
    parser.add_argument(
        "--full-compile",
        action="store_true",
        help="Run full (non-incremental) compile after ingesting",
    )
    args = parser.parse_args()

    config = load_config()
    paths = resolve_paths(config)
    raw_dir = paths["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Knowledge Base Ingest[/bold blue]")

    try:
        if is_url(args.source):
            dest = ingest_url(args.source, raw_dir)
            console.print(f"[green]✓[/green] Fetched URL → [cyan]raw/{dest.name}[/cyan]")
        else:
            src_path = Path(args.source).expanduser().resolve()
            if not src_path.exists():
                console.print(f"[red]File not found: {src_path}[/red]")
                sys.exit(1)
            dest = ingest_file(src_path, raw_dir)
            console.print(f"[green]✓[/green] Copied file → [cyan]raw/{dest.name}[/cyan]")

        if not args.no_compile:
            from utils import REPO_ROOT
            success = trigger_compile(REPO_ROOT, full=args.full_compile)
            if success:
                console.print("\n[green]Ingest and compile complete.[/green]")
            else:
                console.print("\n[yellow]Ingest complete but compile had errors.[/yellow]")
        else:
            console.print(
                "\n[dim]Skipping compile. Run `python tools/compile.py` when ready.[/dim]"
            )

    except Exception as e:
        console.print(f"[red]Ingest failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
