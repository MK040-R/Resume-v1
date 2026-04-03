"""
Shared utilities for the LLM Knowledge Base system.
All other tools import from here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import yaml
from dotenv import load_dotenv
from rich.console import Console

console = Console()

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.resolve()


def load_config() -> dict:
    """Read config.yaml from the repo root."""
    config_path = REPO_ROOT / "config.yaml"
    with config_path.open() as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict) -> dict[str, Path]:
    """Return absolute Path objects for raw, wiki, outputs."""
    base = REPO_ROOT
    return {
        "raw": (base / config["paths"]["raw"]).resolve(),
        "wiki": (base / config["paths"]["wiki"]).resolve(),
        "outputs": (base / config["paths"]["outputs"]).resolve(),
    }


# ── Anthropic client ──────────────────────────────────────────────────────────

def get_anthropic_client() -> anthropic.Anthropic:
    """Create and return an Anthropic client, loading .env if needed."""
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Edit .env and add your key."
        )
    return anthropic.Anthropic(api_key=api_key)


# ── Token usage logging ───────────────────────────────────────────────────────

def log_token_usage(
    operation: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    source_file: str | None = None,
) -> None:
    """Append a token usage line to usage.log in the repo root."""
    log_path = REPO_ROOT / "usage.log"
    now = datetime.now(timezone.utc).isoformat()
    extra = f" | file={source_file}" if source_file else ""
    line = (
        f"{now} | op={operation} | model={model}"
        f" | in={input_tokens} | out={output_tokens}{extra}\n"
    )
    with log_path.open("a") as f:
        f.write(line)


# ── Atomic file write ─────────────────────────────────────────────────────────

def atomic_write(path: Path, content: str) -> None:
    """Write content to a .tmp file then rename — prevents partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── LLM call with retry ───────────────────────────────────────────────────────

def llm_call_with_retry(
    client: anthropic.Anthropic,
    max_retries: int = 3,
    **kwargs: Any,
) -> anthropic.types.Message:
    """
    Call client.messages.create(**kwargs) with exponential back-off retry.
    Logs token usage automatically.
    """
    operation = kwargs.pop("_operation", "llm_call")
    delay = 2.0
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(**kwargs)
            log_token_usage(
                operation=operation,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            return response
        except anthropic.RateLimitError as exc:
            last_exc = exc
            if attempt < max_retries:
                console.print(
                    f"[yellow]Rate limit hit — retrying in {delay:.0f}s "
                    f"(attempt {attempt + 1}/{max_retries})[/yellow]"
                )
                time.sleep(delay)
                delay *= 2
        except anthropic.APIStatusError as exc:
            # 5xx errors are retryable; 4xx (except 429) are not
            if exc.status_code >= 500 and attempt < max_retries:
                last_exc = exc
                console.print(
                    f"[yellow]Server error {exc.status_code} — retrying in "
                    f"{delay:.0f}s[/yellow]"
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise

    raise last_exc  # type: ignore[misc]


# ── File hashing ──────────────────────────────────────────────────────────────

def file_hash(filepath: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Manifest (raw/.manifest.json) ─────────────────────────────────────────────

def load_manifest(raw_dir: Path) -> dict:
    """Load the processing manifest, creating an empty one if absent."""
    path = raw_dir / ".manifest.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {"files": {}}


def save_manifest(raw_dir: Path, manifest: dict) -> None:
    """Atomically save the manifest."""
    path = raw_dir / ".manifest.json"
    atomic_write(path, json.dumps(manifest, indent=2))


# ── Text extraction ───────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".html", ".htm", ".png", ".jpg", ".jpeg"}


def extract_text(filepath: Path, client: anthropic.Anthropic | None = None) -> str:
    """
    Extract text content from a file based on its extension.
    For images, `client` is required to use the vision API.
    Returns a UTF-8 string.
    """
    ext = filepath.suffix.lower()

    if ext in {".md", ".txt"}:
        return filepath.read_text(encoding="utf-8", errors="replace")

    if ext == ".pdf":
        return _extract_pdf(filepath)

    if ext in {".html", ".htm"}:
        return _extract_html(filepath)

    if ext in {".png", ".jpg", ".jpeg"}:
        if client is None:
            raise ValueError("Anthropic client required for image extraction")
        return _extract_image(filepath, client)

    raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(filepath: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")

    reader = PdfReader(str(filepath))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_html(filepath: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")

    raw = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    # Remove scripts and styles
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _extract_image(filepath: Path, client: anthropic.Anthropic) -> str:
    """Use Claude vision to describe an image."""
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    media_type = media_types[filepath.suffix.lower()]
    image_data = base64.standard_b64encode(filepath.read_bytes()).decode("utf-8")

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail. Include all visible text, "
                            "diagrams, charts, key concepts, and any technical information. "
                            "Format your response as structured markdown."
                        ),
                    },
                ],
            }
        ],
    )
    log_token_usage(
        operation="image_extraction",
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        source_file=filepath.name,
    )
    return response.content[0].text


# ── Changelog helper ──────────────────────────────────────────────────────────

def append_changelog(wiki_dir: Path, entries: list[str]) -> None:
    """Append timestamped entries to wiki/_changelog.md."""
    if not entries:
        return
    changelog_path = wiki_dir / "_changelog.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"\n## {now}\n"]
    for entry in entries:
        lines.append(f"- {entry}")
    text = "\n".join(lines) + "\n"
    with changelog_path.open("a") as f:
        f.write(text)


# ── Slug helper ───────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """Convert a concept name to a filesystem-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


# ── JSON extraction from LLM responses ───────────────────────────────────────

def extract_json(text: str) -> Any:
    """
    Extract JSON from an LLM response that may contain markdown fences
    or surrounding prose.
    """
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find a JSON block in markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first [ or { and parse from there
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx = text.find(start_char)
        if idx != -1:
            # Find matching close
            depth = 0
            for i, ch in enumerate(text[idx:], idx):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[idx : i + 1])
                        except json.JSONDecodeError:
                            break

    raise ValueError(f"Could not extract JSON from response:\n{text[:500]}")
