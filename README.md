# LLM-Powered Knowledge Base

A personal knowledge base system where an LLM reads your raw source material and automatically builds and maintains a structured wiki — extracting concepts, connecting ideas, and answering questions against the compiled knowledge.

## How It Works

**Layer 1 — `raw/`:** Drop any file here. PDFs, markdown, text, images, HTML. This is your dump folder. Never organized, never cleaned up.

**Layer 2 — `wiki/`:** The LLM reads everything in `raw/` and builds a structured wiki of concept articles, categories, and summaries. The human never edits this directly.

**Token efficiency:** Queries run against `wiki/` (compact, structured), never against the raw files. The system reads `_index.md` first, then fetches only the relevant articles.

## Folder Structure

```
/
├── raw/                    # Drop your files here
├── wiki/
│   ├── concepts/           # One .md article per concept
│   ├── categories/         # One .md article per category
│   ├── summaries/          # One .md summary per raw source
│   ├── _index.md           # Master index (auto-maintained)
│   └── _changelog.md       # Log of every LLM change
├── outputs/                # Query answers, lint reports
├── tools/
│   ├── utils.py            # Shared helpers
│   ├── compile.py          # Build/update wiki from raw/
│   ├── query.py            # Q&A against wiki
│   ├── search.py           # Full-text search
│   ├── lint.py             # Health checks
│   ├── ingest.py           # Add files/URLs to raw/
│   └── watch.py            # Auto-compile on file changes
├── config.yaml
├── .env                    # Your API key goes here
├── requirements.txt
└── usage.log               # Token usage log (auto-created)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Add files to raw/

```bash
cp ~/Downloads/paper.pdf raw/
cp ~/notes/*.md raw/
```

Or use the ingest tool (see below).

### 4. Compile

```bash
python tools/compile.py
```

This builds your entire `wiki/` from scratch (or incrementally updates it).

## Usage

### Compile — Build the wiki

```bash
# Incremental (only process new/changed files)
python tools/compile.py

# Full recompile (reprocess everything)
python tools/compile.py --full
```

The first run processes all files in `raw/`, extracts concepts, builds wiki articles, and generates `_index.md`.

Subsequent runs are fast — only new or modified files are re-summarized.

### Query — Ask questions

```bash
python tools/query.py "What are the key themes across all my sources?"
python tools/query.py "How does the attention mechanism work?"
python tools/query.py "Compare BERT and GPT architectures" --top-k 8
python tools/query.py "Summarize the main findings" --no-writeback
```

Answers are:
- Printed to the terminal
- Saved as timestamped `.md` files in `outputs/`
- Filed back into relevant concept articles as "Derived Insights"

### Search — Find articles

```bash
python tools/search.py "transformer attention"
python tools/search.py "BERT fine-tuning" --top-k 10
python tools/search.py "attention" --path concepts
```

Also importable in Python:
```python
from tools.search import search
from pathlib import Path
results = search("attention mechanism", wiki_dir=Path("wiki"), top_k=5)
for r in results:
    print(r["title"], r["score"], r["excerpt"])
```

### Lint — Health checks

```bash
# Generate report only
python tools/lint.py

# Generate report and auto-fix issues
python tools/lint.py --fix
```

The linter checks for:
- Terminology inconsistencies across articles
- Concepts mentioned but missing dedicated articles
- Missing connections between related articles
- Thin articles with insufficient content
- Broken wikilinks

Reports are saved to `outputs/lint-report-TIMESTAMP.md`.

### Ingest — Add files or URLs

```bash
# Add a file
python tools/ingest.py /path/to/paper.pdf
python tools/ingest.py ~/Downloads/notes.md

# Fetch a URL (converts HTML to markdown)
python tools/ingest.py https://arxiv.org/abs/1706.03762
python tools/ingest.py https://example.com/article

# Add without compiling immediately
python tools/ingest.py /path/to/file.txt --no-compile
```

### Watch — Auto-compile on changes

```bash
# Watch in foreground (Ctrl+C to stop)
python tools/watch.py

# Run as background daemon
python tools/watch.py --daemon

# Check daemon status
python tools/watch.py --status

# Stop daemon
python tools/watch.py --stop
```

Activity is logged to `watch.log`.

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
llm:
  model: claude-opus-4-5      # Model for all LLM calls
  max_tokens: 4096             # Max tokens per response

paths:
  raw: ./raw
  wiki: ./wiki
  outputs: ./outputs

compile:
  incremental: true            # Only process new/changed files
  auto_lint_after_compile: false

watch:
  enabled: false
  debounce_seconds: 5          # Wait N seconds after last change before compiling
```

## Obsidian Compatibility

The entire `wiki/` folder is Obsidian-compatible:
- All internal links use `[[wikilink]]` format
- `_index.md` serves as the vault home page
- Folder structure works as an Obsidian vault

To open in Obsidian: **Open vault** → select the `wiki/` folder.

## Wiki Article Format

Every concept article follows this structure:

```markdown
# Concept Name

## Definition
One paragraph definition.

## Key Points
- Point 1
- Point 2

## Sources
- [[source-file]] - what this source contributes

## Related Concepts
- [[related-concept]]

## Derived Insights
(Q&A answers filed back here over time)

---
Last updated: YYYY-MM-DD
Source count: N
```

## Supported File Types

| Extension | How it's processed |
|-----------|-------------------|
| `.md`, `.txt` | Read directly |
| `.pdf` | Text extracted with pypdf |
| `.html`, `.htm` | Text extracted with BeautifulSoup |
| `.png`, `.jpg`, `.jpeg` | Described using Claude vision API |

## Token Efficiency

The system is designed to minimize token usage:

- **Incremental compilation:** Files are only re-summarized if their content has changed (tracked by SHA-256 hash in `raw/.manifest.json`)
- **Selective querying:** Queries read `_index.md` first (small), then fetch only relevant articles
- **Smart updates:** Concept articles are only rewritten if new information actually changes them
- **Usage logging:** Every API call logs `input_tokens` and `output_tokens` to `usage.log`

## Files Never Manually Edited

These files are owned by the system — do not edit manually:
- `wiki/**` — all wiki content
- `raw/.manifest.json` — processing state
- `usage.log` — token log
- `watch.log` — watcher log

## Troubleshooting

**"ANTHROPIC_API_KEY not set"** — Edit `.env` and add your key.

**"No files found in raw/"** — Drop some files in `raw/` and try again.

**"Wiki index is empty"** — Run `python tools/compile.py` first.

**compile.py seems slow** — It calls the LLM for each file. Use `--full` only when necessary; the default incremental mode is much faster on repeat runs.

**Wikilinks broken in Obsidian** — Make sure you open `wiki/` as the vault root, not the repo root.
