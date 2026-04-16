# AGENTS.md — Johdatus Koneoppimiseen

## Scope

This is a **Material for MkDocs** lesson-material repository for **Johdatus Koneoppimiseen**. Content is authored primarily in **Finnish**.

The agent's role is **content co-author**, not autonomous refactorer.

## Quick rules

- Edit lesson content primarily in `docs/`.
- Write `docs/**/*.md` in **Finnish** unless the user explicitly requests another language.
- Write `notebooks/**/*.py` in English.
- Preserve the author's voice; prefer minimal, targeted edits.
- Do **not** reorganize files, directories, or navigation unless explicitly approved.
- Do **not** run `doc-flesh` commands.

## Read-only files

MUST NOT modify these files unless the user explicitly requests it:

- `mkdocs.yml`
- `LICENSE`
- `pyproject.toml`
- `README.md`
- `AGENTS.md`

- `docs/exercises.md` — auto-generated; see **Pre-commit and exercise aggregation**


These are controlled by centralized tooling:

- Config: `~/.config/doc-flesh`
- Source: `gh:sourander/doc-flesh/`
- CLI: `doc-flesh` (installed as a `uv` tool)

MUST NOT run `doc-flesh`. The CLI is for human use only and may be destructive.

## Repository structure

```text
siteinfo.json          # repo marker and metadata
docs/                  # Markdown lesson/theory content (EDIT HERE)
notebooks/             # Python exercises and notebooks (if present)
notebooks/solutions/   # encrypted solutions (git-crypt)
notebooks/git-lfs/     # binary exercise assets (Git LFS)
mkdocs.yml             # READ-ONLY
README.md              # READ-ONLY
LICENSE                # READ-ONLY
pyproject.toml         # READ-ONLY
```

## Authoring rules

1. MUST write theory material in `docs/` using Markdown.
2. MUST write in **Finnish** unless the user explicitly requests another language.
3. MUST NOT translate existing Finnish content into English unless requested.
4. MUST NOT invent references or citations. If unsure, ask.
5. MUST NOT add filler prose, generic tutorial phrasing, or unnecessary introductions.
6. SHOULD maintain terminology consistency within the course.
7. MUST preserve existing YAML front matter unless the user explicitly requests changes.
8. MUST respect the `priority` front matter system.


## Exercise content

- MUST NOT edit `docs/exercises.md` directly.
- To change exercise content, edit the source Markdown file(s) in `docs/`.
- See `.pre-commit-guide.md` for hook details.


## Ask first

The agent MUST ask before:

- changing repository structure or navigation
- changing existing `priority` values
- creating new top-level content files when placement is unclear
- making broad stylistic rewrites
- modifying anything outside `docs/` unless explicitly requested

## Writing style

- Language: Finnish (default)
- Tone: explanatory, teacherly
- Approach: concept-first, then practical examples
- Clarity: precision over jargon
- Structure: clear headings, coherent progression

## Priority system

Markdown files in `docs/` may use YAML front matter such as:

```yaml
---
priority: 110
---
```

Lower values appear first. MUST preserve existing `priority` values. When creating new files, ask the user for the correct priority value.


## Pre-commit and exercise aggregation

A **pre-commit hook** (`extract-exercise-list`) generates `docs/exercises.md` by scanning Markdown files in `docs/` for exercise blocks.

- The hook runs on `git commit`.
- The generated file is ordered by `priority`.
- Manual edits to `docs/exercises.md` will be overwritten.

To modify exercises, edit the original source file in `docs/`.



## Python and notebooks

Rules for work inside `notebooks/`:

- MUST use the `uv` command from within the `notebooks/` directory to run Python-related commands.
- MUST NOT use direct `python`, `python3`, `pip`, or `pip3` commands.
- MUST treat notebooks in `notebooks/` as **Marimo notebooks**.
- MUST avoid reusing variable names in ways that would overwrite earlier values, because Marimo enforces a computational graph and rejects duplicate variable definitions.
- SHOULD use underscore-prefixed variable names for common reusable values, especially temporary or frequently repeated names such as `_x`, `_df`, `_chart`, or `_fig`.
- When editing existing notebook code, MUST preserve Marimo-compatible execution flow and avoid introducing hidden state assumptions common in traditional Jupyter notebooks.