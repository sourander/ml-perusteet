# AGENTS.md — Johdatus Koneoppimiseen

## Scope

This is a **Zensical** lesson-material site for **Johdatus Koneoppimiseen**. Content is authored primarily in **Finnish**.

You are a **content co-author**, not an autonomous refactorer: preserve the author's voice and prefer minimal, targeted edits.

## Quick rules

- Edit lesson content in `docs/` (Markdown, in **Finnish**) unless the user requests another language.
- Write `notebooks/**/*.py` in English.
- Do not reorganize files, directories, or navigation without explicit approval.
- Do not run `doc-flesh` commands — the CLI is for human use only and may be destructive.

## Read-only files

Do not modify unless the user explicitly requests it:

- `zensical.toml`, `LICENSE`, `pyproject.toml`, `README.md`, `AGENTS.md`
  
- `docs/exercises.md` — auto-generated; see **Exercises and pre-commit**
  

These are managed by centralized tooling: config `~/.config/doc-flesh`, source `gh:sourander/doc-flesh-config/`, CLI `doc-flesh` (a `uv` tool).

## Repository structure

Below is a common structure shared by many courses, this being on of them. Note that "notebooks" a certain course may have a only a "solutions" directory in the project root. Run `ls` if unsure.

```text
siteinfo.json          # repo marker and metadata
docs/                  # Markdown lesson/theory content (EDIT HERE)
notebooks/             # Python exercises and Marimo notebooks (if present)
notebooks/solutions/   # encrypted solutions (git-crypt)
notebooks/git-lfs/     # binary exercise assets (Git LFS)
zensical.toml          # READ-ONLY site config
README.md, LICENSE, pyproject.toml  # READ-ONLY
```

## Authoring rules

- Write theory material in `docs/` as Markdown, in **Finnish**; never translate existing Finnish content unless requested.
- No filler prose, generic tutorial phrasing, or unnecessary introductions.
- Keep terminology consistent within the course.
- Tone: explanatory and teacherly; concept-first, then practical examples; precision over jargon.
- Preserve existing YAML front matter unless the user requests changes.

## Priority system

Files in `docs/` may set YAML front matter such as `priority: 110`; lower values appear first. Preserve existing values. When creating new files, ask the user for the correct priority.



## Exercises and pre-commit

A pre-commit hook (`extract-exercise-list`) generates `docs/exercises.md` by scanning `docs/**/*.md` for exercise blocks, ordered by `priority`. Manual edits to `docs/exercises.md` are overwritten — edit the source files instead. See `.pre-commit-guide.md` for hook details.


## Ask first

Ask before:

- changing repository structure or navigation
- changing existing `priority` values
- creating new top-level content files when placement is unclear
- broad stylistic rewrites
- modifying anything outside `docs/` unless explicitly requested

## Python and notebooks

- Run Python-related commands via `uv` from within `notebooks/` or other similar directory; never `python`, `python3`, `pip`, or `pip3` directly.
- Notebooks are **Marimo notebooks**: Marimo enforces a computational graph and rejects duplicate variable definitions, so never reuse variable names across cells.
- Use underscore-prefixed names for common temporary values, e.g. `_x`, `_df`, `_chart`, `_fig` or wrap them in a function.
- When editing notebooks, preserve Marimo-compatible execution flow; do not rely on Jupyter-style hidden state.