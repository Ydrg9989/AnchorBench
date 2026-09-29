# CLAUDE.md

AnchorBench: the public code for the COLM 2026 paper. Paper terms are in CONTEXT.md; the current review and refactor plan are in docs/code-notes/review-2026-09-29/REVIEW.md.

## Research code rules

This is a research repo; the code must stay small enough for its author to read and vouch for.

- **Locate before adding.** Search for where a concept already lives and extend that code path.
- **Variants are configs.** A new experiment variant is a config file or a parameter on the existing function.
- **Pure core, thin shell.** Computation lives in `src/<pkg>/` as pure functions; IO, CLI parsing, and model/API calls live in `scripts/`.
- **Paper names.** Use the terms in `CONTEXT.md`. Name files for what they compute.
- **Yell loudly.** Validate inputs where data enters and raise with the offending value. Catch only specific exceptions.
- **Test the science.** Metrics get property tests with hand-computed expected values. Refactors keep the golden test green.
- **Comments give the why and the source** (paper section, equation). History goes in the commit message.
- **Small diffs.** One concept per change, readable in one sitting. End each change with a reading guide: files touched, the function to read first, any changed inputs/outputs/side effects, and the test command.
- **Ask first** before adding a file under `src/`, an entry point or shell script, a dependency, an abstraction layer (base class, registry, factory), or a top-level markdown file. Say what smaller alternative you considered.
- **Delete dead code.** Git remembers.

## Checks that must stay green

```
python -m pytest -q
(cd /data/yiderigun/AnchorBench-paper && python -m pytest -q)   # paper layer (private): golden pins + claim verifier
python -m ruff check src tests scripts datasets
```

The package is not installed in the miniforge env; pytest finds it through pyproject's pythonpath, everything else needs PYTHONPATH=src. Published numbers are frozen: a refactor that moves one is reverted, never re-pinned.
