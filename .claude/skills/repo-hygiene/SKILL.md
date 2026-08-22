---
name: repo-hygiene
description: Repository cleanup/organization procedure for EyeCU4.0 — periodic hygiene, not scientific work.
---

# Repo hygiene

Use for periodic cleanup, not scientific development. Never touch TEST access,
inference, training, or any frozen algorithm/metric while doing this.

1. **Inventory first.** `git status --short --ignored`, `git ls-files`, find
   large files (`find . -type f -size +5M`, excluding `.git`, `eye_env/`,
   `EyeCU_external_data/`), find `__pycache__`/`.pyc` outside the venv.
2. **Classify by role**, not by file type: PRODUCTION, TEST, FINAL SUBMISSION,
   SCIENTIFIC EVIDENCE (required to support a claim), HISTORICAL-NECESSARY
   (explains a rejected/rolled-back decision), REGENERABLE (previews, debug
   renders, caches — safe to delete), DEVELOPMENT SCRATCH.
3. **Evidence is not "keep everything in experiments/."** Within an experiment
   directory, keep the manifest/hashes/final-results/frozen-predictions;
   delete disposable previews, contact sheets, and duplicate renders whose
   loss does not weaken the defensible claim.
4. **Before deleting a demo/preview file, grep all `.md` for its filename.**
   If a doc cites it, either keep the file or edit the doc first — never
   silently break a citation.
5. **`.gitignore` changes don't untrack existing files.** For anything that
   should never have been committed, `git rm --cached <path>` after adding
   the ignore rule.
6. Root should only hold files a new evaluator expects at the top level
   (README, final docs, entry points, standard config). Move stray scratch
   into `docs/archive/` or delete it; don't invent new top-level structure
   for things that already have a home.
7. Work on a dedicated branch (`chore/...`). Never `git reset --hard`,
   `git clean -fdx`, or `git add -A` blindly — stage explicit paths.
8. Verify at the end: `git status --short`, confirm required final assets
   still exist, confirm nothing scientific got `git check-ignore`d, run only
   the targeted tests relevant to touched code (never the full suite here).
