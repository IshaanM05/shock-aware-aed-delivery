# Exporting to the NMIMS course template

The course repository expects each group folder to follow a fixed layout and passes a compliance
audit. `scripts/export_nmims.py` builds that folder from this repository's real results, so the
course deliverable is generated, not hand-copied.

```bash
python scripts/export_nmims.py --out dist/nmims                  # complete export (needs final results)
python scripts/export_nmims.py --out dist/nmims --allow-partial  # unfinished results become labelled PENDING markers
```

Output: `dist/nmims/Group_03_Kashish_Vaishnavi/` with `README.md`, `RESEARCH_AND_IMPLEMENTATION_GUIDE.md`,
`docs/` (roster, literature review, 4-page manuscript, four 300 DPI figures), `models/aed_delivery_amr.xml`,
`src/` (smoke test, controller entry point, a vendored copy of the package) and `analytics/` (economics
script, figure generator, at least 80 real benchmark episodes).

## What it verifies before reporting success

* the course's own `audit_file` (no currency tokens, no emoji, no faculty names, no forbidden tokens) on every text file;
* the roster: the two official students, exactly six foundational papers (two seminal, four recent), every DOI
  present in `docs/references.json` with the Crossref-verified title;
* the authorised title verbatim, the LinkedIn lines matching the course sync script, manuscript with three
  figures and two tables, figures at 300 DPI, the MJCF compiling with plain `mujoco`, and a benchmark CSV of at least 80 rows;
* results are not stale relative to the current model parameters (`--accept-stale` overrides).

## Applying it to the course repository

`--apply` copies **only** `Group_03_Kashish_Vaishnavi/` into the course repository and refuses to run if that
folder has uncommitted changes. It never stages or commits anything: review the diff, create the feature
branch the course asks for, add `Group_03_Kashish_Vaishnavi/` (never `git add -A`) and open the pull request
yourself.
