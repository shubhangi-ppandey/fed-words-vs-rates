# M8 adversarial review (2026-10-16) and fix log

Reviewer: an independent read-only agent. It worked from scratch copies and touched no project file.

**Bottom line:** the core numbers were re-estimated independently and match `results.json` exactly. No look-ahead
or holdout leakage bug was found. The issues concern claims, the pre-registration story, one data fact and
reproducibility. No pre-registered verdict changes.

Status column: **TODO** → **FIXED** (with where), or **NOTED** (disclosed as a limitation).

## Blockers

| ID | Issue | Fix | Status |
| --- | --- | --- | --- |
| B1a | Four pre-registered analyses were silently dropped: the M4 novelty measure, placebo regressions on non-FOMC days, the "1994–1999 statements" robustness check, and the H5 information-effect (JK CBI) interaction | Run them all (none touches the holdout) and log them in deviations | **FIXED**: `10c_plan_completion.py`, deviations #18, paper §6/§10 |
| B1b | H2, H5 and H6 were tested one-sided, but the plan says two-sided unless stated (only H1 is one-sided). The two-sided H2 p is 0.032, so the verdicts are unchanged | Use two-sided p for H2/H5/H6 in the verdicts and Table 3 | **FIXED**: 06/08/09/12, deviations #17; H2 two-sided p 0.034 |
| B1c | "Written before data collection" cannot be shown: the PROJECT_PLAN mtime is 20:45 because of a later title edit, and the folder is not in git | Reword to "written before any outcome regression; not externally registered". Commit the plan when the user approves git | **FIXED** (wording in deviations + paper App. A + limitations); git commit awaits the user's OK |
| B2 | The executive summary is stale ("effective policy tool"; "five measures") | Re-run 13 | **FIXED** (13 rewritten; final rebuild after seeds) |

## Major

| ID | Issue | Fix | Status |
| --- | --- | --- | --- |
| M1 | In the holdout, M2 = M3 and M2b = M3b: the same models (full.csv and chrono_2023.csv are bitwise identical). Their betas differ only through the SD scaling (18%) | Collapse each pair into one row in the holdout table, figure and app, and state this | **FIXED**: 08 flags identical pairs; table/fig/app collapsed |
| M2 | Pre-trend: tone surprises predict yield moves **before** the meeting (DGS2 k=20 pre-window 8.66 bp, p < 0.001). The growth in the daily LPs is not identified; "no reversal" survives a pre-trend control (h20 4.72, p = 0.014) | Report pre-event placebo horizons and a pre-trend-controlled LP; soften the "persistent" wording | **FIXED**: 09, f06, paper §9, deviations #19 |
| M3 | Overclaims: "only transformer measures detect it" (only M3 is significant at 5%); "look-ahead gave a weaker estimate" (false for TF-IDF); "adds noise / blurs it" (a mechanism the test cannot support) | Say "no evidence of inflation" and give the bounds. Add paired measurement contrasts M3−M3b, M3−M1, M3−M2b, M2−M1 | **FIXED**: 07 + paper Table 6a |
| M4 | The originally pre-registered H3 contrast (M2 − M3b: +0.84 [−0.06, +1.68]) is not reported | Report it in the look-ahead section | **FIXED**: M2 − M3b +0.81 [−0.04, +1.65] in §7 |
| M5 | Single fine-tuning seed per model | Re-train M2 + M3 (1999–2022) with seeds 43 and 44 (`04g`), report the beta spread and seed-averaged probabilities (`10b`) | **FIXED** (2026-10-17): 50/50 runs, 0 collapses. **M3 H1 beta: seed 42 +1.04 (p .004), 43 +0.94 (.007), 44 +0.39 (.110, n.s.); seed-averaged +0.86 (.005).** M2: +0.65 / +0.63 / +0.83; averaged +0.85 (.024). The H3 conclusion is unchanged (averaged M2 ≈ M3). **H1 is seed-sensitive (2 of 3 seeds significant): disclosed in the dashboard and in LinkedIn posts.** `results_seed_robustness.json`, deviation #16 |
| M6 | Reproducibility: run_all is missing 04e/04f/07b/x1; 12 does not check the holdout-lock hash; JK is fetched from the `main` branch (unpinned); raw data are gitignored while 09/10 read data/raw | Add the steps; make 12 verify the lock hash; pin the JK commit SHA; commit the small raw market files plus a SHA-256 manifest | **FIXED**: run_all, 12 assert, JK commit f788aef, 02b manifest, .gitignore |

## Minor

| Issue | Status |
| --- | --- |
| "Dropping 1999 lowers to 0.75, inside the LOYO range" is false (the LOYO min is 0.77 under full-sample SD) | **FIXED** |
| Dangling "Its out-of-time accuracy is 0.550" (reads as if about M3) | **FIXED** |
| **2020-03-03 intermeeting emergency cut labelled "Scheduled" in the JK file** sits in the main sample (dropping it gives H1 1.094, p 0.005). Reclassify it using the Fed's own calendar; this also changes prev_id for 2020-04-29 | **FIXED** |
| The first meeting's S compares against 1998-11-17 (a 6-month gap) and is influential (β without it 0.885, p 0.015) | **FIXED** (robustness row 13 + text) |
| H7 rule direction is unspecified (accepts either sign) | **FIXED** |
| Table 5 shows two-sided p but is labelled "wild p", and doesn't say the control is MP1 (not pc1) | **FIXED** |
| Table 6 S&P differences are sign-flipped to the hypothesised direction without a note | **FIXED** |
| Table 7 M5f estimation β 0.534 (1999–2022) vs 0.281 (2000–2022) is not reconciled | **FIXED** |
| The incremental-R² CI lower bound > 0 is uninformative (the statistic is ≥ 0 by construction) | **FIXED** |
| The permutation test is Manly-style (shuffles S, controls fixed); Freedman–Lane is standard. Results agree | **FIXED** (FL p 0.0035) |
| Market LP h=1 +0.27% (p = 0.0495) undercuts "stocks do not respond overall" | **FIXED** |
| Liu et al. and Mosbach et al. are in the references but never cited in the text | **FIXED** |
| Ayivodji et al. description should be softened to what was verified | **FIXED** |
| README says "6-page dashboard" (it has 7); "replicating" should be "in line with" | **FIXED** |
| The exec summary's "works through expected policy path" needs an "exploratory" label | **FIXED** |
| **417 of 2,977 statement sentences appear verbatim in TDW** (73 labelled in the same or a later year): a concrete label-memorisation channel for M2 | **FIXED** (x2: 369/2,977 verbatim, 58 same/later year; exact matching) |
| log_04e_download.txt is truncated (ends at 20231231; the 2024 vintage exists) | NOTED |
| Streamlit Cloud ignores runtime.txt; choose Python ≥ 3.11 (3.12) in the deploy settings | **FIXED** |
| results.json contains NaN tokens | **FIXED** |
| deviations.md row #7 is out of order | **FIXED** |
