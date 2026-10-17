# Deviations from the analysis plan (`PROJECT_PLAN.md`)

Every change to the plan is logged here with its date and reason.

- **Part A** (#1–#13) was made **before any outcome regression was run**. Before then, market reactions had only been loaded and summarised, never regressed on tone.
- **Part B** (#14–#20) was made **after** the first full set of results and the first holdout run. It follows the independent M8 review (`reports/review_M8.md`) and is labelled as such everywhere it is reported.
- The plan was written before any outcome regression but was not externally registered. No time-stamped public registry exists, and the file was later edited for its title (#13).

## Part A: before any outcome regression

| # | Date | Plan section | Change | Reason |
| --- | --- | --- | --- | --- |
| 1 | 2026-10-16 | §4 M2 | M2 is now **RoBERTa-large fine-tuned by us on all 2,480 Trillion Dollar Words labels (1996–2022)**, not the off-the-shelf `gtfintechlab/FOMC-RoBERTa` | The HF model is gated: downloads return 403 without an approved account. Our M2 reproduces the property that matters, maximal look-ahead (labels from the same years and documents as the sample) |
| 2 | 2026-10-16 | §4 M3 | M3 uses **RoBERTa-large** (not roberta-base) | M2 and M3 then share architecture, pretraining and hyperparameters and differ **only** in the label window |
| 3 | 2026-10-16 | §9 layout | `04_tone_models.py` is split into `04a`–`04d` | Separates the long GPU job from the fast steps. No change in content |
| 4 | 2026-10-16 | §3 / §6 | ZLB rule made precise: the policy floor was zero **before** the announcement, i.e. (2008-12-16, 2015-12-16] and (2020-03-15, 2022-03-16] | The plan gave month ranges only |
| 5 | 2026-10-16 | §3 | The 2007-06-28 statement is dated from the page text; the Fed's URL says 2007-06-18 | Data error on the source website |
| 6 | 2026-10-16 | §4 | **Added M2b** (TF-IDF, full sample), giving a 2×2 {TF-IDF, RoBERTa-large} × {full sample, chronological}. **H3 is redefined within architecture** (M2 − M3, M2b − M3b). The plan's original H3 contrast (M2 − M3b) is still reported (results_lookahead.json `original_plan_H3`) | The original contrast mixed look-ahead with model capacity |
| 7 | 2026-10-16 | §5 | Boilerplate filter finalised by inspecting **statement text only, never labels or outcomes**: whole "Voting…", "In a related action…" and "In taking this action…" paragraphs are dropped; sentences without ending punctuation are dropped; minimum length is 3 words; the splitter no longer breaks at name initials | The first version leaked roster fragments and dropped substantive short sentences |
| 8 | 2026-10-16 | §5 | GSS factors built from **MP1, FF4, ED2, ED3** (not ED4); the path factor is scaled to move ED3 one-for-one. Loadings come from the estimation sample only | ED4 is missing for 12 meetings (1999–2003) |
| 9 | 2026-10-16 | §5 | Chronological tone surprises: **one model (labels < year of the meeting, capped at 2023) scores both the current and the previous statement** | Keeps both terms of S_t on the same scale, with no information after Y−1 |
| 10 | 2026-10-16 | §1 | **Primary measure for H1, H2, H4–H7 = M3** (chronological RoBERTa-large, count-based S_t) | The plan did not name a primary measure. It was fixed after the validation scores but **before any outcome regression** |
| 11 | 2026-10-16 | §4 / 04c | Fine-tuning: **5 epochs, but at least 600 optimizer steps**; a job whose three seeds all collapse is flagged rather than stopping the run | `chrono_1999` (224 labels, 70 steps) stayed at the majority class with all three seeds. Seen on training data only (cf. Mosbach et al. 2021) |
| 12 | 2026-10-16 | §7 extensions | **Extension E3: ChronoBERT** (He et al. 2026). M5 = vintage Y−1 + labels < Y (meetings from 2000); M5f = 2024 vintage + all labels. Contrasts M5f − M5, M2 − M3, M3 − M5 on 2000–2022 | Addresses the residual pretraining look-ahead of RoBERTa (≤2019). Specified before any outcome regression |
| 13 | 2026-10-16 | title, §2 | Title changed to *"…Quantifying Look-Ahead Bias in Measured Federal Reserve Communication Effects"*; positioning per `references_verified.md` §C | "Without look-ahead bias" overclaimed |

## Part B: after the first results (M8 review, 2026-10-16 ~22:15)

| # | Change | Reason | Effect |
| --- | --- | --- | --- |
| 14 | **Scheduled-meeting definition enforced with the Fed's own calendar:** a statement is "scheduled" only if both the JK file and the Fed's calendar pages say so. This drops **2020-03-03** (an intermeeting emergency cut that JK labels "Scheduled"). Main sample 220 → **219** (estimation 190 → **189**); prev_id of 2020-04-29 becomes 2020-01-29 | The plan defines the sample as *scheduled* meetings; 2020-03-03 was not one (the Fed lists it as "March 2 (unscheduled)"). It was the only disagreement across 247 statements | All estimation-sample results re-run |
| 15 | **Holdout re-run with `--force`** after #14 and #17. The holdout data are unchanged; its coefficients change only through the estimation-sample SD used for standardisation and the H2/H5 sidedness. The first run is archived as `results_holdout_run1.json` / `holdout_lock_run1.json` | Keeps the holdout consistent with the corrected estimation sample. No specification was chosen by looking at holdout results | Reported transparently |
| 16 | **Seed robustness (not pre-registered):** M2 and M3 re-trained with seeds 43 and 44 (`04g`, `10b`); H1 reported for each seed and for seed-averaged probabilities. Primary results stay seed 42 | Single-seed fine-tuning variance (review M5) | Reported as robustness |
| 17 | **Sidedness aligned with the plan:** the plan says "two-sided unless stated", and only H1 is stated one-sided. H2, H5 and H6 are now tested **two-sided** (the code had used one-sided). H7's rule now has a direction: rate-sensitive industries respond more negatively to hawkish surprises (group − rest < 0, CI excluding 0) | Code/plan mismatch found by the review | H2 still supported (two-sided); H5, H6, H7 unchanged |
| 18 | **Omitted pre-registered analyses now run** (`10c`): (a) M4 novelty, tested on the absolute 30-min response (Ehrmann & Talmi style; M4 is unsigned, so it cannot enter the signed H1 spec); (b) placebo on random non-FOMC days; (c) the 1994–1999 statements; (d) the H5 information-effect (JK CBI) interaction | They were in the plan but missing from the first pipeline | Reported in the paper |
| 19 | **Local projections:** added pre-event placebo horizons and pre-trend-controlled LPs (`09`) | Tone surprises predict yield moves *before* meetings (review M2); the daily LP growth is not an announcement effect | The persistence wording is softened |
| 20 | **Exploratory additions:** paired measurement contrasts (M3 − M3b, M3 − M1, M3 − M2b, M2 − M1, M2 − M2b), Freedman–Lane permutation p, a drop-first-meeting robustness row, and the verbatim overlap between statement sentences and the labelled corpus | To support or qualify claims flagged by the review (M3, minor items) | Labelled exploratory |

## Editorial (not analytical)

| Date | Change |
| --- | --- |
| 2026-10-17 | Before publishing the repository: PROJECT_PLAN.md §12 reworded to drop a third-party program name; local absolute paths in `reports/log_04c_transformers.txt` shortened to project-relative paths. No hypothesis, rule, sample or result changed. |
