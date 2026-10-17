# P3 — Words vs. Rates

**Working title:** *Words vs. Rates: Quantifying Look-Ahead Bias in Measured Federal Reserve Communication Effects* (changed 2026-10-16 from "...Without Look-Ahead Bias", which overclaimed; see reports/references_verified.md §C)
**Author:** Shubhangi Pandey · **Level:** PhD research (financial economics / text-as-data) · **Plan date:** 2026-10-16
**Suggested GitHub repo:** `shubhangi-ppandey/fed-words-vs-rates`

---

## 1. Research questions & pre-registered hypotheses

**RQ1.** Does the hawkish/dovish tone of FOMC statements move asset prices *beyond* the market-measured policy-rate surprise?

**RQ2.** How much do pretrained language models whose training data overlaps the sample inflate the measured effect?
This question asks about look-ahead bias. A related check: does the effect survive in a true out-of-time period (2023–2026)
that falls after the classifier was published?

**RQ3.** Through which channel does tone act: the policy path (2y/5y), long rates/term premium (10y/30y), equities, or the dollar?
Does it interact with the "central-bank information effect"?

**RQ4.** Which industries respond, and is sensitivity larger at the zero lower bound (ZLB)?

The hypotheses are fixed *before* any outcome regressions are run. The decision rules apply as written,
nulls are reported as nulls, and α = 5% (two-sided unless stated).

| ID | Hypothesis | Test | Decision rule |
| --- | --- | --- | --- |
| H1 | A hawkish tone surprise raises the 2-year Treasury yield in the 30-min window, controlling for the target surprise | OLS ΔTFUT02 on ToneSurprise + MP1; HC3 SE + wild bootstrap (Rademacher, 9,999) | Supported if β>0 and the bootstrap p<0.05 (one-sided) |
| H2 | Tone surprise explains part of the Gürkaynak–Sack–Swanson **path factor** | Path factor on ToneSurprise (+ target factor) | Supported if β>0, p<0.05 and the incremental R² is reported with a CI |
| H3 | **Look-ahead inflation:** the effect measured with FOMC-RoBERTa (fine-tuned on in-sample 1996–2022 labels) is larger than with look-ahead-safe measures | Paired event bootstrap of the standardized β difference, M2 − M3b | Supported if the difference is >0 and its 95% CI excludes 0 |
| H4 | In the untouched 2023–2026 holdout the sign of β (H1 spec) is the same as in-sample | Same regression on the holdout | Report β, CI and the minimum detectable effect; "consistent" if the sign matches. Low power is stated, not hidden |
| H5 | A hawkish tone surprise lowers the S&P 500 in the 30-min window, controlling for the policy surprise | ΔSP500 on ToneSurprise + pc1 | β<0, p<0.05. A positive sign is read as an information effect, with the JK CBI interaction checked |
| H6 | The tone effect is larger at the ZLB (2008-12→2015-12, 2020-03→2022-03) | Interaction ToneSurprise × ZLB | Interaction >0, p<0.05 |
| H7 | Rate-sensitive industries respond more (daily window) | 49 industry regressions; Holm-adjusted | Report Holm-significant industries; a pre-specified rate-sensitive group (Utilities, Real Estate, Banks) compared with the rest |

---

## 2. Contribution & positioning (to be verified in the literature review)

- **Classic "words vs. actions":** Gürkaynak, Sack & Swanson (2005) infer communication from *market* factors (target vs path).
  Here, communication is measured *directly from text*, and we test whether the text explains the path factor.
- **Text-based Fed studies:** Hansen & McMahon (2016); Shah, Paturi & Chava (2023, *Trillion Dollar Words*);
  Handlan (text shocks); Gardner, Scotti & Vega (2022); Kansoy (2022, minutes surprises); Rosa (2013, minutes);
  Ehrmann & Talmi (2020, statement similarity).
- **Information effect:** Nakamura & Steinsson (2018); Jarociński & Karadi (2020); Bauer & Swanson (2023).
- **New angle (the main contribution):** a *graded look-ahead design*. Four tone measures with increasing look-ahead exposure,
  plus a true out-of-time holdout. This quantifies how much LLM look-ahead bias (Glasserman & Lin 2023; Sarkar & Vafa 2024)
  inflates estimated communication effects. Novelty risk: someone may already have done this for FOMC text.
  **Every citation is verified before it is used in the paper.**

---

## 3. Data (all free; verified live 2026-10-16)

| Source | What | Coverage | Access |
| --- | --- | --- | --- |
| federalreserve.gov | FOMC post-meeting statements (+ minutes and pressers for extensions) | 1994–2026 | HTML scrape of the calendar and historical pages, at a polite rate |
| Jarociński–Karadi `jkshocks_update_fed` (GitHub) | 30-min surprises: FF1–4, MP1, ED1–4, **TFUT02/05/10/30, SP500, SP500FUT, EUR**; pc1, MP/CBI decomposition | 385 events, 1988 → 2026-09-16 (updated 2026-10-11) | raw CSV |
| SF Fed (Bauer–Swanson 2023) | MPS, MPS_ORTH, high-frequency asset changes | 1988 → 2023-12 | xlsx |
| HF `gtfintechlab/fomc_communication` | 2,480 labelled sentences (train 1,984 / test 496) with **year** (0 dovish, 1 hawkish, 2 neutral; verify) | 1996–2022 | CC BY-NC 4.0 |
| HF `gtfintechlab/FOMC-RoBERTa` | Pretrained hawkish/dovish classifier | trained on the above | CC BY-NC 4.0 |
| FRED | DGS3MO, DGS2, DGS5, DGS10, DGS30, VIXCLS (daily) | decades | CSV endpoint |
| Ken French Data Library | 49 industry portfolios, daily; FF factors, daily | 1926 → | zip |

- **Sample:** scheduled FOMC meetings with a statement, from **May 1999** (statements at every meeting) to **Sep 2026**.
  Unscheduled and pre-1999 events are robustness checks only.
- **Split:**
  - Estimation sample: 1999-05 → 2022-12.
  - **Holdout: 2023-01 → 2026-09.** This is after FOMC-RoBERTa was published, and it is untouched until M4.

---

## 4. Leakage & look-ahead controls

1. **Graded tone measures:**
   - **M1** Dictionary (hawk/dove word lists fixed in advance): no learned look-ahead.
   - **M3b** TF-IDF + logistic regression trained **chronologically**: for statement year *Y*, train only on labelled sentences
     with year < *Y*. No pretrained LM, so no look-ahead.
   - **M3** RoBERTa-base fine-tuned chronologically (year < *Y*). Its labels are look-ahead-safe; its pretraining corpus (≤2019)
     is not, and this is stated.
   - **M2** FOMC-RoBERTa off the shelf. It was fine-tuned on sentences from the same years and documents as the sample,
     so it has **maximal look-ahead**.
   - **M4** Statement similarity/novelty (cosine similarity with the previous statement). No learning.
2. The holdout (2023–2026) is not touched for any choice: tone formula, sentence filter or specification.
3. Specifications are fixed in this plan. Any deviation is logged in `reports/deviations.md` with its reason.
4. **Event timing:** 30-min windows capture the *statement only*, because pressers start at 14:30.
   Daily windows (FRED, Ken French) capture statement + presser, and both are reported.

---

## 5. Measures

- **Sentence filter:** drop voting-roster and boilerplate sentences using a fixed regex list.
  Sensitivity is checked with no filter.
- **Document tone:** τ_t = (#hawkish − #dovish) / #sentences. Alternative: probability-weighted, Σ(p_h − p_d)/N.
- **Tone surprise (primary):** S_t = τ_t − τ_{t−1}, the change versus the previous scheduled statement.
  Markets read statements as edits ("redlines") of the previous one.
  Alternative: the residual from an expanding-window AR(1) of τ.
- **Policy controls:**
  - MP1 (target surprise) and pc1 (JK policy factor).
  - GSS target and path factors, computed by us from FF1/ED2–ED4 with PCA and the GSS rotation.
  - JK MP/CBI (sign-restricted decomposition).
  - Bauer–Swanson MPS_ORTH (to 2023).

---

## 6. Empirical design

- **Event regressions (30-min):** Δy_e = α + β·S_e + γ'·Controls_e + ε_e.
  y ∈ {TFUT02, TFUT05, TFUT10, TFUT30, SP500, EUR}. S is standardized (per 1 SD of tone surprise).
- **Inference:** HC3; wild bootstrap; **permutation test** (shuffle S across events, 9,999×); Holm across outcome families.
- **Path-factor test (H2):** path_e = α + β·S_e + δ·target_e + ε.
- **Look-ahead test (H3):** the same specification for M1–M4, then a paired bootstrap of the standardized β differences.
- **Persistence:** local projections (Jordà 2005) on daily DGS2, DGS10 and the market return, h = 0…20 trading days, Newey–West.
- **Heterogeneity:** ZLB interaction (H6); 49 Ken French industries on the daily window, with the market factor as control (H7, Holm).
- **Placebo:** the same regressions on randomly drawn non-FOMC days (daily data), and permutation as above.

---

## 7. Robustness (pre-specified)

- Exclude 2008–09.
- Include unscheduled meetings.
- Use the 1994–1999 statements.
- Alternative tone formulas and the no-filter variant.
- Alternative surprise definition (AR residual).
- Daily vs 30-min windows.
- Drop the FOMC statement days that had a press conference (2011–2018 subset).
- Leave-one-year-out stability of β.

**Extensions (only if the core is finished):**

- E1: minutes-release days, which carry no rate news, as a cleaner communication test.
- E2: press-conference transcript tone (daily window).

---

## 8. Validation of tone measures

- Macro-F1 and accuracy on the TDW **test split** for M2, M3 and M3b (the full-sample variants), with bootstrap CIs.
- Chronological out-of-time F1: train on labels with year < *Y*, test on labels from year *Y*, averaged across years.
- Agreement between measures: Spearman correlation of τ_t, and the share of sign agreement for S_t.
- Face validity: tone around known episodes (2004 "measured pace", 2013 taper, 2022 hikes).

---

## 9. Repository layout

```
Project3/
├── README.md                    findings-first, numbers auto-filled from results
├── PROJECT_PLAN.md              this file (pre-registration)
├── requirements.txt             app only (streamlit, pandas, numpy, plotly)
├── requirements-analysis.txt    offline pipeline (torch, transformers, sklearn, statsmodels, …)
├── runtime.txt, .streamlit/config.toml, .gitignore
├── analysis/
│   ├── common/                  paths.py, stats_utils.py, figstyle.py
│   ├── 01_fetch_statements.py   scrape FOMC statements → data/raw/statements/*.txt + index
│   ├── 02_fetch_market.py       JK + SF Fed surprises, FRED, Ken French → data/raw/
│   ├── 03_build_events.py       align statements ↔ surprises ↔ daily data → data/processed/events.csv
│   ├── 04_tone_models.py        M1–M4 + validation → sentence_scores.parquet, tone_by_statement.csv
│   ├── 05_surprises.py          tone surprise, GSS factors → events_model.csv
│   ├── 06_event_study.py        H1, H2, H5 (+ holdout lock)
│   ├── 07_lookahead.py          H3 (graded measures)
│   ├── 08_holdout.py            H4 (run once, last)
│   ├── 09_persistence_hetero.py LPs, ZLB (H6), industries (H7)
│   └── 10_robustness.py
├── data/raw/ (gitignored, re-downloadable)   data/processed/ (committed)
├── reports/
│   ├── results.json             every number used by the paper, README and app
│   ├── figures/, research_paper.md, executive_summary.md, deviations.md
│   └── make_research_paper_docx.py   reads results.json; sets docx author = Shubhangi Pandey
└── app/streamlit_app.py
```

---

## 10. Dashboard (Streamlit Cloud; precomputed artifacts only, no model inference)

1. **Tone timeline:** τ_t and S_t for each measure, with policy-rate changes and the ZLB shaded.
2. **Meeting explorer:** pick a meeting to see the statement with sentences coloured hawkish, dovish or neutral,
   the diff vs the previous statement, the surprises, and the 30-min and daily market reactions.
3. **Words vs. Rates results:** coefficient plots by outcome and measure, with CIs and the permutation distribution.
4. **Look-ahead lab:** M1–M4 compared side by side, plus the holdout panel.
5. **Industries & persistence:** an industry-response heatmap and local-projection IRFs.

---

## 11. Paper & executive summary

- **Journal-style paper sections:**
  - Abstract
  - Introduction
  - Literature
  - Data
  - Measuring tone (incl. look-ahead taxonomy)
  - Identification
  - Results (H1, H2, H5)
  - Look-ahead (H3, H4)
  - Heterogeneity and persistence (H6, H7)
  - Robustness
  - Limitations
  - Conclusion
  - References (verified)
  - Appendices
- **Executive summary:** one page for policy and market readers.
- **No causal overreach:**
  - The effects are identified as responses to *communication content conditional on the rate surprise*.
  - Tone is not randomly assigned. The statement text could co-move with other news in the same window, and this is acknowledged.

---

## 12. Lessons applied from the P1/P2 review

- No hand-typed numbers: the paper, README and app all read `reports/results.json`.
- No selection on the test/holdout set. The holdout is locked until M4.
- Every claim is checked against outputs before release, with an adversarial review at M8.
- docx core properties: author and lastModifiedBy = **Shubhangi Pandey**, title set. No program or internship references.
- The README layout matches the actual files. App colours match the meaning (risk up = red).
- Privacy: no personal data is involved (public Fed documents and market data).

---

## 13. Milestones

| # | Milestone | Deliverable | Acceptance check |
| --- | --- | --- | --- |
| M0 | Setup | folder, git init (local), requirements, scaffolding | `py -3.14 analysis/01_…` runs; identity config = Shubhangi |
| M1 | Data | statements 1994–2026, JK/BS surprises, FRED, Ken French; `events.csv` | Every scheduled meeting 1999-05→2026-09 has text + a JK surprise; 5 statements spot-checked against the Fed site |
| M2 | Tone measures | M1–M4 scores + validation table | TDW test macro-F1 reported; chronological models trained only on past years (asserted in code) |
| M3 | Core results | H1, H2, H5 on the estimation sample | Regressions + bootstrap + permutation saved in `results.json` |
| M4 | Look-ahead & holdout | H3, then H4 (holdout unlocked once) | Holdout run logged with a timestamp; no earlier access |
| M5 | Heterogeneity / persistence / robustness | H6, H7, LPs, robustness table | All pre-specified checks present |
| M6 | Dashboard | 5 pages; AppTest passes | 0 exceptions; numbers equal `results.json` |
| M7 | Paper + exec summary + README | md + docx + pdf | Every number traced to `results.json` |
| M8 | Final adversarial review → commit/push/deploy (with the user's OK) | review report | No unresolved high/medium issue |

---

## 14. Risks & fallbacks

- **Fed website layout differs across eras:** use the per-year historical pages, with a fallback list of statement URLs.
  Validate counts against the JK event list.
- **Few holdout events (~30):** report the minimum detectable effect; do not overclaim.
- **Tone surprises are noisy:** use probability-weighted tone and similarity (M4) as complements.
- **RoBERTa fine-tuning time:** use roberta-base; one model per year is about 28 models. If GPU time becomes an issue,
  fine-tune per 3-year block.
- **TDW labels cover minutes, speeches and pressers, not only statements:** statement-level validity is checked with face validity
  and cross-measure agreement.
- **License CC BY-NC 4.0:** non-commercial academic use only, with attribution.

---

## 15. Decisions for the student

1. GitHub repo name (suggested: `fed-words-vs-rates`).
2. Whether the extensions E1 (minutes days) and E2 (pressers) go in v1 or later. Recommendation: later.
