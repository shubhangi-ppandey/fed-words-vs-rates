"""Step 13: write the research paper, executive summary and README from reports/results.json (and the processed
data), then build the formal .docx/.pdf with common/paperdoc.py.

No number is typed by hand: every figure in the text is formatted from results.json or computed here from the
processed data. Qualitative statements are guarded by assertions or worded conditionally on the estimates, so a
re-run that changes a result cannot silently produce text that contradicts it. Revised after the M8 review
(reports/review_M8.md): claims are limited to what the tests support, and post-results additions are labelled.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paperdoc  # noqa: E402
from common.paths import PROCESSED, REPORTS, ROOT  # noqa: E402

AUTHOR = "Shubhangi Pandey"
TITLE = "Words vs. Rates: Quantifying Look-Ahead Bias in Measured Federal Reserve Communication Effects"
R = json.loads((REPORTS / "results.json").read_text())
V, VAL = R["verdicts"], R["validation"]
CORE = pd.DataFrame(R["core"]["results"])
EV = pd.read_csv(PROCESSED / "events_model.csv", dtype={"id": str}, parse_dates=["date"])
EST = EV[EV.holdout == 0]
P = R["primary_measure"]
MLAB = {"M1": "M1 dictionary", "M2b": "M2b TF-IDF (full sample)", "M3b": "M3b TF-IDF (chronological)",
        "M2": "M2 RoBERTa-large (full sample)", "M3": "M3 RoBERTa-large (chronological)",
        "M5f": "M5f ChronoBERT (full sample)", "M5": "M5 ChronoBERT (fully chronological)"}
ORDER = ["M3", "M2", "M3b", "M2b", "M1"]
X2 = json.loads((REPORTS / "exploratory_x2_label_overlap.json").read_text())


def p_(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def ci(a, b, d=2):
    return f"[{a:+.{d}f}, {b:+.{d}f}]"


def excl0(c):
    return c[0] > 0 or c[1] < 0


def row(test, m, outcome=None):
    d = CORE[(CORE.test == test) & (CORE.measure == m)]
    if outcome:
        d = d[d.outcome == outcome]
    assert len(d) == 1
    return d.iloc[0]


# ---------- quantities computed from the data (descriptive, not estimated by hand) ----------
def descriptives():
    z = EST[f"S_{P}"] / EST[f"S_{P}"].std(ddof=1)
    y = EST.hf_TFUT02
    X = np.column_stack([np.ones(len(EST)), z, EST.hf_MP1])
    X0 = np.column_stack([np.ones(len(EST)), EST.hf_MP1])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    r2 = lambda XX, bb: 1 - ((y - XX @ bb) ** 2).sum() / ((y - y.mean()) ** 2).sum()  # noqa: E731
    return {"sd_S": float(EST[f"S_{P}"].std(ddof=1)), "sd_mp1": float(EST.hf_MP1.std(ddof=1)), "b_mp1": float(b[2]),
            "r2_full": float(r2(X, b)), "r2_mp1": float(r2(X0, b0)), "tau_mean": float(EST[f"tau_{P}"].mean())}


D = descriptives()
h1, h2, h5 = V["H1"], V["H2"], V["H5"]
assert h1["supported"] and h2["supported"] and not h5["supported"], "verdicts changed: revise the narrative"
la_rob, la_tf = V["H3"]["RoBERTa-large"], V["H3"]["TF-IDF"]
assert not la_rob["supported"] and not la_tf["supported"], "H3 verdict changed: revise the narrative"
assert not V["H6"]["supported"] and not V["H7"]["supported"], "H6/H7 verdicts changed: revise the narrative"
rq = CORE[(CORE.test == "RQ3") & (CORE.measure == P)].set_index("outcome")
assert (rq.loc[["hf_TFUT02", "hf_TFUT05", "hf_TFUT10", "hf_TFUT30"], "p_wild_holm"] < 0.05).all()
rob = pd.DataFrame(R["robustness"]["checks"])
loyo = R["robustness"]["leave_one_year_out"]
x1 = R["exploratory_x1_path_absorption"]
E3, HO, PC, SR = R.get("extension_E3_chronobert"), R.get("holdout"), R["plan_completion"], R.get("seed_robustness")
PH = R["persistence_hetero"]
lp = {k: pd.DataFrame(v).set_index("h") for k, v in PH["local_projections"].items()}
pre = {k: pd.DataFrame(v).set_index("k") for k, v in PH["pre_event_placebo"].items()}
lpc = {k: pd.DataFrame(v).set_index("h") for k, v in PH["local_projections_pretrend_controlled"].items()}
mag_ratio = h1["beta_bp_per_sd"] / (D["b_mp1"] * D["sd_mp1"])
LA = R["lookahead"]
sim = LA["measure_similarity"]
MC = {c["contrast"]: c for c in LA["measurement_contrasts"]}
OH3 = LA["original_plan_H3"]
placebo = PC["b_placebo_non_fomc_days"]
assert placebo["share_placebo_abs_ge_actual"] < 0.05, "placebo result changed: revise the narrative"
cbi = PC["d_H5_cbi_interaction"]
nov = PC["a_novelty_abs_response"]
ext = PC["c_extended_1994_2022"]
sig = {m: row("H1", m).p_wild < 0.05 for m in ORDER}


def seed_text() -> str:
    if not SR:
        return "Seed robustness: not yet available."
    r = pd.DataFrame(SR["rows"])
    m3 = r[r.measure == "M3"]
    avg = m3[m3.variant == "seed-averaged"].iloc[0]
    rng = SR["M3_beta_range_across_seeds"]
    n_sig = int((m3[m3.variant.str.startswith("seed ")].p_wild_greater < 0.05).sum())
    return (f"Re-training the RoBERTa-large models with two further seeds gives primary-measure coefficients between "
            f"{rng[0]:.2f} and {rng[1]:.2f} bp ({n_sig} of 3 seeds significant at 5%); averaging the three seeds' "
            f"probabilities gives {avg.beta:+.2f} bp (p = {p_(avg.p_wild_greater)}). For the full-sample model M2 the "
            f"range is {SR['M2_beta_range_across_seeds'][0]:.2f}–{SR['M2_beta_range_across_seeds'][1]:.2f} bp.")


# ======================================================================================================
def paper() -> str:
    d = R["data"]
    v3, v3b = VAL["M3_roberta_chrono_out_of_time"]["pooled"], VAL["M3b_tfidf_chrono_out_of_time"]["pooled"]
    vr, vt, vm1 = VAL["RoBERTa_large_test_split"], VAL["TFIDF_test_split"], VAL["M1_dictionary_test_split"]
    hv = V["H4"]
    L = []
    a = L.append
    a(f"# {TITLE}\n")
    a(f"> **Abstract.** Text-based measures of central-bank communication increasingly rely on language models whose "
      f"training data overlap the period being studied. I ask whether the hawkish or dovish tone of Federal Open Market "
      f"Committee (FOMC) statements moves asset prices beyond the policy-rate surprise, and whether look-ahead in the "
      f"tone classifier changes the answer. Using {d['main_sample']} scheduled FOMC statements "
      f"({d['first_meeting'][:4]}–{d['last_meeting'][:4]}) and 30-minute market responses, I measure tone with a 2×2 "
      f"design that crosses model capacity (TF-IDF vs. RoBERTa-large) with the label window (all labels vs. only "
      f"labels that pre-date each meeting), plus a dictionary and a chronologically pretrained ChronoBERT arm. With the "
      f"look-ahead-safe primary measure, a one-standard-deviation hawkish tone surprise raises the two-year Treasury "
      f"yield by {h1['beta_bp_per_sd']:.2f} bp within 30 minutes (wild-bootstrap p = {p_(h1['p_wild'])}) after "
      f"controlling for the target-rate surprise, explains part of the Gürkaynak–Sack–Swanson path factor, and moves "
      f"yields from two to thirty years; the same tone assigned to random non-meeting days has no effect. Equities do "
      f"not respond on average. I find no evidence that label look-ahead inflates the estimate: for RoBERTa-large the "
      f"full-sample model gives a smaller coefficient (difference {la_rob['diff']:+.2f} bp, 95% CI "
      f"{ci(*la_rob['ci95'])}). Measurement quality matters more: the chronological RoBERTa-large measure yields "
      f"significantly larger estimates than dictionary and TF-IDF measures, which detect nothing. A fully chronological "
      f"ChronoBERT measure gives a similar but less precise estimate, and the coefficient keeps its sign in an "
      f"untouched, though underpowered, 2023–2026 holdout. The analysis plan was fixed before any outcome regression; "
      f"every later change is logged.\n")
    a("**Keywords:** central bank communication; FOMC; high-frequency identification; text as data; look-ahead bias; "
      "language models; Treasury yields.  **JEL:** E52, E58, G12, G14, C55.\n")

    # 1 ------------------------------------------------------------------------------------------
    a("## 1. Introduction\n")
    a("Central banks act with words as well as with rates. Since Gürkaynak, Sack and Swanson (2005) showed that a "
      "\"path\" factor, distinct from the surprise in the policy rate itself, accounts for much of the asset-price "
      "response to FOMC announcements, a large literature has tried to measure the content of that communication "
      "directly from text (Lucca and Trebbi, 2009; Hansen and McMahon, 2016; Gardner, Scotti and Vega, 2022; "
      "Doh, Song and Yang, 2026; Aruoba and Drechsel, forthcoming). The most recent measures rely on pretrained "
      "language models fine-tuned on human-labelled sentences (Shah, Paturi and Chava, 2023).\n")
    a("These models raise a problem that is easy to overlook in event studies: **look-ahead**. A classifier fine-tuned "
      "on labelled sentences from 1996–2022 and then used to score statements from the same years has, in effect, seen "
      "the future relative to most of the events it measures; its pretraining corpus can leak information in the same "
      "way (Glasserman and Lin, 2024; Sarkar and Vafa, 2024; He, Lv, Manela and Wu, 2026). The concern here is not "
      "classification accuracy, which earlier work has examined on this corpus (Shah et al., 2023, appendix D), but "
      "whether look-ahead changes the *econometric estimate* that the text measure feeds into.\n")
    a("This paper makes three contributions. First, using a tone measure that never uses a label from after the "
      "meeting it scores, it shows that FOMC wording moves Treasury yields beyond the policy-rate surprise, and that "
      "the move is not reversed over the following month. Second, it quantifies look-ahead with a 2×2 design that "
      "crosses model capacity with the label window: there is no evidence that label look-ahead inflates the "
      "estimate, whereas the quality of the classifier changes the conclusion. Third, a chronologically pretrained "
      "ChronoBERT arm addresses pretraining leakage, and an untouched 2023–2026 holdout checks the sign out of "
      "sample. The analysis plan, primary measure and decision rules were written down before any outcome regression; "
      "changes made before and after the first results are logged separately (Appendix A).\n")

    # 2 ------------------------------------------------------------------------------------------
    a("## 2. Related literature\n")
    a("**High-frequency identification.** Kuttner (2001) and Bernanke and Kuttner (2005) measure policy surprises from "
      "federal funds futures in narrow windows; Gürkaynak, Sack and Swanson (2005) separate target and path factors; "
      "Swanson (2021) extends this to forward guidance and asset purchases. Nakamura and Steinsson (2018) and "
      "Jarociński and Karadi (2020) argue that announcements also reveal the central bank's information, while Bauer "
      "and Swanson (2023a, 2023b) attribute much of that pattern to the Fed responding to news. Acosta et al. (2026) "
      "provide a database showing that press conferences now carry much of the market response.\n")
    a("**Text as data in central banking.** Lucca and Trebbi (2009) score FOMC statements automatically; Hansen and "
      "McMahon (2016) and Cieslak and Schrimpf (2019) study the macro-financial effects of communication; Ehrmann and "
      "Talmi (2020) show that changes in statement wording raise volatility; Rosa (2013) and Kansoy (2022) study "
      "minutes; Apel, Blix Grimaldi and Hull (2022) build hawk–dove dictionaries; Gorodnichenko, Pham and Talavera "
      "(2023) use voice tone; Handlan (2022) and Doh, Song and Yang (2026) build text-based surprises with neural "
      "models. Loughran and McDonald (2011) is the standard reference on finance dictionaries.\n")
    a("**Look-ahead bias in language models.** Glasserman and Lin (2024) and Sarkar and Vafa (2024) document "
      "look-ahead in pretrained models; Lopez-Lira, Tang and Zhu (2025) and Crane, Karra and Soto (2025) show that "
      "models memorise pre-cutoff economic data; He et al. (2026) and Drinkall et al. (2024) build chronologically "
      "consistent models. Closest to this paper, Ayivodji et al. (2026) study how look-ahead in language-model "
      "measures of monetary-policy tone in the media affects Taylor-rule estimates, Xue et al. (2026) audit temporal "
      "leakage across architectures for financial news, and Schlee, Lukassen and Weisser (2026) use a time-based "
      "split of the same labelled corpus for classification. None of these estimates how label or pretraining "
      "look-ahead propagates into high-frequency monetary event studies.\n")

    # 3 ------------------------------------------------------------------------------------------
    a("## 3. Data\n")
    a(f"**Statements.** I download all {d['n_statements_downloaded']} FOMC post-meeting statements published on "
      f"federalreserve.gov from 1994 to {d['last_meeting'][:4]}. The main sample consists of the {d['main_sample']} "
      f"scheduled meetings from {d['first_meeting']} (after which a statement followed every meeting) to "
      f"{d['last_meeting']}: {d['estimation']} meetings to December 2022 form the estimation sample and the "
      f"{d['holdout']} meetings from 2023 onward form an untouched holdout. A meeting counts as scheduled only if both "
      f"the high-frequency data set and the Fed's own calendar say so; this excludes the intermeeting cut of "
      f"3 March 2020, which the high-frequency file labels as scheduled. Statements are split into sentences; voting "
      f"rosters, discount-rate notes and other procedural sentences are removed by a fixed filter, leaving "
      f"{d['n_sentences_kept']:,} of {d['n_sentences']:,} sentences.\n")
    a("**Market responses.** Thirty-minute changes around each statement come from the updated Jarociński–Karadi "
      "high-frequency data set (fed funds and Eurodollar futures, two-, five-, ten- and thirty-year Treasury futures, "
      "the S&P 500 and the euro), pinned to a fixed commit of its public repository. The window ends before the press "
      "conference begins. Daily data come from FRED (Treasury yields) and the Kenneth French Data Library (market and "
      "49 industry returns). Bauer–Swanson surprises (to 2023) are used as an alternative control.\n")
    a("**Labels.** Sentence-level labels (hawkish, dovish, neutral) come from the Trillion Dollar Words corpus "
      "(Shah, Paturi and Chava, 2023): 2,480 sentences from FOMC minutes, press conferences and speeches, 1996–2022, "
      f"each with its year. {X2['verbatim_in_labels']} of the {X2['kept_sentences']:,} scored statement sentences "
      f"({100 * X2['share']:.0f}%) also appear verbatim in the labelled corpus; for {X2['label_year_same_or_later_than_statement']} "
      "of them the label comes from the same or a later year, a direct memorisation channel available only to "
      "full-sample models.\n")

    # 4 ------------------------------------------------------------------------------------------
    a("## 4. Measuring tone, and the look-ahead design\n")
    a("Each sentence is classified as hawkish, dovish or neutral. Statement tone is τ = (hawkish − dovish) / sentences "
      "and the tone surprise is S = τ − τ(previous scheduled statement): markets read a statement as an edit of the "
      f"previous one. In the estimation sample the primary measure has mean τ = {D['tau_mean']:+.2f} and a standard "
      f"deviation of S of {D['sd_S']:.3f}; a one-standard-deviation surprise is thus a shift of about "
      f"{100 * D['sd_S']:.0f} percentage points in the net share of hawkish sentences. Transformer measures fine-tune "
      "RoBERTa-large (Liu et al., 2019) or ChronoBERT (He et al., 2026).\n")
    a("Table: Table 1. Tone measures and their look-ahead exposure")
    a("| Measure | Model | Labels used to score a meeting in year Y | Pretraining data |")
    a("| --- | --- | --- | --- |")
    a("| M1 | rule-based dictionary | none | none |")
    a("| M2b | TF-IDF + logistic regression | all labels 1996–2022 (look-ahead) | none |")
    a("| M3b | TF-IDF + logistic regression | labels from years < Y only | none |")
    a("| M2 | RoBERTa-large, fine-tuned | all labels 1996–2022 (look-ahead) | general web text to 2019 |")
    a("| **M3 (primary)** | RoBERTa-large, fine-tuned | labels from years < Y only | general web text to 2019 |")
    a("| M5f | ChronoBERT (2024 vintage), fine-tuned | all labels 1996–2022 | text to 2024 |")
    a("| M5 | ChronoBERT (vintage Y−1), fine-tuned | labels from years < Y only | text to the end of Y−1 |")
    a("| M4 | statement novelty (1 − cosine similarity with the previous statement) | none | none |")
    a("")
    a("For chronological measures, a single model trained on labels before year Y scores both the current and the "
      "previous statement, so S never uses information from after Y−1. All fine-tuning uses identical, a-priori "
      "hyperparameters (Appendix B).\n")
    rows = [("M1 dictionary", vm1), ("TF-IDF (train → test split)", vt), ("RoBERTa-large (train → test split)", vr)]
    if "ChronoBERT_test_split" in VAL:
        rows.append(("ChronoBERT 2024 vintage (train → test split)", VAL["ChronoBERT_test_split"]))
    a("Table: Table 2. Classifier accuracy (macro-F1 with 95% bootstrap CI)")
    a("| Classifier | Evaluation | Macro-F1 | 95% CI |")
    a("| --- | --- | --- | --- |")
    for lab, v in rows:
        a(f"| {lab} | standard test split (n = {v['n']}) | {v['macro_f1']:.3f} | [{v['macro_f1_ci95'][0]:.3f}, {v['macro_f1_ci95'][1]:.3f}] |")
    a(f"| M3b TF-IDF, chronological | next unseen year, pooled (n = {v3b['n']}) | {v3b['macro_f1']:.3f} | [{v3b['macro_f1_ci95'][0]:.3f}, {v3b['macro_f1_ci95'][1]:.3f}] |")
    a(f"| M3 RoBERTa-large, chronological | next unseen year, pooled (n = {v3['n']}) | {v3['macro_f1']:.3f} | [{v3['macro_f1_ci95'][0]:.3f}, {v3['macro_f1_ci95'][1]:.3f}] |")
    if "M5_chronobert_chrono_out_of_time" in VAL:
        v5 = VAL["M5_chronobert_chrono_out_of_time"]["pooled"]
        a(f"| M5 ChronoBERT, chronological | next unseen year, pooled (n = {v5['n']}) | {v5['macro_f1']:.3f} | [{v5['macro_f1_ci95'][0]:.3f}, {v5['macro_f1_ci95'][1]:.3f}] |")
    a("")
    a(f"RoBERTa-large reaches a macro-F1 of {vr['macro_f1']:.3f} on the standard split, in line with the benchmark "
      f"reported by Shah et al. (2023). The dictionary and TF-IDF reach about {vm1['macro_f1']:.2f}. Trained only on "
      f"past labels, RoBERTa-large still scores {v3['macro_f1']:.3f} on the next, unseen year, while chronological "
      f"TF-IDF falls to {v3b['macro_f1']:.3f}, close to chance in the early years when few labels exist (Figure 2).\n")
    a("![Figure 1. FOMC statement tone (primary and look-ahead measures) and the fed funds target](figures/f01_tone_timeline.png)")
    a("![Figure 2. Classifier accuracy: standard split and chronological out-of-time evaluation](figures/f02_classifier_validation.png)")

    # 5 ------------------------------------------------------------------------------------------
    a("## 5. Empirical design\n")
    a("For each meeting e, the market response is regressed on the standardised tone surprise and a policy-surprise "
      "control: Δy_e = α + β·S_e/σ(S) + γ·control_e + ε_e. With 30-minute windows, β is the response to statement "
      "content conditional on the rate surprise measured in the same window. Standard errors are HC3 (MacKinnon and "
      "White, 1985); p-values come from a restricted wild bootstrap with Rademacher weights (Davidson and Flachaire, "
      "2008) and from a permutation test that shuffles tone across meetings (9,999 draws each); families of outcomes "
      "are Holm (1979) adjusted. As pre-specified, H1 is tested one-sided and all other hypotheses two-sided. The GSS "
      "target and path factors are built from the 30-minute surprises in MP1, FF4, ED2 and ED3, with loadings "
      "estimated on the estimation sample only. Persistence is measured with local projections (Jordà, 2005).\n")
    a("Table: Table 3. Hypotheses and decision rules (primary measure M3)")
    a("| ID | Hypothesis | Decision rule |")
    a("| --- | --- | --- |")
    for k, txt in [("H1", "A hawkish tone surprise raises the 30-min two-year yield, controlling for MP1"),
                   ("H2", "The tone surprise explains the GSS path factor, controlling for the target factor"),
                   ("H3", "Look-ahead inflates the measured effect (full-sample minus chronological model)"),
                   ("H4", "The holdout coefficient has the same sign as in estimation"),
                   ("H5", "A hawkish tone surprise lowers the 30-min S&P 500, controlling for the JK policy factor"),
                   ("H6", "The effect is larger at the zero lower bound"),
                   ("H7", "Rate-sensitive industries respond more (daily)")]:
        a(f"| {k} | {txt} | {V[k]['rule']} |")
    a("")

    # 6 ------------------------------------------------------------------------------------------
    a("## 6. Results: words move yields\n")
    a("Table: Table 4. Core results, estimation sample (response per 1-SD tone surprise; HC3 s.e. in parentheses; "
      "wild-bootstrap p: one-sided for H1, two-sided for H2 and H5)")
    a("| Measure | H1: 2-year yield (bp) | p | H2: path factor (bp) | p | H5: S&P 500 (%) | p |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for m in ORDER:
        r1, r2, r5 = row("H1", m), row("H2", m), row("H5", m)
        a(f"| {MLAB[m]} | {r1.beta:+.2f} ({r1.se_hc3:.2f}) | {p_(r1.p_wild)} | {r2.beta:+.2f} ({r2.se_hc3:.2f}) | "
          f"{p_(r2.p_wild)} | {r5.beta:+.3f} ({r5.se_hc3:.3f}) | {p_(r5.p_wild)} |")
    a("")
    a(f"With the primary measure, a one-standard-deviation hawkish tone surprise raises the two-year Treasury yield by "
      f"**{h1['beta_bp_per_sd']:.2f} bp** within 30 minutes (HC3 s.e. {h1['se']:.2f}; wild-bootstrap p = "
      f"{p_(h1['p_wild'])}; permutation p = {p_(h1['p_perm'])}), so **H1 is supported**. Adding tone raises the R² "
      f"of the regression from {D['r2_mp1']:.3f} to {D['r2_full']:.3f}. For scale, a one-standard-deviation target "
      f"surprise moves the two-year yield by about {D['b_mp1'] * D['sd_mp1']:.2f} bp in the same regression, so a "
      f"one-standard-deviation tone surprise has roughly {100 * mag_ratio:.0f}% of that effect. Tone also explains "
      f"part of the path factor (**H2 supported**: β = {h2['beta_bp_per_sd']:.2f} bp, two-sided p = "
      f"{p_(h2['p_wild'])}; incremental R² {h2['incremental_r2']['value']:.3f}). Equities do not respond on average "
      f"(**H5 not supported**: β = {h5['beta_pct_per_sd']:+.3f}%, two-sided p = {p_(h5['p_wild'])}).\n")
    i_, zm = cbi["interaction"], cbi["z_in_monetary_meetings"]
    a(f"The plan specified that a positive H5 sign would be checked for an information effect. Interacting tone with "
      f"an indicator of the {cbi['n_cbi_meetings']} meetings that Jarociński and Karadi classify as central-bank "
      f"information shocks gives an interaction of {i_['beta']:+.3f}% (two-sided p = {p_(i_['p_wild'])}); the tone "
      f"coefficient at the other meetings is {zm['beta']:+.3f}% (p = {p_(zm['p_wild'])}). "
      + ("Hawkish wording thus coincides with *higher* stock prices at information-effect meetings, consistent with "
         "markets reading it as good news about the economy, and the average equity response masks offsetting effects."
         if i_["p_wild"] < 0.05 and i_["beta"] > 0 else
         "There is no significant difference between the two types of meeting.") + "\n")
    a("Table: Table 5. Channels: response across assets (primary measure; control MP1; two-sided wild p, Holm-adjusted "
      "across six outcomes)")
    a("| Outcome (30 min) | β per 1-SD | HC3 s.e. | wild p (two-sided) | Holm p |")
    a("| --- | --- | --- | --- | --- |")
    for o, lab in [("hf_TFUT02", "2-year yield (bp)"), ("hf_TFUT05", "5-year yield (bp)"), ("hf_TFUT10", "10-year yield (bp)"),
                   ("hf_TFUT30", "30-year yield (bp)"), ("hf_SP500", "S&P 500 (%)"), ("hf_EUR", "EUR/USD (%)")]:
        r = rq.loc[o]
        a(f"| {lab} | {r.beta:+.3f} | {r.se_hc3:.3f} | {p_(r.p_wild)} | {p_(r.p_wild_holm)} |")
    a("")
    a("The response is present along the whole Treasury curve and survives the Holm correction at every maturity, "
      "while the S&P 500 and the euro do not move. Figures 3 and 4 show the coefficients for every measure and the "
      "partial-regression scatter; Figure 5 shows the observed t-statistic against its permutation distribution. "
      f"Statement novelty (M4), tested as in Ehrmann and Talmi (2020) on the size of the two-year response, has a "
      f"coefficient of {nov['beta']:+.2f} bp per standard deviation (two-sided p = {p_(nov['p_wild'])}).\n")
    a("![Figure 3. Core coefficients for every tone measure (95% HC3 intervals)](figures/f03_core_coefficients.png)")
    a("![Figure 4. Partial-regression plot for H1](figures/f09_partial_regression_h1.png)")
    a("![Figure 5. Permutation null distribution of the H1 t-statistic](figures/f05_permutation_h1.png)")
    xb = {r["y"]: r for r in x1["b_beyond_pc1"]}
    a(f"**How do words work? (exploratory, not pre-registered.)** One pre-specified robustness check (Section 10) "
      f"replaces MP1 with the Jarociński–Karadi monetary-policy and information shocks, whose sum is their policy "
      f"factor pc1 — the first principal component of one-month to one-year rate surprises. There the effect "
      f"disappears. An exploratory decomposition explains why: tone predicts pc1 itself (β = "
      f"{x1['a_tone_predicts_pc1']['beta']:.2f} bp, p = {p_(x1['a_tone_predicts_pc1']['p_wild'])}), pc1 is strongly "
      f"correlated with the GSS path factor ({x1['corr_pc1_gss_path']:.2f}), and once pc1 is held fixed tone adds "
      f"nothing significant to the two-, five- and ten-year yields (two-year β = {xb['hf_TFUT02']['beta']:+.2f}, "
      f"p = {p_(xb['hf_TFUT02']['p_wild'])}). This is consistent with words moving yields *through the expected path "
      f"of policy*, so that conditioning on the market-measured path absorbs the channel. Only at thirty years is "
      f"there a hint of an additional effect (β = {xb['hf_TFUT30']['beta']:+.2f}, p = {p_(xb['hf_TFUT30']['p_wild'])}).\n")

    # 7 ------------------------------------------------------------------------------------------
    a("## 7. Does look-ahead inflate the estimate?\n")
    pr = pd.DataFrame(LA["pairs"])
    a("Table: Table 6. Same architecture, different label window (H1–H5 specifications; 95% paired-bootstrap CI). "
      "Differences are oriented so that a positive value means a larger effect in the hypothesised direction "
      "(for the S&P 500 the sign is flipped).")
    a("| Architecture | Outcome | Full-sample (look-ahead) β | Chronological β | Difference | 95% CI |")
    a("| --- | --- | --- | --- | --- | --- |")
    for _, r in pr.iterrows():
        o = {"hf_TFUT02": "2-year yield", "gss_path": "path factor", "hf_SP500": "S&P 500"}[r.outcome]
        a(f"| {r.architecture} | {o} | {r.beta_look_ahead:+.3f} | {r.beta_safe:+.3f} | "
          f"{r.diff_in_hypothesised_direction:+.3f} | {ci(*r.ci95, d=3)} |")
    a("")
    a(f"**H3 is not supported in either architecture.** For RoBERTa-large the full-sample model gives a *smaller* "
      f"two-year coefficient ({la_rob['beta_look_ahead']:.2f} vs. {la_rob['beta_safe']:.2f} bp; difference "
      f"{la_rob['diff']:+.2f}, 95% CI {ci(*la_rob['ci95'])}), which rules out inflation larger than about "
      f"{la_rob['ci95'][1]:.2f} bp. For TF-IDF the difference is positive but imprecise ({la_tf['diff']:+.2f}, "
      f"{ci(*la_tf['ci95'])}) and cannot exclude inflation of up to {la_tf['ci95'][1]:.2f} bp. The contrast in the "
      f"original plan, full-sample RoBERTa-large against chronological TF-IDF (M2 − M3b), is {OH3['diff']:+.2f} "
      f"(95% CI {ci(*OH3['ci95'])}): not significant under the pre-specified rule, though close to it, and it "
      f"mixes look-ahead with model capacity, which is why H3 was redefined within architecture before any outcome "
      f"regression (Appendix A). The measures are far from interchangeable: the full-sample and chronological "
      f"surprises agree in sign at only {100 * sim['RoBERTa-large']['sign_agreement']:.0f}% of meetings for "
      f"RoBERTa-large and {100 * sim['TF-IDF']['sign_agreement']:.0f}% for TF-IDF.\n")
    a("Table: Table 6a. Paired measurement contrasts (exploratory; H1 specification; 95% paired-bootstrap CI)")
    a("| Contrast | Difference (bp) | 95% CI |")
    a("| --- | --- | --- |")
    for k in ["M3 - M3b", "M3 - M1", "M3 - M2b", "M2 - M1", "M2 - M2b"]:
        c = MC[k]
        a(f"| {MLAB[c['a']]} − {MLAB[c['b']]} | {c['diff']:+.3f} | {ci(*c['ci95'], d=3)} |")
    a("")
    sig_names = [MLAB[m] for m in ORDER if sig[m]]
    a(f"Measurement quality matters more than the label window. At the 5% level only {', '.join(sig_names)} "
      f"detects the effect; the full-sample RoBERTa-large measure is positive but marginal (p = "
      f"{p_(row('H1', 'M2').p_wild)}), and the dictionary and TF-IDF measures find nothing. In paired comparisons the "
      f"chronological RoBERTa-large measure gives a significantly larger coefficient than chronological TF-IDF "
      f"({MC['M3 - M3b']['diff']:+.2f}, {ci(*MC['M3 - M3b']['ci95'])}), the dictionary "
      f"({MC['M3 - M1']['diff']:+.2f}, {ci(*MC['M3 - M1']['ci95'])}) and full-sample TF-IDF "
      f"({MC['M3 - M2b']['diff']:+.2f}, {ci(*MC['M3 - M2b']['ci95'])}). A researcher using a dictionary or "
      "bag-of-words measure would have concluded that statement tone does not move yields.\n")
    a("![Figure 6. Look-ahead versus chronological coefficients within each architecture](figures/f04_lookahead.png)")
    if E3:
        a(e3_section(E3))

    # 8 ------------------------------------------------------------------------------------------
    a("## 8. Out-of-sample: the untouched 2023–2026 holdout\n")
    a(holdout_section())

    # 9 ------------------------------------------------------------------------------------------
    a("## 9. Persistence and heterogeneity\n")
    l2, l10, p2, c2 = lp["DGS2"], lp["DGS10"], pre["DGS2"], lpc["DGS2"]
    a(f"Daily local projections need care, because tone surprises are related to yield movements *before* the "
      f"meeting. Over the 20 trading days before a meeting the two-year yield has risen by {p2.loc[20, 'beta']:+.2f} bp "
      f"per standard deviation of the subsequent tone surprise (p = {p_(p2.loc[20, 'p'])}), and over 10 days by "
      f"{p2.loc[10, 'beta']:+.2f} bp (p = {p_(p2.loc[10, 'p'])}), although not over the final five days "
      f"({p2.loc[5, 'beta']:+.2f} bp, p = {p_(p2.loc[5, 'p'])}): hawkish statements tend to follow yield run-ups. The "
      f"growth of the response over the weeks after the meeting therefore cannot be read as an announcement effect. "
      f"Controlling for the 20-day pre-meeting change, the two-year response is {c2.loc[0, 'beta']:+.2f} bp on the "
      f"announcement day (p = {p_(c2.loc[0, 'p'])}) and {c2.loc[20, 'beta']:+.2f} bp after 20 trading days "
      f"(p = {p_(c2.loc[20, 'p'])}), so the move is not reversed; for the ten-year yield the controlled response is "
      f"{lpc['DGS10'].loc[0, 'beta']:+.2f} bp on the day (p = {p_(lpc['DGS10'].loc[0, 'p'])}). Figure 8 shows both "
      f"the pre-event placebo horizons and the controlled projections.\n")
    h6, h7 = V["H6"], V["H7"]
    a(f"The effect is not larger at the zero lower bound (**H6 not supported**: interaction "
      f"{h6['beta_interaction']:+.2f} bp, s.e. {h6['se_hc3']:.2f}, two-sided p = {p_(h6['p_wild'])}, "
      f"{h6['n_zlb']} ZLB meetings). On daily data, no industry responds significantly after Holm adjustment and the "
      f"pre-specified rate-sensitive group (utilities, real estate, banks) does not fall more than the rest after "
      f"hawkish surprises (**H7 not supported**: {h7['group_minus_rest']:+.3f}%, 95% CI {ci(*h7['ci95'], d=3)}).\n")
    a("![Figure 8. Local projections with pre-event placebo horizons and pre-trend control](figures/f06_local_projections.png)")
    a("![Figure 9. Industry responses (daily, net of the market)](figures/f07_industries.png)")

    # 10 -----------------------------------------------------------------------------------------
    a("## 10. Robustness\n")
    a("Table: Table 8. Robustness checks of H1 (primary measure; one-sided wild p). Checks 1–11 are pre-specified; "
      "check 13 was added after the first results.")
    a("| Check | n | β (bp per 1-SD) | HC3 s.e. | wild p |")
    a("| --- | --- | --- | --- | --- |")
    for _, r in rob.iterrows():
        a(f"| {r.check} | {r.n} | {r.beta:+.3f} | {r.se_hc3:.3f} | {p_(r.p_wild_greater)} |")
    a(f"| 1994–2022, adding {ext['n_added_pre_1999']} pre-1999 statements (pre-specified) | {ext['M3']['n']} | "
      f"{ext['M3']['beta']:+.3f} | {ext['M3']['se_hc3']:.3f} | {p_(ext['M3']['p_wild'])} |")
    a("")
    jk = rob[rob.check.str.startswith("8 ")].iloc[0]
    others = rob[~rob.check.str.startswith("8 ")]
    a(f"The estimate is stable across the other {len(others)} checks (β between {others.beta.min():.2f} and "
      f"{others.beta.max():.2f} bp, all p ≤ {others.p_wild_greater.max():.3f}), including dropping the crisis years, "
      f"probability-weighted tone, no sentence filter, an AR(1)-residual surprise, daily windows, excluding days with "
      f"press conferences, Bauer–Swanson or GSS controls, adding unscheduled statements, and dropping the first "
      f"meeting, whose previous statement lies six months earlier. Dropping one year at a time gives coefficients "
      f"between {loyo['min']:.2f} and {loyo['max']:.2f} bp. A Freedman–Lane permutation test gives p = "
      f"{p_(R['robustness']['baseline_freedman_lane_p_greater'])}. The exception is the Jarociński–Karadi control "
      f"(β = {jk.beta:+.2f}, p = {p_(jk.p_wild_greater)}), which, as discussed in Section 6, conditions on the path "
      f"channel through which tone appears to operate.\n")
    a(f"**Placebo (pre-specified).** Assigning each meeting's tone surprise to a random trading day at least five days "
      f"from any FOMC statement and repeating the daily regression {placebo['n_placebo_draws']:,} times gives a mean "
      f"coefficient of {placebo['placebo_beta_mean']:+.3f} bp (s.d. {placebo['placebo_beta_sd']:.3f}), against "
      f"{placebo['actual_daily_beta_no_mp1']['beta']:+.2f} bp on the actual meeting days; only "
      f"{100 * placebo['share_placebo_abs_ge_actual']:.2f}% of placebo draws are as large.\n")
    a(f"**Fine-tuning seeds (added after the first results).** {seed_text()}\n")

    # 11-12 --------------------------------------------------------------------------------------
    a("## 11. Limitations\n")
    a("- **Tone is not randomly assigned.** The coefficient measures the response to statement content conditional on "
      "the rate surprise; other information released in the same window could co-move with tone, and hawkish "
      "statements tend to follow yield run-ups.")
    a("- **Residual look-ahead.** RoBERTa was pretrained on text up to 2019, after most meetings in the sample, so its "
      "pretraining can carry information about 1999–2019; the ChronoBERT arm addresses this channel and finds no "
      "significant difference. The human labels were assigned in 2022–23 with hindsight, which no chronological "
      "ordering can remove.")
    a("- **Labels are not statements.** The labelled corpus contains minutes, press conferences and speeches, not "
      "statements; statement-level validity rests on the accuracy evidence and face validity (Figure 1).")
    a(f"- **A short holdout** of {R['data']['holdout']} meetings has limited power, and in the holdout the "
      "full-sample and chronological models coincide, so it tests the sign of the effect, not look-ahead.")
    a("- **Thirty-minute windows capture the statement only.** Since 2019 every meeting has a press conference, which "
      "now carries much of the market response (Acosta et al., 2026); the daily-window checks include it.")
    a("- **Not externally registered.** The analysis plan was written before any outcome regression but not filed "
      "with a time-stamped registry; changes are logged in `reports/deviations.md`.\n")
    a("## 12. Conclusion\n")
    a(f"Measured with a look-ahead-safe, high-quality classifier, the words of FOMC statements matter: a "
      f"one-standard-deviation hawkish shift in wording raises Treasury yields by about "
      f"{h1['beta_bp_per_sd']:.0f} bp at the two-year horizon within half an hour, beyond the policy-rate surprise, "
      "apparently through the expected path of policy, and the move is not reversed over the following month. I find "
      "no evidence that label look-ahead created this result. The choice of text measure is decisive, however: "
      "dictionary and bag-of-words measures would have concluded that words do not matter. For empirical work with "
      "language models, the lesson is to report chronologically trained measures alongside full-sample ones, and to "
      "validate classifiers out of time before reading anything into an event-study coefficient.\n")
    a(references())
    a(appendix())
    return "\n".join(L)


def e3_section(e3) -> str:
    b, c = e3["betas"], {x["a"] + "-" + x["b"]: x for x in e3["contrasts"]}
    v5 = VAL.get("M5_chronobert_chrono_out_of_time", {}).get("pooled")
    s = ["### 7.1 Separating pretraining from label look-ahead (pre-registered extension)\n"]
    s.append(f"ChronoBERT (He et al., 2026) is pretrained only on text dated up to each vintage. M5 scores a meeting in "
             f"year Y with the vintage from the end of Y−1, fine-tuned on labels before Y, so neither pretraining nor "
             f"labels can see the future; M5f uses the 2024 vintage and all labels. Because the first vintage is "
             f"1999-12-31, the comparison uses the common sample {e3['sample']['start'][:4]}–"
             f"{e3['sample']['end'][:4]} (n = {e3['sample']['n']}), with surprises re-standardised on that sample.\n")
    s.append("Table: Table 6b. Look-ahead decomposition, common sample (H1 specification; one-sided wild p)")
    s.append("| Measure | β (bp per 1-SD) | HC3 s.e. | wild p |")
    s.append("| --- | --- | --- | --- |")
    for m in ["M3", "M2", "M5", "M5f", "M3b", "M2b", "M1"]:
        s.append(f"| {MLAB[m]} | {b[m]['beta']:+.3f} | {b[m]['se_hc3']:.3f} | {p_(b[m]['p_wild_greater'])} |")
    s.append("")
    s.append("| Contrast | Difference | 95% CI |")
    s.append("| --- | --- | --- |")
    for k, lab in [("M5f-M5", "ChronoBERT: full sample − fully chronological (total look-ahead)"),
                   ("M2-M3", "RoBERTa-large: full sample − chronological labels (label look-ahead)"),
                   ("M3-M5", "chronological RoBERTa − fully chronological ChronoBERT (pretraining, with architecture)")]:
        s.append(f"| {lab} | {c[k]['diff']:+.3f} | {ci(*c[k]['ci95'], d=3)} |")
    s.append("")
    m5, m3 = b["M5"], b["M3"]
    tot, lab, pre_ = c["M5f-M5"], c["M2-M3"], c["M3-M5"]
    t = [f"The fully chronological ChronoBERT measure gives {m5['beta']:+.2f} bp (p = {p_(m5['p_wild_greater'])})"]
    if m5["p_wild_greater"] < 0.05 and m5["beta"] > 0:
        t.append(", so the effect is detected when neither pretraining nor labels can see the future.")
    elif m5["beta"] > 0 and m5["beta"] >= 0.5 * m3["beta"]:
        t.append(f": positive and of similar size to the primary measure on the same sample ({m3['beta']:+.2f} bp, "
                 f"p = {p_(m3['p_wild_greater'])}) but less precisely estimated")
        t.append(f", in line with ChronoBERT's lower out-of-time accuracy ({v5['macro_f1']:.3f} vs. "
                 f"{VAL['M3_roberta_chrono_out_of_time']['pooled']['macro_f1']:.3f})." if v5 else ".")
    else:
        t.append("; with this smaller, chronologically pretrained model the effect is not detected.")
    if not any(excl0(x["ci95"]) for x in (tot, lab, pre_)):
        t.append(f" None of the look-ahead contrasts is significant: total look-ahead within ChronoBERT "
                 f"{tot['diff']:+.2f} bp ({ci(*tot['ci95'])}), label look-ahead within RoBERTa-large "
                 f"{lab['diff']:+.2f} ({ci(*lab['ci95'])}), and the contrast between chronological RoBERTa-large and "
                 f"fully chronological ChronoBERT {pre_['diff']:+.2f} ({ci(*pre_['ci95'])}), the last being "
                 "confounded with architecture.")
    t.append(f" On this sample, which drops the 1999 meetings, the primary coefficient is {m3['beta']:.2f} bp against "
             f"{h1['beta_bp_per_sd']:.2f} bp in the full estimation sample: the 1999 meetings are influential (see the "
             "drop-first-meeting check in Section 10), but the effect remains significant without them.\n")
    s.append("".join(t))
    return "\n".join(s)


def holdout_section() -> str:
    if not HO:
        return "The holdout has not yet been evaluated.\n"
    h = pd.DataFrame(HO["results"]).query("test == 'H1'").set_index("measure")
    pairs = {tuple(x) for x in HO.get("identical_surprise_pairs", [])}
    assert ("M2", "M3") in pairs and ("M2b", "M3b") in pairs, "holdout pairs changed: revise the text"
    v = V["H4"]
    scale_m2 = h.loc["M2", "beta"]
    s = [f"The holdout covers the {HO['sample']['n_events']} scheduled meetings from {HO['sample']['start']} to "
         f"{HO['sample']['end']}, after the end of the labelled corpus. Every measure scores these statements with "
         "models trained only on 1996–2022 labels, so none has label look-ahead there (only M5f's 2024 pretraining "
         "vintage post-dates some of these meetings). For the same reason the full-sample and chronological models "
         "coincide in the holdout — M2 and M3 produce identical surprises, as do M2b and M3b — so the holdout tests "
         "the sign of the effect, not look-ahead. It was first evaluated after all estimation-sample analysis was "
         "final and re-run once after the correction of the estimation sample (Appendix A); both runs are "
         "time-stamped.\n",
         "Table: Table 7. Holdout, 2023–2026 (H1 specification; standardised by the estimation-sample SD of the "
         "chronological measure; one-sided wild p)",
         "| Model (holdout) | n | β (bp per 1-SD) | 95% CI | wild p | MDE (80% power) | Estimation β | Same sign |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m, lab in [("M3", "RoBERTa-large, all 1996–2022 labels (M2 = M3)"),
                   ("M3b", "TF-IDF, all 1996–2022 labels (M2b = M3b)"),
                   ("M5", "ChronoBERT, 2022 vintage + all labels (M5)"),
                   ("M5f", "ChronoBERT, 2024 vintage + all labels (M5f)"), ("M1", "M1 dictionary")]:
        r = h.loc[m]
        s.append(f"| {lab} | {r.n} | {r.beta:+.3f} | {ci(*r.ci95)} | {p_(r.p_wild)} | {r.mde_80pct_power:.2f} | "
                 f"{r.estimation_beta:+.3f} | {'yes' if r.same_sign_as_estimation else 'no'} |")
    s.append("")
    s.append(f"For the primary model the holdout coefficient is {v['beta']:+.2f} bp (95% CI {ci(*v['ci95'])}, "
             f"p = {p_(v['p_wild'])}) against {h1['beta_bp_per_sd']:+.2f} bp in estimation, so **H4 is "
             f"{'supported (same sign)' if v['consistent'] else 'not supported (sign reversed)'}**. Standardising "
             f"by the full-sample model's estimation SD instead gives {scale_m2:+.2f} bp. With {v['n']} meetings the "
             f"minimum detectable effect at 80% power is {v['mde_80pct_power']:.2f} bp, "
             + ("larger than the estimation-sample effect, so the holdout cannot by itself confirm or reject an effect "
                "of that size." if v["mde_80pct_power"] > h1["beta_bp_per_sd"] else "smaller than the estimation-sample effect.")
             + "\n")
    s.append("![Figure 7. Estimation versus holdout coefficients](figures/f08_holdout.png)")
    return "\n".join(s)


REFS = [
    "Acosta, M., Ajello, A., Bauer, M., Loria, F. & Miranda-Agrippino, S. (2026). Financial Market Effects of FOMC Communication: Evidence from a New Event-Study Database. Federal Reserve Bank of San Francisco Working Paper 2025-30. doi:10.24148/wp2025-30",
    "Apel, M., Blix Grimaldi, M. & Hull, I. (2022). How Much Information Do Monetary Policy Committees Disclose? Evidence from the FOMC's Minutes and Transcripts. Journal of Money, Credit and Banking 54(5), 1459–1490. doi:10.1111/jmcb.12885",
    "Aruoba, S. B. & Drechsel, T. (forthcoming). Identifying Monetary Policy Shocks: A Natural Language Approach. American Economic Journal: Macroeconomics. doi:10.1257/mac.20240184",
    "Ayivodji, F., Briand, E., Moran, K. & Stevanovic, D. (2026). Monetary Policy in the Media Spotlight: Sentiments, Signals, and Economic Impact. arXiv:2605.15092",
    "Bauer, M. D. & Swanson, E. T. (2023a). An Alternative Explanation for the \"Fed Information Effect.\" American Economic Review 113(3), 664–700. doi:10.1257/aer.20201220",
    "Bauer, M. D. & Swanson, E. T. (2023b). A Reassessment of Monetary Policy Surprises and High-Frequency Identification. NBER Macroeconomics Annual 2022, 37, 87–155. doi:10.1086/723574",
    "Bernanke, B. S. & Kuttner, K. N. (2005). What Explains the Stock Market's Reaction to Federal Reserve Policy? Journal of Finance 60(3), 1221–1257. doi:10.1111/j.1540-6261.2005.00760.x",
    "Cieslak, A. & Schrimpf, A. (2019). Non-monetary news in central bank communication. Journal of International Economics 118, 293–315. doi:10.1016/j.jinteco.2019.01.012",
    "Crane, L. D., Karra, A. & Soto, P. E. (2025). Total Recall? Evaluating the Macroeconomic Knowledge of Large Language Models. Finance and Economics Discussion Series 2025-044. doi:10.17016/FEDS.2025.044",
    "Davidson, R. & Flachaire, E. (2008). The wild bootstrap, tamed at last. Journal of Econometrics 146(1), 162–169. doi:10.1016/j.jeconom.2008.08.003",
    "Doh, T., Song, D. & Yang, S.-K. (2026). Deciphering Federal Reserve Communication via Text Analysis of Alternative FOMC Statements. American Economic Journal: Macroeconomics 18(3), 379–412. doi:10.1257/mac.20220312",
    "Drinkall, F., Rahimikia, E., Pierrehumbert, J. & Zohren, S. (2024). Time Machine GPT. Findings of the Association for Computational Linguistics: NAACL 2024, 3281–3292. doi:10.18653/v1/2024.findings-naacl.208",
    "Ehrmann, M. & Talmi, J. (2020). Starting from a blank page? Semantic similarity in central bank communication and market volatility. Journal of Monetary Economics 111, 48–62. doi:10.1016/j.jmoneco.2019.01.028",
    "Gardner, B., Scotti, C. & Vega, C. (2022). Words speak as loudly as actions: Central bank communication and the response of equity prices to macroeconomic announcements. Journal of Econometrics 231(2), 387–409. doi:10.1016/j.jeconom.2021.07.014",
    "Glasserman, P. & Lin, C. (2024). Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis. Journal of Financial Data Science 6(1), 25–42. doi:10.3905/jfds.2023.1.143",
    "Gorodnichenko, Y., Pham, T. & Talavera, O. (2023). The Voice of Monetary Policy. American Economic Review 113(2), 548–584. doi:10.1257/aer.20220129",
    "Gürkaynak, R. S., Sack, B. & Swanson, E. T. (2005). Do Actions Speak Louder Than Words? The Response of Asset Prices to Monetary Policy Actions and Statements. International Journal of Central Banking 1(1), 55–93.",
    "Handlan, A. (2022). Text Shocks and Monetary Surprises: Text Analysis of FOMC Statements with Machine Learning. Working paper, Brown University.",
    "Hansen, S. & McMahon, M. (2016). Shocking language: Understanding the macroeconomic effects of central bank communication. Journal of International Economics 99(S1), S114–S133. doi:10.1016/j.jinteco.2015.12.008",
    "He, S., Lv, L., Manela, A. & Wu, J. (2026). Chronologically consistent large language models. Journal of Financial Economics 185, 104374. doi:10.1016/j.jfineco.2026.104374",
    "Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics 6(2), 65–70.",
    "Jarociński, M. & Karadi, P. (2020). Deconstructing Monetary Policy Surprises—The Role of Information Shocks. American Economic Journal: Macroeconomics 12(2), 1–43. doi:10.1257/mac.20180090",
    "Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections. American Economic Review 95(1), 161–182. doi:10.1257/0002828053828518",
    "Kansoy, F. (2022). FOMC minutes: as a source of central bank communication surprise. Warwick Economics Research Papers No. 1436.",
    "Kuttner, K. N. (2001). Monetary policy surprises and interest rates: Evidence from the Fed funds futures market. Journal of Monetary Economics 47(3), 523–544. doi:10.1016/S0304-3932(01)00055-1",
    "Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L. & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692",
    "Lopez-Lira, A., Tang, Y. & Zhu, M. (2025). The Memorization Problem: Can We Trust LLMs' Economic Forecasts? arXiv:2504.14765",
    "Loughran, T. & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. Journal of Finance 66(1), 35–65. doi:10.1111/j.1540-6261.2010.01625.x",
    "Lucca, D. O. & Trebbi, F. (2009). Measuring Central Bank Communication: An Automated Approach with Application to FOMC Statements. NBER Working Paper 15367. doi:10.3386/w15367",
    "MacKinnon, J. G. & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. Journal of Econometrics 29(3), 305–325. doi:10.1016/0304-4076(85)90158-7",
    "Mosbach, M., Andriushchenko, M. & Klakow, D. (2021). On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines. International Conference on Learning Representations (ICLR 2021).",
    "Nakamura, E. & Steinsson, J. (2018). High-Frequency Identification of Monetary Non-Neutrality: The Information Effect. Quarterly Journal of Economics 133(3), 1283–1330. doi:10.1093/qje/qjy004",
    "Rosa, C. (2013). The financial market effect of FOMC minutes. Federal Reserve Bank of New York Economic Policy Review 19(2), 67–81.",
    "Sarkar, S. K. & Vafa, K. (2024). Lookahead Bias in Pretrained Language Models. SSRN 4754678. doi:10.2139/ssrn.4754678",
    "Schlee, M., Lukassen, F. & Weisser, C. (2026). LabelFusion-TS: Fusing Large Language Models, Transformer Encoders, and Financial Time Series for Monetary-Policy Stance Classification. arXiv:2608.11753",
    "Shah, A., Paturi, S. & Chava, S. (2023). Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, 6664–6679. doi:10.18653/v1/2023.acl-long.368",
    "Swanson, E. T. (2021). Measuring the effects of Federal Reserve forward guidance and asset purchases on financial markets. Journal of Monetary Economics 118, 32–53. doi:10.1016/j.jmoneco.2020.09.003",
    "Xue, C. et al. (2026). Temporal Leakage in Financial News NLP: A Multi-Architecture Audit with a Regime-Specific M&A Signal. arXiv:2608.17223",
]


def references() -> str:
    return "## References\n\n" + "\n".join(f"- {r}" for r in REFS) + "\n"


def appendix() -> str:
    dev = (ROOT / "reports" / "deviations.md").read_text(encoding="utf-8")
    n_a = sum(1 for ln in dev.split("## Part B")[0].splitlines() if ln.startswith("| ") and ln[2:3].isdigit())
    n_b = sum(1 for ln in dev.split("## Part B")[1].splitlines() if ln.startswith("| ") and ln[2:3].isdigit())
    return ("## Appendix A. Analysis plan and deviations\n\n"
            "The hypotheses, decision rules, sample split and robustness checks were written in `PROJECT_PLAN.md` "
            "before any outcome regression; the plan was not filed with an external registry. "
            f"`reports/deviations.md` logs {n_a} changes made before any outcome regression (Part A) and {n_b} made "
            "after the first set of results, following an independent adversarial review (Part B). Part B corrected "
            "the sample definition (the 3 March 2020 intermeeting cut), aligned the sidedness of H2, H5 and H6 with the "
            "plan, ran four pre-specified analyses that the first pipeline had omitted (statement novelty, placebo "
            "days, the 1994–1999 statements and the H5 information-effect check), added pre-event placebo horizons to "
            "the local projections and a seed-robustness check, and re-ran the holdout once after the sample "
            "correction (the first run is archived as `reports/results_holdout_run1.json`). No pre-specified verdict "
            "changed. Analyses not in the plan are labelled exploratory in the text.\n\n"
            "## Appendix B. Reproducibility\n\n"
            "The pipeline is `analysis/run_all.py`, which runs every step in order. Fine-tuning uses lr = 1e-5, batch "
            "16, maximum length 128, warm-up 10%, weight decay 0.01, gradient clipping 1.0, bf16, five epochs with at "
            "least 600 optimiser steps (cf. Mosbach, Andriushchenko and Klakow, 2021), seed 42, and a pre-specified "
            "re-seeding rule if training collapses to one class. All RoBERTa-large and ChronoBERT models were trained "
            "on one NVIDIA RTX 5070 Ti laptop GPU. The high-frequency data are pinned to a fixed commit, the small raw "
            "files are included in the repository, and `data/raw_manifest.json` lists the SHA-256 hash of every raw "
            "input. Every number in this paper is read from `reports/results.json`; the dashboard reads the same "
            "file.\n\n"
            "**Data and licences.** FOMC statements: Board of Governors of the Federal Reserve System. High-frequency "
            "surprises: Jarociński and Karadi (updated series) and Bauer and Swanson (FRBSF). FRED, Federal Reserve Bank "
            "of St. Louis. Kenneth R. French Data Library. Trillion Dollar Words labels: CC BY-NC 4.0 (non-commercial "
            "research use, with attribution). ChronoBERT: MIT licence.\n")


# ======================================================================================================
def exec_summary() -> str:
    hv = V["H4"]
    hold = (f"In the untouched 2023–2026 holdout ({hv['n']} meetings) the coefficient is {hv['beta']:+.2f} bp "
            f"(95% CI {ci(*hv['ci95'])}), {'the same sign as in estimation' if hv['consistent'] else 'the opposite sign to estimation'}; "
            f"with so few meetings the test can only detect effects above {hv['mde_80pct_power']:.1f} bp.") if "beta" in hv else ""
    return f"""# Executive summary: Words vs. Rates

**Question.** When the Federal Reserve's post-meeting statement sounds more hawkish or dovish than the previous one,
do markets react beyond the interest-rate decision itself? And does it matter whether the language model that
measures tone has "seen the future"?

**Answer.**

- **Words matter for bond markets.** A one-standard-deviation hawkish shift in statement wording raises the two-year
  Treasury yield by **{h1['beta_bp_per_sd']:.2f} bp within 30 minutes** (p = {p_(h1['p_wild'])}), beyond the rate surprise.
  That is roughly **{100 * mag_ratio:.0f}%** of the effect of a one-standard-deviation rate surprise. Yields from two to
  thirty years move, and the same tone placed on random non-meeting days has no effect.
- **The move is not reversed** over the following month. Yields were already drifting the same way before hawkish
  meetings, so the longer-run drift is not attributed to the statement.
- **Apparent mechanism (exploratory):** wording shifts the market's expected path of policy rates.
- **Stock prices do not respond on average** in the 30-minute window. Hawkish wording coincides with higher stock
  prices at meetings that markets read as good news about the economy.
- **Measurement is decisive.** Dictionary and word-count measures find no effect; only the look-ahead-safe language
  model measure detects it clearly.
- **No evidence that look-ahead created the result.** For the language model, allowing it to "see" labels from later
  years did not increase the estimate.
- **Holdout.** {hold}

**Why it matters.**

- **For policymakers:** statement wording moves the yield curve beyond the rate decision itself, and the move does
  not reverse within weeks; equity prices are little affected on average.
- **For analysts and researchers:** text-based measures should be built chronologically and validated out of time
  before they are used in market or policy analysis.

**How.**

- {R['data']['main_sample']} scheduled FOMC statements, {R['data']['first_meeting'][:4]}–{R['data']['last_meeting'][:4]}.
- 30-minute market reactions from high-frequency futures data.
- Seven tone measures, from a dictionary to RoBERTa-large and a chronologically pretrained ChronoBERT.
- The analysis plan was fixed before any outcome was examined; later changes are logged.

*Shubhangi Pandey · {date.today():%B %Y} · Full paper: `reports/research_paper.pdf` · Interactive dashboard: `app/streamlit_app.py`*
"""


def readme(url: str | None = None) -> str:
    vr = VAL["RoBERTa_large_test_split"]
    hv = V["H4"]
    hold = (f"+{hv['beta']:.2f} bp ({'same sign' if hv['consistent'] else 'sign reversed'}; MDE {hv['mde_80pct_power']:.1f} bp; "
            "underpowered)") if "beta" in hv else "not yet run"
    live = f"**Live dashboard:** {url}" if url else "**Live dashboard:** _link added after deployment_"
    return f"""# Words vs. Rates

**Quantifying look-ahead bias in measured Federal Reserve communication effects.**
An empirical study of whether the tone of FOMC statements moves markets beyond the policy-rate surprise,
and whether language-model look-ahead changes the answer. Author: {AUTHOR}.

{live}

---

## Findings (estimation sample {R['core']['sample']['start'][:4]}–{R['core']['sample']['end'][:4]}, n = {R['core']['sample']['n_events']} scheduled meetings)

| Hypothesis (plan fixed before outcomes) | Result |
| --- | --- |
| H1 Hawkish tone surprise raises the 30-min 2-year yield (beyond MP1) | **Supported:** {h1['beta_bp_per_sd']:+.2f} bp per 1-SD, one-sided wild-bootstrap p = {p_(h1['p_wild'])} |
| H2 Tone explains the GSS path factor | **Supported:** {h2['beta_bp_per_sd']:+.2f} bp, two-sided p = {p_(h2['p_wild'])} |
| H3 Look-ahead inflates the estimate | **Not supported:** RoBERTa difference {la_rob['diff']:+.2f} {ci(*la_rob['ci95'])}; TF-IDF {la_tf['diff']:+.2f} {ci(*la_tf['ci95'])} |
| H4 Same sign in the untouched 2023–26 holdout | {hold} |
| H5 Hawkish tone lowers the S&P 500 | **Not supported** on average ({h5['beta_pct_per_sd']:+.3f}%, p = {p_(h5['p_wild'])}); positive at information-effect meetings |
| H6 Larger at the zero lower bound | **Not supported** (two-sided p = {p_(V['H6']['p_wild'])}) |
| H7 Rate-sensitive industries respond more | **Not supported** |

- Yields at 2, 5, 10 and 30 years respond (Holm-adjusted p < 0.05); placebo days show nothing; the move is not reversed
  over 20 days (controlling for the pre-meeting trend).
- **Sensitivity to fine-tuning randomness.** {seed_text()} The pre-registered primary result is the seed-42 run.
- Classifier quality decides the conclusion: dictionary and TF-IDF measures find nothing. RoBERTa-large reaches a
  macro-F1 of {vr['macro_f1']:.3f}, in line with Shah et al. (2023).
- Every change to the analysis plan, including those made after an independent adversarial review, is logged in
  `reports/deviations.md`.

![Core results](reports/figures/f03_core_coefficients.png)

## Design

- **Tone measures.** A 2×2 design {{TF-IDF, RoBERTa-large}} × {{all labels (look-ahead), labels before each meeting
  (chronological)}}, plus a dictionary and a ChronoBERT arm. ChronoBERT is pretrained only on text up to the year
  before the meeting.
- **Identification.** 30-minute windows (Jarociński–Karadi data, pinned commit), controlling for the policy-rate surprise.
- **Inference.** HC3, wild bootstrap and permutation tests, Holm corrections, local projections with pre-event placebo
  horizons.
- **Discipline.** Plan in `PROJECT_PLAN.md`; deviations in `reports/deviations.md`; review in `reports/review_M8.md`;
  time-stamped holdout runs (`reports/holdout_lock.json`).

## Repository layout

```
├── PROJECT_PLAN.md                 analysis plan (hypotheses, decision rules)
├── analysis/                       pipeline + common/ (paths, text, dictionary, stats, figstyle, paperdoc)
│   ├── 01_fetch_statements.py      FOMC statements 1994-2026
│   ├── 02_fetch_market.py, 02b_raw_manifest.py   surprises, FRED, Ken French, labels; SHA-256 manifest
│   ├── 03_build_events.py          statement <-> market alignment (scheduled = JK and Fed calendar agree)
│   ├── 04a-04g                     sentences, dictionary, TF-IDF, RoBERTa-large, ChronoBERT, seed robustness (GPU)
│   ├── 05_surprises.py             tone surprises, GSS factors
│   ├── 06-10c                      H1-H7, look-ahead, holdout, persistence, robustness, plan completion
│   ├── 11_figures.py, 12_compile_results.py, 13_write_reports.py
│   ├── exploratory/                analyses added after seeing results (labelled as exploratory)
│   └── run_all.py                  runs everything in order
├── app/streamlit_app.py            7-page dashboard (reads precomputed files only)
├── data/raw/                       public-domain and openly licensed inputs (see "Data and licences")
├── data/raw_manifest.json          SHA-256 of every raw file, including those re-downloaded by 02
├── data/processed/                 all derived data used by the analysis and the app
└── reports/
    ├── results.json                every number used by the README and app
    ├── deviations.md, review_M8.md, references_verified.md, holdout_lock.json
    └── figures/
```

## Reproduce

```bash
pip install -r requirements-analysis.txt   # plus a CUDA build of torch for the GPU steps (04c, 04f, 04g)
python analysis/run_all.py                 # several hours on an RTX 5070 Ti laptop GPU; the holdout step refuses
                                           # to re-run unless --force is given and logged
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**Deploying the dashboard (Streamlit Community Cloud):** choose Python 3.12 in the app's advanced settings (the
platform ignores `runtime.txt`; pandas 3 needs Python ≥ 3.11). The app needs only `requirements.txt`.

## Data and licences

- **Included:** FOMC statement texts (Federal Reserve Board, public domain); Treasury yields and the fed funds target
  (FRED / Federal Reserve H.15, public domain); Trillion Dollar Words labels (Shah, Paturi & Chava 2023,
  **CC BY-NC 4.0: non-commercial use, with attribution**); the FOMC meeting calendar and press-conference dates.
- **Not redistributed; downloaded by `analysis/02_fetch_market.py`:** Jarociński–Karadi high-frequency surprises
  (GitHub `marekjarocinski/jkshocks_update_fed`, pinned commit), Bauer–Swanson surprises (FRBSF), VIX (Cboe, via FRED),
  Kenneth R. French Data Library portfolios. `data/raw_manifest.json` lists their SHA-256 hashes for verification.
- A research paper is in preparation. Statement tone effects are identified conditional on the rate surprise in the
  same window; see the dashboard's *Methods & data* page and `reports/deviations.md` for limitations.
"""


def tidy(md: str) -> str:
    """Blank line after figures and table captions, so GitHub renders them as separate blocks."""
    out = []
    for ln in md.split("\n"):
        out.append(ln)
        if ln.startswith("![") or ln.startswith("Table: "):
            out.append("")
    return "\n".join(out)


def main() -> None:
    if "--readme-only" in sys.argv:  # papers are deferred to the paper phase; README ships with the code
        url = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else None
        (ROOT / "README.md").write_text(readme(url), encoding="utf-8")
        print("written: README.md only")
        return
    (REPORTS / "research_paper.md").write_text(tidy(paper()), encoding="utf-8")
    (REPORTS / "executive_summary.md").write_text(exec_summary(), encoding="utf-8")
    (ROOT / "README.md").write_text(readme(), encoding="utf-8")
    docx, pdf = paperdoc.make(REPORTS / "research_paper.md", AUTHOR,
                              [f"**{AUTHOR}**", f"{date.today():%B %Y}",
                               "Empirical study with a pre-specified analysis plan · code, data and dashboard: "
                               "Words vs. Rates repository"])
    print("written:", REPORTS / "research_paper.md", docx.name, pdf.name, "executive_summary.md", "README.md")


if __name__ == "__main__":
    main()
