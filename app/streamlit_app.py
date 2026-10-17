"""Words vs. Rates — interactive companion to the research paper.

Reads only precomputed artifacts (data/processed/*.csv, reports/results.json); no model runs in the app,
so it fits comfortably within Streamlit Community Cloud limits. Dependencies: streamlit, pandas, numpy, plotly.
"""
import difflib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theme  # noqa: E402  ORBIT design system (app/theme.py)

ROOT = Path(__file__).resolve().parents[1]
PROC, REP = ROOT / "data" / "processed", ROOT / "reports"

# ORBIT palette: one accent; everything else is ink/grey on black
NAVY, TEAL, AMBER, GREY, SKY, RED, BLUE = (theme.ACCENT, theme.GREY_A, "#F0F0FA", theme.GREY_B, theme.GREY_C,
                                          theme.DANGER, theme.ACCENT)
LINE = theme.INK35
MEASURES = ["M1", "M2b", "M3b", "M2", "M3"]
MCOLOR = {"M1": GREY, "M2b": SKY, "M3b": TEAL, "M2": AMBER, "M3": NAVY, "M5": theme.ACCENT_TINT}
MLABEL = {"M1": "M1 · dictionary (no learning)", "M2b": "M2b · TF-IDF, full sample (look-ahead)",
          "M3b": "M3b · TF-IDF, chronological (safe)", "M2": "M2 · RoBERTa-large, full sample (look-ahead)",
          "M3": "M3 · RoBERTa-large, chronological (safe) — primary",
          "M5": "M5 · ChronoBERT, fully chronological (pretraining + labels)"}
MSHORT = {"M1": "M1 dictionary", "M2b": "M2b TF-IDF full", "M3b": "M3b TF-IDF chrono", "M2": "M2 RoBERTa full",
          "M3": "M3 RoBERTa chrono", "M5": "M5 ChronoBERT chrono"}
VIEW_MEASURES = MEASURES + ["M5"]  # timeline and explorer; M5 is defined from 2000
TONE_COLOR = {1: RED, 0: BLUE, 2: GREY, -1: GREY}   # hawkish red, dovish blue, neutral grey
TONE_NAME = {1: "hawkish", 0: "dovish", 2: "neutral", -1: "n/a"}

st.set_page_config(page_title="Words vs. Rates", page_icon="▲", layout="wide")
theme.inject()


@st.cache_data
def load():
    ev = pd.read_csv(PROC / "events_model.csv", dtype={"id": str, "prev_id": str}, parse_dates=["date"])
    ss = pd.read_csv(PROC / "sentence_scores.csv", dtype={"id": str})
    idx = pd.read_csv(PROC / "statements_index.csv", dtype={"id": str})
    res = json.loads((REP / "results.json").read_text())
    return ev, ss, idx, res


ev, ss, idx, R = load()
V = R["verdicts"]


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def shade(fig, holdout=True):
    for a, b in [("2008-12-16", "2015-12-16"), ("2020-03-15", "2022-03-16")]:
        fig.add_vrect(x0=a, x1=b, fillcolor="#F0F0FA", opacity=0.05, line_width=0)
    if holdout:
        fig.add_vrect(x0="2023-01-01", x1=str(ev.date.max().date()), fillcolor=theme.ACCENT, opacity=0.12, line_width=0)


def coef_chart(rows, title, xlabel):
    fig = go.Figure()
    for i, m in enumerate(MEASURES):
        r = rows[m]
        fig.add_trace(go.Scatter(x=[r["beta"]], y=[MLABEL[m]], mode="markers", marker=dict(color=MCOLOR[m], size=11),
                                 error_x=dict(type="data", array=[1.96 * r["se_hc3"]], color=MCOLOR[m]),
                                 hovertemplate=f"β = {r['beta']:.3f}<br>HC3 SE = {r['se_hc3']:.3f}<br>"
                                               f"wild-bootstrap p = {fmt_p(r['p_wild'])}<br>permutation p = {fmt_p(r['p_perm'])}"
                                               "<extra></extra>", showlegend=False))
    fig.add_vline(x=0, line_color=LINE, line_width=1)
    fig.update_layout(title=title, xaxis_title=xlabel, height=330, margin=dict(l=10, r=10, t=50, b=40))
    return fig


PAGES = ["Overview", "Tone timeline", "Meeting explorer", "Words vs. Rates results",
         "Look-ahead lab", "Industries & persistence", "Methods & data"]
SLUG = {p: p.lower().replace(" & ", "-").replace(".", "").replace(" ", "-") for p in PAGES}  # ?page=look-ahead-lab
if "nav" not in st.session_state:  # deep link on first load only; afterwards the radio owns the state
    st.session_state.nav = next((p for p in PAGES if SLUG[p] == st.query_params.get("page")), PAGES[0])
st.sidebar.markdown("<div class='orbit-kicker' style='margin-top:.6rem'>Words vs. Rates</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", PAGES, key="nav")
st.query_params["page"] = SLUG[page]
st.sidebar.markdown("---")
st.sidebar.caption(f"{R['data']['main_sample']} scheduled FOMC meetings, {R['data']['first_meeting']} → "
                   f"{R['data']['last_meeting']}. Estimation {R['data']['estimation']}, untouched holdout "
                   f"{R['data']['holdout']} (2023+). Primary tone measure: **{R['primary_measure']}**.")

# ------------------------------------------------------------------------------------------------
if page == "Overview":
    h1, h3 = V["H1"], V["H3"]["RoBERTa-large"]
    theme.hero(
        kicker=f"FOMC communication · {R['data']['first_meeting'][:4]}–{R['data']['last_meeting'][:4]}",
        title="Words vs. Rates",
        subtitle="Do Federal Reserve tone surprises move markets beyond the rate surprise — and does language-model "
                 "look-ahead change the answer?",
        stats=[(f"{h1['beta_bp_per_sd']:+.2f}", "bp", "2-year yield per 1-SD hawkish surprise"),
               (f"{R['data']['main_sample']}", "", "scheduled FOMC statements"),
               (fmt_p(h1["p_wild"]), "", "H1 wild-bootstrap p"),
               (f"{h3['diff']:+.2f}", "bp", "look-ahead minus safe (RoBERTa)"),
               (f"{R['validation']['RoBERTa_large_test_split']['macro_f1']:.3f}", "", "classifier macro-F1")])
    spark = go.Figure(go.Scatter(x=ev.date, y=ev[f"tau_{R['primary_measure']}"], mode="lines",
                                 line=dict(color="#F0F0FA", width=1.2), hoverinfo="skip"))
    spark.add_vrect(x0="2023-01-01", x1=str(ev.date.max().date()), fillcolor=theme.ACCENT, opacity=0.18, line_width=0)
    spark.update_layout(height=170, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False),
                        yaxis=dict(visible=False), showlegend=False)
    st.plotly_chart(spark, width="stretch", config={"displayModeBar": False})
    st.caption("Statement tone, 1999–2026 (look-ahead-safe measure). Blue: untouched holdout.")
    theme.section("Verdicts · applied mechanically from reports/results.json")
    rows = []
    for k, label in [("H1", "Hawkish tone surprise raises the 2-year yield (30 min)"),
                     ("H2", "Tone surprise explains the GSS path factor"),
                     ("H3", "Look-ahead inflates the measured effect"),
                     ("H4", "Same sign in the untouched 2023-26 holdout"),
                     ("H5", "Hawkish tone surprise lowers the S&P 500 (30 min)"),
                     ("H6", "Effect larger at the zero lower bound"),
                     ("H7", "Rate-sensitive industries respond more")]:
        v = V[k]
        if k == "H3":
            arch = {a: v[a]["supported"] for a in ("RoBERTa-large", "TF-IDF")}
            verdict = ("not supported (both architectures)" if not any(arch.values()) else
                       ", ".join(f"{a}: {'supported' if s else 'not supported'}" for a, s in arch.items()))
        elif k == "H4":
            verdict = v.get("status") or ("consistent" if v["consistent"] else "not consistent")
        else:
            verdict = "supported" if v["supported"] else "not supported"
        rows.append({"Hypothesis": k, "Statement": label, "Verdict": verdict, "Decision rule": v["rule"]})
    theme.table(pd.DataFrame(rows), hi={"Verdict": lambda s: s in ("supported", "consistent")})
    st.caption("Effects are identified as market responses to statement content, conditional on the policy-rate "
               "surprise measured in the same 30-minute window. Tone is not randomly assigned — see Methods & data.")
    if R.get("seed_robustness"):
        m3 = [r for r in R["seed_robustness"]["rows"] if r["measure"] == "M3"]
        single = [r for r in m3 if r["variant"].startswith("seed ")]
        avg = next(r for r in m3 if r["variant"] == "seed-averaged")
        st.caption(f"Robustness to training randomness: H1 holds for {sum(r['p_wild_greater'] < 0.05 for r in single)} of "
                   f"{len(single)} fine-tuning seeds and for seed-averaged tone ({avg['beta']:+.2f} bp, "
                   f"p = {fmt_p(avg['p_wild_greater'])}). Details on the Results page.")

# ------------------------------------------------------------------------------------------------
elif page == "Tone timeline":
    theme.page_title("Signal", "Statement tone, 1999–2026")
    pick = st.multiselect("Measures", VIEW_MEASURES, default=["M3", "M2"], format_func=lambda m: MSHORT[m])
    kind = st.radio("Show", ["tone level τ", "tone surprise S (change vs previous statement)"], horizontal=True)
    fig = go.Figure()
    for m in pick:
        col = f"tau_{m}" if kind.startswith("tone level") else f"S_{m}"
        fig.add_trace(go.Scatter(x=ev.date, y=ev[col], mode="lines+markers", name=MLABEL[m],
                                 line=dict(color=MCOLOR[m], width=2 if m == "M3" else 1.2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=ev.date, y=ev.fed_target_upper, name="fed funds target (upper), %", yaxis="y2",
                             line=dict(color=TEAL, width=1, shape="hv", dash="dot")))
    shade(fig)
    fig.update_layout(height=480, yaxis_title="(hawkish − dovish) / sentences", hovermode="x unified",
                      yaxis2=dict(overlaying="y", side="right", title="target, %", showgrid=False),
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Grey bands: zero lower bound. Blue band: untouched holdout (2023+), after every tone classifier's "
               "label data (1996–2022).")

# ------------------------------------------------------------------------------------------------
elif page == "Meeting explorer":
    theme.page_title("Inspect", "Meeting explorer")
    dates = ev.sort_values("date", ascending=False)
    choice = st.selectbox("FOMC meeting", dates.id, format_func=lambda i: str(ev.set_index("id").loc[i, "date"].date()))
    e = ev.set_index("id").loc[choice]
    m = st.radio("Colour sentences by", VIEW_MEASURES, index=4, format_func=lambda x: MLABEL[x], horizontal=False)
    c = st.columns(5)
    na = "n/a" if pd.isna(e[f"tau_{m}"]) else None
    c[0].metric("Tone τ", na or f"{e[f'tau_{m}']:+.3f}")
    c[1].metric("Tone surprise S", na or f"{e[f'S_{m}']:+.3f}")
    if na:
        st.caption("ChronoBERT's first vintage is 1999-12-31, so M5 starts with the 2000 meetings.")
    c[2].metric("Target surprise MP1", f"{e.hf_MP1:+.1f} bp")
    c[3].metric("2y yield, 30 min", f"{e.hf_TFUT02:+.1f} bp")
    c[4].metric("S&P 500, 30 min", f"{e.hf_SP500:+.2f} %")
    url = idx.set_index("id").loc[choice, "url"]
    st.markdown(f"Official statement: [{url}]({url}) · previous statement: {ev.set_index('id').loc[choice, 'prev_id']}")
    sent = ss[ss.id == choice].sort_values("sent_idx")
    html = []
    for _, r in sent.iterrows():
        lab = int(r[m])
        html.append(f"<span style='background-color:{TONE_COLOR[lab]}1F;border-left:2px solid {TONE_COLOR[lab]};"
                    f"padding:6px 12px;display:block;margin:4px 0;color:rgba(240,240,250,.85)' "
                    f"title='{TONE_NAME[lab]}'>{r.text}</span>")
    st.markdown("<div style='font-size:1rem;line-height:1.55'>" + "".join(html) + "</div>", unsafe_allow_html=True)
    st.caption("Red = hawkish · blue = dovish · grey = neutral. Voting rosters and procedural sentences are excluded "
               "(pre-specified filter).")
    with st.expander("What changed versus the previous statement?"):
        prev = ss[ss.id == e.prev_id].sort_values("sent_idx").text.tolist()
        cur = sent.text.tolist()
        diff = [d for d in difflib.ndiff(prev, cur) if d[:2] in ("+ ", "- ")]
        if not diff:
            st.write("No sentence-level changes.")
        for d in diff:
            st.markdown(("**＋ ADDED** · " if d.startswith("+") else "**－ REMOVED** · ") + d[2:])

# ------------------------------------------------------------------------------------------------
elif page == "Words vs. Rates results":
    theme.page_title("Results · estimation sample 1999–2022", "Words vs. Rates")
    h1, h2, h5 = V["H1"], V["H2"], V["H5"]
    theme.stat_row([(f"{h1['beta_bp_per_sd']:+.2f}", "bp", f"2-year yield · H1 · p {fmt_p(h1['p_wild'])} (one-sided)"),
                    (f"{h2['beta_bp_per_sd']:+.2f}", "bp", f"policy-path factor · H2 · p {fmt_p(h2['p_wild'])}"),
                    (f"{h2['incremental_r2']['value']:.3f}", "", "incremental R² of tone for the path"),
                    (f"{h5['beta_pct_per_sd']:+.3f}", "%", f"S&P 500 · H5 · p {fmt_p(h5['p_wild'])}")])
    core = pd.DataFrame(R["core"]["results"])
    for t, title, xl in [("H1", "H1 · 30-min 2-year Treasury yield, controlling for MP1", "bp per 1-SD tone surprise"),
                         ("H2", "H2 · Gürkaynak–Sack–Swanson path factor, controlling for the target factor", "bp per 1-SD tone surprise"),
                         ("H5", "H5 · 30-min S&P 500 return, controlling for the JK policy factor", "% per 1-SD tone surprise")]:
        rows = {r["measure"]: r for r in core[core.test == t].to_dict("records")}
        st.plotly_chart(coef_chart(rows, title, xl), width="stretch")
    theme.section("Channels · primary measure · Holm-adjusted across six outcomes")
    rq = core[(core.test == "RQ3") & (core.measure == R["primary_measure"])][["outcome", "n", "beta", "se_hc3", "p_wild", "p_wild_holm"]]
    rq["outcome"] = rq.outcome.map({"hf_TFUT02": "2-year (bp)", "hf_TFUT05": "5-year (bp)", "hf_TFUT10": "10-year (bp)",
                                    "hf_TFUT30": "30-year (bp)", "hf_SP500": "S&P 500 (%)", "hf_EUR": "EUR/USD (%)"})
    rq.columns = ["Outcome", "n", "Beta per 1-SD", "HC3 s.e.", "Wild p (two-sided)", "Holm p"]
    theme.table(rq, fmt={"Beta per 1-SD": "{:+.3f}", "HC3 s.e.": "{:.3f}", "Wild p (two-sided)": fmt_p, "Holm p": fmt_p},
                hi={"Holm p": lambda p: p < 0.05})
    sr = R.get("seed_robustness")
    if sr:
        theme.section("Seed robustness · the same models re-trained with two more random seeds")
        t = pd.DataFrame(sr["rows"])[["variant", "measure", "beta", "se_hc3", "p_wild_greater", "corr_with_seed42_S"]]
        t["measure"] = t.measure.map({"M2": "M2 · look-ahead", "M3": "M3 · safe (primary)"})
        t.columns = ["Training seed", "Measure", "Beta per 1-SD", "HC3 s.e.", "Wild p (one-sided)", "Corr. with seed-42 surprise"]
        theme.table(t, fmt={"Beta per 1-SD": "{:+.3f}", "HC3 s.e.": "{:.3f}", "Wild p (one-sided)": fmt_p,
                            "Corr. with seed-42 surprise": "{:.2f}"}, hi={"Wild p (one-sided)": lambda p: p < 0.05},
                    nowrap=("Training seed", "Measure"))
        m3 = [r for r in sr["rows"] if r["measure"] == "M3"]
        single = [r for r in m3 if r["variant"].startswith("seed ")]
        avg = next(r for r in m3 if r["variant"] == "seed-averaged")
        lo, hi = sr["M3_beta_range_across_seeds"]
        st.caption(f"Across {len(single)} fine-tuning seeds the primary estimate ranges from {lo:+.2f} to {hi:+.2f} bp and is "
                   f"significant for {sum(r['p_wild_greater'] < 0.05 for r in single)} of {len(single)}. Averaging the seeds' "
                   f"sentence probabilities gives {avg['beta']:+.2f} bp (p = {fmt_p(avg['p_wild_greater'])}). Fine-tuning "
                   "randomness is a real source of uncertainty in text measures: reported, not hidden. The pre-registered "
                   "primary result stays seed 42 (deviation #16).")
    theme.section("Partial-regression view of H1")
    z = f"S_{R['primary_measure']}"
    d = ev[ev.holdout == 0][["date", "hf_TFUT02", z, "hf_MP1"]].dropna()
    Xc = np.column_stack([np.ones(len(d)), d.hf_MP1])
    ry = d.hf_TFUT02 - Xc @ np.linalg.lstsq(Xc, d.hf_TFUT02, rcond=None)[0]
    rx = (d[z] - Xc @ np.linalg.lstsq(Xc, d[z], rcond=None)[0]) / d[z].std(ddof=1)
    fig = go.Figure(go.Scatter(x=rx, y=ry, mode="markers", marker=dict(color=NAVY, size=7, opacity=0.6),
                               name=f"meetings 1999–2022 (n = {len(d)})",
                               text=d.date.dt.date.astype(str), hovertemplate="%{text}<br>S=%{x:.2f} SD<br>Δ2y=%{y:.1f} bp<extra></extra>"))
    b = float(rx @ ry / (rx @ rx))
    xs = np.linspace(rx.min(), rx.max(), 20)
    fig.add_trace(go.Scatter(x=xs, y=b * xs, mode="lines", line=dict(color="#F0F0FA", width=1.5), name=f"slope {b:.2f}"))
    fig.update_layout(height=420, xaxis_title="tone surprise net of MP1 (SD)", yaxis_title="2-year yield change net of MP1 (bp)",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------------------------
elif page == "Look-ahead lab":
    theme.page_title("Measurement", "Look-ahead lab")
    h3, h4 = V["H3"], V["H4"]
    theme.stat_row([(f"{h3['RoBERTa-large']['diff']:+.2f}", "bp", f"RoBERTa look-ahead − safe · {theme.ci(h3['RoBERTa-large']['ci95'])}"),
                    (f"{h3['TF-IDF']['diff']:+.2f}", "bp", f"TF-IDF look-ahead − safe · {theme.ci(h3['TF-IDF']['ci95'])}"),
                    (f"{h4['beta']:+.2f}", "bp", f"holdout 2023–26 beta · n {h4['n']} · same sign"),
                    (f"{h4['mde_80pct_power']:.2f}", "bp", "holdout minimum detectable effect")])
    st.markdown("Each architecture is trained twice from the same starting point: once on **all** labelled "
                "sentences 1996–2022 (look-ahead: labels from the same years as the sample), once **chronologically** "
                "(for a meeting in year Y, only labels from years before Y).")
    v = R["validation"]
    oot = pd.DataFrame(v["M3_roberta_chrono_out_of_time"]["by_year"])
    oot_b = pd.DataFrame(v["M3b_tfidf_chrono_out_of_time"]["by_year"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=oot.year, y=oot.n_train, name="labels available (year < Y)", marker_color="rgba(240,240,250,.10)", yaxis="y2"))
    fig.add_trace(go.Scatter(x=oot.year, y=oot.macro_f1, name="M3 RoBERTa-large", line=dict(color=NAVY)))
    fig.add_trace(go.Scatter(x=oot_b.year, y=oot_b.macro_f1, name="M3b TF-IDF", line=dict(color=TEAL)))
    fig.update_layout(height=380, yaxis=dict(title="out-of-time macro-F1", range=[0, 1]),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="labels"),
                      legend=dict(orientation="h", y=-0.25), title="Accuracy of the look-ahead-safe classifiers on the next, unseen year")
    st.plotly_chart(fig, width="stretch")
    pr = pd.DataFrame(R["lookahead"]["pairs"])
    pr["outcome"] = pr.outcome.map({"hf_TFUT02": "H1 · 2-year yield", "gss_path": "H2 · path factor", "hf_SP500": "H5 · S&P 500"})
    theme.section("Same architecture, different label window")
    pr = pr[["architecture", "outcome", "beta_look_ahead", "beta_safe", "diff_in_hypothesised_direction", "ci95"]]
    pr.columns = ["Architecture", "Outcome", "Beta look-ahead", "Beta safe", "Look-ahead − safe", "95% CI"]
    theme.table(pr, fmt={"Beta look-ahead": "{:+.3f}", "Beta safe": "{:+.3f}", "Look-ahead − safe": "{:+.3f}", "95% CI": theme.ci})
    st.caption("Differences are signed in the hypothesised direction (for the S&P 500, a more negative look-ahead β counts "
               "as positive). Paired event bootstrap; a 95% CI that includes 0 means no evidence of look-ahead inflation.")
    e3 = R.get("extension_E3_chronobert")
    if e3:
        theme.section(f"Pretraining vs label look-ahead · ChronoBERT · {e3['sample']['start'][:4]}–{e3['sample']['end'][:4]} · n = {e3['sample']['n']}")
        theme.table(pd.DataFrame([{"Measure": MLABEL.get(k, k + " · ChronoBERT, full sample (look-ahead)"), "Beta per 1-SD (bp)": v["beta"],
                                   "HC3 s.e.": v["se_hc3"], "Wild p (one-sided)": v["p_wild_greater"]} for k, v in e3["betas"].items()]),
                    fmt={"Beta per 1-SD (bp)": "{:+.3f}", "HC3 s.e.": "{:.3f}", "Wild p (one-sided)": fmt_p},
                    hi={"Wild p (one-sided)": lambda p: p < 0.05})
        theme.table(pd.DataFrame([{"Contrast": c["contrast"], "Difference (bp)": c["diff"], "95% CI": c["ci95"]}
                                  for c in e3["contrasts"]]), fmt={"Difference (bp)": "{:+.3f}", "95% CI": theme.ci})
    theme.section("Untouched holdout · 2023–2026")
    if R.get("holdout"):
        h = pd.DataFrame(R["holdout"]["results"]).query("test == 'H1'").set_index("measure")
        names = {"M3": "RoBERTa-large, all 1996-2022 labels (M2 = M3 in the holdout)",
                 "M3b": "TF-IDF, all 1996-2022 labels (M2b = M3b in the holdout)",
                 "M5": "ChronoBERT, 2022 vintage (M5)", "M5f": "ChronoBERT, 2024 vintage (M5f)", "M1": "M1 dictionary"}
        tab = h.loc[[m for m in names if m in h.index], ["n", "beta", "se_hc3", "p_wild", "mde_80pct_power",
                                                         "estimation_beta", "same_sign_as_estimation"]]
        tab.insert(0, "Model", [names[m] for m in tab.index])
        tab.columns = ["Model", "n", "Holdout beta", "HC3 s.e.", "Wild p", "MDE (80% power)", "Estimation beta", "Same sign"]
        theme.table(tab, fmt={"Holdout beta": "{:+.3f}", "HC3 s.e.": "{:.3f}", "Wild p": fmt_p, "MDE (80% power)": "{:.2f}",
                              "Estimation beta": "{:+.3f}", "Same sign": lambda b: "✓" if b else "✗"},
                    hi={"Same sign": bool})
        st.caption("In the holdout the full-sample and chronological models coincide (both use all 1996-2022 labels), so "
                   "it tests the sign of the effect, not look-ahead. About 30 meetings: the minimum detectable effect "
                   "(80% power) is reported so the holdout is not over-read.")
    else:
        st.write("The holdout is locked until all estimation-sample analysis is final.")

# ------------------------------------------------------------------------------------------------
elif page == "Industries & persistence":
    theme.page_title("Dynamics", "Persistence and heterogeneity")
    lp = R["persistence_hetero"]["local_projections"]
    pre = R["persistence_hetero"].get("pre_event_placebo", {})
    ctl = R["persistence_hetero"].get("local_projections_pretrend_controlled", {})
    h6, h7 = V["H6"], V["H7"]
    c2 = {r["h"]: r for r in ctl.get("DGS2", [])}
    stats = ([(f"{c2[0]['beta']:+.2f}", "bp", f"2-year · day 0 · pre-trend controlled · p {fmt_p(c2[0]['p'])}"),
              (f"{c2[20]['beta']:+.2f}", "bp", f"2-year · day 20 · pre-trend controlled · p {fmt_p(c2[20]['p'])}")]
             if 0 in c2 and 20 in c2 else [])
    theme.stat_row(stats + [(f"{h6['beta_interaction']:+.2f}", "bp", f"zero-lower-bound interaction · H6 · p {fmt_p(h6['p_wild'])}"),
                            (f"{h7['n_holm_significant']}", f"/ {len(R['persistence_hetero']['H7']['industries'])}",
                             "industries Holm-significant · H7")])
    st.markdown("Grey points are **pre-event placebo** horizons (the change over the k days *before* the meeting): "
                "hawkish statements tend to follow yield run-ups, so the drift after the meeting is not attributed to it. "
                "The red dashed line controls for the 20-day pre-meeting trend.")
    names = {"DGS2": "2-year yield (bp)", "DGS10": "10-year yield (bp)", "mkt": "cumulative market return (%)"}
    cols = st.columns(3)
    for col, (k, t) in zip(cols, names.items()):
        d = pd.DataFrame(lp[k])
        fig = go.Figure([go.Scatter(x=d.h, y=d.beta + 1.96 * d.se, line=dict(width=0), showlegend=False),
                         go.Scatter(x=d.h, y=d.beta - 1.96 * d.se, fill="tonexty", fillcolor="rgba(62,106,225,0.20)",
                                    line=dict(width=0), showlegend=False),
                         go.Scatter(x=d.h, y=d.beta, line=dict(color=NAVY), showlegend=False)])
        if k in pre:
            q = pd.DataFrame(pre[k])
            fig.add_trace(go.Scatter(x=-q.k, y=q.beta, mode="markers", marker=dict(color=GREY, size=5),
                                     error_y=dict(array=1.96 * q.se, color=GREY), showlegend=False))
        if k in ctl:
            c = pd.DataFrame(ctl[k])
            fig.add_trace(go.Scatter(x=c.h, y=c.beta, line=dict(color=RED, dash="dash"), showlegend=False))
        fig.add_hline(y=0, line_color=LINE, line_width=1)
        fig.update_layout(title=t, height=300, xaxis_title="trading days relative to the meeting", margin=dict(t=40))
        col.plotly_chart(fig, width="stretch")
    st.markdown(f"**Zero lower bound (H6):** interaction {h6['beta_interaction']:+.2f} bp (HC3 SE {h6['se_hc3']:.2f}), "
                f"two-sided wild-bootstrap p = {fmt_p(h6['p_wild'])}; {h6['n_zlb']} ZLB meetings.")
    ind = pd.DataFrame(R["persistence_hetero"]["H7"]["industries"]).sort_values("beta")
    grp = set(R["persistence_hetero"]["H7"]["rate_sensitive_group"])
    fig = go.Figure(go.Bar(x=ind.industry, y=ind.beta, error_y=dict(array=1.96 * ind.se_hc3),
                           marker_color=[RED if i in grp else (NAVY if p < 0.05 else GREY) for i, p in zip(ind.industry, ind.p_holm)]))
    fig.update_layout(height=420, title="Daily industry responses net of the market (red = pre-specified rate-sensitive; blue = Holm p<0.05)",
                      yaxis_title="% per 1-SD tone surprise")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------------------------
else:
    theme.page_title("Method", "Methods & data")
    st.markdown(f"""
**Sample.** {R['data']['main_sample']} scheduled FOMC meetings with a post-meeting statement
({R['data']['first_meeting']} → {R['data']['last_meeting']}): {R['data']['estimation']} for estimation (to 2022) and
{R['data']['holdout']} in an untouched holdout (2023+). {R['data']['n_statements_downloaded']} statements downloaded
from federalreserve.gov; {R['data']['n_sentences_kept']} non-boilerplate sentences scored.

**Market reactions.** 30-minute windows around each statement (Jarociński–Karadi updated high-frequency data:
fed funds and Eurodollar futures, Treasury futures 2/5/10/30y, S&P 500, EUR). Daily data (FRED, Kenneth French
Data Library) for persistence and industries.

**Tone.** Sentences are labelled hawkish / dovish / neutral; statement tone τ = (hawkish − dovish) / sentences;
tone surprise S = τ − τ(previous scheduled statement). Classifiers are trained on the Trillion Dollar Words corpus
(Shah, Paturi & Chava 2023; CC BY-NC 4.0).

**Inference.** HC3 standard errors, restricted wild bootstrap and permutation tests (9,999 draws), Holm corrections,
Newey–West for local projections. All hypotheses, decision rules and the primary measure were fixed before any
outcome regression; every change to the plan is logged in `reports/deviations.md`.

**Limitations.** Tone is not randomly assigned: statement text may co-move with other information released in the
same window. The holdout is short (~30 meetings). RoBERTa's general pretraining corpus pre-dates 2020, a residual
look-ahead channel that the chronological fine-tuning does not remove.
""")
    st.caption("Sources: Board of Governors of the Federal Reserve System; M. Jarociński & P. Karadi (jkshocks_update_fed); "
               "M. Bauer & E. Swanson (FRBSF); FRED, Federal Reserve Bank of St. Louis; Kenneth R. French Data Library; "
               "gtfintechlab/fomc_communication.")
