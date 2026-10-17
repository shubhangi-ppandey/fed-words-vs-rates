"""Step 11: every report figure, drawn from the saved results (no estimation happens here except the permutation
null distribution in f05, which re-uses the step-6 specification and seed).

reports/figures/
  f01_tone_timeline.png        primary (M3) and look-ahead (M2) statement tone, fed funds target, ZLB and holdout
  f02_classifier_validation.png test-split macro-F1 by classifier; chronological out-of-time F1 by year
  f03_core_coefficients.png    H1 / H2 / H5 coefficients (95% HC3 CI) for every measure
  f04_lookahead.png            look-ahead vs safe coefficients within each architecture
  f05_permutation_h1.png       permutation null distribution of the H1 t-statistic (primary measure)
  f06_local_projections.png    persistence of the response (2y, 10y yields; market return), h = 0..20 days
  f07_industries.png           49 industry responses (daily), rate-sensitive group highlighted
  f08_holdout.png              estimation vs holdout H1 coefficients by measure (only after step 8 has run)
  f09_partial_regression_h1.png 30-min 2y yield change (net of MP1) against the primary tone surprise
"""
import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import figstyle  # noqa: E402
from common.figstyle import (ACCENT, AMBER, CARBON, GRAPHITE, GREY, MEASURE_COLOR, MEASURE_LABEL, NAVY,  # noqa: E402
                             RED, SOFT, TEAL)
from common.paths import FIGURES, MARKET, PROCESSED, REPORTS  # noqa: E402

es = importlib.import_module("06_event_study")
MEASURES = ["M1", "M2b", "M3b", "M2", "M3"]
figstyle.apply()


def suptitle(fig, text, **kw):
    """GROUND headline: left-aligned, Carbon, regular weight."""
    fig.suptitle(text, x=0.012, ha="left", color=CARBON, **kw)


def load(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def f01_timeline(ev):
    fed = []
    for sid in ("DFEDTAR", "DFEDTARU"):
        f = pd.read_csv(MARKET / f"fred_{sid}.csv")
        f.columns = ["date", "v"]
        f["date"] = pd.to_datetime(f["date"])
        fed.append(f.assign(v=pd.to_numeric(f.v, errors="coerce")))
    fed = pd.concat(fed).dropna().sort_values("date")
    fed = fed[fed.date >= "1999-01-01"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for a, b in [("2008-12-16", "2015-12-16"), ("2020-03-15", "2022-03-16")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=SOFT, alpha=1.0, lw=0)
    ax.axvspan(pd.Timestamp("2023-01-01"), ev.date.max() + pd.Timedelta(days=30), color=ACCENT, alpha=0.07, lw=0)
    ax.plot(ev.date, ev.tau_M2, color=AMBER, lw=1.0, alpha=0.8, label=MEASURE_LABEL["M2"])
    ax.plot(ev.date, ev.tau_M3, color=NAVY, lw=1.6, label=MEASURE_LABEL["M3"])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("statement tone  (hawkish - dovish) / sentences")
    ax2 = ax.twinx()
    ax2.step(fed.date, fed.v, where="post", color=GREY, lw=1.0, ls=":", label="fed funds target (upper), %")
    ax2.set_ylabel("fed funds target, %", color=TEAL)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.07))
    ax.set_title("FOMC statement tone, 1999–2026   ·   grey band: zero lower bound   ·   blue band: untouched holdout")
    fig.savefig(FIGURES / "f01_tone_timeline.png")
    plt.close(fig)


def f02_validation(v):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.2, 2.2]})
    items = [("M1 dictionary", v["M1_dictionary_test_split"], GREY), ("TF-IDF", v["TFIDF_test_split"], TEAL),
             ("RoBERTa-\nlarge", v["RoBERTa_large_test_split"], NAVY)]
    if "ChronoBERT_test_split" in v:
        items.append(("ChronoBERT\n(2024 vintage)", v["ChronoBERT_test_split"], GRAPHITE))
    for i, (lab, d, c) in enumerate(items):
        a.bar(i, d["macro_f1"], color=c, width=0.6)
        a.errorbar(i, d["macro_f1"], yerr=[[d["macro_f1"] - d["macro_f1_ci95"][0]], [d["macro_f1_ci95"][1] - d["macro_f1"]]],
                   color="black", capsize=4, lw=1)
        a.text(i, d["macro_f1_ci95"][1] + 0.015, f"{d['macro_f1']:.3f}", ha="center", fontsize=9)
    a.set_xticks(range(len(items)), [x[0] for x in items], fontsize=8.5)
    a.set_ylim(0, 0.9)
    a.set_ylabel("macro-F1 (TDW test split, n = 496)")
    a.set_title("Classifier accuracy (standard split)")
    m3 = pd.DataFrame(v["M3_roberta_chrono_out_of_time"]["by_year"])
    m3b = pd.DataFrame(v["M3b_tfidf_chrono_out_of_time"]["by_year"])
    b.plot(m3.year, m3.macro_f1, "-o", color=NAVY, ms=4, label=f"M3 RoBERTa-large (pooled {v['M3_roberta_chrono_out_of_time']['pooled']['macro_f1']:.3f})")
    b.plot(m3b.year, m3b.macro_f1, "-o", color=TEAL, ms=4, label=f"M3b TF-IDF (pooled {v['M3b_tfidf_chrono_out_of_time']['pooled']['macro_f1']:.3f})")
    if "M5_chronobert_chrono_out_of_time" in v:
        m5 = pd.DataFrame(v["M5_chronobert_chrono_out_of_time"]["by_year"])
        b.plot(m5.year, m5.macro_f1, "-o", color=GRAPHITE, ms=4,
               label=f"M5 ChronoBERT, vintage Y-1 (pooled {v['M5_chronobert_chrono_out_of_time']['pooled']['macro_f1']:.3f})")
    b.axhline(1 / 3, color=GREY, ls="--", lw=0.8)
    b.text(m3.year.min(), 1 / 3 + 0.01, "1/3", color=GREY, fontsize=8)
    b2 = b.twinx()
    b2.bar(m3.year, m3.n_train, color=GREY, alpha=0.18, width=0.8)
    b2.set_ylabel("labelled sentences available (year < Y)", color=GREY)
    b2.grid(False)
    b2.spines["right"].set_visible(True)
    b.set_zorder(b2.get_zorder() + 1)
    b.patch.set_visible(False)
    b.set_ylim(0, 0.9)
    b.set_xlabel("year Y (model trained only on labels from years < Y)")
    b.set_ylabel("out-of-time macro-F1 on year-Y labels")
    b.set_title("Look-ahead-safe classifiers: accuracy on the next, unseen year")
    b.legend(loc="lower right", fontsize=8.5)
    fig.savefig(FIGURES / "f02_classifier_validation.png")
    plt.close(fig)


def f03_core(core):
    res = pd.DataFrame(core["results"])
    tests = [("H1", "H1: 30-min 2-year yield (bp)"), ("H2", "H2: GSS path factor (bp)"), ("H5", "H5: 30-min S&P 500 (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (t, title) in zip(axes, tests):
        d = res[res.test == t].set_index("measure").loc[MEASURES]
        for i, m in enumerate(MEASURES):
            ax.errorbar(d.loc[m, "beta"], i, xerr=1.96 * d.loc[m, "se_hc3"], fmt="o", color=MEASURE_COLOR[m], capsize=3,
                        ms=6 if m == "M3" else 5)
            ax.text(d.loc[m, "beta"], i - 0.32, f"p={d.loc[m, 'p_wild']:.3f}", ha="center", fontsize=7.5, color="#333")
        ax.set_ylim(-0.7, len(MEASURES) - 0.5)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("response per 1-SD tone surprise")
    axes[0].set_yticks(range(len(MEASURES)), [MEASURE_LABEL[m] for m in MEASURES], fontsize=8.5)
    suptitle(fig, f"Core results, estimation sample {core['sample']['start'][:4]}-{core['sample']['end'][:4]} "
                 f"(n = {core['sample']['n_events']}); bars = 95% HC3 CI; wild-bootstrap p: H1 one-sided, H2 and H5 two-sided", fontsize=10.5)
    fig.savefig(FIGURES / "f03_core_coefficients.png")
    plt.close(fig)


def f04_lookahead(la):
    pr = pd.DataFrame(la["pairs"])
    outs = {"hf_TFUT02": "H1: 2y yield", "gss_path": "H2: path factor", "hf_SP500": "H5: S&P 500"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True, gridspec_kw={"wspace": 0.15})
    for ax, (o, title) in zip(axes, outs.items()):
        d = pr[pr.outcome == o]
        for i, (_, r) in enumerate(d.iterrows()):
            ax.plot([r.beta_safe, r.beta_look_ahead], [i, i], color=GREY, lw=1.2)
            ax.plot(r.beta_safe, i, "o", color=MEASURE_COLOR[r.safe], ms=7, label="safe" if i == 0 else None)
            ax.plot(r.beta_look_ahead, i, "D", color=MEASURE_COLOR[r.look_ahead], ms=6, label="look-ahead" if i == 0 else None)
            ax.text((r.beta_safe + r.beta_look_ahead) / 2, i - 0.3,
                    f"diff {r.diff_in_hypothesised_direction:+.2f} [{r.ci95[0]:+.2f}, {r.ci95[1]:+.2f}]", fontsize=7.5, ha="center")
        ax.set_yticks(range(len(d)), d.architecture.tolist())
        ax.set_ylim(-0.6, len(d) - 0.3)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("coefficient per 1-SD surprise")
    axes[0].legend(fontsize=8, loc="upper left")
    suptitle(fig, "H3: same architecture, different label window (diff = look-ahead minus safe, hypothesised direction; 95% bootstrap CI)",
                 fontsize=10)
    fig.savefig(FIGURES / "f04_lookahead.png")
    plt.close(fig)


def f05_permutation(est):
    from common.stats import _hc3_t
    z = f"z_{es.PRIMARY}"
    d = est[["hf_TFUT02", z, "hf_MP1"]].dropna()
    Y = d.hf_TFUT02.to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d[z], d.hf_MP1])
    t_obs = _hc3_t(X, Y, np.linalg.inv(X.T @ X), 1)[1]
    rng = np.random.default_rng(2)
    ts = []
    Xp = X.copy()
    for _ in range(9999):
        Xp[:, 1] = rng.permutation(X[:, 1])
        ts.append(_hc3_t(Xp, Y, np.linalg.inv(Xp.T @ Xp), 1)[1])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(ts, bins=80, color=GREY, alpha=0.7)
    ax.axvline(t_obs, color=ACCENT, lw=2, label=f"observed t = {t_obs:.2f}")
    ax.set_xlabel("HC3 t-statistic when tone surprises are shuffled across meetings")
    ax.set_ylabel("count")
    ax.set_title(f"H1 permutation test, primary measure {es.PRIMARY} (9,999 shuffles)")
    ax.legend()
    fig.savefig(FIGURES / "f05_permutation_h1.png")
    plt.close(fig)


def f06_lp(ph):
    lp, pre, ctl = ph["local_projections"], ph.get("pre_event_placebo", {}), ph.get("local_projections_pretrend_controlled", {})
    titles = {"DGS2": "2-year Treasury yield (bp)", "DGS10": "10-year Treasury yield (bp)", "mkt": "cumulative market return (%)"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, (k, t) in zip(axes, titles.items()):
        d = pd.DataFrame(lp[k])
        ax.fill_between(d.h, d.beta - 1.96 * d.se, d.beta + 1.96 * d.se, color=NAVY, alpha=0.13)
        ax.plot(d.h, d.beta, "-o", color=NAVY, ms=3, label="after the meeting (h = 0..20)")
        if k in pre:
            q = pd.DataFrame(pre[k])
            ax.errorbar(-q.k, q.beta, yerr=1.96 * q.se, fmt="o", color=GREY, ms=3, lw=0.8, capsize=0,
                        label="pre-event placebo (change over the k days before)")
        if k in ctl:
            c = pd.DataFrame(ctl[k])
            ax.plot(c.h, c.beta, "--", color=RED, lw=1.4, label="controlling for the 20-day pre-trend")
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(-0.5, color="black", lw=0.5, ls=":")
        ax.set_title(t, fontsize=10.5)
        ax.set_xlabel("trading days relative to the meeting")
    axes[0].set_ylabel("response to a 1-SD tone surprise")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    suptitle(fig, f"Local projections on the primary tone surprise ({es.PRIMARY}) + MP1, with pre-event placebo horizons; "
                 "95% Newey-West bands", fontsize=10.5)
    fig.savefig(FIGURES / "f06_local_projections.png")
    plt.close(fig)


def f07_industries(ph):
    d = pd.DataFrame(ph["H7"]["industries"]).sort_values("beta")
    grp = set(ph["H7"]["rate_sensitive_group"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    cols = [RED if r.industry in grp else (NAVY if r.p_holm < 0.05 else GREY) for r in d.itertuples()]
    ax.bar(range(len(d)), d.beta, color=cols)
    ax.errorbar(range(len(d)), d.beta, yerr=1.96 * d.se_hc3, fmt="none", ecolor="black", lw=0.6, capsize=1.5)
    ax.set_xticks(range(len(d)), d.industry, rotation=90, fontsize=7.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("daily return response, % per 1-SD surprise")
    ax.set_title("H7: daily industry responses net of the market (dark = pre-specified rate-sensitive group; "
                 "blue = Holm p<0.05)\n"
                 f"Group minus rest = {ph['H7']['group_minus_rest']:+.3f} "
                 f"[{ph['H7']['group_minus_rest_ci95'][0]:+.3f}, {ph['H7']['group_minus_rest_ci95'][1]:+.3f}]", fontsize=9.5)
    fig.savefig(FIGURES / "f07_industries.png")
    plt.close(fig)


def f08_holdout(core, ho):
    """In the holdout, full-sample and chronological models coincide (identical surprises): one row per distinct model."""
    est = pd.DataFrame(core["results"]).query("test == 'H1'").set_index("measure")
    h = pd.DataFrame(ho["results"]).query("test == 'H1'").set_index("measure")
    rows = [("M3", "RoBERTa-large, all 1996-2022 labels (= M2 = M3 in the holdout)", "M3"),
            ("M3b", "TF-IDF, all 1996-2022 labels (= M2b = M3b in the holdout)", "M3b"),
            ("M1", MEASURE_LABEL["M1"], "M1")]
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    for i, (m, lab, c) in enumerate(rows):
        ax.errorbar(est.loc[m, "beta"], i + 0.12, xerr=1.96 * est.loc[m, "se_hc3"], fmt="o", color=MEASURE_COLOR[c], capsize=3)
        ax.errorbar(h.loc[m, "beta"], i - 0.12, xerr=1.96 * h.loc[m, "se_hc3"], fmt="s", mfc="white", color=MEASURE_COLOR[c], capsize=3)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8.5)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("H1: bp change in the 2-year yield per 1-SD tone surprise (estimation: chronological measure)")
    ax.set_title(f"Estimation (filled, n={core['sample']['n_events']}) vs untouched holdout 2023-2026 (open, n={ho['sample']['n_events']})")
    fig.savefig(FIGURES / "f08_holdout.png")
    plt.close(fig)


def f09_partial(est):
    z = f"z_{es.PRIMARY}"
    d = est[["hf_TFUT02", z, "hf_MP1"]].dropna()
    Xc = np.column_stack([np.ones(len(d)), d.hf_MP1])
    ry = d.hf_TFUT02 - Xc @ np.linalg.lstsq(Xc, d.hf_TFUT02, rcond=None)[0]
    rx = d[z] - Xc @ np.linalg.lstsq(Xc, d[z], rcond=None)[0]
    b = float(rx @ ry / (rx @ rx))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(rx, ry, s=18, color=NAVY, alpha=0.6)
    xs = np.linspace(rx.min(), rx.max(), 50)
    ax.plot(xs, b * xs, color=RED, lw=1.5, label=f"slope = {b:.2f} bp per SD")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(f"tone surprise ({es.PRIMARY}), net of MP1, SD units")
    ax.set_ylabel("30-min 2-year yield change, net of MP1 (bp)")
    ax.set_title("H1 partial-regression plot, estimation sample")
    ax.legend()
    fig.savefig(FIGURES / "f09_partial_regression_h1.png")
    plt.close(fig)


def main() -> None:
    ev = pd.read_csv(PROCESSED / "events_model.csv", parse_dates=["date"])
    est = es.load_estimation()
    f01_timeline(ev)
    f02_validation(load("validation_summary.json"))
    core = load("results_core.json")
    f03_core(core)
    f04_lookahead(load("results_lookahead.json"))
    f05_permutation(est)
    f06_lp(load("results_persistence_hetero.json"))
    f07_industries(load("results_persistence_hetero.json"))
    ho = load("results_holdout.json")
    if ho:
        f08_holdout(core, ho)
    f09_partial(est)
    print("figures:", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
