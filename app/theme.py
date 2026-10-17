"""ORBIT theme (DESIGN_SYSTEM.md): SpaceX-inspired dark interface for Streamlit dashboards.

Black canvas, off-white ink in opacity steps, D-DIN (bundled, SIL OFL) in UPPERCASE with wide tracking for headings,
Roboto Mono for numbers, hairline dividers, big stat blocks, and one accent colour (#3E6AE1).
Fonts are served from app/static/fonts (requires `server.enableStaticServing = true` in .streamlit/config.toml).
Reusable across portfolio projects: copy this file plus app/static/fonts/.
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG, PANEL = "#000000", "#0B0C0E"
INK = "rgb(240,240,250)"
INK70, INK50, INK35, INK15, INK08 = ("rgba(240,240,250,.70)", "rgba(240,240,250,.50)", "rgba(240,240,250,.35)",
                                     "rgba(240,240,250,.15)", "rgba(240,240,250,.08)")
ACCENT, ACCENT_TINT, ACCENT2, DANGER = "#3E6AE1", "#8EA8EC", "#E8B04B", "#E5484D"
GREY_A, GREY_B, GREY_C = "#A3A3AD", "#6E6E78", "#4A4A52"

_CSS = f"""
<style>
@font-face {{ font-family: 'D-DIN'; src: url('app/static/fonts/D-DIN.woff2') format('woff2'); font-weight: 400; }}
@font-face {{ font-family: 'D-DIN'; src: url('app/static/fonts/D-DIN-Bold.woff2') format('woff2'); font-weight: 700; }}
@font-face {{ font-family: 'Roboto Mono'; src: url('app/static/fonts/RobotoMono.ttf') format('truetype'); }}

html, body, [class*="css"], .stApp, .stMarkdown, p, li, label, span, div {{
  font-family: 'D-DIN', 'Inter', -apple-system, Arial, sans-serif; }}
.stApp {{ background: {BG}; color: {INK70}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1380px; padding-top: 2.2rem; padding-bottom: 4rem; }}
h1, h2, h3, h4 {{ font-family: 'D-DIN', sans-serif !important; text-transform: uppercase; letter-spacing: .06em;
  color: {INK} !important; font-weight: 400 !important; }}
h1 {{ font-size: 2.6rem !important; }} h2 {{ font-size: 1.7rem !important; }} h3 {{ font-size: 1.15rem !important; }}
p, li {{ color: {INK70}; font-size: 1.0rem; line-height: 1.55; }}
a {{ color: {INK} !important; text-decoration: none; border-bottom: 1px solid {INK35}; }}
hr {{ border: none; border-top: 1px solid {INK15}; margin: 2rem 0; }}

/* sidebar = minimal mission nav */
[data-testid="stSidebar"] {{ background: {BG}; border-right: 1px solid {INK15}; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{ text-transform: uppercase; letter-spacing: .12em;
  font-size: .78rem; color: {INK70}; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: .35rem 0; }}

/* metrics = big numbers with tiny uppercase labels */
[data-testid="stMetricLabel"] p {{ text-transform: uppercase; letter-spacing: .14em; font-size: .72rem !important;
  color: {INK50} !important; }}
[data-testid="stMetricValue"] {{ font-family: 'D-DIN', sans-serif; font-size: 2.6rem !important; color: {INK};
  letter-spacing: .01em; }}
[data-testid="stMetricDelta"] {{ font-family: 'Roboto Mono', monospace; font-size: .75rem; }}

/* widgets */
.stButton > button, .stDownloadButton > button {{ background: transparent; color: {INK}; border: 1px solid {INK};
  border-radius: 4px; text-transform: uppercase; letter-spacing: .14em; font-size: .78rem; padding: .55rem 1.4rem; }}
.stButton > button:hover {{ background: {INK}; color: {BG}; }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{ background: {PANEL}; border-color: {INK15}; border-radius: 4px; }}
[data-testid="stExpander"] {{ border: 1px solid {INK15}; border-radius: 4px; background: {BG}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {INK15}; border-radius: 4px; }}
code {{ font-family: 'Roboto Mono', monospace; color: {INK}; background: {PANEL}; }}
.stAlert {{ background: {PANEL}; border: 1px solid {INK15}; border-radius: 4px; color: {INK70}; }}
[data-testid="stCaptionContainer"] p {{ color: {INK50}; font-size: .82rem; letter-spacing: .02em; }}

/* ORBIT components */
.orbit-hero {{ padding: 3.2rem 0 2.4rem 0; border-bottom: 1px solid {INK15}; margin-bottom: 2.2rem; }}
.orbit-kicker {{ text-transform: uppercase; letter-spacing: .32em; font-size: .78rem; color: {INK50}; margin-bottom: 1rem; }}
.orbit-title {{ font-family: 'D-DIN', sans-serif; text-transform: uppercase; letter-spacing: .05em; font-size: 4.6rem;
  line-height: 1.0; color: {INK}; margin: 0 0 1.1rem 0; }}
.orbit-sub {{ font-size: 1.15rem; color: {INK70}; max-width: 880px; line-height: 1.55; }}
.orbit-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0;
  margin-top: 2.6rem; border-top: 1px solid {INK15}; }}
.orbit-stat {{ padding: 1.4rem 1.2rem 0 0; }}
.orbit-num {{ font-family: 'D-DIN', sans-serif; font-size: 3.1rem; color: {INK}; letter-spacing: .01em; line-height: 1.05; }}
.orbit-num .unit {{ font-size: 1.2rem; color: {INK50}; margin-left: .25rem; letter-spacing: .08em; }}
.orbit-label {{ text-transform: uppercase; letter-spacing: .16em; font-size: .7rem; color: {INK50}; margin-top: .5rem; }}
.orbit-section {{ text-transform: uppercase; letter-spacing: .3em; font-size: .75rem; color: {INK50};
  border-top: 1px solid {INK15}; padding-top: 1rem; margin: 2.4rem 0 1rem 0; }}
.orbit-accent {{ color: {ACCENT}; }}
.orbit-stats.bare {{ margin: 0 0 1.6rem 0; padding-bottom: 1.6rem; border-bottom: 1px solid {INK15}; }}

/* ORBIT table: micro uppercase headers, hairlines, mono right-aligned numbers, wrapping text */
.orbit-scroll {{ overflow-x: auto; margin: .2rem 0 1.4rem 0; }}
table.orbit-table {{ width: 100%; border-collapse: collapse; font-size: .92rem; border: none !important; }}
table.orbit-table th, table.orbit-table td {{ border-top: none !important; border-left: none !important;
  border-right: none !important; background: transparent; }}
table.orbit-table th {{ text-transform: uppercase; letter-spacing: .14em; font-size: .66rem; font-weight: 400;
  color: {INK50}; text-align: left; padding: .55rem .8rem; border-bottom: 1px solid {INK35}; white-space: nowrap; }}
table.orbit-table td {{ color: {INK70}; padding: .6rem .8rem; border-bottom: 1px solid {INK08}; vertical-align: top;
  line-height: 1.45; }}
table.orbit-table .num {{ text-align: right; font-family: 'Roboto Mono', monospace; font-size: .84rem; white-space: nowrap; }}
table.orbit-table td.num {{ color: {INK}; }}
table.orbit-table td.hi {{ color: {ACCENT}; }}
table.orbit-table tr:hover td {{ background: rgba(240,240,250,.03); }}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    pio.templates["orbit"] = go.layout.Template(layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="D-DIN, Inter, Arial, sans-serif", color="rgba(240,240,250,.75)", size=13),
        title=dict(font=dict(size=14, color=INK), x=0, xanchor="left"),
        colorway=[ACCENT, INK, GREY_A, ACCENT2, ACCENT_TINT, GREY_B],
        xaxis=dict(gridcolor=INK08, zerolinecolor=INK35, linecolor=INK15, tickcolor=INK15, showgrid=False),
        yaxis=dict(gridcolor=INK08, zerolinecolor=INK35, linecolor=INK15, tickcolor=INK15),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#111214", bordercolor=INK15, font=dict(family="Roboto Mono, monospace", color=INK)),
        margin=dict(l=10, r=10, t=50, b=40)))
    pio.templates.default = "orbit"


def hero(kicker: str, title: str, subtitle: str, stats: list) -> None:
    """Full-width hero: kicker, giant uppercase title, one-line subtitle, row of (number, unit, label) stats."""
    st.markdown(f"<div class='orbit-hero'><div class='orbit-kicker'>{kicker}</div>"
                f"<div class='orbit-title'>{title}</div><div class='orbit-sub'>{subtitle}</div>"
                f"<div class='orbit-stats'>{_cells(stats)}</div></div>", unsafe_allow_html=True)


def _cells(stats: list) -> str:
    return "".join(f"<div class='orbit-stat'><div class='orbit-num'>{n}<span class='unit'>{u}</span></div>"
                   f"<div class='orbit-label'>{lab}</div></div>" for n, u, lab in stats)


def stat_row(stats: list) -> None:
    """Big-number-first row for the top of a page: list of (number, unit, label)."""
    st.markdown(f"<div class='orbit-stats bare'>{_cells(stats)}</div>", unsafe_allow_html=True)


def table(df: pd.DataFrame, fmt: dict | None = None, hi: dict | None = None, nowrap: tuple = ()) -> None:
    """ORBIT HTML table. fmt: column -> format string ('{:+.3f}') or callable; formatted and numeric columns are
    right-aligned in Roboto Mono. hi: column -> predicate(value) that paints the cell in the accent colour.
    nowrap: short text columns that should stay on one line."""
    fmt, hi = fmt or {}, hi or {}
    num = {c for c in df.columns if c in fmt or pd.api.types.is_numeric_dtype(df[c])}

    def cell(c, v):
        if c in fmt:
            s = fmt[c](v) if callable(fmt[c]) else fmt[c].format(v)
        elif v is None or (isinstance(v, float) and np.isnan(v)):
            s = "—"
        elif isinstance(v, (float, np.floating)):
            s = f"{v:,.3f}"
        else:
            s = str(v)
        if c in num:
            s = s.replace("-", "−")  # typographic minus in numbers
        cls = " ".join(k for k, on in [("num", c in num), ("hi", c in hi and hi[c](v))] if on)
        style = " style='white-space:nowrap'" if c in nowrap else ""
        return f"<td class='{cls}'{style}>{html.escape(s)}</td>"

    head = "".join(f"<th class='{'num' if c in num else ''}'>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(cell(c, r[c]) for c in df.columns) + "</tr>" for _, r in df.iterrows())
    st.markdown(f"<div class='orbit-scroll'><table class='orbit-table'><thead><tr>{head}</tr></thead>"
                f"<tbody>{body}</tbody></table></div>", unsafe_allow_html=True)


def ci(c) -> str:
    """Format a 2-element CI as [lo, hi]."""
    return f"[{c[0]:+.2f}, {c[1]:+.2f}]"


def section(label: str) -> None:
    st.markdown(f"<div class='orbit-section'>{label}</div>", unsafe_allow_html=True)


def page_title(kicker: str, title: str) -> None:
    st.markdown(f"<div class='orbit-kicker'>{kicker}</div><h1 style='margin-top:0'>{title}</h1>", unsafe_allow_html=True)
