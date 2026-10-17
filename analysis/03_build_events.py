"""Step 3: align every FOMC statement with its market data.

One row per statement with:
- the Jarocinski-Karadi (JK) 30-minute surprises (fed funds / Eurodollar futures, Treasury futures 2/5/10/30y,
  S&P 500, EUR) and the JK policy factor pc1 with its monetary-policy / information decomposition
- Bauer-Swanson MPS and MPS_ORTH (available to 2023-12)
- daily changes (statement day close minus previous trading day close) in FRED Treasury yields and VIX
- the daily market return (Ken French Mkt-RF + RF)
- sample flags: event type, main sample, holdout, ZLB

Units: JK interest-rate surprises are in percentage points (pp) and are converted here to basis points (bp);
S&P 500 surprises are 100 x log change (percent). FRED yield changes are converted to bp.

Outputs
- data/processed/events.csv
- data/processed/events_industries_daily.csv   (49 Ken French industry returns on each statement day, %)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import MARKET, PROCESSED, USER_AGENT  # noqa: E402

# Pre-specified sample definitions (PROJECT_PLAN.md sections 3 and 6)
MAIN_START = "1999-05-18"          # first meeting after which a statement was released at every meeting
HOLDOUT_START = "2023-01-01"       # after FOMC-RoBERTa's training data (1996-2022); locked until milestone M4
# ZLB: the policy-rate floor was already zero *before* the announcement
ZLB_WINDOWS = [("2008-12-16", "2015-12-16"), ("2020-03-15", "2022-03-16")]

JK_RATE_COLS = ["FF1", "FF2", "FF3", "FF4", "MP1", "ED1", "ED2", "ED3", "ED4",
                "TFUT02", "TFUT05", "TFUT10", "TFUT30"]
JK_PCT_COLS = ["SP500", "SP500FUT", "EUR"]
FRED_YIELDS = ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30"]


def classify(event: str) -> str:
    if not isinstance(event, str):
        return "no_hf_data"
    if event == "FOMC Rate Decision (Scheduled)":
        return "scheduled"
    return "unscheduled_or_other"


def fed_scheduled_end_dates() -> set:
    """End dates of SCHEDULED FOMC meetings according to the Fed's own calendar pages (cached in data/raw).

    Historical pages (1994-2020) head every event as '<Month D[-D]> Meeting - YYYY', with '(unscheduled)',
    'Conference Call' or '(notation vote)' for the others; the current calendar (2021+) lists scheduled meetings only.
    Added after the M8 review: the JK file labels the 2020-03-03 emergency cut 'Scheduled' (deviations.md #14).
    """
    import json
    import re
    import time

    import requests
    from bs4 import BeautifulSoup

    cache = MARKET.parent / "fed_scheduled_meetings.json"
    if cache.exists():
        return set(json.loads(cache.read_text()))
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    def end_date(text: str, year: str):
        """Last month token + last day number, e.g. 'January 31-February 1' or 'Jan/Feb 31-1' -> Feb 1."""
        mon = re.findall(r"[A-Z][a-z]{2,8}", text)
        day = re.findall(r"\d{1,2}", text)
        return pd.Timestamp(f"{mon[-1]} {day[-1]} {year}").date().isoformat() if mon and day else None

    out = set()
    for y in range(1994, 2021):
        soup = BeautifulSoup(s.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm", timeout=40).text, "lxml")
        for h in soup.find_all(["h3", "h4", "h5"]):
            t = h.get_text(" ", strip=True)
            m = re.match(r"(.+) Meeting - (\d{4})$", t)
            if m and "(" not in m.group(1):  # '(unscheduled)', '(cancelled)' etc. are not regular meetings
                d = end_date(m.group(1), m.group(2))
                if d:
                    out.add(d)
        time.sleep(0.6)
    soup = BeautifulSoup(s.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", timeout=40).text, "lxml")
    for panel in soup.select("div.panel.panel-default"):
        yr = re.search(r"(\d{4})", panel.select_one(".panel-heading").get_text())
        for mtg in panel.select("div.fomc-meeting"):
            month = mtg.select_one(".fomc-meeting__month").get_text(" ", strip=True)
            date_txt = mtg.select_one(".fomc-meeting__date").get_text(" ", strip=True)
            if yr and "(" not in date_txt:  # skip '(notation vote)' and similar
                d = end_date(f"{month} {date_txt}", yr.group(1))
                if d:
                    out.add(d)
    cache.write_text(json.dumps(sorted(out), indent=0))
    return out


def daily_change(series: pd.Series, dates: pd.Series) -> pd.Series:
    """Close on the event date minus the previous available close (NaN if the event day has no close)."""
    s = series.dropna().sort_index()
    out = []
    for d in dates:
        if d not in s.index:
            out.append(np.nan)
            continue
        pos = s.index.get_loc(d)
        out.append(s.iloc[pos] - s.iloc[pos - 1] if pos > 0 else np.nan)
    return pd.Series(out, index=dates.index)


def in_zlb(d: pd.Timestamp) -> bool:
    return any(pd.Timestamp(a) < d <= pd.Timestamp(b) for a, b in ZLB_WINDOWS)


def main() -> None:
    st = pd.read_csv(PROCESSED / "statements_index.csv", dtype={"id": str})
    st["date"] = pd.to_datetime(st["date"])

    # --- JK 30-minute surprises and shocks ---------------------------------------------------
    jk = pd.read_csv(MARKET / "jk_surprises.csv")
    jk["start"] = pd.to_datetime(jk["start"])
    jk["date"] = jk["start"].dt.normalize()
    dup = jk[jk.duplicated("date", keep=False)]
    if len(dup):
        print("JK dates with >1 event (keeping the FOMC rate decision, else the first):")
        print(dup[["start", "Event"]].to_string())
        jk["_prio"] = (~jk["Event"].str.startswith("FOMC Rate Decision")).astype(int)
        jk = jk.sort_values(["date", "_prio", "start"]).drop_duplicates("date").drop(columns="_prio")
    for c in JK_RATE_COLS:
        jk[c] = jk[c] * 100.0  # pp -> bp
    jk = jk.rename(columns={c: f"hf_{c}" for c in JK_RATE_COLS + JK_PCT_COLS})

    shocks = pd.read_csv(MARKET / "jk_shocks.csv")
    shocks["start"] = pd.to_datetime(shocks["start"])
    for c in ["pc1", "MP_pm", "CBI_pm", "MP_median", "CBI_median"]:
        shocks[c] = shocks[c] * 100.0  # pp -> bp
    shocks = shocks.drop(columns=["SP500"]).rename(columns=lambda c: c if c == "start" else f"jk_{c}")
    jk = jk.merge(shocks, on="start", how="left")

    ev = st.merge(jk.rename(columns={"start": "hf_time", "Event": "hf_event"}), on="date", how="left")
    ev["event_type"] = ev["hf_event"].map(classify)
    # cross-check with the Fed's own calendar: 'scheduled' requires BOTH sources (deviations.md #14)
    fed_sched = fed_scheduled_end_dates()
    ev["fed_scheduled"] = ev["date"].dt.strftime("%Y-%m-%d").isin(fed_sched).astype(int)
    disagree = ev[(ev.event_type == "scheduled") != (ev.fed_scheduled == 1)]
    if len(disagree):
        print("JK vs Fed calendar disagreements (scheduled only if both agree):")
        print(disagree[["id", "date", "hf_event", "fed_scheduled"]].to_string())
    ev.loc[(ev.event_type == "scheduled") & (ev.fed_scheduled == 0), "event_type"] = "unscheduled_or_other"
    ev.loc[ev["title"].str.contains("Longer-Run Goals and Monetary Policy Strategy", case=False, na=False),
           "event_type"] = "strategy_statement"

    # --- Bauer-Swanson (SF Fed) ---------------------------------------------------------------
    bs = pd.read_excel(MARKET / "bauer_swanson.xlsx", sheet_name="FOMC (update 2023)")
    bs["date"] = pd.to_datetime(bs["Date"])
    bs = bs.groupby("date", as_index=False)[["MPS", "MPS_ORTH"]].first()
    bs[["MPS", "MPS_ORTH"]] = bs[["MPS", "MPS_ORTH"]] * 100.0  # pp -> bp
    ev = ev.merge(bs.rename(columns={"MPS": "bs_MPS", "MPS_ORTH": "bs_MPS_ORTH"}), on="date", how="left")

    # --- Daily FRED changes ------------------------------------------------------------------
    for sid in FRED_YIELDS + ["VIXCLS"]:
        f = pd.read_csv(MARKET / f"fred_{sid}.csv")
        f.columns = ["date", "value"]
        f["date"] = pd.to_datetime(f["date"])
        f["value"] = pd.to_numeric(f["value"], errors="coerce")
        s = f.set_index("date")["value"]
        chg = daily_change(s, ev["date"])
        ev[f"d_{sid}"] = chg * (100.0 if sid != "VIXCLS" else 1.0)  # yields: pp -> bp; VIX in points

    # --- Fed funds target (upper bound from 2008-12-16) just after the decision (FRED dates a change
    #     from its effective day, often the day after the meeting, so look 3 days ahead) -----------
    tgt = []
    for sid in ("DFEDTAR", "DFEDTARU"):
        f = pd.read_csv(MARKET / f"fred_{sid}.csv")
        f.columns = ["date", "v"]
        f["date"] = pd.to_datetime(f["date"])
        tgt.append(f.assign(v=pd.to_numeric(f["v"], errors="coerce")).dropna())
    tgt = pd.concat(tgt).drop_duplicates("date", keep="last").set_index("date")["v"].sort_index()
    ev["fed_target_upper"] = [tgt.loc[:d + pd.Timedelta(days=3)].iloc[-1] for d in ev["date"]]

    # --- Ken French daily market and industries ----------------------------------------------
    ff3 = pd.read_csv(MARKET / "kf_ff3_daily.csv", parse_dates=["date"]).set_index("date")
    ev["d_mkt"] = ev["date"].map(ff3["Mkt-RF"] + ff3["RF"])
    ind = pd.read_csv(MARKET / "kf_49ind_daily.csv", parse_dates=["date"]).set_index("date")
    ind_ev = ev[["id", "date"]].join(ind, on="date")

    # --- Flags ---------------------------------------------------------------------------------
    ev["year"] = ev["date"].dt.year
    ev["zlb"] = ev["date"].map(in_zlb).astype(int)
    ev["holdout"] = (ev["date"] >= HOLDOUT_START).astype(int)
    ev["main_sample"] = ((ev["event_type"] == "scheduled") & (ev["date"] >= MAIN_START)).astype(int)

    ev.to_csv(PROCESSED / "events.csv", index=False)
    ind_ev.to_csv(PROCESSED / "events_industries_daily.csv", index=False)

    # --- Report --------------------------------------------------------------------------------
    print(f"\nstatements: {len(ev)} | by type: {ev.event_type.value_counts().to_dict()}")
    main_s = ev[ev.main_sample == 1]
    print(f"main sample (scheduled, >= {MAIN_START}): {len(main_s)} "
          f"= estimation {int((main_s.holdout == 0).sum())} + holdout {int(main_s.holdout.sum())}; "
          f"ZLB events {int(main_s.zlb.sum())}")
    cols = ["hf_MP1", "hf_TFUT02", "hf_TFUT10", "hf_SP500", "jk_pc1", "bs_MPS", "d_DGS2", "d_DGS10", "d_mkt"]
    print("non-missing in main sample:", {c: int(main_s[c].notna().sum()) for c in cols})
    print(main_s[cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
