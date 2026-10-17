"""Step 2: download all market data, policy-surprise series and the labelled sentence corpus.

Outputs (data/raw/market/ unless noted)
- jk_surprises.csv        Jarocinski-Karadi 30-min surprises around FOMC announcements (FF, ED, Treasury futures,
                          S&P 500, EUR), 1988 onwards; github.com/marekjarocinski/jkshocks_update_fed
- jk_shocks.csv           JK policy factor pc1 and the MP / CBI decomposition, by announcement
- bauer_swanson.xlsx      SF Fed monetary policy surprises (Bauer & Swanson 2023), to 2023
- fred_<ID>.csv           FRED daily series (Treasury yields, VIX, fed funds target: DFEDTAR to 2008, DFEDTARU/L after)
- kf_49ind_daily.csv      Ken French 49 industry portfolios, value-weighted daily returns (%)
- kf_ff3_daily.csv        Ken French Fama-French 3 factors + RF, daily (%)
- data/raw/tdw/{train,test}.csv   Trillion Dollar Words labelled sentences (Shah, Paturi & Chava 2023;
                          huggingface.co/datasets/gtfintechlab/fomc_communication; CC BY-NC 4.0)
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import MARKET, RAW, USER_AGENT  # noqa: E402

# pinned to the commit used for the paper (pushed 2026-10-11); change deliberately to update the data
JK_COMMIT = "f788aef7d573f87f28b70ffc3763e73c4886a53f"
JK_BASE = f"https://raw.githubusercontent.com/marekjarocinski/jkshocks_update_fed/{JK_COMMIT}/"
SF_FED_XLSX = "https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx"
FRED_SERIES = ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30", "VIXCLS", "DFEDTAR", "DFEDTARU", "DFEDTARL"]
KF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
TDW_BASE = "https://huggingface.co/datasets/gtfintechlab/fomc_communication/resolve/main/"

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def get(url: str, tries: int = 5) -> bytes:
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def read_kf_daily(zip_bytes: bytes) -> pd.DataFrame:
    """Parse the first (value-weighted) daily block of a Ken French CSV inside a zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith(",") or ln.lower().startswith("date,"))
    header = [h.strip() for h in lines[start].split(",")]
    header[0] = "date"
    rows = []
    for ln in lines[start + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            break  # end of the first block
        rows.append(parts)
    df = pd.DataFrame(rows, columns=header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.replace([-99.99, -999.0], pd.NA)


def main() -> None:
    (MARKET / "jk_surprises.csv").write_bytes(get(JK_BASE + "source_data/fomc_surprises_jk.csv"))
    (MARKET / "jk_shocks.csv").write_bytes(get(JK_BASE + "shocks_fed_jk_t.csv"))
    (MARKET / "bauer_swanson.xlsx").write_bytes(get(SF_FED_XLSX))
    for sid in FRED_SERIES:
        (MARKET / f"fred_{sid}.csv").write_bytes(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"))

    ind = read_kf_daily(get(KF_BASE + "49_Industry_Portfolios_daily_CSV.zip"))
    ind.to_csv(MARKET / "kf_49ind_daily.csv", index=False)
    ff3 = read_kf_daily(get(KF_BASE + "F-F_Research_Data_Factors_daily_CSV.zip"))
    ff3.to_csv(MARKET / "kf_ff3_daily.csv", index=False)

    tdw = RAW / "tdw"
    tdw.mkdir(exist_ok=True)
    for split in ("train", "test"):
        (tdw / f"{split}.csv").write_bytes(get(TDW_BASE + f"{split}.csv"))

    # Short provenance report
    jk = pd.read_csv(MARKET / "jk_surprises.csv")
    print(f"JK surprises: {len(jk)} events, {jk['start'].iloc[0]} .. {jk['start'].iloc[-1]}")
    print(f"KF 49 industries: {ind.shape}, {ind.date.min().date()} .. {ind.date.max().date()}")
    print(f"KF FF3: {ff3.shape}, {ff3.date.min().date()} .. {ff3.date.max().date()}")
    for sid in FRED_SERIES:
        f = pd.read_csv(MARKET / f"fred_{sid}.csv")
        print(f"FRED {sid}: {len(f)} rows, last {f.iloc[-1, 0]}")
    for split in ("train", "test"):
        t = pd.read_csv(tdw / f"{split}.csv")
        print(f"TDW {split}: {t.shape}, years {t['year'].min()}-{t['year'].max()}, labels {t['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
