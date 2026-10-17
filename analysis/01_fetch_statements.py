"""Step 1: download every FOMC post-meeting statement (1994 onwards) from federalreserve.gov.

Sources
- Historical year pages /monetarypolicy/fomchistoricalYYYY.htm (links labelled "Statement"), 1994-2020.
- The current calendar page /monetarypolicy/fomccalendars.htm (statement HTML links), 2021 onwards.

Outputs
- data/raw/statements_html/<id>.html   raw pages (cached; re-runs skip downloads)
- data/raw/statements_txt/<id>.txt     cleaned statement text, one paragraph per line
- data/processed/statements_index.csv  id, date, url, n_paragraphs, n_words

<id> is the release date YYYYMMDD; a second statement on the same date gets a suffix (_2, ...).
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED, STATEMENTS_HTML, STATEMENTS_TXT, USER_AGENT  # noqa: E402

BASE = "https://www.federalreserve.gov"
FIRST_YEAR, LAST_HISTORICAL_YEAR = 1994, 2020
SLEEP = 0.6  # seconds between requests (polite crawling)

# Lines that belong to the page frame, not to the statement itself.
HEADER_PATTERNS = [
    r"^release date:", r"^for immediate release", r"^for release at", r"^share$",
    r"^fomc statement$", r"^federal reserve issues fomc statement", r"^press release$", r"^frb: press release",
    r"^(january|february|march|april|may|june|july|august|september|october|november|december) \d{1,2}, \d{4}$",
]
FOOTER_PATTERNS = [
    r"^for media inquiries", r"^implementation note", r"^last update", r"^home\b", r"^accessibility",
    r"^contact us", r"^\d{4} monetary policy\b", r"^news and events$", r"^press releases$",
    r"^board of governors of the federal reserve system$", r"^related information",
    r"^attachment", r"^\(?\d+(\.\d+)? ?kb pdf\)?",
]
_header_re = re.compile("|".join(HEADER_PATTERNS), re.I)
_footer_re = re.compile("|".join(FOOTER_PATTERNS), re.I)

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def fetch(url: str) -> str:
    """GET a page with retries; decode as UTF-8 (the site declares it inconsistently)."""
    for attempt in range(4):
        try:
            r = session.get(url, timeout=40)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r.text
            if r.status_code == 404:
                raise FileNotFoundError(url)
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after retries: {url}")


def statement_links() -> list[tuple[str, str]]:
    """Return sorted (date YYYYMMDD, absolute URL) pairs for every statement link."""
    links = []
    for year in range(FIRST_YEAR, LAST_HISTORICAL_YEAR + 1):
        soup = BeautifulSoup(fetch(f"{BASE}/monetarypolicy/fomchistorical{year}.htm"), "lxml")
        for a in soup.find_all("a", href=True):
            if a.get_text(strip=True).lower() == "statement":
                links.append(a["href"])
        time.sleep(SLEEP)
    soup = BeautifulSoup(fetch(f"{BASE}/monetarypolicy/fomccalendars.htm"), "lxml")
    for a in soup.find_all("a", href=True):
        if re.search(r"/newsevents/pressreleases/monetary\d{8}a\.htm$", a["href"]):
            links.append(a["href"])

    out = {}
    for href in links:
        url = href if href.startswith("http") else BASE + href
        m = re.search(r"(\d{8})", url)
        if not m:
            continue
        out[url] = m.group(1)
    return sorted(((d, u) for u, d in out.items()), key=lambda x: (x[0], x[1]))


def _clean(s: str) -> str:
    s = s.replace("\xa0", " ").replace("‑", "-").replace("–", "-").replace("—", " - ")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def extract_paragraphs(html: str) -> list[str]:
    """Pull the statement paragraphs out of a page from any era of the Fed website."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    node = soup.select_one("div#article") or soup.select_one("div#content") or soup.body
    # Old pages mix <p> paragraphs with bare text and hard-wrapped source lines, so mark every
    # block-level boundary explicitly and join everything else with spaces.
    for tag in node.find_all(["p", "br", "div", "td", "tr", "li", "h1", "h2", "h3", "h4", "table"]):
        tag.insert_before("\n¶\n")
    blocks = [_clean(x) for x in node.get_text(" ").split("¶")]
    blocks = [b for b in blocks if b]

    paras, started = [], False
    for b in blocks:
        if _footer_re.search(b) and started:
            break
        if _header_re.search(b):
            continue
        if len(b) < 25 and not started:  # stray navigation fragments before the text
            continue
        if len(b.split()) < 5 and not re.search(r"[.:;]$", b):  # link-list items (joint CB releases)
            continue
        started = True
        paras.append(b)
    return paras


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_date_re = re.compile(rf"({_MONTHS}) (\d{{1,2}}), (\d{{4}})")


def page_release_date(html: str) -> str | None:
    """Release date printed at the top of the page (the URL date is wrong for at least one page)."""
    soup = BeautifulSoup(html, "lxml")
    node = soup.select_one("div#article") or soup.body
    m = _date_re.search(_clean(node.get_text(" "))[:400])
    return pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}").date().isoformat() if m else None


def main() -> None:
    pairs = statement_links()
    print(f"found {len(pairs)} statement links ({pairs[0][0]} .. {pairs[-1][0]})")
    rows, seen = [], {}
    for date, url in pairs:
        seen[date] = seen.get(date, 0) + 1
        sid = date if seen[date] == 1 else f"{date}_{seen[date]}"
        html_path = STATEMENTS_HTML / f"{sid}.html"
        if not html_path.exists():
            html_path.write_text(fetch(url), encoding="utf-8")
            time.sleep(SLEEP)
        html = html_path.read_text(encoding="utf-8")
        paras = extract_paragraphs(html)
        (STATEMENTS_TXT / f"{sid}.txt").write_text("\n".join(paras) + "\n", encoding="utf-8")
        url_date = pd.Timestamp(date).date().isoformat()
        rel_date = page_release_date(html) or url_date
        if rel_date != url_date:
            print(f"  date fix {sid}: URL says {url_date}, page says {rel_date} -> using page date")
        rows.append({
            "id": sid,
            "date": rel_date,
            "url_date": url_date,
            "url": url,
            "title": paras[0][:160] if paras else "",
            "n_paragraphs": len(paras),
            "n_words": sum(len(p.split()) for p in paras),
        })
    idx = pd.DataFrame(rows)
    idx.to_csv(PROCESSED / "statements_index.csv", index=False)
    print(idx.groupby(idx["date"].str[:4]).size().to_string())
    print(f"saved {len(idx)} statements; words min/median/max = "
          f"{idx.n_words.min()}/{int(idx.n_words.median())}/{idx.n_words.max()}")


if __name__ == "__main__":
    main()
