"""Single place for every path used by the pipeline (all relative to the project root)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS_CACHE = ROOT / "models_cache"

STATEMENTS_HTML = RAW / "statements_html"
STATEMENTS_TXT = RAW / "statements_txt"
MARKET = RAW / "market"

for _p in (RAW, PROCESSED, FIGURES, STATEMENTS_HTML, STATEMENTS_TXT, MARKET):
    _p.mkdir(parents=True, exist_ok=True)

# One polite, identifiable User-Agent for every request to public sites.
USER_AGENT = "Mozilla/5.0 (academic research; FOMC communication study)"
