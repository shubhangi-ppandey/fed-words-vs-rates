"""Text utilities shared by the tone pipeline: repair, sentence splitting, boilerplate filter.

The filter rules are fixed in advance (PROJECT_PLAN.md section 5) and are never tuned on outcomes.
"""
import re

# Abbreviations whose trailing period must not end a sentence.
_ABBREV = ["U.S.", "Jr.", "Sr.", "Mr.", "Ms.", "Dr.", "St.", "Inc.", "Co.", "e.g.", "i.e.", "No.", "vs.", "Corp.",
           "a.m.", "p.m."]
# Split only after a lowercase letter, digit, '%', ')' or quote + terminal punctuation, so that name initials
# ("Timothy F. Geithner") and acronyms never end a sentence.
_SPLIT = re.compile(r"(?<=[a-z0-9%)\"'][.!?])\s+(?=[A-Z(\"'])")

# Pre-specified boilerplate: roster / procedural sentences that carry no policy-stance content.
BOILERPLATE_PATTERNS = [
    r"^voting\b",
    r"\bvoted (for|against)\b",
    r"^in a related action,? the board of governors",
    r"\bdiscount rate\b",
    r"\bboards? of directors of the federal reserve banks?\b",
    r"\brequests? submitted by\b",
    r"approved the following statement for release",
    r"^for media inquiries",
    r"implementation note",
    r"^the federal reserve (today )?released the following statement",
    r"^the federal reserve today announced the following",
]
_boiler = re.compile("|".join(BOILERPLATE_PATTERNS), re.I)
MIN_WORDS = 3


def repair(s: str) -> str:
    """Fix UTF-8 text that was decoded as cp1252 (mojibake such as 'publicâ€™s')."""
    if any(m in s for m in ("â€", "Ã", "Â")):
        try:
            s = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    s = (s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", " - ").replace("\xa0", " "))
    return re.sub(r"\s+", " ", s).strip()


def split_sentences(paragraph: str) -> list[str]:
    protected = paragraph
    for i, a in enumerate(_ABBREV):
        protected = protected.replace(a, a.replace(".", f"\u0000{i}\u0000"))
    parts = _SPLIT.split(protected)
    out = []
    for p in parts:
        for i, a in enumerate(_ABBREV):
            p = p.replace(a.replace(".", f"\u0000{i}\u0000"), a)
        p = p.strip()
        if p:
            out.append(p)
    return out


def is_boilerplate(sentence: str) -> bool:
    return (bool(_boiler.search(sentence)) or len(sentence.split()) < MIN_WORDS
            or not re.search(r"[.!?:;\"')]$", sentence.strip()))  # headings / link text have no end punctuation


def is_boilerplate_paragraph(paragraph: str) -> bool:
    """Whole paragraphs that are rosters or procedural notes (every sentence in them is boilerplate)."""
    return bool(re.match(r"\s*(voting\b|in a related action|in taking (this|the discount) action)", paragraph, re.I))
