"""M1: a transparent, rule-based hawkish/dovish dictionary (no learning, hence no look-ahead).

Design (fixed 2026-10-16, before any evaluation; NOT tuned on labelled data or outcomes):
a sentence receives hawkish and dovish "hits" from three families of rules, in the spirit of the
noun-plus-modifier approach to central-bank text:
  1. Policy-stance language (tightening/firming vs easing/accommodation).
  2. Inflation nouns with upward (hawkish) or downward (dovish) modifiers.
  3. Real-activity nouns with strong (hawkish) or weak (dovish) modifiers; unemployment is reversed.
Label = hawkish if hits_h > hits_d, dovish if hits_d > hits_h, else neutral.
"""
import re

_w = lambda pat: re.compile(pat, re.I)  # noqa: E731

POLICY_HAWK = _w(r"\b(tighten\w*|firming|firm(ed)? the stance|rais(e|ed|ing) (the|its) target|increas\w* (in )?(the|its) target"
                 r"|remov\w* (of )?(policy )?accommodation|less accommodative|restrictive|reduc\w* (its |the )?(securities )?"
                 r"holdings|balance sheet reduction|taper\w*|additional (policy )?firming|further (policy )?firming"
                 r"|increase in the target range|raise the target range|rate increases?)\b")
POLICY_DOVE = _w(r"\b(eas(e|ed|ing) (the stance|of monetary policy|monetary policy)|lower(ed|ing)? (its |the )?target|"
                 r"reduc\w* (its |the )?target|decrease in the target range|lower the target range|cut\b|accommodative"
                 r"|highly accommodative|exceptionally low|patient|asset purchases|purchas\w* (additional )?"
                 r"(treasury|agency|mortgage|longer-term)|maintain\w* (the|its|this) target range|support (the economy|"
                 r"economic activity|the recovery))")
INFL_NOUN = _w(r"\b(inflation\w*|price pressures?|prices|wages?|compensation|costs?|inflationary)\b")
INFL_UP = _w(r"\b(ris(e|es|ing|en)|rose|increas\w*|elevated|high(er)?|accelerat\w*|upward|upside|picked up|firm(ed|ing)?"
             r"|imbalances|pressures?|stoked|moved up|above)\b")
INFL_DOWN = _w(r"\b(low(er)?|subdued|moderat\w*|declin\w*|fell|fall(en|ing)?|decreas\w*|soft(er|ened|ening)?|muted"
               r"|below|contained|eas(ed|ing)|downward|running below|restrained|benign)\b")
ACT_NOUN = _w(r"\b(economic activity|economy|growth|spending|labor markets?|employment|job gains|payroll\w*|demand|output"
              r"|investment|production|hiring)\b")
ACT_STRONG = _w(r"\b(strong(er|ly)?|solid|robust|improv\w*|accelerat\w*|tight(en(ed|ing))?|resilient|picked up|brisk"
                r"|expand\w* at a (solid|strong|robust)|vigorous|firm(ed)?)\b")
ACT_WEAK = _w(r"\b(weak\w*|slow\w*|soft(en\w*)?|declin\w*|contract\w*|deteriorat\w*|downside|stress\w*|strain\w*|slack"
              r"|sluggish|subdued|depressed|disrupt\w*|uncertain\w*|harmed|fell|decreas\w*)\b")
UNEMP_UP = _w(r"unemployment rate[^.]*\b(ris|rose|increas|elevated|high)")
UNEMP_DOWN = _w(r"unemployment rate[^.]*\b(declin|fell|low|decreas|moved down)")


def score(sentence: str) -> tuple[int, int]:
    """Return (hawkish hits, dovish hits) for one sentence."""
    h = d = 0
    h += bool(POLICY_HAWK.search(sentence))
    d += bool(POLICY_DOVE.search(sentence))
    if INFL_NOUN.search(sentence):
        h += bool(INFL_UP.search(sentence))
        d += bool(INFL_DOWN.search(sentence))
    if ACT_NOUN.search(sentence):
        h += bool(ACT_STRONG.search(sentence))
        d += bool(ACT_WEAK.search(sentence))
    h += bool(UNEMP_DOWN.search(sentence))
    d += bool(UNEMP_UP.search(sentence))
    return h, d


def label(sentence: str) -> int:
    """TDW coding: 0 = dovish, 1 = hawkish, 2 = neutral."""
    h, d = score(sentence)
    return 1 if h > d else 0 if d > h else 2
