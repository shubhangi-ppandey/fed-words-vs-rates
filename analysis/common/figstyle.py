"""GROUND figure style (DESIGN_SYSTEM.md): Tesla-inspired subtractive minimalism for report and paper figures.

White canvas; Carbon / Graphite / Pewter / Silver neutrals; ONE accent colour (#3E6AE1) for the primary result.
No top or right spines, hairline y-grid only, no boxes, shadows or gradients. Typeface: D-DIN (bundled, SIL OFL),
with Roboto Mono available for numbers. Measures are distinguished by tone and marker before colour.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONT_DIR = Path(__file__).resolve().parents[2] / "app" / "static" / "fonts"
for _f in ("D-DIN.ttf", "D-DIN-Bold.ttf", "Inter.ttf", "RobotoMono.ttf"):
    if (FONT_DIR / _f).exists():
        font_manager.fontManager.addfont(str(FONT_DIR / _f))

# GROUND tokens
WHITE, SOFT = "#FFFFFF", "#F4F4F4"
CARBON, GRAPHITE, PEWTER, SILVER = "#171A20", "#393C41", "#5C5E62", "#8E8E8E"
HAIR, HAIR_STRONG = "#EEEEEE", "#D0D1D2"
ACCENT, ACCENT_TINT = "#3E6AE1", "#9DB3EF"
LIGHT = "#BDBEC0"

# Names used by the figure code, mapped onto the GROUND palette
NAVY, AMBER, TEAL, SKY, GREY, RED = ACCENT, CARBON, PEWTER, LIGHT, SILVER, CARBON

# Primary result = accent; look-ahead counterparts = dark neutrals; weaker classifiers = light neutrals
MEASURE_COLOR = {"M3": ACCENT, "M2": CARBON, "M3b": PEWTER, "M2b": LIGHT, "M1": SILVER,
                 "M5": ACCENT_TINT, "M5f": GRAPHITE}
MEASURE_LABEL = {
    "M1": "M1 dictionary (no learning)",
    "M2b": "M2b TF-IDF, full sample (look-ahead)",
    "M3b": "M3b TF-IDF, chronological (safe)",
    "M2": "M2 RoBERTa-large, full sample (look-ahead)",
    "M3": "M3 RoBERTa-large, chronological (safe, primary)",
    "M5": "M5 ChronoBERT, fully chronological",
    "M5f": "M5f ChronoBERT, full sample",
}


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight", "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE, "axes.facecolor": WHITE,
        "font.family": ["D-DIN", "DejaVu Sans"], "font.size": 10.5,
        "text.color": GRAPHITE, "axes.labelcolor": GRAPHITE, "xtick.color": PEWTER, "ytick.color": PEWTER,
        "axes.titlesize": 12.5, "axes.titleweight": "normal", "axes.titlecolor": CARBON, "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": HAIR_STRONG, "axes.linewidth": 0.8,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": HAIR, "grid.linewidth": 0.9, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": GRAPHITE,
        "lines.linewidth": 1.8, "lines.markersize": 5,
        "errorbar.capsize": 0,
        "axes.prop_cycle": matplotlib.cycler(color=[ACCENT, CARBON, PEWTER, SILVER, ACCENT_TINT, LIGHT]),
        "figure.titlesize": 11.5, "figure.titleweight": "normal",
    })
