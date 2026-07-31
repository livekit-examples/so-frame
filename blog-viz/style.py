"""Shared visual style for the blog figures, so the whole set reads as one system.

Every figure hangs its title, subtitle, footnote and plotting area off the same left rail, uses one
type scale, and draws on the same warm cream page. Import `apply()` first, then build the figure
with `title_block()` / `footnote()` / `clean_axes()` / `frame_image()`.

The categorical series colours are the validated four-slot order (blue, orange, aqua, yellow),
checked against THIS page colour rather than a generic white one: worst adjacent CVD dE 9.1, worst
adjacent normal-vision dE 22.9. Aqua and yellow sit under 3:1 against cream, so any figure using
them must label its series directly rather than leaning on a legend swatch alone. Every line chart
here does.
"""
import matplotlib
import matplotlib.patheffects as pe

# ---------------------------------------------------------------------------------------------
# Page and ink
# ---------------------------------------------------------------------------------------------
BG = "#FFFBF0"          # warm cream page
INK = "#14130F"         # near-black, warm
MUTED = "#57534A"       # secondary text
FAINT = "#8A8477"       # tertiary text, footnotes
GRID = "#E9E1CE"        # cream-tinted gridlines
AXIS = "#C9C0AA"        # axis lines and panel borders

# ---------------------------------------------------------------------------------------------
# Categorical series colours, in fixed slot order. Assign by slot, never cycle.
# ---------------------------------------------------------------------------------------------
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
BLUE, ORANGE, AQUA, GOLD = SERIES

# The four encoders always take the same slot, in every figure, so a colour means one architecture
# across the whole post.
ENCODER_COLOR = {
    "dino_patch": BLUE,
    "dino_global_mean": ORANGE,
    "dino_global_cls": AQUA,
    "squint": GOLD,
}
ENCODER_LABEL = {
    "dino_patch": "dino_patch",
    "dino_global_mean": "dino_global (mean)",
    "dino_global_cls": "dino_global (cls)",
    "squint": "squint CNN",
}

# ---------------------------------------------------------------------------------------------
# Rig hues: annotations that mirror physical objects, NOT chart accents. Sourced from config.py's
# linear base colours, converted to sRGB, so an arrow pointing at the cube is the cube's colour.
# ---------------------------------------------------------------------------------------------
CUBE = "#3d61cc"        # config.BLUE  (0.04, 0.12, 0.60) linear
BIN = "#e8d13a"         # config.YELLOW (0.80, 0.62, 0.04) linear
SURFACE = "#e6e4de"     # the lightbox panels

def sequential_blue():
    """One-hue light-to-dark ramp, for magnitude rather than identity."""
    from matplotlib.colors import LinearSegmentedColormap
    steps = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    return LinearSegmentedColormap.from_list("soframe_blue", steps)


# Gold and aqua sit under 3:1 against cream, so they carry a hairline of a darker relative to hold
# their edge. Kept thin on purpose: a fat halo would make those two series look heavier than the
# blue and orange they are being compared against.
RELIEF = {
    GOLD: [pe.withStroke(linewidth=2.9, foreground="#8a6100")],
    AQUA: [pe.withStroke(linewidth=2.9, foreground="#0f6b4a")],
}

# ---------------------------------------------------------------------------------------------
# Type scale (pt)
# ---------------------------------------------------------------------------------------------
T_TITLE = 15
T_SUB = 10.5
T_LABEL = 11
T_TICK = 10
T_FOOT = 9

# Inter, with fallbacks. matplotlib falls back per glyph down this list, so anything Inter lacks
# borrows from further along instead of dropping to a tofu box.
#
#     brew install --cask font-inter
#
# Without it the figures silently render in Helvetica, which is close enough to miss at a glance
# and wrong enough to look off next to the ones that have it, so check_font() says so out loud.
FONT_STACK = ["Inter", "Helvetica", "Arial", "DejaVu Sans"]


def check_font():
    """Warn if Inter is missing rather than quietly falling back to Helvetica."""
    from matplotlib import font_manager as fm
    if not any(f.name == "Inter" for f in fm.fontManager.ttflist):
        print("[style] Inter not found, falling back to Helvetica. "
              "Install it with: brew install --cask font-inter")


def apply():
    """Set the global rcParams every figure shares."""
    check_font()
    matplotlib.rcParams.update({
        "font.family": FONT_STACK,
        "font.size": T_TICK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1.0,
        "axes.labelcolor": INK,
        "axes.labelsize": T_LABEL,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": T_TICK,
        "ytick.labelsize": T_TICK,
        "text.color": INK,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titlesize": T_LABEL,
        "axes.titlecolor": INK,
        "figure.dpi": 130,
        "savefig.dpi": 130,
    })


def title_block(fig, title, subtitle=None, *, left, top=0.955, gap=0.052):
    """Left-align a title and optional subtitle to the `left` rail (figure fraction).

    Pass the same `left` the plotting area uses, so text and plot share one vertical guide.
    """
    fig.text(left, top, title, ha="left", va="top", fontsize=T_TITLE, fontweight="bold", color=INK)
    if subtitle:
        fig.text(left, top - gap, subtitle, ha="left", va="top", fontsize=T_SUB, color=MUTED)
        return top - gap
    return top


def footnote(fig, text, *, left, y=0.02):
    """Left-align a footnote to the same rail as the title block."""
    fig.text(left, y, text, ha="left", va="bottom", fontsize=T_FOOT, color=FAINT)


def clean_axes(ax, grid_axis="y"):
    """Recessive grid, no top/right spines, no tick marks: the editorial default."""
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(length=0)


def frame_image(ax, label=None, *, color=None):
    """Style an image panel: no ticks, a thin warm border, an optional caption under it."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(AXIS)
        s.set_linewidth(1.0)
    if label:
        ax.set_xlabel(label, fontsize=T_TICK, color=color or MUTED, labelpad=6)


def panel_title(ax, text, *, color=None):
    """A short heading above an image panel."""
    ax.set_title(text, fontsize=T_TICK, color=color or MUTED, pad=6, loc="left")
