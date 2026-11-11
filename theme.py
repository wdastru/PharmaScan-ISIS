"""
theme.py
Global matplotlib theme for PharmaScan-ISIS.

Call apply_theme() once at application start. All subsequent figures
inherit the style via plt.rcParams.

Style target: clean modern scientific-report look
(GraphPad Prism / BioRender inspired):
  * sans-serif font (Inter / Helvetica / Arial, DejaVu Sans fallback)
  * spines: only left + bottom, no top/right
  * grid: off by default; subtle if enabled by an individual plot
  * ticks: outward, no top/right
  * lines: thin, solid, markers with thin white edge
  * legend: no frame, tight
  * palette: muted pastel tones, color-blind friendly

The palette constants (BIO_PALETTE, ELEMENT_COLORS) live here so that
plotting.py can import them without duplicating color definitions.
"""

import matplotlib as mpl


# ----------------------------------------------------------------------
# Palettes
# ----------------------------------------------------------------------
# Group palette: Okabe-Ito inspired, muted for a BioRender-style look.
# The reference group uses a near-black defined separately.
BIO_PALETTE = [
    "#4E79A7",  # muted blue
    "#E15759",  # muted red
    "#59A14F",  # muted green
    "#F28E2B",  # muted orange
    "#B07AA1",  # muted purple
    "#76B7B2",  # muted teal
    "#EDC948",  # muted gold
    "#9C755F",  # muted brown
    "#BAB0AC",  # muted gray
    "#FF9DA7",  # muted pink
]

REFERENCE_COLOR = "#1a1a1a"

# Per-element colors used across plot functions (points / spline /
# Lorentzian / sigmoid / diff). Keeping them centralised makes every
# plot use the same visual language.
ELEMENT_COLORS = {
    "points":     "#2C3E50",   # dark slate (experimental data)
    "spline":     "#C0392B",   # deep red (spline fit)
    "lorentzian": "#16A085",   # teal (Lorentzian envelope)
    "sigmoid":    "#8E44AD",   # purple (sigmoid envelope)
    "diff":       "#D35400",   # orange (Lorentzian - spline)
    "mean":       "#2C3E50",   # dark slate (bar-chart mean)
    "sd":         "#7F8C8D",   # gray (error bars)
}


# ----------------------------------------------------------------------
# Theme application
# ----------------------------------------------------------------------
def apply_theme() -> None:
    """
    Apply the global rcParams for the PharmaScan-ISIS theme.

    Idempotent: safe to call multiple times. The change is purely
    visual; the analysis pipeline is unaffected.
    """
    mpl.rcParams.update({
        # ---- font ----
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.titleweight":  "semibold",
        "axes.labelsize":    10,
        "axes.labelweight":  "medium",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,

        # ---- spines: only left + bottom (GraphPad style: thicker and black) ----
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
        "axes.linewidth":     1.2, # Assi più spessi
        "axes.edgecolor":     "black",

        # ---- ticks: outward, no top/right ----
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.top":          False,
        "ytick.right":        False,
        "xtick.major.size":   6,
        "ytick.major.size":   6,
        "xtick.major.width":  1.2,
        "ytick.major.width":  1.2,

        # ---- grid: off by default; subtle when enabled ----
        "axes.grid":          False,
        "grid.color":         "#D5D8DC",
        "grid.linestyle":     "-",
        "grid.linewidth":     0.5,
        "grid.alpha":         0.4,

        # ---- lines: thicker, distinct markers ----
        "lines.linewidth":        2.0, # Linee più spesse
        "lines.markersize":       7,   # Marker più grandi
        "lines.markeredgewidth":  1.5,
        "lines.markeredgecolor":  "none",

        # ---- legend: no frame, tight ----
        "legend.frameon":        False,
        "legend.fontsize":       9,
        "legend.title_fontsize": 9,
        "legend.handlelength":   1.8,
        "legend.handletextpad":  0.6,
        "legend.labelspacing":   0.4,
        "legend.borderaxespad":  0.8,

        # ---- figure ----
        "figure.dpi":       100,
        "savefig.dpi":      300,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",

        # ---- errorbar ----
        "errorbar.capsize": 3,
        # Suppress the "More than 20 figures" warning. The analysis can
        # legitimately produce many figures (one per replicate); the user
        # is expected to close them when done.
        "figure.max_open_warning": 50,
    })

# ----------------------------------------------------------------------
# Widget styling (CheckButtons / Button)
# ----------------------------------------------------------------------
def style_checkbuttons(cb):
    """
    Clean, minimal styling for CheckButtons (squares, but modern).
    """
    cb.ax.set_xticks([])
    cb.ax.set_yticks([])
    cb.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in cb.ax.spines.values():
        spine.set_visible(False)
    cb.ax.set_facecolor("none")

    # Access internals
    rects  = getattr(cb, "rectangles", getattr(cb, "_rectangles", []))
    lines  = getattr(cb, "lines", getattr(cb, "_lines", []))
    labels = getattr(cb, "labels", getattr(cb, "_labels", []))

    for rect in rects:
        rect.set_facecolor("white")
        rect.set_edgecolor("#C7CDD4") # Grigio chiaro
        rect.set_linewidth(1.0)
        
    for line in lines:
        line.set_color(ELEMENT_COLORS["points"])
        line.set_linewidth(1.5)

    for text in labels:
        text.set_fontsize(10)
        text.set_color("#2C3E50")
        text.set_fontweight("medium")

def style_button(btn):
    btn.color      = "#F4F6F8"
    btn.hovercolor = "#E4E8EC"
    if btn.ax is not None:
        btn.ax.set_xticks([])
        btn.ax.set_yticks([])
        btn.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in btn.ax.spines.values():
            spine.set_visible(False)
            
    btn.label.set_fontsize(8)
    btn.label.set_color("#2C3E50")