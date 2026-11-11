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

        # ---- spines: only left + bottom ----
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
        "axes.linewidth":     0.8,

        # ---- ticks: outward, no top/right ----
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.top":          False,
        "ytick.right":        False,
        "xtick.major.size":   4,
        "ytick.major.size":   4,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,

        # ---- grid: off by default; subtle when enabled ----
        "axes.grid":          False,
        "grid.color":         "#D5D8DC",
        "grid.linestyle":     "-",
        "grid.linewidth":     0.5,
        "grid.alpha":         0.4,

        # ---- lines: thin, solid, markers with thin white edge ----
        "lines.linewidth":        1.5,
        "lines.markersize":       5,
        "lines.markeredgewidth":  1.2,
        "lines.markeredgecolor":  "white",

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
    Apply the PharmaScan-ISIS theme to a matplotlib CheckButtons widget:
      * transparent axes with no spines (the widget floats over the figure)
      * soft rounded-look boxes (filled white with light gray edge)
      * thicker check marks in the theme's "points" color
      * slightly larger label font

    Compatible with both legacy (matplotlib < 3.7) and current
    (matplotlib >= 3.7) CheckButtons APIs: the later versions renamed
    `rectangles`, `lines`, `labels` to `_rectangles`, `_lines`, `_labels`.
    """
    # The widget axes: no frame, transparent
    cb.ax.set_facecolor("none")
    for spine in cb.ax.spines.values():
        spine.set_visible(False)
    cb.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # --- Access internals across matplotlib versions ---
    rects  = getattr(cb, "rectangles", None)
    if rects is None:
        rects = getattr(cb, "_rectangles", [])
    lines  = getattr(cb, "lines", None)
    if lines is None:
        lines = getattr(cb, "_lines", [])
    labels = getattr(cb, "labels", None)
    if labels is None:
        labels = getattr(cb, "_labels", [])

    # Boxes and check marks
    box_face    = "#FFFFFF"
    box_edge    = "#C7CDD4"
    check_color = ELEMENT_COLORS["points"]

    for rect in rects:
        rect.set_facecolor(box_face)
        rect.set_edgecolor(box_edge)
        rect.set_linewidth(1.0)

    for line in lines:
        line.set_color(check_color)
        line.set_linewidth(1.6)

    # Labels
    for text in labels:
        text.set_fontsize(9)
        text.set_color("#2C3E50")


def style_button(btn):
    """
    Apply the PharmaScan-ISIS theme to a matplotlib Button widget:
      * light gray fill, subtle hover, no border
      * bolder, slightly larger label
    """
    btn.color      = "#F4F6F8"
    btn.hovercolor = "#E4E8EC"
    if btn.ax is not None:
        for spine in btn.ax.spines.values():
            spine.set_visible(False)
    btn.label.set_fontsize(9)
    btn.label.set_color("#2C3E50")