import shutil
import matplotlib
from cycler import cycler

FONT_SIZES: dict[str, int] = {
  "small":  16,
  "medium": 20,
  "large":  25,
}

def _get_base_rc_params(
  use_tex : bool = True
) -> dict[str, object]:
  rc_params: dict[str, object] = {
    ## font
    "font.family": "serif",
    "font.size": FONT_SIZES["large"],
    "axes.titlesize": FONT_SIZES["large"],
    "axes.labelsize": FONT_SIZES["large"],
    "xtick.labelsize": FONT_SIZES["medium"],
    "ytick.labelsize": FONT_SIZES["medium"],
    "figure.titlesize": FONT_SIZES["small"],
    "legend.fontsize": FONT_SIZES["large"],
    ## lines + axes
    "lines.linewidth": 1.5,
    "axes.linewidth": 1.0,
    ## ticks
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.minor.width": 0.75,
    "ytick.minor.width": 0.75,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "xtick.minor.pad": 5,
    "ytick.minor.pad": 5,
    ## legend
    "legend.labelspacing": 0.2,
    "legend.loc": "upper right",
    "legend.frameon": False,
    ## figure + saving
    "figure.figsize": (8.0, 6.0),
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "savefig.pad_inches": 0.1,
  }
  if use_tex and shutil.which("latex") is not None:
    rc_params.update({
      "text.usetex": True,
      "text.latex.preamble": r"""
        \usepackage{bm,amsmath,mathrsfs,amssymb,url,xfrac}
        \providecommand{\mathdefault}[1]{#1}
      """,
    })
  else: rc_params.update({"text.usetex": False})
  return rc_params

LIGHT_RC_PARAMS: dict[str, object] = {
  ## backgrounds
  "figure.facecolor": "white",
  "axes.facecolor": "white",
  "savefig.facecolor": "white",
  ## foreground
  "axes.edgecolor": "#222222",
  "axes.labelcolor": "#222222",
  "text.color": "#222222",
  "axes.titlecolor": "#222222",
  "xtick.color": "#333333",
  "ytick.color": "#333333",
  ## grid
  "grid.color": "#dddddd",
  "grid.alpha": 0.6,
  ## default colours
  "axes.prop_cycle": cycler(color=[
    "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"
  ]),
}

DARK_RC_PARAMS: dict[str, object] = {
  ## backgrounds
  "figure.facecolor": "#0b0b0e",
  "axes.facecolor": "#0b0b0e",
  "savefig.facecolor": "#0b0b0e",
  ## foreground
  "axes.edgecolor": "#e6e6e6",
  "axes.labelcolor": "#e6e6e6",
  "text.color": "#e6e6e6",
  "axes.titlecolor": "#e6e6e6",
  "xtick.color": "#cfcfd2",
  "ytick.color": "#cfcfd2",
  ## grid
  "grid.color": "#2e2e35",
  "grid.alpha": 0.3,
  ## default colours
  "axes.prop_cycle": cycler(color=[
    "#7aa2f7", "#9ece6a", "#f7768e", "#e0af68", "#bb9af7",
    "#7dcfff", "#f6bd60", "#c0caf5", "#89ddff", "#ff9e64"
  ]),
}

THEMES: dict[str, dict[str, object]] = {
  "light": LIGHT_RC_PARAMS,
  "dark": DARK_RC_PARAMS,
}

def _compose_rc_params(
  theme   : str = "light",
  use_tex : bool = True,
) -> dict[str, object]:
  rc_params = _get_base_rc_params(use_tex=use_tex).copy()
  rc_params.update(THEMES[theme])
  return rc_params

def apply_theme_globally(
  theme   : str = "light",
  use_tex : bool = True,
) -> None:
  """Apply theme for the whole process (global)."""
  if theme == "dark":
    try:
      import matplotlib.pyplot as _mpl_plot
      _mpl_plot.style.use("dark_background")
    except Exception:
      pass
  matplotlib.rcParams.update(_compose_rc_params(theme=theme, use_tex=use_tex))

## END OF PARAMETERS