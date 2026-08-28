# Figure layout: fit the figure to what it draws

Built. `panel_aspect_ratio` and `FigureLayout.figure_margins` are gone, so this is the only
way a figure is sized. See "As built" for where it differed from the spec.

## The problem

Two complaints, one cause.

1. Margins are guessed. You type four pt values, render, look for clipping, adjust, repeat.
2. `panel_aspect_ratio` has to be re-typed to three decimals every time a margin moves.

The cause is that `panel_aspect_ratio` describes the panel's *share* of the figure, with the
margins taken out of that share (see `_compute_panel_shape`). So it is not a property of the
panel you can decide: it is the solution to an equation involving the margins. Calibrating one
figure in `kriel-quokka-mhd` took five re-solves of the same number, none of which meant
anything on its own.

The margins are guessed for a different reason: what they hold is text, and a label's extent
is not knowable before it is drawn. A stacked-fraction axis label measures 30 pt across; a
`$x_1$` measures 12. No default serves both.

## The model

Split what is chosen from what follows.

Chosen:

- `figure_width`, from the page, as now.
- `panel_aspect`, the width / height of the **drawn** panel. Stable: `1.0` for a square
  image panel, and it never changes again.
- `figure_padding`, the clear space left outside everything the figure draws, in pt. One
  small constant, the same for every figure.
- gaps between panels, in pt, as now.

Derived:

- the four margins, from measuring what sits outside each panel, plus the padding
- the panel width, from the figure width less the margins and gaps
- the figure height, from the panel height and the vertical margins

The width and the height are not symmetric, and that is deliberate. The page pins the width,
so horizontally the panel is what gives. Nothing pins the height, so vertically the figure is
what gives.

## The algorithm

Measure once, solve once, apply. No loop.

1. **Measure.** With a renderer, take the extent of everything that sits outside each panel:
   tick labels, axis labels, and any colorbar's own labels. Call the total on each side `K`.
   `K` is text at absolute pt sizes, so it does not change when the panel resizes. That is
   what makes one measurement enough.
2. **Solve.** For the constrained (horizontal) direction, with `W` the figure width, `P` the
   panel width, and `t` a colorbar's thickness as a fraction of the panel:

   ```
   W = left_padding + K_left + P + gap + t*P + K_right + right_padding
   P = (W - padding - K_left - K_right - gap) / (1 + t)
   ```

   Linear, so closed form, for any number of colorbars on any sides: every term is either a
   measured constant or proportional to `P`, giving `P * (1 + sum of t) = constants`.
3. **Apply.** `panel_height = P / panel_aspect`, then the figure height follows by forward
   substitution, since nothing constrains it. Set the geometry and save.

### Two things the measurement must get right

- A top or bottom colorbar contributes horizontally as well, through its end tick labels: the
  first label is centred on the bar's end, so half of it hangs past the panel. Treating top
  bars as vertical-only clips them.
- Do not measure past the *panel* on a side that carries a colorbar, or the bar's own
  thickness gets counted as part of `K` when it is the `P`-proportional term being solved for.
  Measure past the *colorbar's* box instead, and compute the bar and its gap analytically.
  The layout pass therefore needs to know which neighbouring axes are colorbars and on which
  side; `add_colorbar` creates them, so that is already known.

### What does not need iterating

Colorbar thickness stays a fraction of the panel. It appears on both sides of the equation,
which is not the same as being unsolvable. This supersedes the reasoning behind
`eb3a9f56`/`609cca73`: there is no need to move thickness to pt to break a circularity,
because there is no circularity to break.

## Where it runs

Inside `save_figure`. By then every artist exists, so nothing has to be declared up front:
adding a colorbar just fits, the same way adding a longer tick label just fits. The caller
still writes build-then-save and sees none of it.

## What breaks

- `panel_aspect_ratio` changes meaning. Rename it to `panel_aspect` so the break is loud.
  Silently reinterpreting the old numbers would produce wrong figures rather than errors.
- `FigureMargins` becomes an override rather than the normal path. Keep it for the cases
  where a figure wants a margin larger than its contents need.
- Scripts that read panel positions between create and save would see them move. In
  `kriel-quokka-mhd` this is `current-sheet/plot_evolution.py`, which reads
  `get_position()` to place a shared colorbar and two shared axis labels.

## As built

Resolved along the way:

- **Shared colorbars.** `add_colorbar` takes `panels`, so it spans whatever it describes.
  No caller reads a panel position any more, and the dummy anchor axis is gone.
- **Shared labels.** `add_shared_axis_label` names an axis once for a grid, placed outside
  the panels' own labels by the fit. This replaced `figure.supylabel`, which is anchored to
  the figure and so was left behind when the panels moved.
- **Auto tick locators.** Not rare: two of fourteen figures clipped. The fit pins the ticks
  it measured, so what was measured is what is drawn, and one pass still suffices.
- **Colorbar thickness.** Now a ratio of the bar's own length, so a bar keeps its
  proportions whatever it spans. The bar no longer grows thicker for describing a grid.
- **The colorbar guard is unnecessary.** A fitted figure leaves room for its bars by
  construction, so the off-page case it would have caught cannot arise.

Still open:

- **Padding is measured against text boxes, not ink**, so visual clearance varies by a few
  pt with whatever is outermost on a side. Only fixable by rasterising.
- **`usetex` sizes are optically banded.** LaTeX takes metrics from the nearest whole-point
  design, and small designs are drawn wider, so a smaller requested size can give a *longer*
  label. Measure, do not infer length from size. `fix-cm` does not help; this was tested.
- **Thickness follows length**, so a bar spanning a wide row is proportionally thick. Fine
  when bars are of similar length, as in a paper; the demos need per-call overrides.
- **Is padding constant in drawn pt or printed pt?** Only the same thing once the base width
  equals the page's text width. Today a single-column figure is magnified 1.072 on the page
  and a full-width one 1.116, so a 6 pt padding prints as 6.4 or 6.7. Separable.
