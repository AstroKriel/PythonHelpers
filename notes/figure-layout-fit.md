# Figure layout: fit the figure to what it draws

Draft spec. Not implemented.

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

## Open calls

- **Shared colorbars.** Spanning a panel grid is currently done by the caller, by reading
  panel positions and adding a dummy axis. That is the one thing a save-time fit would
  disturb. Either `add_colorbar` grows a way to span a grid, or the fit is triggered
  explicitly before such reads.
- **Auto tick locators.** `MaxNLocator` picks tick values from the axes size, so a panel that
  resizes between the measure and the final draw can change its widest label. Anything on an
  explicit `MultipleLocator` is immune. Pin the locator during layout, or accept a rare
  one-character miss. Neither needs a loop.
- **Is padding constant in drawn pt or printed pt?** Only the same thing once the base width
  equals the page's text width. Today a single-column figure is magnified 1.072 on the page
  and a full-width one 1.116, so a 6 pt padding prints as 6.4 or 6.7. Related but separable.
- **Colorbar thickness: self-similar or uniform?** As a fraction of the panel, a bar in a
  single-column figure is physically thinner than one in a full-width figure. Taste call.

## Suggested order

1. Guard `add_colorbar` against placing its axes outside the figure. Independent of all of
   the above, and worth having on its own: it failed silently on four figures in
   `kriel-quokka-mhd`, one of which reached the published paper.
2. Prototype the fit on one figure with an awkward label, to check the measurement lands
   without hand-tuning.
3. The rename and the rollout.
4. The base-width fix, after the rename, so no aspect has to be re-solved.
