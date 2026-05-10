#!/usr/bin/env python3
"""GNG Visualizer – Growing Neural Gas animation for 1-D scalar data.

Run ``python gng_app.py --help`` for all options.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from gng import GrowingNeuralGas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Animate a Growing Neural Gas (GNG) learning a 1-D scalar dataset.\n"
            "Neurons are colour-coded by their total accumulated lifetime."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    data_grp = p.add_argument_group("Dataset")
    data_grp.add_argument(
        "--n-samples", type=int, default=5000,
        help="Number of data samples to generate.",
    )
    data_grp.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset generation and GNG initialisation.",
    )

    # GNG hyperparameters
    gng_grp = p.add_argument_group("GNG hyperparameters")
    gng_grp.add_argument(
        "--epsilon-b", type=float, default=0.2,
        help="Winner neuron learning rate (ε_b).",
    )
    gng_grp.add_argument(
        "--epsilon-n", type=float, default=0.006,
        help="Topological-neighbour learning rate (ε_n).",
    )
    gng_grp.add_argument(
        "--max-age", type=int, default=50,
        help="Maximum edge age before the edge is removed.",
    )
    gng_grp.add_argument(
        "--lam", type=int, default=100,
        help="Neuron-insertion interval λ (steps between insertions).",
    )
    gng_grp.add_argument(
        "--alpha", type=float, default=0.5,
        help="Error-reduction factor applied to q and f on neuron insertion.",
    )
    gng_grp.add_argument(
        "--d", type=float, default=0.995,
        help="Per-step error-decay factor applied to all neurons.",
    )

    # Visualisation
    vis_grp = p.add_argument_group("Visualisation")
    vis_grp.add_argument(
        "--fps", type=float, default=10.0,
        help="Animation frame rate (frames per second).",
    )
    vis_grp.add_argument(
        "--steps-per-frame", type=int, default=50,
        help="Dataset steps processed between animation frames.",
    )

    # Behaviour
    p.add_argument(
        "--loop", action="store_true",
        help=(
            "Loop over the dataset indefinitely instead of stopping at the "
            "end of one pass."
        ),
    )

    return p


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def generate_dataset(n_samples: int, seed: int = 42) -> np.ndarray:
    """Return *n_samples* scalar values from a 3-component Gaussian mixture.

    The three components give the network a non-trivial multimodal
    distribution to learn, making the growth animation visually interesting.
    """
    rng = np.random.default_rng(seed)
    weights = np.array([0.40, 0.35, 0.25])
    means   = np.array([-2.0,  0.5,  3.5])
    stds    = np.array([ 0.6,  1.0,  0.5])
    components = rng.choice(len(weights), size=n_samples, p=weights)
    samples = rng.normal(means[components], stds[components])
    return samples.astype(np.float64)


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class GNGVisualizer:
    """Manages the matplotlib figure and ``FuncAnimation``.

    Parameters
    ----------
    data:
        Full 1-D dataset the GNG will learn from.
    gng:
        An already-initialised :class:`GrowingNeuralGas` instance.
    fps:
        Target animation frame rate.
    steps_per_frame:
        How many dataset samples to process between successive frames.
    loop:
        When *True* the dataset is replayed from the start after each pass.
    """

    _NEURON_Y: float = 0.0
    _CMAP = plt.cm.plasma

    def __init__(
        self,
        data: np.ndarray,
        gng: GrowingNeuralGas,
        fps: float,
        steps_per_frame: int,
        loop: bool,
    ) -> None:
        self.data = data
        self.gng = gng
        self.fps = fps
        self.steps_per_frame = steps_per_frame
        self.loop = loop

        self._data_idx: int = 0
        self._done: bool = False
        self._edge_lines: list[matplotlib.lines.Line2D] = []

        self._build_figure()

    # ----------------------------------------------------------------
    # Figure construction
    # ----------------------------------------------------------------

    def _build_figure(self) -> None:
        self.fig, (self.ax_hist, self.ax_gng) = plt.subplots(
            2, 1,
            figsize=(11, 6),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        self.fig.suptitle(
            "Growing Neural Gas — 1-D Scalar Data", fontsize=13, fontweight="bold"
        )

        xmin = float(self.data.min()) - 0.4
        xmax = float(self.data.max()) + 0.4

        # ---- top panel: data histogram ----------------------------------
        self.ax_hist.hist(
            self.data, bins=60, density=True,
            color="steelblue", alpha=0.35, label="Data distribution",
        )
        self.ax_hist.set_xlim(xmin, xmax)
        self.ax_hist.set_ylabel("Density")
        self.ax_hist.legend(loc="upper right", fontsize=8)
        self._info_text = self.ax_hist.text(
            0.02, 0.97, "",
            transform=self.ax_hist.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
        )

        # ---- bottom panel: GNG network ----------------------------------
        self.ax_gng.set_xlim(xmin, xmax)
        self.ax_gng.set_ylim(-0.5, 0.5)
        self.ax_gng.set_yticks([])
        self.ax_gng.axhline(0, color="lightgray", lw=1, zorder=0)
        self.ax_gng.set_xlabel("Value")
        self.ax_gng.set_ylabel(
            "GNG\nnetwork", rotation=0, labelpad=45, va="center", fontsize=8
        )

        self._scatter = self.ax_gng.scatter(
            [], [], c=[], cmap=self._CMAP,
            s=130, zorder=5, linewidths=0.6, edgecolors="black",
        )

        # Colour-bar legend for neuron lifetime
        self._sm = ScalarMappable(cmap=self._CMAP, norm=Normalize(vmin=0, vmax=1))
        self._sm.set_array([])
        self._cbar = self.fig.colorbar(
            self._sm, ax=self.ax_gng,
            orientation="horizontal", fraction=0.6, pad=0.55, aspect=40,
        )
        self._cbar.set_label("Neuron lifetime (steps)", fontsize=8)

        self.fig.tight_layout()

    # ----------------------------------------------------------------
    # Animation update callback
    # ----------------------------------------------------------------

    def _update(self, _frame: int) -> list:
        # Once all data has been consumed (and loop=False) freeze the display.
        if self._done:
            return [self._scatter] + self._edge_lines

        # Advance the GNG by *steps_per_frame* signals.
        for _ in range(self.steps_per_frame):
            if self._data_idx >= len(self.data):
                if self.loop:
                    self._data_idx = 0
                else:
                    self._done = True
                    break
            if not self._done:
                self.gng.step(float(self.data[self._data_idx]))
                self._data_idx += 1

        # ---- update scatter (neurons) -----------------------------------
        positions = self.gng.neuron_positions
        lifetimes = self.gng.neuron_lifetimes

        if len(positions) > 0:
            lt_max = int(lifetimes.max())
            clim_max = max(lt_max, 1)

            xy = np.column_stack(
                [positions, np.full(len(positions), self._NEURON_Y)]
            )
            self._scatter.set_offsets(xy)
            self._scatter.set_array(lifetimes.astype(float))
            self._scatter.set_clim(0, clim_max)

            # Keep the colorbar in sync with the current lifetime range.
            self._sm.set_clim(0, clim_max)
            self._cbar.update_normal(self._sm)

        # ---- update edge lines ------------------------------------------
        for ln in self._edge_lines:
            ln.remove()
        self._edge_lines = []

        for p_i, p_j in self.gng.edge_segments():
            (ln,) = self.ax_gng.plot(
                [p_i, p_j], [self._NEURON_Y, self._NEURON_Y],
                color="dimgray", lw=2, alpha=0.55, zorder=3,
            )
            self._edge_lines.append(ln)

        # ---- info text --------------------------------------------------
        pct = 100.0 * self._data_idx / len(self.data)
        if self.loop:
            status = f"looping (pass {self._data_idx // len(self.data) + 1})"
        elif self._done:
            status = "done"
        else:
            status = f"{pct:.0f}%"

        self._info_text.set_text(
            f"Step {self.gng.step_count} | "
            f"Samples {self._data_idx}/{len(self.data)} ({status}) | "
            f"Neurons {len(positions)} | "
            f"Edges {len(self.gng.edges)}"
        )

        return [self._scatter] + self._edge_lines

    # ----------------------------------------------------------------
    # Entry point
    # ----------------------------------------------------------------

    def run(self) -> None:
        """Start the animation and block until the window is closed."""
        interval_ms = 1000.0 / self.fps
        self._ani = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Generate the scalar dataset.
    data = generate_dataset(args.n_samples, seed=args.seed)

    # Build and initialise the GNG.
    gng = GrowingNeuralGas(
        epsilon_b=args.epsilon_b,
        epsilon_n=args.epsilon_n,
        max_age=args.max_age,
        lam=args.lam,
        alpha=args.alpha,
        d=args.d,
    )
    gng.initialize(data)

    # Run the animated visualisation.
    viz = GNGVisualizer(
        data=data,
        gng=gng,
        fps=args.fps,
        steps_per_frame=args.steps_per_frame,
        loop=args.loop,
    )
    viz.run()


if __name__ == "__main__":
    main()
