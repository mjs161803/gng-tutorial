"""Growing Neural Gas (GNG) algorithm for 1-D scalar data.

Reference
---------
Fritzke, B. (1995). A growing neural gas network learns topologies.
In Advances in Neural Information Processing Systems (NIPS), 7, 625-632.
"""

from __future__ import annotations

import numpy as np


class GrowingNeuralGas:
    """Growing Neural Gas for 1-D (scalar) input signals.

    Parameters
    ----------
    epsilon_b:
        Learning rate for the winner neuron (best matching unit).
    epsilon_n:
        Learning rate for topological neighbours of the winner.
    max_age:
        Maximum edge age; edges older than this are removed.
    lam:
        Number of steps between neuron insertions (λ in the paper).
    alpha:
        Error-reduction factor applied to *q* and *f* on insertion.
    d:
        Per-step error-decay factor applied to all neurons.
    """

    def __init__(
        self,
        epsilon_b: float = 0.2,
        epsilon_n: float = 0.006,
        max_age: int = 50,
        lam: int = 100,
        alpha: float = 0.5,
        d: float = 0.995,
    ) -> None:
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.max_age = max_age
        self.lam = lam
        self.alpha = alpha
        self.d = d

        self.step_count: int = 0
        self.positions: list[float] = []
        self.errors: list[float] = []
        # Accumulated lifetime: number of algorithm steps since each neuron
        # was created.
        self.lifetimes: list[int] = []
        # Adjacency map: (i, j) -> edge_age, with i < j always.
        self.edges: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initialize(self, data: np.ndarray) -> None:
        """Seed the network with two neurons drawn uniformly from the data range."""
        lo, hi = float(data.min()), float(data.max())
        rng = np.random.default_rng()
        p1, p2 = rng.uniform(lo, hi, size=2)
        self.positions = [float(p1), float(p2)]
        self.errors = [0.0, 0.0]
        self.lifetimes = [0, 0]
        self.edges = {}
        self.step_count = 0

    def step(self, x: float) -> None:
        """Process one training signal *x*."""
        s1, s2 = self._two_nearest(x)

        # Increment lifetime of every neuron by one step.
        for k in range(len(self.lifetimes)):
            self.lifetimes[k] += 1

        # Accumulate quantisation error at the winner.
        self.errors[s1] += (self.positions[s1] - x) ** 2

        # Move winner towards the input signal.
        self.positions[s1] += self.epsilon_b * (x - self.positions[s1])

        # Age edges leaving s1; collect surviving neighbours.
        neighbours: list[int] = []
        stale: list[tuple[int, int]] = []
        for (i, j), age in list(self.edges.items()):
            if i == s1 or j == s1:
                new_age = age + 1
                if new_age > self.max_age:
                    stale.append((i, j))
                else:
                    self.edges[(i, j)] = new_age
                    neighbours.append(j if i == s1 else i)

        # Move surviving neighbours towards the input signal.
        for nb in neighbours:
            self.positions[nb] += self.epsilon_n * (x - self.positions[nb])

        # Create / refresh the edge between s1 and s2.
        key = (min(s1, s2), max(s1, s2))
        self.edges[key] = 0

        # Remove stale edges (protect the freshly created/reset one).
        for k in stale:
            if k != key:
                self.edges.pop(k, None)

        # Remove neurons that have become isolated (no remaining edges).
        connected: set[int] = set()
        for i, j in self.edges:
            connected.add(i)
            connected.add(j)
        isolated = sorted(
            [k for k in range(len(self.positions)) if k not in connected],
            reverse=True,
        )
        if isolated:
            self._remove(isolated)

        self.step_count += 1

        # Possibly insert a new neuron every *lam* steps.
        if self.step_count % self.lam == 0:
            self._insert()

        # Decay accumulated errors for all neurons.
        for k in range(len(self.errors)):
            self.errors[k] *= self.d

    # ------------------------------------------------------------------
    # Read-only properties for the visualiser
    # ------------------------------------------------------------------

    @property
    def neuron_positions(self) -> np.ndarray:
        """1-D array of neuron positions."""
        return np.array(self.positions, dtype=float)

    @property
    def neuron_lifetimes(self) -> np.ndarray:
        """1-D array of neuron lifetimes (steps since creation)."""
        return np.array(self.lifetimes, dtype=int)

    @property
    def neuron_errors(self) -> np.ndarray:
        """1-D array of accumulated quantisation errors."""
        return np.array(self.errors, dtype=float)

    def edge_segments(self) -> list[tuple[float, float]]:
        """Return ``[(pos_i, pos_j)]`` for every edge in the network."""
        return [(self.positions[i], self.positions[j]) for i, j in self.edges]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _two_nearest(self, x: float) -> tuple[int, int]:
        dists = [abs(p - x) for p in self.positions]
        order = sorted(range(len(dists)), key=lambda i: dists[i])
        return order[0], order[1]

    def _remove(self, desc_indices: list[int]) -> None:
        """Remove neurons at *desc_indices* (must be sorted descending).

        Processing from highest to lowest index ensures that each ``pop``
        does not disturb the positions of neurons with smaller indices that
        still need to be removed in subsequent iterations.
        """
        for idx in desc_indices:
            self.positions.pop(idx)
            self.errors.pop(idx)
            self.lifetimes.pop(idx)
            new_edges: dict[tuple[int, int], int] = {}
            for (i, j), age in self.edges.items():
                if i == idx or j == idx:
                    continue  # edge touching the removed neuron
                ni = i - (1 if i > idx else 0)
                nj = j - (1 if j > idx else 0)
                new_edges[(min(ni, nj), max(ni, nj))] = age
            self.edges = new_edges

    def _insert(self) -> None:
        """Insert a new neuron between the two highest-error connected neurons."""
        if len(self.positions) < 2:
            return

        # Neuron with the highest accumulated quantisation error.
        q = max(range(len(self.errors)), key=lambda i: self.errors[i])

        # Neighbour of *q* with the highest error.
        nbrs: list[int] = []
        for i, j in self.edges:
            if i == q:
                nbrs.append(j)
            elif j == q:
                nbrs.append(i)
        if not nbrs:
            return
        f = max(nbrs, key=lambda i: self.errors[i])

        # Place the new neuron at the midpoint of q and f.
        new_pos = (self.positions[q] + self.positions[f]) / 2.0
        r = len(self.positions)
        self.positions.append(new_pos)

        # Reduce errors of q and f; new neuron inherits q's reduced error.
        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha
        self.errors.append(self.errors[q])
        self.lifetimes.append(0)

        # Rewire: remove q–f edge, add q–r and f–r edges.
        self.edges.pop((min(q, f), max(q, f)), None)
        self.edges[(min(q, r), max(q, r))] = 0
        self.edges[(min(f, r), max(f, r))] = 0
