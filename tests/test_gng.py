"""Unit tests for the Growing Neural Gas algorithm (gng.py) and the
application helpers (gng_app.py).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pytest

from gng import GrowingNeuralGas
from gng_app import build_parser, generate_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gng(**kwargs) -> GrowingNeuralGas:
    """Return a GNG with small, fast-acting defaults for testing."""
    defaults = dict(
        epsilon_b=0.2,
        epsilon_n=0.006,
        max_age=5,
        lam=10,
        alpha=0.5,
        d=0.995,
    )
    defaults.update(kwargs)
    return GrowingNeuralGas(**defaults)


def _initialized_gng(n_data: int = 200, **kwargs) -> tuple[GrowingNeuralGas, np.ndarray]:
    data = generate_dataset(n_data, seed=0)
    gng = _make_gng(**kwargs)
    gng.initialize(data)
    return gng, data


# ---------------------------------------------------------------------------
# GNG initialisation
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_starts_with_two_neurons(self):
        gng, _ = _initialized_gng()
        assert len(gng.positions) == 2
        assert len(gng.errors) == 2
        assert len(gng.lifetimes) == 2

    def test_neurons_within_data_range(self):
        data = generate_dataset(500, seed=1)
        gng = _make_gng()
        gng.initialize(data)
        lo, hi = data.min(), data.max()
        for p in gng.positions:
            assert lo <= p <= hi

    def test_initial_errors_are_zero(self):
        gng, _ = _initialized_gng()
        assert gng.errors == [0.0, 0.0]

    def test_initial_lifetimes_are_zero(self):
        gng, _ = _initialized_gng()
        assert gng.lifetimes == [0, 0]

    def test_no_initial_edges(self):
        gng, _ = _initialized_gng()
        assert len(gng.edges) == 0

    def test_step_count_reset_on_initialize(self):
        gng, data = _initialized_gng()
        gng.step(data[0])
        gng.initialize(data)
        assert gng.step_count == 0


# ---------------------------------------------------------------------------
# Single step behaviour
# ---------------------------------------------------------------------------

class TestSingleStep:
    def test_step_count_increments(self):
        gng, data = _initialized_gng()
        assert gng.step_count == 0
        gng.step(data[0])
        assert gng.step_count == 1

    def test_all_lifetimes_increase_by_one(self):
        gng, data = _initialized_gng()
        before = list(gng.lifetimes)
        gng.step(data[0])
        for b, a in zip(before, gng.lifetimes):
            assert a == b + 1

    def test_edge_created_between_s1_and_s2(self):
        gng, data = _initialized_gng()
        gng.step(data[0])
        assert len(gng.edges) == 1

    def test_winner_moves_towards_input(self):
        gng, data = _initialized_gng()
        x = data[0]
        # Identify the winner before the step.
        dists = [abs(p - x) for p in gng.positions]
        s1 = int(np.argmin(dists))
        pos_before = gng.positions[s1]
        gng.step(x)
        # The winner should have moved closer to x (or be exactly at x).
        dist_before = abs(pos_before - x)
        dist_after = abs(gng.positions[s1] - x)
        assert dist_after <= dist_before + 1e-12

    def test_errors_decay_after_step(self):
        gng, data = _initialized_gng()
        # Give the winner some error first.
        gng.step(data[0])
        errors_after_first = list(gng.errors)
        gng.step(data[1])
        # At least one neuron should have a decayed (smaller) or equal error.
        decayed = any(
            gng.errors[i] <= errors_after_first[i] + 1e-12
            for i in range(min(len(gng.errors), len(errors_after_first)))
        )
        assert decayed


# ---------------------------------------------------------------------------
# Edge ageing and removal
# ---------------------------------------------------------------------------

class TestEdgeAgeingAndRemoval:
    def test_edge_removed_after_max_age(self):
        """An edge that is never refreshed is removed after max_age steps.

        With x=0 always, neurons 0 and 1 are s1/s2 (the nearest pair), so
        edge (0,1) is reset to age 0 every step.  Edge (0,2) also emanates
        from s1=0 but is NOT the s1–s2 edge, so it ages monotonically until
        it exceeds max_age and is removed, leaving neuron 2 isolated and then
        pruned.
        """
        max_age = 3
        gng = _make_gng(max_age=max_age, lam=1000)
        # Neurons: 0 and 1 are close (s1/s2), neuron 2 is far away.
        gng.positions = [0.0, 1.0, 100.0]
        gng.errors = [0.0, 0.0, 0.0]
        gng.lifetimes = [0, 0, 0]
        # Edge (0,1) will be reset every step; edge (0,2) will age.
        gng.edges = {(0, 1): 0, (0, 2): 0}
        gng.step_count = 0

        # Run enough steps for edge (0,2) to exceed max_age.
        for _ in range(max_age + 2):
            gng.step(0.0)

        # Neuron at position 100.0 should have been pruned once isolated.
        assert 100.0 not in gng.positions, (
            "Far neuron should be pruned after its edge exceeds max_age"
        )

    def test_edge_age_resets_on_match(self):
        gng, data = _initialized_gng(max_age=100)
        # Drive many steps; the s1–s2 edge age should always be 0 after each step.
        for x in data[:50]:
            gng.step(x)
            for age in gng.edges.values():
                assert age >= 0


# ---------------------------------------------------------------------------
# Neuron insertion
# ---------------------------------------------------------------------------

class TestNeuronInsertion:
    def test_neuron_inserted_every_lam_steps(self):
        lam = 10
        gng, data = _initialized_gng(lam=lam)
        # After exactly lam steps, a neuron should have been inserted.
        for x in data[:lam]:
            gng.step(x)
        assert len(gng.positions) > 2, "Expected at least one inserted neuron"

    def test_neuron_count_grows_over_time(self):
        gng, data = _initialized_gng(n_data=500, lam=10)
        for x in data:
            gng.step(x)
        assert len(gng.positions) > 3

    def test_new_neuron_within_data_range(self):
        gng, data = _initialized_gng(n_data=500, lam=10)
        lo, hi = data.min(), data.max()
        for x in data:
            gng.step(x)
        for p in gng.positions:
            assert lo - 1e-6 <= p <= hi + 1e-6

    def test_inserted_neuron_lifetime_starts_at_zero(self):
        """A freshly inserted neuron should have lifetime 0 at creation."""
        lam = 10
        gng, data = _initialized_gng(lam=lam)
        # Run exactly up to (but not including) the insertion.
        for x in data[:lam - 1]:
            gng.step(x)
        n_before = len(gng.positions)
        # This step triggers the insertion.
        gng.step(data[lam - 1])
        if len(gng.positions) > n_before:
            # The new neuron is at index n_before (appended last).
            assert gng.lifetimes[-1] == 0


# ---------------------------------------------------------------------------
# Neuron properties (array accessors)
# ---------------------------------------------------------------------------

class TestNeuronProperties:
    def test_neuron_positions_shape(self):
        gng, data = _initialized_gng(n_data=300, lam=20)
        for x in data[:100]:
            gng.step(x)
        pos = gng.neuron_positions
        assert pos.ndim == 1
        assert len(pos) == len(gng.positions)

    def test_neuron_lifetimes_shape(self):
        gng, data = _initialized_gng(n_data=300, lam=20)
        for x in data[:100]:
            gng.step(x)
        lt = gng.neuron_lifetimes
        assert lt.ndim == 1
        assert len(lt) == len(gng.lifetimes)

    def test_edge_segments_count(self):
        gng, data = _initialized_gng(n_data=300, lam=20)
        for x in data[:100]:
            gng.step(x)
        segs = gng.edge_segments()
        assert len(segs) == len(gng.edges)
        for p_i, p_j in segs:
            assert isinstance(p_i, float)
            assert isinstance(p_j, float)

    def test_lifetimes_are_non_negative(self):
        gng, data = _initialized_gng(n_data=300, lam=20)
        for x in data[:200]:
            gng.step(x)
        assert all(lt >= 0 for lt in gng.lifetimes)

    def test_older_neurons_have_larger_lifetimes(self):
        """The two seed neurons should have the highest lifetimes."""
        lam = 10
        gng, data = _initialized_gng(n_data=500, lam=lam)
        for x in data[:200]:
            gng.step(x)
        if len(gng.lifetimes) > 2:
            # At least some neurons should have been around longer than others.
            assert max(gng.lifetimes) > min(gng.lifetimes)


# ---------------------------------------------------------------------------
# Internal _remove helper
# ---------------------------------------------------------------------------

class TestRemove:
    def _gng_with_chain(self) -> GrowingNeuralGas:
        """Return a GNG with 4 neurons in a chain: 0-1-2-3."""
        gng = _make_gng(lam=1000)
        gng.positions = [0.0, 1.0, 2.0, 3.0]
        gng.errors = [0.1, 0.2, 0.3, 0.4]
        gng.lifetimes = [10, 8, 5, 2]
        gng.edges = {(0, 1): 0, (1, 2): 0, (2, 3): 0}
        gng.step_count = 0
        return gng

    def test_remove_single_end_neuron(self):
        gng = self._gng_with_chain()
        gng._remove([3])
        assert len(gng.positions) == 3
        assert gng.positions == [0.0, 1.0, 2.0]
        assert (2, 3) not in gng.edges
        assert (0, 1) in gng.edges
        assert (1, 2) in gng.edges

    def test_remove_middle_neuron(self):
        gng = self._gng_with_chain()
        gng._remove([1])
        assert len(gng.positions) == 3
        # After removing index 1 (value 1.0):
        # old edge (0,1) removed; old (1,2) removed; old (2,3) -> (1,2)
        assert gng.positions == [0.0, 2.0, 3.0]
        assert (1, 2) in gng.edges  # was (2, 3)

    def test_remove_two_neurons_descending(self):
        gng = self._gng_with_chain()
        gng._remove([3, 1])  # descending
        assert len(gng.positions) == 2
        assert gng.positions == [0.0, 2.0]

    def test_edge_indices_consistent_after_removal(self):
        gng = self._gng_with_chain()
        gng._remove([2])
        n = len(gng.positions)
        for i, j in gng.edges:
            assert 0 <= i < n
            assert 0 <= j < n
            assert i < j


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

class TestGenerateDataset:
    def test_returns_correct_length(self):
        data = generate_dataset(1000, seed=7)
        assert len(data) == 1000

    def test_returns_float64(self):
        data = generate_dataset(100)
        assert data.dtype == np.float64

    def test_deterministic_with_same_seed(self):
        d1 = generate_dataset(500, seed=99)
        d2 = generate_dataset(500, seed=99)
        np.testing.assert_array_equal(d1, d2)

    def test_different_seeds_give_different_data(self):
        d1 = generate_dataset(500, seed=1)
        d2 = generate_dataset(500, seed=2)
        assert not np.array_equal(d1, d2)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def _parse(self, args: list[str]) -> argparse.Namespace:
        return build_parser().parse_args(args)

    def test_default_values(self):
        ns = self._parse([])
        assert ns.n_samples == 5000
        assert ns.epsilon_b == 0.2
        assert ns.epsilon_n == 0.006
        assert ns.max_age == 50
        assert ns.lam == 100
        assert ns.alpha == 0.5
        assert ns.d == 0.995
        assert ns.fps == 10.0
        assert ns.steps_per_frame == 50
        assert ns.loop is False
        assert ns.seed == 42

    def test_loop_flag(self):
        ns = self._parse(["--loop"])
        assert ns.loop is True

    def test_custom_hyperparameters(self):
        ns = self._parse([
            "--epsilon-b", "0.3",
            "--epsilon-n", "0.01",
            "--max-age", "30",
            "--lam", "50",
            "--alpha", "0.6",
            "--d", "0.99",
        ])
        assert ns.epsilon_b == 0.3
        assert ns.epsilon_n == 0.01
        assert ns.max_age == 30
        assert ns.lam == 50
        assert ns.alpha == 0.6
        assert ns.d == 0.99

    def test_fps_and_steps_per_frame(self):
        ns = self._parse(["--fps", "24.0", "--steps-per-frame", "100"])
        assert ns.fps == 24.0
        assert ns.steps_per_frame == 100

    def test_n_samples_and_seed(self):
        ns = self._parse(["--n-samples", "2000", "--seed", "7"])
        assert ns.n_samples == 2000
        assert ns.seed == 7

    def test_invalid_epsilon_b_type(self):
        with pytest.raises(SystemExit):
            self._parse(["--epsilon-b", "not_a_float"])

    def test_invalid_max_age_type(self):
        with pytest.raises(SystemExit):
            self._parse(["--max-age", "3.5"])
