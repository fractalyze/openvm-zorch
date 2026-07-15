"""Tests for the timing harnesses' shared array walk.

Backend-agnostic host Python (no cuda deps), so this runs in CI on any machine —
which is the point: `array_leaves` drives every per-stage number the milestone-4
"beat native" issues read, and until it was a library nothing could check it.
"""

from __future__ import annotations

from absl.testing import absltest

import frx.numpy as jnp
from zk_dtypes import babybear_mont as F

from openvm_zorch.bench_common import _NO_ARRAY_TYPES, array_leaves
from openvm_zorch.commit.stacking import StackedLayout
from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag, Interaction


def _dag() -> ConstraintsDag:
    """A DAG shaped like fixture-gen's dump: JSON-ish node dicts + ints."""
    return ConstraintsDag(
        nodes=(
            {"kind": "main", "part_index": 0, "offset": 0, "index": 3},
            {"kind": "constant", "value": 7},
            {"kind": "add", "lhs": 0, "rhs": 1},
        ),
        constraint_idx=(2,),
        interactions=(
            Interaction(bus_index=1, message=(0, 1), count=2, count_weight=1),
        ),
    )


class ArrayLeavesTest(absltest.TestCase):
    def test_finds_arrays_through_containers(self):
        a, b, c = jnp.zeros((2,), F), jnp.zeros((3,), F), jnp.zeros((4,), F)
        self.assertLen(array_leaves([a, (b,), {"k": c}]), 3)

    def test_skipped_types_contribute_no_leaf(self):
        self.assertEmpty(array_leaves(_dag()))
        layout = StackedLayout.new(2, 4, [(2, 3), (1, 2)])
        self.assertEmpty(array_leaves(layout))

    def test_skip_does_not_hide_a_sibling_array(self):
        """The skip must drop only the skipped object, never its neighbours —
        the walk still has to reach every array a stage left un-materialized."""
        trace = jnp.zeros((4, 2), F)
        leaves = array_leaves({"dag": _dag(), "trace": trace})
        self.assertLen(leaves, 1)

    def test_skipped_types_really_hold_no_array(self):
        """The load-bearing invariant behind `_NO_ARRAY_TYPES`.

        Skipping these is only sound while they truly cannot reach an array. Walk
        them with the skip disabled: if one ever grows an array field, this fails
        — which is the signal to drop it from the tuple rather than let the
        harness silently stop its timer at dispatch.
        """
        layout = StackedLayout.new(2, 4, [(2, 3), (1, 2)])
        for obj in (_dag(), layout):
            self.assertEmpty(
                array_leaves(obj, skip=()),
                msg=f"{type(obj).__name__} now reaches an array; remove it from "
                "_NO_ARRAY_TYPES",
            )

    def test_no_array_types_is_not_empty(self):
        self.assertIn(ConstraintsDag, _NO_ARRAY_TYPES)
        self.assertIn(StackedLayout, _NO_ARRAY_TYPES)


if __name__ == "__main__":
    absltest.main()
