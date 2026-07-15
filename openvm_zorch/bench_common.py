"""Shared plumbing for the prover's timing harnesses (``verify_prove``, the
per-stage benches).

A stage's output (carry, transcript, message) is plain ``@dataclass`` objects --
``ProveCarry``, ``StackedPcsData``, the proof messages -- deliberately NOT
registered FRX pytrees (``ProveCarry``'s docstring: the carry is host-side Python
that never crosses a ``frx.jit`` boundary). So ``frx.tree_util`` cannot see the
arrays inside them and ``frx.block_until_ready`` on one is a silent no-op --
which would stop a timer at dispatch rather than at compute completion. Hence the
hand-walk here.
"""

from __future__ import annotations

import dataclasses

import frx

from openvm_zorch.commit.stacking import StackedLayout
from openvm_zorch.logup_zerocheck.constraints import ConstraintsDag

# Host-side plans that hold no device array: walking one can only return ``[]``,
# and the walk runs inside the harness's timer, where it cost more than the stage
# it was timing. Only add a type that cannot reach an array -- one that can would
# drop leaves and stop the timer at dispatch, the bug ``array_leaves`` exists to
# prevent.
_NO_ARRAY_TYPES = (ConstraintsDag, StackedLayout)


def array_leaves(obj, skip: tuple[type, ...] = _NO_ARRAY_TYPES) -> list[frx.Array]:
    """The FRX arrays reachable in ``obj`` -- what a harness must block on to
    time a stage honestly.

    ``skip`` exists so the test can re-walk the skipped types with ``skip=()``
    and prove they really hold no array; callers want the default.
    """
    if isinstance(obj, frx.Array):
        return [obj]
    if skip and isinstance(obj, skip):
        return []
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [
            a
            for f in dataclasses.fields(obj)
            for a in array_leaves(getattr(obj, f.name), skip)
        ]
    if isinstance(obj, (list, tuple)):
        return [a for x in obj for a in array_leaves(x, skip)]
    if isinstance(obj, dict):
        return [a for x in obj.values() for a in array_leaves(x, skip)]
    return []
