"""Symbolic constraint DAG: loading and vectorized evaluation.

The reference keygen flattens each AIR's constraints and interaction
expressions into one ``SymbolicExpressionDag`` — nodes in topological order,
plus the node indices asserted to zero and the interactions referencing nodes
by index (air_builders/symbolic/dag.rs). fixture-gen dumps it as
canonical-u32 JSON (``constraints_{air}.json``).

Evaluation mirrors the reference's ``SymbolicEvaluator::eval_nodes``
(symbolic_expression.rs) but vectorized: every node value is an array over
whatever leading batch shape the trace parts carry — (rows,) cells in the
univariate round, (domain, rows) cells in the MLE rounds — so one pass
evaluates the whole grid. Works in either field: pass base-field parts for
the univariate round, folded extension-field parts afterwards.

Reference:
https://github.com/openvm-org/stark-backend/blob/16d60de7/crates/stark-backend/src/air_builders/symbolic/dag.rs
https://github.com/openvm-org/stark-backend/blob/16d60de7/crates/stark-backend/src/prover/logup_zerocheck/single.rs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import frx.numpy as fnp
from frx import Array

from openvm_zorch.fields import F, f_const, f_to_ef


@dataclass(frozen=True)
class Interaction:
    """One interaction, expressions referenced by node index."""

    bus_index: int
    message: tuple[int, ...]
    count: int
    count_weight: int


@dataclass(frozen=True)
class ConstraintsDag:
    """One AIR's symbolic constraints, as dumped by fixture-gen."""

    nodes: tuple[dict[str, Any], ...]
    constraint_idx: tuple[int, ...]
    interactions: tuple[Interaction, ...]

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "ConstraintsDag":
        return cls(
            nodes=tuple(obj["nodes"]),
            constraint_idx=tuple(obj["constraint_idx"]),
            interactions=tuple(
                Interaction(
                    bus_index=i["bus_index"],
                    message=tuple(i["message"]),
                    count=i["count"],
                    count_weight=i["count_weight"],
                )
                for i in obj["interactions"]
            ),
        )


def _promote(x: Array) -> Array:
    """Lift a base-field array into the extension when needed."""
    return f_to_ef(x) if x.dtype == F else x


def eval_nodes(
    dag: ConstraintsDag,
    sels: Array,
    parts: Sequence[tuple[Array, Array | None]],
    public_values: Sequence[int],
) -> list[Array]:
    """Evaluate every DAG node over the parts' leading batch shape.

    ``sels`` is ``(..., 3)`` in the evaluator's order [is_first_row,
    is_transition, is_last_row]; ``parts[p]`` is the (local, next) pair of the
    p-th partitioned-main matrix, each ``(..., width)`` (``next`` may be None
    when the AIR never rotates). Constants and public values are embedded in
    the parts' dtype, matching the reference's ``EF::from`` promotions.
    """
    dtype = sels.dtype

    def embed(value: int) -> Array:
        c = f_const(value)
        return c if dtype == F else f_to_ef(c)

    out: list[Array] = []
    for node in dag.nodes:
        kind = node["kind"]
        if kind == "variable":
            if node["entry"] == "main":
                local, nxt = parts[node["part_index"]]
                mat = local if node["offset"] == 0 else nxt
                value = mat[..., node["index"]]
            elif node["entry"] == "public":
                value = embed(public_values[node["index"]])
            else:
                raise NotImplementedError(f"entry {node['entry']!r} not supported")
        elif kind == "is_first_row":
            value = sels[..., 0]
        elif kind == "is_transition":
            value = sels[..., 1]
        elif kind == "is_last_row":
            value = sels[..., 2]
        elif kind == "constant":
            value = embed(node["value"])
        elif kind == "add":
            value = out[node["left"]] + out[node["right"]]
        elif kind == "sub":
            value = out[node["left"]] - out[node["right"]]
        elif kind == "neg":
            value = -out[node["idx"]]
        elif kind == "mul":
            value = out[node["left"]] * out[node["right"]]
        else:
            raise ValueError(f"unknown node kind {kind!r}")
        out.append(value)
    return out


def acc_constraints(
    dag: ConstraintsDag, node_vals: Sequence[Array], lambda_pows: Sequence[Array]
) -> Array:
    """``Σ_k λ^k · C_k`` over the constraint nodes (single.rs
    ``acc_constraints``); always extension-valued."""
    acc = fnp.zeros((), lambda_pows[0].dtype)
    for lam_pow, idx in zip(lambda_pows, dag.constraint_idx):
        acc = acc + lam_pow * _promote(node_vals[idx])
    return acc


def eval_interactions(
    dag: ConstraintsDag, node_vals: Sequence[Array], beta_pows: Sequence[Array]
) -> list[tuple[Array, Array]]:
    """Per interaction, the (count, h_β(message ‖ bus)) pair (single.rs
    ``eval_interactions``); the denominator carries no α term."""
    out = []
    for interaction in dag.interactions:
        denom = beta_pows[len(interaction.message)] * f_to_ef(
            f_const(interaction.bus_index + 1)
        )
        for beta_pow, msg_idx in zip(beta_pows, interaction.message):
            denom = denom + beta_pow * _promote(node_vals[msg_idx])
        out.append((node_vals[interaction.count], denom))
    return out


def acc_interactions(
    dag: ConstraintsDag,
    node_vals: Sequence[Array],
    beta_pows: Sequence[Array],
    eq_3bs: Sequence[Array],
) -> tuple[Array, Array]:
    """Interactions accumulated under their ``eq(ξ_3, b)`` weights as a
    (numerator, denominator) pair (single.rs ``acc_interactions``)."""
    pairs = eval_interactions(dag, node_vals, beta_pows)
    numer = fnp.zeros((), eq_3bs[0].dtype) if eq_3bs else fnp.zeros((), beta_pows[0].dtype)
    denom = numer
    for eq_3b, (count, h_beta) in zip(eq_3bs, pairs):
        numer = numer + eq_3b * _promote(count)
        denom = denom + eq_3b * h_beta
    return numer, denom
