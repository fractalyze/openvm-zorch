"""Prismalinear / univariate-skip toolbox for the batch sumcheck.

SWIRL's round 0 treats each trace column as a prismalinear polynomial —
degree < ``2^l_skip`` in the univariate variable ``Z`` over the two-adic
subgroup ``D``, multilinear in the rest. This module holds the SWIRL-specific
univariate machinery the reference keeps in ``poly_common.rs`` /
``prover/sumcheck.rs``: the ``eq_D`` kernels (plain and ♯-twisted), their
coefficient-form polynomials, PLE folding at a challenge, and the
coset-evaluation/interpolation pair behind ``sumcheck_uni_round0_poly``.

Smallness is structural: every transform here is over ``D`` (or ``d`` cosets
of it), so interpolation matrices are built host-side from canonical integers
and applied as unrolled sums — exact field arithmetic, no NTT-convention
risk. The one ``lax.fft`` call derives ω so the subgroup generator is pinned
to the zkx-native NTT convention (== plonky3's, byte-matched in Stage 1).

The multiplicative-coset generator is plonky3 BabyBear's ``GENERATOR = 31``.

Reference:
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/poly_common.rs
https://github.com/openvm-org/stark-backend/blob/f6a84921/crates/stark-backend/src/prover/sumcheck.rs#L47
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
from jax import Array, lax

from openvm_zorch.fields import EF, F, MODULUS, f_const, f_inv_const, f_to_ef
from zorch.poly.eq import expand_eq_to_hypercube

# plonky3 BabyBear `F::GENERATOR` (baby_bear.rs `MONTY_GEN`).
GENERATOR = 31


@lru_cache(maxsize=None)
def omega_int(l_skip: int) -> int:
    """Canonical generator of the order-``2^l_skip`` subgroup, extracted from
    ``lax.fft`` (evaluating the polynomial ``Z`` on ``D``) so it can never
    drift from the NTT the rest of the prover stands on."""
    coeffs = jnp.zeros((1, 1 << l_skip), F).at[0, 1].set(jnp.ones((), F))
    evals = lax.fft(coeffs, "FFT", 1 << l_skip)[0]
    omega = int(
        jnp.asarray(lax.bitcast_convert_type(evals[1:2], F).astype(jnp.uint32))[0]
    )
    assert pow(omega, 1 << l_skip, MODULUS) == 1
    assert pow(omega, 1 << (l_skip - 1), MODULUS) != 1
    return omega


def omega_pows_f(l_skip: int) -> Array:
    """``[1, ω, ..., ω^{2^l_skip - 1}]`` as base-field constants."""
    w = omega_int(l_skip)
    return jnp.array(
        [pow(w, k, MODULUS) for k in range(1 << l_skip)], F
    )


def _idft_rows(l_skip: int, chunks: Array) -> list[Array]:
    """iDFT over the trailing-window axis: ``chunks`` is ``(..., 2^l_skip)``
    evaluations on ``D`` (index k ↦ ω^k); returns the ``2^l_skip`` coefficient
    slices ``[c_0, c_1, ...]``, each ``(...,)``.

    Unrolled inverse-Vandermonde rows from host integers — exact and
    dtype-agnostic (extension evaluations promote the base-field weights).
    """
    size = 1 << l_skip
    w = omega_int(l_skip)
    inv_n = pow(size, MODULUS - 2, MODULUS)
    ext = chunks.dtype != F
    out = []
    for t in range(size):
        acc = None
        for k in range(size):
            weight = f_const(inv_n * pow(w, (-t * k) % size, MODULUS))
            if ext:
                weight = f_to_ef(weight)
            term = chunks[..., k] * weight
            acc = term if acc is None else acc + term
        out.append(acc)
    return out


def eval_eq_uni(l_skip: int, x: Array, y: Array) -> Array:
    """``eq_D(x, y)`` — the Lagrange-kernel of the skip domain
    (poly_common.rs ``eval_eq_uni``)."""
    one = jnp.ones((), x.dtype)
    res = one
    xp, yp = x, y
    for _ in range(l_skip):
        res = (xp + yp) * res + (xp - one) * (yp - one)
        xp = xp * xp
        yp = yp * yp
    inv = f_inv_const(1 << l_skip)
    return res * (f_to_ef(inv) if x.dtype == EF else inv)


def eval_eq_uni_at_one(l_skip: int, x: Array) -> Array:
    """``eq_D(x, 1)`` (poly_common.rs ``eval_eq_uni_at_one``)."""
    one = jnp.ones((), x.dtype)
    res = one
    xp = x
    for _ in range(l_skip):
        res = res * (xp + one)
        xp = xp * xp
    inv = f_inv_const(1 << l_skip)
    return res * (f_to_ef(inv) if x.dtype == EF else inv)


def eval_in_uni(l_skip: int, n: int, z: Array) -> Array:
    """``in_{D,n}(z)`` — the short-trace stride indicator (poly_common.rs
    ``eval_in_uni``): 1 for ``n >= 0``; for ``n < 0`` it is ``eq_{D'}(z', 1)``
    over the order-``2^{-n}`` subgroup with ``z' = z^{2^{l_skip + n}}``."""
    if n >= 0:
        return jnp.ones((), z.dtype)
    if n < -l_skip:
        raise ValueError(f"n ({n}) < -l_skip ({l_skip})")
    zp = z
    for _ in range(l_skip + n):
        zp = zp * zp
    return eval_eq_uni_at_one(-n, zp)


def eval_eq_mle(x: list[Array], y: list[Array]) -> Array:
    """``∏_i (1 − y_i − x_i + 2·x_i·y_i)`` — the multilinear equality kernel
    evaluated at two points (poly_common.rs ``eval_eq_mle``)."""
    acc = jnp.ones((), EF)
    for x_i, y_i in zip(x, y):
        acc = acc * (jnp.ones((), EF) - y_i - x_i + (x_i * y_i) * 2)
    return acc


def eval_eq_prism(l_skip: int, x: list[Array], y: list[Array]) -> Array:
    """``eq_D(x_0, y_0)·eq(x[1:], y[1:])`` — the prismalinear equality kernel
    (poly_common.rs ``eval_eq_prism``)."""
    return eval_eq_uni(l_skip, x[0], y[0]) * eval_eq_mle(x[1:], y[1:])


def eval_eq_rot_cube(x: list[Array], y: list[Array]) -> tuple[Array, Array]:
    """The (eq, rot) cube kernels (poly_common.rs ``eval_eq_rot_cube``):
    ``eq`` is the multilinear equality, ``rot`` the cyclic-rotation MLE, both
    on ``{0,1}^len(x)``."""
    one = jnp.ones((), EF)
    rot = one
    eq = one
    for x_i, y_i in zip(reversed(x), reversed(y)):
        rot = x_i * (one - y_i) * eq + (one - x_i) * y_i * rot
        eq = eq * (x_i * y_i + (one - x_i) * (one - y_i))
    return eq, rot


def eval_rot_kernel_prism(l_skip: int, x: list[Array], y: list[Array]) -> Array:
    """``κ_rot(x, y)`` — the prismalinear rotation kernel (poly_common.rs
    ``eval_rot_kernel_prism``): rotate within ``D`` off the boundary, rotate
    the cube at the boundary."""
    omega = f_to_ef(f_const(omega_int(l_skip)))
    eq_cube, rot_cube = eval_eq_rot_cube(x[1:], y[1:])
    return eval_eq_uni(l_skip, x[0], y[0] * omega) * eq_cube + eval_eq_uni_at_one(
        l_skip, x[0]
    ) * eval_eq_uni_at_one(l_skip, y[0] * omega) * (rot_cube - eq_cube)


def eq_cube_table(point: list[Array]) -> Array:
    """eq(point, y) for y on the hypercube, LSB-first in ``point`` (index bit
    i ↔ point[i]) — the reference's ``evals_eq_hypercube_serial`` layout."""
    if not point:
        return jnp.ones((1,), EF)
    return expand_eq_to_hypercube(
        jnp.stack(point[::-1]), jnp.ones((), point[0].dtype)
    )


def eval_eq_sharp_uni(l_skip: int, xi_1: list[Array], z: Array) -> Array:
    """``eq♯_D(ξ_1, z)`` (poly_common.rs ``eval_eq_sharp_uni``)."""
    eq_evals = eq_cube_table(xi_1)
    omega = omega_pows_f(l_skip)
    acc = jnp.zeros((), EF)
    for k in range(1 << l_skip):
        acc = acc + eval_eq_uni(l_skip, z, f_to_ef(omega[k])) * eq_evals[k]
    return acc


def eq_uni_poly(l_skip: int, x: Array) -> list[Array]:
    """``eq_D(x, Z)`` in coefficient form (poly_common.rs ``eq_uni_poly``):
    ``coeff_0 = 1/N``, ``coeff_j = x^{N-j}/N``."""
    size = 1 << l_skip
    inv_n = f_to_ef(f_inv_const(size))
    x_pows = [jnp.ones((), x.dtype)]
    for _ in range(size):
        x_pows.append(x_pows[-1] * x)
    return [inv_n] + [x_pows[size - j] * inv_n for j in range(1, size)]


def eq_sharp_uni_poly(l_skip: int, xi_1: list[Array]) -> list[Array]:
    """``eq♯_D(ξ_1, Z)`` in coefficient form — the iDFT of the eq-table of
    ξ_1 read as evaluations on ``D`` (prover/poly.rs ``eq_sharp_uni_poly``)."""
    return _idft_rows(l_skip, eq_cube_table(xi_1))


def fold_ple_evals(l_skip: int, mat: Array, r: Array) -> Array:
    """Fold prismalinear evaluations at ``r`` in the univariate variable
    (prover/sumcheck.rs ``fold_ple_evals``): each column's ``2^l_skip``
    window interpolates to a univariate in ``Z``, evaluated at ``r``.

    ``mat`` is ``(lifted_height, width)`` base-field; rotation/lifting are the
    caller's concern (pass the rolled/tiled matrix). Returns
    ``(lifted_height / 2^l_skip, width)`` extension-valued.
    """
    height, width = mat.shape
    chunks = mat.reshape(height >> l_skip, 1 << l_skip, width)
    coeffs = _idft_rows(l_skip, jnp.moveaxis(chunks, 1, -1))
    acc = f_to_ef(coeffs[0])
    r_pow = r
    for c in coeffs[1:]:
        acc = acc + f_to_ef(c) * r_pow
        r_pow = r_pow * r
    return acc


def coset_evals(l_skip: int, mat: Array, num_cosets: int) -> Array:
    """Evaluations of each column-window polynomial on the geometric cosets
    ``g^{c+1}·D`` for ``c < num_cosets`` (the per-x iDFT + coset-DFT inside
    ``sumcheck_uni_round0_poly``; coset ``g^0·D = D`` is skipped to dodge the
    zerofier's zeros).

    ``mat`` is ``(lifted_height, width)`` base-field. Returns
    ``(num_cosets, 2^l_skip, lifted_height / 2^l_skip, width)`` where axis 1
    indexes ``z = g^{c+1}·ω^k``.
    """
    size = 1 << l_skip
    w = omega_int(l_skip)
    height, width = mat.shape
    chunks = mat.reshape(height >> l_skip, size, width)
    coeffs = _idft_rows(l_skip, jnp.moveaxis(chunks, 1, -1))
    out = []
    for c in range(num_cosets):
        shift = pow(GENERATOR, c + 1, MODULUS)
        evals_z = []
        for k in range(size):
            acc = None
            for t in range(size):
                weight = f_const(pow(shift, t, MODULUS) * pow(w, (t * k) % size, MODULUS))
                term = coeffs[t] * weight
                acc = term if acc is None else acc + term
            evals_z.append(acc)
        out.append(jnp.stack(evals_z))
    return jnp.stack(out)


def geometric_cosets_to_coeffs(
    l_skip: int, evals: Array, num_cosets: int
) -> list[Array]:
    """Interpolate a degree ``< num_cosets·2^l_skip`` polynomial from its
    evaluations on the cosets ``g^{c+1}·D`` (prover/poly.rs
    ``from_geometric_cosets_evals_idft`` with ``shift = init = g``).

    ``evals`` is ``(num_cosets, 2^l_skip)`` (axis 1: z-index on the coset).
    Per coset: iDFT then unshift by ``g^{-(c+1)t}`` gives ``Q_t(s_c^N)`` where
    the polynomial splits as ``P(Z) = Σ_t Z^t·Q_t(Z^N)``; the ``Q_t`` then
    Lagrange-interpolate across the coset offsets ``s_c^N``. The cross-coset
    basis lives on host integers — exact, and free of the reference's
    chirp-z bookkeeping (interpolation through the same points is unique).
    """
    size = 1 << l_skip
    shifts = [pow(GENERATOR, c + 1, MODULUS) for c in range(num_cosets)]
    # Q_t(s_c^N): per-coset iDFT, unshifted.
    q_at = []  # [coset][t]
    for c in range(num_cosets):
        rows = _idft_rows(l_skip, evals[c])
        inv_s = pow(shifts[c], MODULUS - 2, MODULUS)
        q_at.append(
            [row * f_const(pow(inv_s, t, MODULUS)) for t, row in enumerate(rows)]
        )
    # Lagrange basis over the points s_c^N, in coefficient form (host ints).
    points = [pow(s, size, MODULUS) for s in shifts]
    basis: list[list[int]] = []
    for i in range(num_cosets):
        coeffs = [1]
        denom = 1
        for j in range(num_cosets):
            if j == i:
                continue
            coeffs = [0] + coeffs[:]
            lower = [(-points[j]) * c % MODULUS for c in coeffs[1:] + [0]]
            coeffs = [(a + b) % MODULUS for a, b in zip(coeffs, lower)]
            denom = denom * (points[i] - points[j]) % MODULUS
        inv_denom = pow(denom, MODULUS - 2, MODULUS)
        basis.append([c * inv_denom % MODULUS for c in coeffs])
    # coeffs[m·N + t] = Σ_c Q_t(s_c^N) · basis[c][m]
    out: list[Array] = []
    for m in range(num_cosets):
        for t in range(size):
            acc = None
            for c in range(num_cosets):
                term = q_at[c][t] * f_const(basis[c][m])
                acc = term if acc is None else acc + term
            out.append(acc)
    return out
