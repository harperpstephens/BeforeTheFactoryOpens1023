"""Before the Factory Opens: the two-dimensional extension (Appendix B).

Runs the same information experiment as the one-dimensional benchmark, on the
same Central Ohio corridor, but keeps both map coordinates:

    x = distance along the Intel-Columbus axis / 45 miles
    y = distance across the axis / 45 miles

Distance from the Intel site is therefore true straight-line distance. The
project premium fades with that distance in every direction, and the local
outcome is the mass inside a real five-mile circle instead of a projected band.
The 1D benchmark is imported, not modified.

As in 1D, the pre-project baseline is chosen so the 2021 residence pattern is an
exact equilibrium of the same discrete model used for the transition. Any
movement before opening is then anticipation, not the baseline drifting.

Usage:
    python before_factory_opens_2d_geographic_extension.py --demo
    python before_factory_opens_2d_geographic_extension.py --output-dir Before_the_Factory_Opens_Preferred_2D_Output \
        --one-d-headline Before_the_Factory_Opens_Benchmark_1D_Output/Benchmark_1D_Headline_Metrics.json

The paper's preferred run uses --nx 121 --ny 81 --dt 0.10.

A live run downloads the same LODES and QCEW files as before_factory_opens_main_model.py.
The --demo output is synthetic and for testing code only; never report it as Ohio evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix, eye, lil_matrix
from scipy.sparse.linalg import spsolve

import before_factory_opens_main_model as core

VERSION = "2026.10.04-appendix2-2d-r2"


@dataclass(frozen=True)
class Calibration2D:
    """Inputs for the 2D extension. Economic values must match the 1D benchmark."""

    # Default grid: 21 x 15 cells keeps cells close to square (90/21 = 4.29
    # miles by 60/15 = 4.00 miles). The paper's preferred run overrides this
    # from the command line. A 1D comparison is solved at the same nx and dt,
    # so the 1D-vs-2D gap reflects geography rather than grid choice.
    nx: int = 21
    ny: int = 15
    half_length_miles: float = 45.0
    cross_halfwidth_miles: float = 30.0
    circle_radius_miles: float = 5.0
    smoothing_miles: float = 3.5
    project_scale_miles: float = 20.0

    # Same economic parameters as the 1D benchmark (checked at startup).
    rho: float = 0.05
    c: float = 1.50
    sigma2: float = 0.15
    kappa_1d: float = 0.30
    gamma: float = 1.0
    dt: float = 0.20
    tail_years: float = 12.0

    # Same news dates and expected openings as the 1D benchmark.
    announcement: str = "2022-01-21"
    revision1_date: str = "2024-03-20"
    revision2_date: str = "2025-02-28"
    initial_opening_lag: float = 4.0
    revision1_opening_lag: float = 6.0
    revision2_opening_lag: float = 9.0

    baseline_fit_tv_tol: float = 1e-6
    terminal_tv_tol: float = 2e-3


@dataclass(frozen=True)
class Model2D:
    """The solver's version of the 2D calibration: grid units plus convergence tolerances."""

    nx: int
    ny: int
    dt: float
    horizon: float
    opening: float
    rho: float
    c: float
    sigma2: float
    kappa: float
    gamma: float
    omega: float
    shock_h: float
    circle_h: float
    y_half: float
    b: tuple[float, ...]
    eps: float = 1e-10
    stat_relax: float = 0.15
    dyn_relax: float = 0.50
    stat_tol: float = 1e-8
    dyn_tol: float = 8e-7
    stat_max: int = 900
    dyn_max: int = 800
    policy_tol: float = 5e-9
    policy_max: int = 100
    baseline_fit_tv_tol: float = 1e-6
    terminal_tv_tol: float = 2e-3


@dataclass
class Grid2D:
    """Cell centers, the time grid, and neighbor tables for building generators fast."""

    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    dx: float
    dy: float
    t: np.ndarray
    # One row per (cell, neighbor) pair, used to build every generator quickly.
    src: np.ndarray
    dst: np.ndarray
    direction: np.ndarray
    step: np.ndarray
    diff_rate: np.ndarray
    # One row per shared cell face, used by the exact baseline inversion.
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_h: np.ndarray

    @property
    def n(self) -> int:
        return self.X.size


@dataclass
class Stat2D:
    """A 2D stationary equilibrium and how closely it was solved."""

    value: np.ndarray
    mass: np.ndarray
    control: np.ndarray
    Q: csr_matrix
    fp: float
    hjb: float
    kfe: float


@dataclass
class Path2D:
    """A 2D transition path and how closely it was solved."""

    name: str
    value: np.ndarray
    mass: np.ndarray
    control: np.ndarray
    fp: float
    hjb: float
    kfe: float


def check_reference_primitives(cal: Calibration2D) -> None:
    """Stop if any 2D economic parameter or date has drifted from the 1D benchmark."""
    ref = core.Calibration()
    pairs = {
        "half_length_miles": (cal.half_length_miles, ref.half_length_miles),
        "cross_halfwidth_miles": (cal.cross_halfwidth_miles, ref.cross_halfwidth_miles),
        "smoothing_miles": (cal.smoothing_miles, ref.smoothing_miles),
        "project_scale_miles": (cal.project_scale_miles, ref.project_scale_miles),
        "rho": (cal.rho, ref.rho),
        "c": (cal.c, ref.c),
        "sigma2": (cal.sigma2, ref.sigma2),
        "gamma": (cal.gamma, ref.gamma),
        "initial_opening_lag": (cal.initial_opening_lag, ref.initial_opening_lag),
        "revision1_opening_lag": (cal.revision1_opening_lag, ref.revision1_opening_lag),
        "revision2_opening_lag": (cal.revision2_opening_lag, ref.revision2_opening_lag),
    }
    bad = [k for k, (a, b) in pairs.items() if not np.isclose(a, b, rtol=0, atol=1e-12)]
    if bad:
        raise RuntimeError(f"The 2D parameters no longer match the 1D benchmark: {bad}")
    if (cal.announcement, cal.revision1_date, cal.revision2_date) != (
        ref.announcement,
        ref.revision1_date,
        ref.revision2_date,
    ):
        raise RuntimeError("The 2D news dates no longer match the 1D benchmark.")


def make_grid(m: Model2D) -> Grid2D:
    """Build the 2D cells, time grid, and neighbor tables."""
    if m.nx < 9 or m.ny < 7 or m.nx % 2 == 0 or m.ny % 2 == 0:
        raise ValueError("nx and ny must be odd, with nx>=9 and ny>=7.")
    dx = 2.0 / m.nx
    dy = 2.0 * m.y_half / m.ny
    x = -1.0 + (np.arange(m.nx) + 0.5) * dx
    y = -m.y_half + (np.arange(m.ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)
    nt = int(round(m.horizon / m.dt)) + 1
    t = np.arange(nt, dtype=float) * m.dt

    # Move directions: 0 = left, 1 = right, 2 = down, 3 = up.
    src, dst, direction, step, rate = [], [], [], [], []
    eu, ev, eh = [], [], []
    D = 0.5 * m.sigma2
    idx = lambda j, i: j * m.nx + i
    for j in range(m.ny):
        for i in range(m.nx):
            a = idx(j, i)
            for di, dj, d, h in ((-1, 0, 0, dx), (1, 0, 1, dx), (0, -1, 2, dy), (0, 1, 3, dy)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m.nx and 0 <= jj < m.ny:
                    src.append(a)
                    dst.append(idx(jj, ii))
                    direction.append(d)
                    step.append(h)
                    rate.append(D / h**2)
            if i + 1 < m.nx:
                eu.append(a)
                ev.append(idx(j, i + 1))
                eh.append(dx)
            if j + 1 < m.ny:
                eu.append(a)
                ev.append(idx(j + 1, i))
                eh.append(dy)
    return Grid2D(
        x=x,
        y=y,
        X=X,
        Y=Y,
        dx=dx,
        dy=dy,
        t=t,
        src=np.asarray(src, int),
        dst=np.asarray(dst, int),
        direction=np.asarray(direction, int),
        step=np.asarray(step, float),
        diff_rate=np.asarray(rate, float),
        edge_u=np.asarray(eu, int),
        edge_v=np.asarray(ev, int),
        edge_h=np.asarray(eh, float),
    )


def density(p: np.ndarray, g: Grid2D) -> np.ndarray:
    """Convert cell probabilities to a density per unit area."""
    return np.asarray(p, float) / (g.dx * g.dy)


def premium(g: Grid2D, m: Model2D) -> np.ndarray:
    """Intel's operating premium: a bump that fades with straight-line distance from the site."""
    r = np.hypot(g.X.ravel(), g.Y.ravel())
    return m.omega * np.exp(-0.5 * (r / m.shock_h) ** 2)


def flow_utility(p: np.ndarray, g: Grid2D, m: Model2D, post: bool) -> np.ndarray:
    """Flow payoff per cell: baseline amenity, plus the premium once open, minus congestion."""
    F = m.kappa * (density(p, g) + m.eps) ** m.gamma
    return np.asarray(m.b) + (premium(g, m) if post else 0.0) - F


def control_from_value(V: np.ndarray, g: Grid2D, m: Model2D) -> np.ndarray:
    """Optimal moving speed toward each of the four neighbors.

    Workers move toward any neighbor with higher value, at speed (value gap) / c,
    the same rule as the 1D model, applied in four directions.
    """
    V = np.asarray(V, float).ravel()
    a = np.zeros((g.n, 4), dtype=float)
    gain = np.maximum((V[g.dst] - V[g.src]) / g.step, 0.0) / m.c
    a[g.src, g.direction] = gain
    return a


def control_cost(a: np.ndarray, m: Model2D) -> np.ndarray:
    """Quadratic moving cost: (c / 2) times squared speed, summed over directions."""
    return 0.5 * m.c * np.sum(np.asarray(a, float) ** 2, axis=1)


def controlled_Q(a: np.ndarray, g: Grid2D, m: Model2D) -> csr_matrix:
    """Generator for random moves plus optimal moves, with walls at the domain edges.

    Checks that the result is valid: nonnegative rates and rows that sum to zero.
    """
    a = np.asarray(a, float)
    if a.shape != (g.n, 4):
        raise ValueError("2D control must have shape (number_of_cells, 4).")
    off = g.diff_rate + a[g.src, g.direction] / g.step
    diag = -np.bincount(g.src, weights=off, minlength=g.n)
    rows = np.r_[g.src, np.arange(g.n)]
    cols = np.r_[g.dst, np.arange(g.n)]
    data = np.r_[off, diag]
    Q = csr_matrix((data, (rows, cols)), shape=(g.n, g.n))
    if off.min(initial=0.0) < -1e-13:
        raise RuntimeError("Generator contains a negative off-diagonal transition rate.")
    if np.max(np.abs(np.asarray(Q.sum(axis=1)).ravel())) > 1e-10:
        raise RuntimeError("Generator row sums are not zero.")
    return Q


def inverse_baseline(target: np.ndarray, m0: Model2D, g: Grid2D):
    """Find the baseline amenity ``b`` that makes ``target`` exactly stationary in 2D.

    In 2D there is no closed form like the 1D detailed-balance formula. But
    once you know which neighbor of each face has the higher value, the
    zero-flow condition is linear in V. So: guess which way each face points,
    solve the linear system, recheck the directions, and repeat until they
    stop changing and the target is held perfectly still. Then back out ``b``
    from the stationary HJB, exactly as in 1D.
    """
    p = np.asarray(target, float).ravel()
    p = np.maximum(p, 1e-15)
    p /= p.sum()
    D = 0.5 * m0.sigma2
    V = m0.c * D * np.log(p)
    V -= p @ V
    last_sign = None

    for _ in range(60):
        sign = V[g.edge_v] >= V[g.edge_u]
        weight = np.where(sign, p[g.edge_u], p[g.edge_v])
        coef = weight / g.edge_h**2

        A = lil_matrix((g.n, g.n))
        rhs = np.zeros(g.n)
        # Zero net flow across each face, using the weight of the cell that moves.
        for u, v, h, w in zip(g.edge_u, g.edge_v, g.edge_h, coef):
            A[u, u] -= w
            A[u, v] += w
            A[v, v] -= w
            A[v, u] += w
            d = m0.c * D * (p[v] - p[u]) / h**2
            rhs[u] += d
            rhs[v] -= d

        # V is only pinned down up to a constant; fix it so the population-weighted mean is zero.
        A[-1, :] = p
        rhs[-1] = 0.0
        Vn = np.asarray(spsolve(A.tocsr(), rhs), float)
        a = control_from_value(Vn, g, m0)
        Q = controlled_Q(a, g, m0)
        kfe = float(np.max(np.abs(Q.T @ p)))
        new_sign = Vn[g.edge_v] >= Vn[g.edge_u]
        V = Vn
        if kfe < 1e-10 and np.array_equal(sign, new_sign):
            break
        if last_sign is not None and np.array_equal(last_sign, new_sign) and kfe >= 1e-10:
            raise RuntimeError(f"The 2D baseline inversion stopped improving (KFE residual {kfe:.3e}).")
        last_sign = sign.copy()
    else:
        raise RuntimeError("The 2D baseline inversion never settled on a direction for every face.")

    F = m0.kappa * (density(p, g) + m0.eps) ** m0.gamma
    cst = control_cost(a, m0)
    b = m0.rho * V - Q @ V + F + cst
    shift = 1.0 - float(p @ b)
    b = b + shift
    V = V + shift / m0.rho
    hjb = m0.rho * V - (Q @ V + (b - F) - cst)
    if np.max(np.abs(hjb)) > 1e-10:
        raise RuntimeError("2D inverse HJB identity failed.")
    if np.max(np.abs(Q.T @ p)) > 1e-10:
        raise RuntimeError("2D inverse KFE identity failed.")
    return replace(m0, b=tuple(float(z) for z in b)), V


def stationary_mass(Q: csr_matrix) -> np.ndarray:
    """The population distribution that ``Q`` leaves unchanged."""
    n = Q.shape[0]
    A = Q.T.tolil()
    A[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    p = np.asarray(spsolve(A.tocsr(), rhs), float)
    if p.min() < -1e-9:
        raise RuntimeError("2D stationary mass contains a negative probability.")
    p = np.maximum(p, 0.0)
    return p / p.sum()


def stationary_hjb(p, g, m, post, V0=None):
    """Solve the 2D stationary HJB for a fixed population ``p`` by policy iteration."""
    I = eye(g.n, format="csr")
    V = flow_utility(p, g, m, post) / m.rho if V0 is None else np.asarray(V0, float).copy()
    for _ in range(m.policy_max):
        a = control_from_value(V, g, m)
        Q = controlled_Q(a, g, m)
        z = np.asarray(spsolve(m.rho * I - Q, flow_utility(p, g, m, post) - control_cost(a, m)), float)
        if np.max(np.abs(z - V)) < m.policy_tol:
            V = z
            break
        V = z
    else:
        raise RuntimeError("2D stationary HJB policy iteration failed.")
    a = control_from_value(V, g, m)
    Q = controlled_Q(a, g, m)
    r = m.rho * V - (Q @ V + flow_utility(p, g, m, post) - control_cost(a, m))
    return V, a, Q, float(np.max(np.abs(r)))


def stationary_mfg(m, g, post, p0=None, V0=None):
    """Find a 2D stationary equilibrium: the population that workers' best moves reproduce."""
    p = np.ones(g.n) / g.n if p0 is None else np.asarray(p0, float) / np.sum(p0)
    V = V0
    fp = np.inf
    for _ in range(m.stat_max):
        V, a, Q, hjb = stationary_hjb(p, g, m, post, V)
        br = stationary_mass(Q)
        fp = float(np.max(np.abs(br - p)))
        p = (1.0 - m.stat_relax) * p + m.stat_relax * br
        p /= p.sum()
        if fp < m.stat_tol:
            break
    else:
        raise RuntimeError(f"2D stationary MFG failed: {fp:.3e}")
    V, a, Q, hjb = stationary_hjb(p, g, m, post, V)
    br = stationary_mass(Q)
    return Stat2D(V, p, a, Q, float(np.max(np.abs(br - p))), hjb, float(np.max(np.abs(Q.T @ p))))


def solve_path(name, m, g, p0, post_flags, terminal):
    """Solve a 2D transition from ``p0`` to the ``terminal`` continuation value.

    Same backward-HJB / forward-KFE loop as the 1D model. Large grids converge
    slowly with a fixed step, so the update size adapts each round (see below).
    """
    I = eye(g.n, format="csr")
    p0 = np.asarray(p0, float)

    def backward(pp):
        V = np.zeros_like(pp)
        V[-1] = terminal
        A = np.zeros((len(g.t) - 1, g.n, 4))
        max_hjb = 0.0
        for n in range(len(g.t) - 2, -1, -1):
            u = flow_utility(pp[n], g, m, bool(post_flags[n]))
            Vn = V[n + 1].copy()
            for _ in range(m.policy_max):
                a = control_from_value(Vn, g, m)
                Q = controlled_Q(a, g, m)
                z = np.asarray(
                    spsolve((1.0 / m.dt + m.rho) * I - Q, V[n + 1] / m.dt + u - control_cost(a, m)), float
                )
                if np.max(np.abs(z - Vn)) < m.policy_tol:
                    Vn = z
                    break
                Vn = z
            else:
                raise RuntimeError(f"2D dynamic HJB failed at time step {n}.")
            a = control_from_value(Vn, g, m)
            Q = controlled_Q(a, g, m)
            rr = (Vn - V[n + 1]) / m.dt + m.rho * Vn - Q @ Vn - u + control_cost(a, m)
            max_hjb = max(max_hjb, float(np.max(np.abs(rr))))
            V[n] = Vn
            A[n] = a
        return V, A, max_hjb

    def forward(A):
        pp = np.zeros((len(g.t), g.n))
        pp[0] = p0
        max_kfe = 0.0
        for n, a in enumerate(A):
            Q = controlled_Q(a, g, m)
            z = np.asarray(spsolve(I - m.dt * Q.T, pp[n]), float)
            if z.min() < -1e-10:
                raise RuntimeError("2D dynamic KFE produced negative mass.")
            z = np.maximum(z, 0.0)
            rr = z - pp[n] - m.dt * (Q.T @ z)
            max_kfe = max(max_kfe, float(np.max(np.abs(rr))))
            pp[n + 1] = z
        return pp, max_kfe

    pp = np.repeat(p0[None, :], len(g.t), axis=0)
    fp = np.inf
    # On fine grids a fixed step barely makes progress, so the step size adapts
    # each round (Aitken acceleration). This changes how fast the loop gets
    # there, not where it ends up.
    relax = min(m.dyn_relax, 0.20 if g.n > 1500 else m.dyn_relax)
    min_relax, max_relax = 0.01, m.dyn_relax
    prev_r = None
    prev_fp = np.inf
    max_iter = max(m.dyn_max, 2500 if g.n > 1500 else m.dyn_max)
    for it in range(1, max_iter + 1):
        V, A, max_hjb = backward(pp)
        br, max_kfe = forward(A)
        r = br - pp
        fp = float(np.max(np.abs(r)))
        if fp < m.dyn_tol:
            break
        if prev_r is not None:
            dr = (r - prev_r).ravel()
            denom = float(dr @ dr)
            if denom > 1e-30:
                # Aitken step, clipped hard to keep mass positive and the loop stable.
                num = float(prev_r.ravel() @ dr)
                candidate = -relax * num / denom
                if np.isfinite(candidate):
                    relax = float(np.clip(candidate, min_relax, max_relax))
            if fp > 1.05 * prev_fp:
                relax = max(min_relax, 0.5 * relax)
        pp = pp + relax * r
        pp[0] = p0
        prev_r = r.copy()
        prev_fp = fp
        if it % 100 == 0:
            print(f"  {name}: iter={it}, fp={fp:.3e}, relax={relax:.3f}", flush=True)
    else:
        raise RuntimeError(
            f"The 2D {name} transition did not converge after {max_iter} rounds (fixed-point gap {fp:.3e}, step {relax:.3g})."
        )

    V, A, max_hjb = backward(pp)
    br, max_kfe = forward(A)
    fp = float(np.max(np.abs(br - pp)))
    return Path2D(name, V, br, A, fp, max_hjb, max_kfe)


def time_index(g: Grid2D, t: float) -> int:
    """Index of the grid time closest to ``t``."""
    return int(np.argmin(np.abs(g.t - t)))


def subgrid(g: Grid2D, start: int) -> Grid2D:
    """The same grid, with the clock restarted at time step ``start``."""
    return Grid2D(
        g.x,
        g.y,
        g.X,
        g.Y,
        g.dx,
        g.dy,
        g.t[start:] - g.t[start],
        g.src,
        g.dst,
        g.direction,
        g.step,
        g.diff_rate,
        g.edge_u,
        g.edge_v,
        g.edge_h,
    )


def revision_path(m, g, initial, post, revisions):
    """Apply each delay announcement in turn, as in the 1D model."""
    mass = initial.mass.copy()
    value = initial.value.copy()
    details = []
    for label, news, new_open in revisions:
        i = min(int(np.searchsorted(g.t, news - 1e-12, side="left")), len(g.t) - 1)
        sg = subgrid(g, i)
        seg = solve_path(label, m, sg, mass[i], sg.t >= (new_open - g.t[i]), post.value)
        before = mass[i].copy()
        mass[i:] = seg.mass
        value[i:] = seg.value
        details.append(
            {
                "label": label,
                "news_grid": float(g.t[i]),
                "opening": float(new_open),
                "inheritance_error": float(np.max(np.abs(mass[i] - before))),
                "fp": float(seg.fp),
                "hjb": float(seg.hjb),
                "kfe": float(seg.kfe),
                "min_mass": float(seg.mass.min()),
            }
        )
    return mass, details


def _rect_circle_area(x0, x1, y0, y1, r):
    """Area of the part of a grid cell that lies inside a circle centered on the site.

    Cells fully inside or fully outside are handled directly; cells the circle
    cuts through are integrated numerically to high accuracy.
    """
    corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], float)
    if np.max(np.sum(corners * corners, axis=1)) <= r * r:
        return (x1 - x0) * (y1 - y0)
    cx = 0.0 if x0 <= 0 <= x1 else min(abs(x0), abs(x1))
    cy = 0.0 if y0 <= 0 <= y1 else min(abs(y0), abs(y1))
    if cx * cx + cy * cy >= r * r:
        return 0.0
    lo, hi = max(x0, -r), min(x1, r)
    if lo >= hi:
        return 0.0

    def height(x):
        yy = math.sqrt(max(r * r - x * x, 0.0))
        return max(0.0, min(y1, yy) - max(y0, -yy))

    return float(quad(height, lo, hi, epsabs=1e-12, epsrel=1e-11, limit=100)[0])


def circle_weights(g: Grid2D, radius: float) -> np.ndarray:
    """Share of each cell's area inside the five-mile circle."""
    w = np.zeros(g.n)
    k = 0
    for y in g.y:
        for x in g.x:
            area = _rect_circle_area(x - g.dx / 2, x + g.dx / 2, y - g.dy / 2, y + g.dy / 2, radius)
            w[k] = area / (g.dx * g.dy)
            k += 1
    return w


def tv(a, b) -> float:
    """Total variation distance between two distributions."""
    return float(0.5 * np.sum(np.abs(np.asarray(a) - np.asarray(b))))


def metrics(p, g, circle_w):
    """Mass inside the five-mile circle, plus the average location."""
    p = np.asarray(p, float)
    return {
        "circle_mass": float(p @ circle_w),
        "mean_x": float(p @ g.X.ravel()),
        "mean_y": float(p @ g.Y.ravel()),
    }


def headline(m, g, pre, post, known, surprise, circle_w):
    """2D headline numbers at the original expected opening (same definitions as 1D)."""
    i = time_index(g, m.opening)
    kp, up = known.mass[i], surprise.mass[i]
    bu = metrics(up, g, circle_w)["circle_mass"]
    bk = metrics(kp, g, circle_w)["circle_mass"]
    bp = metrics(post.mass, g, circle_w)["circle_mass"]
    den = bp - bu
    long_tv = tv(post.mass, up)
    return {
        "opening_time": m.opening,
        "geographic_5mi_circle_gap_pp": 100.0 * (bk - bu),
        "opening_tv": tv(kp, up),
        "long_run_circle_gap_pp": 100.0 * den,
        "circle_anticipation_ratio": float((bk - bu) / den) if abs(den) > 1e-12 else None,
        "long_run_tv_from_surprise_state": long_tv,
        "tv_anticipation_ratio": float(tv(kp, up) / long_tv) if long_tv > 1e-12 else None,
        "ratio_warning": "Ratios are normalizations, not guaranteed fractions in [0,1].",
    }


def load_lodes_2d(cal: Calibration2D, refresh=False):
    """Bin 2021 LODES worker residences onto the 2D grid, over the same rectangle as 1D."""
    c1 = replace(
        core.Calibration(),
        nx=cal.nx,
        half_length_miles=cal.half_length_miles,
        cross_halfwidth_miles=cal.cross_halfwidth_miles,
    )
    xw = pd.read_csv(
        core.download(core.URLS["xwalk"], "oh_xwalk.csv.gz", refresh),
        compression="gzip",
        dtype={"tabblk2020": str},
        usecols=["tabblk2020", "blklatdd", "blklondd"],
    )
    xw["tabblk2020"] = xw.tabblk2020.str.zfill(15)
    xw = core.add_corridor(xw, "blklatdd", "blklondd", c1)
    rac = pd.read_csv(
        core.download(core.URLS["rac"].format(year=c1.year), f"oh_rac_{c1.year}.csv.gz", refresh),
        compression="gzip",
        dtype={"h_geocode": str},
        usecols=["h_geocode", "C000"],
    )
    rac["h_geocode"] = rac.h_geocode.str.zfill(15)
    rac["C000"] = pd.to_numeric(rac.C000, errors="coerce").fillna(0.0)
    d = rac.merge(xw, left_on="h_geocode", right_on="tabblk2020", how="inner", validate="one_to_one")
    d["y"] = d.cross_miles / cal.half_length_miles
    y_half = cal.cross_halfwidth_miles / cal.half_length_miles
    xe = np.linspace(-1, 1, cal.nx + 1)
    ye = np.linspace(-y_half, y_half, cal.ny + 1)
    raw, _, _ = np.histogram2d(d.y, d.x, bins=[ye, xe], weights=d.C000)
    if raw.sum() <= 0:
        raise RuntimeError("No LODES residences fell inside the 2D domain.")
    raw = raw.astype(float) / raw.sum()
    total = float(d.C000.sum())
    geo5 = float(d.loc[d.radius_miles <= cal.circle_radius_miles, "C000"].sum() / total)

    # How far workers live from the axis: shows what the 1D projection collapses.
    order = np.argsort(np.abs(d.cross_miles.to_numpy(float)))
    z = np.abs(d.cross_miles.to_numpy(float))[order]
    w = d.C000.to_numpy(float)[order]
    cw = np.cumsum(w) / w.sum()
    qs = {str(q): float(z[min(np.searchsorted(cw, q), len(z) - 1)]) for q in (0.50, 0.75, 0.90, 0.95)}
    return raw, {
        "lodes_workers": int(round(total)),
        "raw_geographic_5mi_share": geo5,
        "abs_cross_distance_weighted_quantiles_miles": qs,
    }


def demo_inputs(cal: Calibration2D):
    """Synthetic two-hump residence map and made-up QCEW values, for testing only."""
    y_half = cal.cross_halfwidth_miles / cal.half_length_miles
    x = -1 + (np.arange(cal.nx) + 0.5) * (2 / cal.nx)
    y = -y_half + (np.arange(cal.ny) + 0.5) * (2 * y_half / cal.ny)
    X, Y = np.meshgrid(x, y)
    raw = (
        np.exp(-0.5 * (((X - 0.28) / 0.26) ** 2 + ((Y + 0.08) / 0.24) ** 2))
        + 0.38 * np.exp(-0.5 * (((X + 0.42) / 0.31) ** 2 + ((Y - 0.22) / 0.19) ** 2))
        + 0.025
    )
    raw /= raw.sum()
    return (
        raw,
        {"employment": 66325.0, "pay": 46448.0},
        {"data_mode": "synthetic_demo_FOR_CODE_VALIDATION_ONLY"},
    )


def build_model(cal: Calibration2D, raw, qcew, nx=None, ny=None, dt=None):
    """Turn the 2D calibration and data into a solvable ``Model2D``."""
    check_reference_primitives(cal)
    nx, ny, dt = nx or cal.nx, ny or cal.ny, dt or cal.dt
    if qcew["employment"] <= 0 or qcew["pay"] <= 0:
        raise ValueError("QCEW employment and pay must be positive.")
    y_half = cal.cross_halfwidth_miles / cal.half_length_miles
    opening = core.grid_at_or_after(cal.initial_opening_lag, dt)
    latest = max(cal.initial_opening_lag, cal.revision1_opening_lag, cal.revision2_opening_lag)
    horizon = core.grid_at_or_after(latest + cal.tail_years, dt)

    # Match congestion to the 1D model. 1D density is 2D density summed across
    # the corridor, so if people were spread evenly across a width W, 2D density
    # would be 1D density / W. Scaling kappa by W^gamma keeps congestion costs
    # the same in that case.
    cross_width = 2.0 * y_half
    kappa_2d = cal.kappa_1d * cross_width**cal.gamma

    local_share = core.INTEL_JOBS / qcew["employment"]
    pay_gap = max(core.INTEL_AVG_PAY - qcew["pay"], 0.0) / qcew["pay"]
    omega = local_share * pay_gap
    m0 = Model2D(
        nx,
        ny,
        dt,
        horizon,
        opening,
        cal.rho,
        cal.c,
        cal.sigma2,
        kappa_2d,
        cal.gamma,
        omega,
        cal.project_scale_miles / cal.half_length_miles,
        cal.circle_radius_miles / cal.half_length_miles,
        y_half,
        tuple(np.ones(nx * ny)),
        baseline_fit_tv_tol=cal.baseline_fit_tv_tol,
        terminal_tv_tol=cal.terminal_tv_tol,
    )
    g = make_grid(m0)

    if raw.shape != (ny, nx):
        raise ValueError(f"raw 2D mass shape {raw.shape} does not match requested grid {(ny, nx)}.")

    sx = cal.smoothing_miles / (g.dx * cal.half_length_miles)
    sy = cal.smoothing_miles / (g.dy * cal.half_length_miles)
    target = gaussian_filter(raw, sigma=(sy, sx), mode="reflect")
    target = np.maximum(target, 1e-15)
    target /= target.sum()
    m, V0 = inverse_baseline(target.ravel(), m0, g)
    diag = {
        "local_job_share": local_share,
        "pay_gap_ratio": pay_gap,
        "omega": omega,
        "kappa_1d_reference": cal.kappa_1d,
        "kappa_2d_scaled": kappa_2d,
        "cross_width_normalized": cross_width,
        "smoothing_miles": cal.smoothing_miles,
        "project_scale_miles": cal.project_scale_miles,
    }
    return m, g, target.ravel(), V0, diag


def matched_1d_headline(cal: Calibration2D, raw2d: np.ndarray, qcew: dict) -> dict:
    """Solve the 1D model on the same nx and dt as the 2D run.

    This gives the fair 1D-vs-2D comparison. The paper's main 1D results use
    a finer grid and time step, so they aren't directly comparable.
    """
    raw1d = np.asarray(raw2d, float).sum(axis=0)
    raw1d /= raw1d.sum()
    c1 = replace(core.Calibration(), nx=cal.nx, dt=cal.dt, tail_years=cal.tail_years)
    m1, g1, _, pre1, post1, known1, surprise1, _ = core.solve_all(c1, raw1d, qcew, nx=cal.nx, dt=cal.dt)
    return core.headline(m1, g1, pre1, post1, known1, surprise1)


def validate_solution(m, g, target, pre, post, known, surprise, no, circle_w, diag):
    """Stop the run if any 2D equilibrium check fails, and say which ones."""
    failures = []
    if pre.hjb >= 1e-7 or post.hjb >= 1e-7:
        failures.append("stationary HJB residual")
    if pre.kfe >= 1e-6 or post.kfe >= 1e-6:
        failures.append("stationary KFE residual")
    if max(known.fp, surprise.fp, no.fp) >= 2e-6:
        failures.append("dynamic fixed-point error")
    if max(known.hjb, surprise.hjb, no.hjb) >= 1e-7:
        failures.append("dynamic HJB residual")
    if max(known.kfe, surprise.kfe, no.kfe) >= 1e-10:
        failures.append("dynamic KFE residual")
    mass_error = max(
        np.max(np.abs(known.mass.sum(1) - 1)),
        np.max(np.abs(surprise.mass.sum(1) - 1)),
        np.max(np.abs(no.mass.sum(1) - 1)),
    )
    if mass_error >= 1e-10 or min(known.mass.min(), surprise.mass.min(), no.mass.min()) < -1e-12:
        failures.append("population accounting")
    baseline_tv = tv(target, pre.mass)
    if baseline_tv > m.baseline_fit_tv_tol:
        failures.append(f"baseline fit TV={baseline_tv:.3e}")
    i = time_index(g, m.opening)
    no_open = tv(no.mass[i], pre.mass)
    if no_open > 1e-5:
        failures.append(f"no-project drift at opening={no_open:.3e}")
    terminal = max(tv(no.mass[-1], pre.mass), tv(known.mass[-1], post.mass), tv(surprise.mass[-1], post.mass))
    if terminal > m.terminal_tv_tol:
        failures.append(f"terminal continuation TV={terminal:.3e}")

    # Geometry check: with people spread evenly, the circle should hold exactly
    # its share of the total area.
    uniform_share = float((np.ones(g.n) / g.n) @ circle_w)
    exact_share = math.pi * m.circle_h**2 / (2.0 * 2.0 * m.y_half)
    circle_error = abs(uniform_share - exact_share)
    if circle_error > 2e-8:
        failures.append(f"circle-integration error={circle_error:.3e}")

    diag.update(
        {
            "baseline_target_vs_stationary_tv": baseline_tv,
            "no_project_tv_at_opening": no_open,
            "no_project_terminal_tv_to_pre": tv(no.mass[-1], pre.mass),
            "known_terminal_tv_to_post": tv(known.mass[-1], post.mass),
            "surprise_terminal_tv_to_post": tv(surprise.mass[-1], post.mass),
            "max_mass_error": float(mass_error),
            "circle_uniform_area_error": circle_error,
        }
    )
    if failures:
        raise RuntimeError("2D equilibrium validation failed: " + ", ".join(failures))


def solve_all(cal, raw, qcew, nx=None, ny=None, dt=None):
    """Solve the 2D model: both stationary states and the forward-looking, surprise, and no-project paths."""
    m, g, target, V0, diag = build_model(cal, raw, qcew, nx=nx, ny=ny, dt=dt)
    pre = stationary_mfg(m, g, False, target, V0)
    initial = pre.mass.copy()
    post = stationary_mfg(m, g, True, initial, pre.value)
    known = solve_path("forward_looking", m, g, initial, g.t >= m.opening, post.value)
    no = solve_path("no_project", m, g, initial, np.zeros(len(g.t), bool), pre.value)
    i = time_index(g, m.opening)
    sg = subgrid(g, i)
    seg = solve_path("surprise_post", m, sg, no.mass[i], np.ones(len(sg.t), bool), post.value)
    surprise = Path2D(
        "surprise_opening",
        np.vstack([no.value[:i], seg.value]),
        np.vstack([no.mass[:i], seg.mass]),
        np.vstack([no.control[:i], seg.control]),
        max(no.fp, seg.fp),
        max(no.hjb, seg.hjb),
        max(no.kfe, seg.kfe),
    )
    cw = circle_weights(g, m.circle_h)
    validate_solution(m, g, target, pre, post, known, surprise, no, cw, diag)
    return m, g, target, pre, post, known, surprise, no, cw, diag


def unit_checks() -> dict:
    """Quick checks with known answers, run before every demo or live solve."""
    cal = replace(Calibration2D(), nx=9, ny=7, dt=0.25, tail_years=2.0)
    q = {"employment": 66325.0, "pay": 46448.0}
    raw = np.ones((cal.ny, cal.nx))
    raw /= raw.sum()
    m, g, target, V0, _ = build_model(cal, raw, q)
    a = control_from_value(np.ones(g.n), g, m)
    Q = controlled_Q(a, g, m)
    generator_rows = float(np.max(np.abs(np.asarray(Q.sum(axis=1)).ravel())))
    inverse_kfe = float(np.max(np.abs(Q.T @ (np.ones(g.n) / g.n))))
    cw = circle_weights(g, m.circle_h)
    uniform_share = float((np.ones(g.n) / g.n) @ cw)
    exact_share = math.pi * m.circle_h**2 / (4 * m.y_half)
    return {
        "generator_row_sum": generator_rows,
        "uniform_diffusion_kfe": inverse_kfe,
        "circle_area_error": abs(uniform_share - exact_share),
        "all_passed": generator_rows < 1e-10
        and inverse_kfe < 1e-10
        and abs(uniform_share - exact_share) < 2e-8,
    }


def write_results(
    out,
    cal,
    m,
    g,
    target,
    pre,
    post,
    known,
    surprise,
    no,
    cw,
    diag,
    meta,
    revision,
    matched_1d: dict,
    one_d_headline: Optional[Path],
):
    """Write the 2D output files and Appendix Figures B1a, B1b, and B2."""
    out.mkdir(parents=True, exist_ok=True)
    head = headline(m, g, pre, post, known, surprise, cw)
    comparison = {
        "matched_1d_nx": cal.nx,
        "matched_1d_dt": cal.dt,
        "matched_1d_opening_tv": matched_1d.get("opening_tv"),
        "two_d_opening_tv": head["opening_tv"],
        "matched_1d_tv_anticipation_ratio": matched_1d.get("tv_anticipation_ratio"),
        "two_d_tv_anticipation_ratio": head["tv_anticipation_ratio"],
        "note": "Matched 1D and 2D TV metrics use the same nx and dt. Local 1D projected-band and 2D circle outcomes are different geographic statistics and are not directly compared.",
    }
    if one_d_headline is not None:
        h1 = json.loads(one_d_headline.read_text())
        comparison.update(
            {
                "frozen_r11_opening_tv": h1.get("opening_tv"),
                "frozen_r11_tv_anticipation_ratio": h1.get("tv_anticipation_ratio"),
                "frozen_r11_note": "Frozen r11 uses its own finer reference discretization; use the matched 1D values above to isolate dimensionality.",
            }
        )

    validation = {
        "version": VERSION,
        "baseline_hjb": pre.hjb,
        "baseline_kfe": pre.kfe,
        "post_hjb": post.hjb,
        "post_kfe": post.kfe,
        "known_fp": known.fp,
        "surprise_fp": surprise.fp,
        "no_project_fp": no.fp,
        "baseline_target_vs_stationary_tv": diag["baseline_target_vs_stationary_tv"],
        "no_project_tv_at_opening": diag["no_project_tv_at_opening"],
        "known_terminal_tv_to_post": diag["known_terminal_tv_to_post"],
        "surprise_terminal_tv_to_post": diag["surprise_terminal_tv_to_post"],
        "max_mass_error": diag["max_mass_error"],
        "circle_uniform_area_error": diag["circle_uniform_area_error"],
        "all_passed": True,
    }
    calibration = {
        "calibration": asdict(cal),
        "model": asdict(m),
        "diagnostics": diag,
        "data": meta,
        "interpretation": "Appendix II 2D robustness. Same retained corridor and information structure as frozen r11; radial project exposure and true five-mile circular outcome.",
    }
    (out / "Preferred_2D_Headline_Metrics.json").write_text(json.dumps(head, indent=2))
    (out / "Preferred_2D_Validation.json").write_text(json.dumps(validation, indent=2))
    (out / "Preferred_2D_Calibration.json").write_text(json.dumps(calibration, indent=2))
    (out / "Preferred_2D_Revision_Validation.json").write_text(json.dumps(revision, indent=2))
    (out / "Matched_1D_vs_Preferred_2D_Comparison.json").write_text(json.dumps(comparison, indent=2))
    tex = (
        "% generated by Before the Factory Opens 2D geographic extension\n"
        f"\\newcommand{{\\TwoDOpeningTVPct}}{{{100*head['opening_tv']:.2f}}}\n"
        f"\\newcommand{{\\TwoDTVAnticipationRatioPct}}{{{100*head['tv_anticipation_ratio']:.1f}}}\n"
        f"\\newcommand{{\\TwoDCircleGapPP}}{{{head['geographic_5mi_circle_gap_pp']:.3f}}}\n"
        f"\\newcommand{{\\TwoDCircleAnticipationRatioPct}}{{{100*head['circle_anticipation_ratio']:.1f}}}\n"
        f"\\newcommand{{\\MatchedOneDTVPct}}{{{100*matched_1d['opening_tv']:.2f}}}\n"
        f"\\newcommand{{\\MatchedOneDTVAnticipationRatioPct}}{{{100*matched_1d['tv_anticipation_ratio']:.1f}}}\n"
    )
    (out / "Preferred_2D_Manuscript_Results.tex").write_text(tex)

    rows = []
    for name, sol in (("forward_looking", known), ("surprise_opening", surprise)):
        for i, t in enumerate(g.t):
            rows.append({"scenario": name, "time": float(t), **metrics(sol.mass[i], g, cw)})
    pd.DataFrame(rows).to_csv(out / "Preferred_2D_Paths.csv", index=False)
    pd.DataFrame(
        {
            "x": g.X.ravel(),
            "y": g.Y.ravel(),
            "smoothed_target_mass": target,
            "discrete_pre_stationary_mass": pre.mass,
            "baseline_b": m.b,
            "project_premium": premium(g, m),
        }
    ).to_csv(out / "Preferred_2D_Baseline.csv", index=False)

    # Figure B2: mass inside the true five-mile circle over time.
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    for name, z in df.groupby("scenario"):
        plt.plot(z.time, z.circle_mass, label=name)
    plt.axvline(m.opening, ls="--", lw=1)
    plt.xlabel("Years since Jan. 2022")
    plt.ylabel("Mass within true 5-mile circle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "Figure_B2_Anticipatory_Adjustment_2D.png", dpi=220)
    plt.close()

    # Figure B1a: the smoothed 2021 baseline the calibration matches.
    plt.figure(figsize=(8, 5.6))
    plt.imshow(
        target.reshape(m.ny, m.nx),
        origin="lower",
        aspect="auto",
        extent=[
            -cal.half_length_miles,
            cal.half_length_miles,
            -cal.cross_halfwidth_miles,
            cal.cross_halfwidth_miles,
        ],
    )
    plt.colorbar(label="Cell probability")
    plt.scatter([0], [0], marker="x", s=60, label="Intel site")
    plt.xlabel("Miles along Intel-Columbus axis")
    plt.ylabel("Miles perpendicular to axis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "Figure_B1a_Smoothed_2021_Baseline_2D.png", dpi=220)
    plt.close()

    # Figure B1b: where forward-looking and surprise populations differ at opening.
    io = time_index(g, m.opening)
    diff = (known.mass[io] - surprise.mass[io]).reshape(m.ny, m.nx)
    vmax = float(np.max(np.abs(diff)))
    plt.figure(figsize=(8, 5.6))
    plt.imshow(
        diff,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[
            -cal.half_length_miles,
            cal.half_length_miles,
            -cal.cross_halfwidth_miles,
            cal.cross_halfwidth_miles,
        ],
    )
    plt.colorbar(label="Forward-looking minus surprise cell mass")
    plt.scatter([0], [0], marker="x", s=60)
    plt.xlabel("Miles along Intel-Columbus axis")
    plt.ylabel("Miles perpendicular to axis")
    plt.tight_layout()
    plt.savefig(out / "Figure_B1b_Opening_Distribution_Difference_2D.png", dpi=220)
    plt.close()

    return head, validation, comparison


def main():
    ap = argparse.ArgumentParser(description="Run the 2D extension of Before the Factory Opens (Appendix B).")
    ap.add_argument(
        "--output-dir",
        default="Before_the_Factory_Opens_Preferred_2D_Output",
        help="Where to write results. An existing folder is replaced.",
    )
    ap.add_argument("--demo", action="store_true", help="Use synthetic inputs. For testing code only.")
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the Census and BLS files instead of using the cache.",
    )
    ap.add_argument("--nx", type=int, default=None, help="Cells along the axis (odd). Paper: 121.")
    ap.add_argument("--ny", type=int, default=None, help="Cells across the axis (odd). Paper: 81.")
    ap.add_argument("--dt", type=float, default=None, help="Time step in years. Paper: 0.10.")
    ap.add_argument(
        "--one-d-headline",
        type=Path,
        default=None,
        help="Optional: the 1D Benchmark_1D_Headline_Metrics.json, to add the main 1D result to the comparison.",
    )
    args = ap.parse_args()

    checks = unit_checks()
    if not checks["all_passed"]:
        raise RuntimeError(f"2D startup checks failed: {checks}")

    cal = replace(Calibration2D(), nx=21, ny=15, dt=0.20, tail_years=4.0) if args.demo else Calibration2D()
    if args.nx is not None:
        cal = replace(cal, nx=args.nx)
    if args.ny is not None:
        cal = replace(cal, ny=args.ny)
    if args.dt is not None:
        cal = replace(cal, dt=args.dt)
    out = Path(args.output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if args.demo:
        raw, qcew, meta = demo_inputs(cal)
    else:
        raw, meta = load_lodes_2d(cal, args.refresh)
        qcew = core.qcew_summary(core.Calibration(), args.refresh)
        meta["data_mode"] = "live_public_data"
    meta["qcew"] = qcew

    matched_1d = matched_1d_headline(cal, raw, qcew)
    m, g, target, pre, post, known, surprise, no, cw, diag = solve_all(cal, raw, qcew)

    # Sequential revisions, with the same news dates and expected openings as 1D.
    r1 = core.grid_at_or_after(core.years(cal.announcement, cal.revision1_date), m.dt)
    r2 = core.grid_at_or_after(core.years(cal.announcement, cal.revision2_date), m.dt)
    o1 = core.grid_at_or_after(cal.revision1_opening_lag, m.dt)
    o2 = core.grid_at_or_after(cal.revision2_opening_lag, m.dt)
    rm, details = revision_path(m, g, known, post, [("2024_revision", r1, o1), ("2025_revision", r2, o2)])
    i0 = time_index(g, m.opening)
    revision = {
        "events": details,
        "max_mass_error": float(np.max(np.abs(rm.sum(1) - 1))),
        "min_mass": float(rm.min()),
        "circle_gap_at_initial_expected_opening_pp": 100.0
        * (metrics(known.mass[i0], g, cw)["circle_mass"] - metrics(rm[i0], g, cw)["circle_mass"]),
        "tv_at_initial_expected_opening": tv(known.mass[i0], rm[i0]),
        "note": "Same representative information dates and maintained expected-opening regimes as frozen r11.",
    }
    revision["all_passed"] = bool(
        revision["max_mass_error"] < 1e-10
        and revision["min_mass"] >= -1e-12
        and all(
            e["inheritance_error"] < 1e-12
            and e["fp"] < 2e-6
            and e["hjb"] < 1e-7
            and e["kfe"] < 1e-10
            and e["min_mass"] >= -1e-12
            for e in details
        )
    )
    if not revision["all_passed"]:
        raise RuntimeError(f"2D revision-path validation failed: {revision}")

    pd.DataFrame([{"time": float(t), **metrics(rm[i], g, cw)} for i, t in enumerate(g.t)]).to_csv(
        out / "Preferred_2D_Sequential_Revision_Path.csv", index=False
    )
    head, val, comparison = write_results(
        out,
        cal,
        m,
        g,
        target,
        pre,
        post,
        known,
        surprise,
        no,
        cw,
        diag,
        meta,
        revision,
        matched_1d,
        args.one_d_headline,
    )
    print(
        json.dumps(
            {
                "version": VERSION,
                "unit_checks": checks,
                "headline": head,
                "validation": val,
                "comparison_to_1d": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
