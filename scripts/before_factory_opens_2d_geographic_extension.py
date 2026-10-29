"""Before the Factory Opens: the two-dimensional extension (Appendix B).

Runs the same information experiment as the one-dimensional benchmark, on the
same Central Ohio corridor, but keeps both map coordinates:

    x = distance along the Intel-Columbus axis / 45 miles
    y = distance across the axis / 45 miles

Distance from the Intel site is therefore true straight-line distance. The
project premium fades with that distance in every direction, and the local
outcome is the mass inside a real five-mile circle instead of a projected band.
The 1D benchmark's solvers are reused unchanged; this file supplies the 2D grid,
generator, baseline inversion, and outcome measures.

As in 1D, the pre-project baseline is chosen so the 2021 residence pattern is an
exact equilibrium of the same discrete model used for the transition. Any
movement before opening is then anticipation, not the baseline drifting.

Usage:
    python before_factory_opens_2d_geographic_extension.py --demo
    python before_factory_opens_2d_geographic_extension.py --output-dir Before_the_Factory_Opens_Preferred_2D_Output \\
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
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

import before_factory_opens_main_model as core
from before_factory_opens_main_model import time_index, tv

VERSION = "2026.10.04-appendix2-2d-r2"
REF = core.Calibration()


@dataclass(frozen=True)
class Calibration2D:
    """Inputs for the 2D extension. Economic values and dates are taken from the 1D benchmark."""

    # Default grid: 21 x 15 cells keeps cells close to square (90/21 = 4.29
    # miles by 60/15 = 4.00 miles). The paper's preferred run overrides this
    # from the command line. A 1D comparison is solved at the same nx and dt,
    # so the 1D-vs-2D gap reflects geography rather than grid choice.
    nx: int = 21
    ny: int = 15
    half_length_miles: float = REF.half_length_miles
    cross_halfwidth_miles: float = REF.cross_halfwidth_miles
    circle_radius_miles: float = REF.central_band_miles
    smoothing_miles: float = REF.smoothing_miles
    project_scale_miles: float = REF.project_scale_miles
    rho: float = REF.rho
    c: float = REF.c
    sigma2: float = REF.sigma2
    kappa_1d: float = REF.kappa
    gamma: float = REF.gamma
    dt: float = 0.20
    tail_years: float = REF.tail_years
    announcement: str = REF.announcement
    revision1_date: str = REF.revision1_date
    revision2_date: str = REF.revision2_date
    initial_opening_lag: float = REF.initial_opening_lag
    revision1_opening_lag: float = REF.revision1_opening_lag
    revision2_opening_lag: float = REF.revision2_opening_lag
    baseline_fit_tv_tol: float = REF.baseline_fit_tv_tol
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
    adaptive = True  # Aitken-accelerated relaxation in core.solve_path (class attribute, kept out of asdict)

    def utility(self, p, g, post):
        """Flow payoff per cell: baseline amenity, plus the premium once open, minus congestion."""
        F = self.kappa * (density(p, g) + self.eps) ** self.gamma
        return np.asarray(self.b) + (premium(g, self) if post else 0.0) - F

    def drift(self, V, g):
        """Optimal moving speed toward each of the four neighbors.

        Workers move toward any neighbor with higher value, at speed (value gap) / c,
        the same rule as the 1D model, applied in four directions.
        """
        V = np.asarray(V, float).ravel()
        a = np.zeros((g.n, 4), dtype=float)
        a[g.src, g.direction] = np.maximum((V[g.dst] - V[g.src]) / g.step, 0.0) / self.c
        return a

    def Q(self, a, g):
        """Generator for random moves plus optimal moves, with walls at the domain edges.

        Checks that the result is valid: nonnegative rates and rows that sum to zero.
        """
        a = np.asarray(a, float)
        if a.shape != (g.n, 4):
            raise ValueError("2D control must have shape (number_of_cells, 4).")
        off = g.diff_rate + a[g.src, g.direction] / g.step
        diag = -np.bincount(g.src, weights=off, minlength=g.n)
        Q = csr_matrix((np.r_[off, diag], (np.r_[g.src, np.arange(g.n)], np.r_[g.dst, np.arange(g.n)])),
                       shape=(g.n, g.n))
        if off.min(initial=0.0) < -1e-13:
            raise RuntimeError("Generator contains a negative off-diagonal transition rate.")
        if np.max(np.abs(np.asarray(Q.sum(axis=1)).ravel())) > 1e-10:
            raise RuntimeError("Generator row sums are not zero.")
        return Q

    def cost(self, a):
        """Quadratic moving cost: (c / 2) times squared speed, summed over directions."""
        return 0.5 * self.c * np.sum(np.asarray(a, float) ** 2, axis=1)


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


def make_grid(m: Model2D) -> Grid2D:
    """Build the 2D cells, time grid, and neighbor tables."""
    if m.nx < 9 or m.ny < 7 or m.nx % 2 == 0 or m.ny % 2 == 0:
        raise ValueError("nx and ny must be odd, with nx>=9 and ny>=7.")
    dx = 2.0 / m.nx
    dy = 2.0 * m.y_half / m.ny
    x = -1.0 + (np.arange(m.nx) + 0.5) * dx
    y = -m.y_half + (np.arange(m.ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)
    t = np.arange(int(round(m.horizon / m.dt)) + 1, dtype=float) * m.dt

    # Move directions: 0 = left, 1 = right, 2 = down, 3 = up.
    nbr, edges = [], []
    D = 0.5 * m.sigma2
    idx = lambda j, i: j * m.nx + i
    for j in range(m.ny):
        for i in range(m.nx):
            for di, dj, d, h in ((-1, 0, 0, dx), (1, 0, 1, dx), (0, -1, 2, dy), (0, 1, 3, dy)):
                if 0 <= i + di < m.nx and 0 <= j + dj < m.ny:
                    nbr.append((idx(j, i), idx(j + dj, i + di), d, h, D / h**2))
            if i + 1 < m.nx:
                edges.append((idx(j, i), idx(j, i + 1), dx))
            if j + 1 < m.ny:
                edges.append((idx(j, i), idx(j + 1, i), dy))
    src, dst, direction, step, rate = zip(*nbr)
    eu, ev, eh = zip(*edges)
    i_, f_ = lambda v: np.asarray(v, int), lambda v: np.asarray(v, float)
    return Grid2D(x, y, X, Y, dx, dy, t, i_(src), i_(dst), i_(direction), f_(step), f_(rate),
                  i_(eu), i_(ev), f_(eh))


def density(p: np.ndarray, g: Grid2D) -> np.ndarray:
    """Convert cell probabilities to a density per unit area."""
    return np.asarray(p, float) / (g.dx * g.dy)


def premium(g: Grid2D, m: Model2D) -> np.ndarray:
    """Intel's operating premium: a bump that fades with straight-line distance from the site."""
    r = np.hypot(g.X.ravel(), g.Y.ravel())
    return m.omega * np.exp(-0.5 * (r / m.shock_h) ** 2)


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
        coef = np.where(sign, p[g.edge_u], p[g.edge_v]) / g.edge_h**2

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
        a = m0.drift(Vn, g)
        Q = m0.Q(a, g)
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
    cst = m0.cost(a)
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
    return np.array([_rect_circle_area(x - g.dx / 2, x + g.dx / 2, y - g.dy / 2, y + g.dy / 2, radius)
                     / (g.dx * g.dy) for y in g.y for x in g.x])


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
    d = core.lodes_blocks(replace(REF, nx=cal.nx), refresh=refresh)
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
    return raw, {"employment": 66325.0, "pay": 46448.0}, {"data_mode": "synthetic_demo_FOR_CODE_VALIDATION_ONLY"}


def build_model(cal: Calibration2D, raw, qcew, nx=None, ny=None, dt=None):
    """Turn the 2D calibration and data into a solvable ``Model2D``."""
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
        nx, ny, dt, horizon, opening, cal.rho, cal.c, cal.sigma2, kappa_2d, cal.gamma, omega,
        cal.project_scale_miles / cal.half_length_miles, cal.circle_radius_miles / cal.half_length_miles,
        y_half, tuple(np.ones(nx * ny)),
        baseline_fit_tv_tol=cal.baseline_fit_tv_tol, terminal_tv_tol=cal.terminal_tv_tol,
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
    c1 = replace(REF, nx=cal.nx, dt=cal.dt, tail_years=cal.tail_years)
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
    mass_error = max(np.max(np.abs(s.mass.sum(1) - 1)) for s in (known, surprise, no))
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

    diag.update({
        "baseline_target_vs_stationary_tv": baseline_tv,
        "no_project_tv_at_opening": no_open,
        "no_project_terminal_tv_to_pre": tv(no.mass[-1], pre.mass),
        "known_terminal_tv_to_post": tv(known.mass[-1], post.mass),
        "surprise_terminal_tv_to_post": tv(surprise.mass[-1], post.mass),
        "max_mass_error": float(mass_error),
        "circle_uniform_area_error": circle_error,
    })
    if failures:
        raise RuntimeError("2D equilibrium validation failed: " + ", ".join(failures))


def solve_all(cal, raw, qcew, nx=None, ny=None, dt=None):
    """Solve the 2D model: both stationary states and the forward-looking, surprise, and no-project paths."""
    m, g, target, V0, diag = build_model(cal, raw, qcew, nx=nx, ny=ny, dt=dt)
    pre = core.stationary_mfg(m, g, False, target, V0)
    initial = pre.mass.copy()
    post = core.stationary_mfg(m, g, True, initial, pre.value)
    known = core.solve_path("forward_looking", m, g, initial, g.t >= m.opening, post.value)
    no = core.solve_path("no_project", m, g, initial, np.zeros(len(g.t), bool), pre.value)
    surprise = core.surprise_path(m, g, no, post)
    cw = circle_weights(g, m.circle_h)
    validate_solution(m, g, target, pre, post, known, surprise, no, cw, diag)
    return m, g, target, pre, post, known, surprise, no, cw, diag


def unit_checks() -> dict:
    """Quick checks with known answers, run before every demo or live solve."""
    cal = replace(Calibration2D(), nx=9, ny=7, dt=0.25, tail_years=2.0)
    raw = np.ones((cal.ny, cal.nx))
    raw /= raw.sum()
    m, g, target, V0, _ = build_model(cal, raw, {"employment": 66325.0, "pay": 46448.0})
    Q = m.Q(m.drift(np.ones(g.n), g), g)
    generator_rows = float(np.max(np.abs(np.asarray(Q.sum(axis=1)).ravel())))
    inverse_kfe = float(np.max(np.abs(Q.T @ (np.ones(g.n) / g.n))))
    uniform_share = float((np.ones(g.n) / g.n) @ circle_weights(g, m.circle_h))
    exact_share = math.pi * m.circle_h**2 / (4 * m.y_half)
    return {
        "generator_row_sum": generator_rows,
        "uniform_diffusion_kfe": inverse_kfe,
        "circle_area_error": abs(uniform_share - exact_share),
        "all_passed": generator_rows < 1e-10 and inverse_kfe < 1e-10 and abs(uniform_share - exact_share) < 2e-8,
    }


def heatmap(arr, cal, label, **kw):
    """Draw one cell-level map of the corridor (Figures B1a and B1b)."""
    plt.figure(figsize=(8, 5.6))
    plt.imshow(arr, origin="lower", aspect="auto", **kw, extent=[
        -cal.half_length_miles, cal.half_length_miles, -cal.cross_halfwidth_miles, cal.cross_halfwidth_miles])
    plt.colorbar(label=label)
    plt.xlabel("Miles along Intel-Columbus axis")
    plt.ylabel("Miles perpendicular to axis")


def write_results(out, cal, m, g, target, pre, post, known, surprise, no, cw, diag, meta, revision,
                  matched_1d: dict, one_d_headline: Optional[Path]):
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
        comparison.update({
            "frozen_r11_opening_tv": h1.get("opening_tv"),
            "frozen_r11_tv_anticipation_ratio": h1.get("tv_anticipation_ratio"),
            "frozen_r11_note": "Frozen r11 uses its own finer reference discretization; use the matched 1D values above to isolate dimensionality.",
        })

    validation = {
        "version": VERSION,
        "baseline_hjb": pre.hjb,
        "baseline_kfe": pre.kfe,
        "post_hjb": post.hjb,
        "post_kfe": post.kfe,
        "known_fp": known.fp,
        "surprise_fp": surprise.fp,
        "no_project_fp": no.fp,
        **{k: diag[k] for k in ("baseline_target_vs_stationary_tv", "no_project_tv_at_opening",
                                "known_terminal_tv_to_post", "surprise_terminal_tv_to_post",
                                "max_mass_error", "circle_uniform_area_error")},
        "all_passed": True,
    }
    calibration = {
        "calibration": asdict(cal),
        "model": asdict(m),
        "diagnostics": diag,
        "data": meta,
        "interpretation": "Appendix II 2D robustness. Same retained corridor and information structure as frozen r11; radial project exposure and true five-mile circular outcome.",
    }
    for name, obj in (("Preferred_2D_Headline_Metrics", head), ("Preferred_2D_Validation", validation),
                      ("Preferred_2D_Calibration", calibration), ("Preferred_2D_Revision_Validation", revision),
                      ("Matched_1D_vs_Preferred_2D_Comparison", comparison)):
        (out / f"{name}.json").write_text(json.dumps(obj, indent=2))
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
    pd.DataFrame({
        "x": g.X.ravel(),
        "y": g.Y.ravel(),
        "smoothed_target_mass": target,
        "discrete_pre_stationary_mass": pre.mass,
        "baseline_b": m.b,
        "project_premium": premium(g, m),
    }).to_csv(out / "Preferred_2D_Baseline.csv", index=False)

    # Figure B2: mass inside the true five-mile circle over time.
    plt.figure(figsize=(8, 5))
    for name, z in pd.DataFrame(rows).groupby("scenario"):
        plt.plot(z.time, z.circle_mass, label=name)
    plt.axvline(m.opening, ls="--", lw=1)
    plt.xlabel("Years since Jan. 2022")
    plt.ylabel("Mass within true 5-mile circle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "Figure_B2_Anticipatory_Adjustment_2D.png", dpi=220)
    plt.close()

    # Figure B1a: the smoothed 2021 baseline the calibration matches.
    heatmap(target.reshape(m.ny, m.nx), cal, "Cell probability")
    plt.scatter([0], [0], marker="x", s=60, label="Intel site")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "Figure_B1a_Smoothed_2021_Baseline_2D.png", dpi=220)
    plt.close()

    # Figure B1b: where forward-looking and surprise populations differ at opening.
    io = time_index(g, m.opening)
    diff = (known.mass[io] - surprise.mass[io]).reshape(m.ny, m.nx)
    vmax = float(np.max(np.abs(diff)))
    heatmap(diff, cal, "Forward-looking minus surprise cell mass", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.scatter([0], [0], marker="x", s=60)
    plt.tight_layout()
    plt.savefig(out / "Figure_B1b_Opening_Distribution_Difference_2D.png", dpi=220)
    plt.close()

    return head, validation, comparison


def main():
    ap = argparse.ArgumentParser(description="Run the 2D extension of Before the Factory Opens (Appendix B).")
    ap.add_argument("--output-dir", default="Before_the_Factory_Opens_Preferred_2D_Output",
                    help="Where to write results. An existing folder is replaced.")
    ap.add_argument("--demo", action="store_true", help="Use synthetic inputs. For testing code only.")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-download the Census and BLS files instead of using the cache.")
    ap.add_argument("--nx", type=int, default=None, help="Cells along the axis (odd). Paper: 121.")
    ap.add_argument("--ny", type=int, default=None, help="Cells across the axis (odd). Paper: 81.")
    ap.add_argument("--dt", type=float, default=None, help="Time step in years. Paper: 0.10.")
    ap.add_argument("--one-d-headline", type=Path, default=None,
                    help="Optional: the 1D Benchmark_1D_Headline_Metrics.json, to add the main 1D result to the comparison.")
    args = ap.parse_args()

    checks = unit_checks()
    if not checks["all_passed"]:
        raise RuntimeError(f"2D startup checks failed: {checks}")

    cal = replace(Calibration2D(), nx=21, ny=15, dt=0.20, tail_years=4.0) if args.demo else Calibration2D()
    cal = replace(cal, **{k: v for k in ("nx", "ny", "dt") if (v := getattr(args, k)) is not None})
    out = Path(args.output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if args.demo:
        raw, qcew, meta = demo_inputs(cal)
    else:
        raw, meta = load_lodes_2d(cal, args.refresh)
        qcew = core.qcew_summary(REF, args.refresh)
        meta["data_mode"] = "live_public_data"
    meta["qcew"] = qcew

    matched_1d = matched_1d_headline(cal, raw, qcew)
    m, g, target, pre, post, known, surprise, no, cw, diag = solve_all(cal, raw, qcew)

    # Sequential revisions, with the same news dates and expected openings as 1D.
    rm, revision = core.sequential_revisions(
        cal, m, g, known, post, lambda p: metrics(p, g, cw)["circle_mass"],
        "circle_gap_at_initial_expected_opening_pp",
        "Same representative information dates and maintained expected-opening regimes as frozen r11.")
    pd.DataFrame([{"time": float(t), **metrics(rm[i], g, cw)} for i, t in enumerate(g.t)]).to_csv(
        out / "Preferred_2D_Sequential_Revision_Path.csv", index=False)
    head, val, comparison = write_results(out, cal, m, g, target, pre, post, known, surprise, no, cw, diag,
                                          meta, revision, matched_1d, args.one_d_headline)
    print(json.dumps({"version": VERSION, "unit_checks": checks, "headline": head, "validation": val,
                      "comparison_to_1d": comparison}, indent=2))


if __name__ == "__main__":
    main()
