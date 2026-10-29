"""Before the Factory Opens: the one-dimensional benchmark model (r11).

This script produces the headline results in "Before the Factory Opens:
Anticipation and Spatial Adjustment to Announced and Delayed Local Shocks."

The model is kept deliberately narrow:

- Workers live along one corridor: the line from Intel's Ohio site toward
  downtown Columbus.
- The pre-project residence pattern is 2021 Census LODES data, smoothed. The
  baseline is chosen so that this pattern is an exact stationary equilibrium of
  the same discrete model used for the transition.
- Intel's effect enters as a reduced-form operating premium that fades with
  distance. Its spatial scale is a maintained assumption, not an estimate.
- Two information experiments: forward-looking workers know the expected
  opening date from the January 2022 announcement; surprise workers ignore the
  project until it opens.
- Two public revisions push the expected opening back (March 2024 and
  February 2025).

If the live Census or BLS downloads fail, the program stops; it never fills in
missing data. The ``--demo`` flag runs on synthetic inputs for code testing
only. Those numbers are not Ohio evidence and should never be reported as such.

Usage:
    python before_factory_opens_main_model.py --output-dir outputs/live_1d_results
    python before_factory_opens_main_model.py --demo
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve
from urllib3.util.retry import Retry

VERSION = "2026.10.04-r11"

# Intel Ohio One site and downtown Columbus (latitude, longitude).
SITE = (40.1178, -82.7112)
COLUMBUS = (39.9612, -82.9988)

# Figures from the January 2022 announcement: direct Intel jobs and average pay.
INTEL_JOBS, INTEL_AVG_PAY = 3000.0, 135_000.0

URLS = {
    "xwalk": "https://lehd.ces.census.gov/data/lodes/LODES8/oh/oh_xwalk.csv.gz",
    "rac": "https://lehd.ces.census.gov/data/lodes/LODES8/oh/rac/oh_rac_S000_JT01_{year}.csv.gz",
    "qcew": "https://data.bls.gov/cew/data/api/{year}/a/area/39089.csv",
}
INFO_SOURCES = {
    "announcement_jobs_pay": "https://www.jobsohio.com/newsroom/news-press/intel-chooses-ohio-for-chip-manufacturing",
    "initial_production_timing": "https://www.intc.com/news-events/press-releases/detail/1521/intel-announces-next-us-site-with-landmark-investment-in",
    "2024_revision": "https://www.reuters.com/technology/intel-prepares-100-bln-spending-spree-across-four-us-states-2024-03-20/",
    "2025_revision": "https://www.intel.com/content/www/us/en/newsroom/news/corporate/ohio-one-construction-timeline-update.html",
}
CACHE = Path(".hps_mfg_cache")


# ---------------------------------------------------------------------------
# Parameters and result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """Every input the benchmark needs, in the units the paper reports."""

    year: int = 2021
    nx: int = 61
    half_length_miles: float = 45.0
    cross_halfwidth_miles: float = 30.0
    central_band_miles: float = 5.0
    smoothing_miles: float = 3.5
    project_scale_miles: float = 20.0  # maintained assumption, not estimated
    rho: float = 0.05
    c: float = 1.50
    sigma2: float = 0.15
    kappa: float = 0.30              # a normalization; the baseline cross-section can't pin it down
    gamma: float = 1.0               # congestion curvature, held fixed
    dt: float = 0.10
    tail_years: float = 12.0
    baseline_fit_tv_tol: float = 1e-6

    # News dates. These are the only calendar dates that change what workers know.
    announcement: str = "2022-01-21"
    revision1_date: str = "2024-03-20"
    revision2_date: str = "2025-02-28"

    # Expected opening times, in years after the January 2022 announcement.
    # They stand for the public timing regimes (roughly end of 2025, 2027-28,
    # and 2030-31); none of them claims to be the true operating day.
    initial_opening_lag: float = 4.0
    revision1_opening_lag: float = 6.0
    revision2_opening_lag: float = 9.0


@dataclass(frozen=True)
class Model:
    """The solver's version of the calibration: grid units, plus convergence tolerances."""

    nx: int
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
    band_h: float
    b: tuple[float, ...]
    eps: float = 1e-10
    stat_relax: float = 0.15
    dyn_relax: float = 0.50
    stat_tol: float = 1e-8
    dyn_tol: float = 8e-7
    stat_max: int = 800
    dyn_max: int = 700
    policy_tol: float = 5e-9
    policy_max: int = 100
    terminal_tv_tol: float = 1e-3
    baseline_fit_tv_tol: float = 1e-6
    adaptive = False  # fixed relaxation in solve_path (class attribute, kept out of asdict)

    # The solvers below call these four, so the 2D model only has to supply its own versions.
    def utility(self, p, g, post):
        """Flow payoff in each cell: baseline amenity, plus the premium once open, minus congestion."""
        F = self.kappa * (density(p, g) + self.eps) ** self.gamma
        return np.asarray(self.b) + (premium(g.x, self) if post else 0.0) - F

    def drift(self, V, g):
        return drift(V, g, self)

    def Q(self, a, g):
        return generator(a, g, self)

    def cost(self, a):
        """Quadratic moving cost: (c / 2) times squared speed."""
        a = np.asarray(a, float)
        return .5 * self.c * np.sum(a * a, axis=-1)


@dataclass
class Grid:
    x: np.ndarray   # cell centers on [-1, 1]
    dx: float
    t: np.ndarray   # time grid in years since the announcement

    @property
    def n(self) -> int:
        return len(self.x)


@dataclass
class Stat:
    """A stationary equilibrium and how closely it was solved."""

    value: np.ndarray
    mass: np.ndarray
    drift: np.ndarray
    Q: csr_matrix
    fp: float    # fixed-point gap
    hjb: float   # HJB residual
    kfe: float   # KFE residual


@dataclass
class PathSol:
    """A transition path over the full time grid and how closely it was solved."""

    name: str
    value: np.ndarray
    mass: np.ndarray
    drift: np.ndarray
    fp: float
    hjb: float
    kfe: float


# ---------------------------------------------------------------------------
# Data: LODES residences and QCEW pay
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    """An HTTP session that retries transient Census/BLS server errors."""
    s = requests.Session()
    retry = Retry(
        total=4, connect=4, read=4, status=4, backoff_factor=.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "HPS-Ohio-MFG-r11 academic research"
    return s


HTTP = _session()


def download(url: str, name: str, refresh: bool = False) -> Path:
    """Download a file once and reuse the cached copy unless ``refresh`` is set."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / name
    if p.exists() and p.stat().st_size and not refresh:
        return p
    try:
        r = HTTP.get(url, timeout=180)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Could not download {url}: {e}") from e
    if not r.content:
        raise RuntimeError(f"The download from {url} was empty.")
    p.write_bytes(r.content)
    return p


def years(a: str, b: str) -> float:
    """Years between two ISO dates."""
    return (date.fromisoformat(b) - date.fromisoformat(a)).days / 365.2425


def grid_at_or_after(x: float, dt: float) -> float:
    """Round a news date up to the next grid time, so news never arrives early."""
    return math.ceil((x - 1e-12) / dt) * dt


def local_xy(lat, lon, lat0=SITE[0], lon0=SITE[1]):
    """Miles east and north of the Intel site (flat-earth approximation)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return ((lon - lon0) * 69.172 * math.cos(math.radians(lat0)), (lat - lat0) * 69.0)


def add_corridor(df: pd.DataFrame, lat: str, lon: str, cal: Calibration) -> pd.DataFrame:
    """Project block locations onto the Intel-Columbus axis and keep the corridor.

    ``along_miles`` is distance along the axis (positive toward Columbus) and
    ``x`` rescales it to [-1, 1]. Blocks farther than ``cross_halfwidth_miles``
    from the axis are dropped.
    """
    e, n = local_xy(df[lat], df[lon])
    ce, cn = local_xy([COLUMBUS[0]], [COLUMBUS[1]])
    axis = np.array([ce[0], cn[0]])
    axis /= np.linalg.norm(axis)
    perp = np.array([-axis[1], axis[0]])
    pts = np.c_[e, n]
    z = df.copy()
    z["along_miles"], z["cross_miles"] = pts @ axis, pts @ perp
    z["radius_miles"] = np.hypot(e, n)  # descriptive only; the model state is x
    z["x"] = z["along_miles"] / cal.half_length_miles
    return z[(z.x.abs() <= 1) & (z.cross_miles.abs() <= cal.cross_halfwidth_miles)].copy()


def lodes_blocks(cal: Calibration, year: Optional[int] = None, refresh=False) -> pd.DataFrame:
    """LODES worker counts (``C000``) for every residence block inside the corridor."""
    y = year or cal.year
    xw = pd.read_csv(
        download(URLS["xwalk"], "oh_xwalk.csv.gz", refresh), compression="gzip",
        dtype={"tabblk2020": str}, usecols=["tabblk2020", "blklatdd", "blklondd"],
    )
    xw["tabblk2020"] = xw.tabblk2020.str.zfill(15)
    xw = add_corridor(xw, "blklatdd", "blklondd", cal)

    rac = pd.read_csv(
        download(URLS["rac"].format(year=y), f"oh_rac_{y}.csv.gz", refresh),
        compression="gzip", dtype={"h_geocode": str}, usecols=["h_geocode", "C000"],
    )
    rac["h_geocode"] = rac.h_geocode.str.zfill(15)
    rac["C000"] = pd.to_numeric(rac.C000, errors="coerce").fillna(0)

    return rac.merge(xw, left_on="h_geocode", right_on="tabblk2020", how="inner", validate="one_to_one")


def lodes_mass(cal: Calibration, year: Optional[int] = None, refresh=False):
    """Bin 2021 LODES worker residences into the model's corridor cells.

    Returns the share of workers in each cell, the total worker count in the
    corridor, and the raw share living within five miles of the site.
    """
    d = lodes_blocks(cal, year, refresh)
    edges = np.linspace(-1, 1, cal.nx + 1)
    mass, _ = np.histogram(d.x, bins=edges, weights=d.C000)
    if mass.sum() <= 0:
        raise RuntimeError("No LODES residences fell inside the corridor.")
    mass = mass.astype(float) / mass.sum()
    geo5 = float(d.loc[d.radius_miles <= cal.central_band_miles, "C000"].sum() / d.C000.sum())
    return mass, int(round(d.C000.sum())), geo5


def qcew_summary(cal: Calibration, refresh=False):
    """Licking County employment and average annual pay from the QCEW.

    Aggregation level 70 is County, Total Covered. It is used only to scale the
    size of Intel's pay premium; it is not the same population as LODES residences.
    """
    d = pd.read_csv(
        download(URLS["qcew"].format(year=cal.year), f"qcew_{cal.year}_39089.csv", refresh),
        dtype=str,
    )
    r = d[d.agglvl_code.str.strip() == "70"]
    if len(r) != 1:
        raise RuntimeError(f"Expected one Licking County Total Covered row in the QCEW file; found {len(r)}.")
    r = r.iloc[0]
    return {"employment": float(r.annual_avg_emplvl), "pay": float(r.avg_annual_pay)}


# ---------------------------------------------------------------------------
# Model pieces: grid, payoffs, and the transition generator
# ---------------------------------------------------------------------------

def grid(model: Model) -> Grid:
    """Finite-volume cells on [-1, 1] and the time grid."""
    if model.nx < 21 or model.nx % 2 == 0:
        raise ValueError("nx must be odd and at least 21.")
    dx = 2 / model.nx
    x = -1 + (np.arange(model.nx) + .5) * dx
    nt = int(round(model.horizon / model.dt)) + 1
    return Grid(x, dx, np.arange(nt) * model.dt)


def density(p, g):
    """Convert cell probabilities to a density."""
    return np.asarray(p) / g.dx


def premium(x, m: Model):
    """Intel's operating premium: a Gaussian bump centered on the site."""
    return m.omega * np.exp(-0.5 * (np.asarray(x) / m.shock_h) ** 2)


def drift(V, g, m):
    """Optimal moving speeds toward the left and right neighbor of each cell.

    Each row holds two nonnegative speeds, (left, right). Workers move toward
    the neighbor with higher value, at speed (value gap) / c. Away from a local
    minimum of the value function only one speed is positive; at a minimum both
    can be. This is the finite-state version of a = V_x / c, consistent with
    the generator and the quadratic moving cost below.
    """
    V = np.asarray(V, float)
    left = np.zeros(m.nx)
    right = np.zeros(m.nx)
    left[1:] = np.maximum((V[:-1] - V[1:]) / g.dx, 0.0) / m.c
    right[:-1] = np.maximum((V[1:] - V[:-1]) / g.dx, 0.0) / m.c
    return np.column_stack([left, right])


def generator(a, g, m):
    """Random moves between neighbors (no flux at the edges) plus the optimal moves in ``a``.

    Checks that the result is a valid generator: nonnegative off-diagonal
    rates and rows that sum to zero.
    """
    a = np.asarray(a, float)
    if a.shape != (m.nx, 2):
        raise ValueError("Controls must have shape (nx, 2): left and right speeds.")
    r = .5 * m.sigma2 / g.dx ** 2
    down = r + a[1:, 0] / g.dx    # rate from cell i to i-1
    up = r + a[:-1, 1] / g.dx     # rate from cell i to i+1
    # Summed in the same order as the original cell-by-cell build, so results match bit for bit.
    diag = -(((np.r_[0.0, np.full(m.nx - 1, r)] + np.r_[np.full(m.nx - 1, r), 0.0])
              + a[:, 0] / g.dx * (np.arange(m.nx) > 0)) + a[:, 1] / g.dx * (np.arange(m.nx) < m.nx - 1))
    Q = diags([down, diag, up], [-1, 0, 1], format="csr")
    if min(down.min(), up.min()) < -1e-12:
        raise RuntimeError("The generator has a negative off-diagonal rate.")
    if np.max(np.abs(np.asarray(Q.sum(1)).ravel())) > 1e-10:
        raise RuntimeError("A generator row does not sum to zero.")
    return Q


# ---------------------------------------------------------------------------
# Baseline calibration: make the 2021 residence pattern an exact equilibrium
# ---------------------------------------------------------------------------

def baseline_from_density(target, m0: Model):
    """Find the baseline amenity ``b`` that makes ``target`` exactly stationary.

    Step 1: across each cell face, pick the value difference that gives zero
    net flow of workers when diffusion and optimal moves are combined. Because
    this is a birth-death chain, detailed balance gives that difference in
    closed form.

    Step 2: back out ``b`` from the discrete stationary HJB equation.

    Both steps use the same generator as the transition solver, so the
    baseline is an exact equilibrium of the discrete model that gets solved.
    """
    g = grid(m0)
    p = np.asarray(target, float)
    p = np.maximum(p, 1e-15)
    p /= p.sum()
    D = .5 * m0.sigma2
    V = np.zeros(m0.nx)

    # Detailed balance at face i+1/2. Where p rises across the face, only the
    # rightward move is active; where it falls, only the leftward one.
    dp = p[1:] - p[:-1]
    V[1:] = np.cumsum(m0.c * D * dp / np.minimum(p[:-1], p[1:]))
    V -= p @ V

    a = drift(V, g, m0)
    Q = generator(a, g, m0)
    kfe = float(np.max(np.abs(Q.T @ p)))
    if kfe > 1e-10:
        raise RuntimeError(f"The inverted baseline does not hold the target still (KFE residual {kfe:.3e}).")

    F = m0.kappa * (density(p, g) + m0.eps) ** m0.gamma
    b = m0.rho * V - Q @ V + F + m0.cost(a)

    # Normalize so the population-weighted amenity is 1. Shifting b and V by
    # constants leaves the moves unchanged and keeps the HJB identity.
    shift = 1.0 - float(p @ b)
    b = b + shift
    V = V + shift / m0.rho
    hjb = m0.rho * V - (Q @ V + (b - F) - m0.cost(a))
    if np.max(np.abs(hjb)) > 1e-10:
        raise RuntimeError("The inverted baseline fails the stationary HJB identity.")
    return replace(m0, b=tuple(b)), V


# ---------------------------------------------------------------------------
# Stationary equilibria (before the project, and long after it opens)
# ---------------------------------------------------------------------------

def stationary_mass(Q):
    """The population distribution that ``Q`` leaves unchanged."""
    n = Q.shape[0]
    A = Q.T.tolil()
    A[-1, :] = 1
    rhs = np.zeros(n)
    rhs[-1] = 1
    p = np.asarray(spsolve(A.tocsr(), rhs))
    if p.min() < -1e-9:
        raise RuntimeError("The stationary distribution has negative mass.")
    p = np.maximum(p, 0)
    return p / p.sum()


def stationary_hjb(p, g, m, post, V0=None):
    """Solve the stationary HJB for a fixed population ``p`` by policy iteration."""
    I = eye(g.n, format="csr")
    V = m.utility(p, g, post) / m.rho if V0 is None else V0.copy()
    for _ in range(m.policy_max):
        a = m.drift(V, g)
        Q = m.Q(a, g)
        Vn = np.asarray(spsolve(m.rho * I - Q, m.utility(p, g, post) - m.cost(a)))
        if np.max(np.abs(Vn - V)) < m.policy_tol:
            V = Vn
            break
        V = Vn
    else:
        raise RuntimeError("Policy iteration for the stationary HJB did not converge.")
    a = m.drift(V, g)
    Q = m.Q(a, g)
    r = m.rho * V - (Q @ V + m.utility(p, g, post) - m.cost(a))
    return V, a, Q, float(np.max(np.abs(r)))


def stationary_mfg(m, g, post, p0=None, V0=None):
    """Find a stationary equilibrium: the population that workers' best moves reproduce.

    Alternates between solving workers' problem for the current population and
    moving the population partway toward the distribution those choices imply.
    """
    p = np.ones(g.n) / g.n if p0 is None else np.asarray(p0, float) / np.sum(p0)
    V = V0
    fp = np.inf
    for _ in range(m.stat_max):
        V, a, Q, hjb = stationary_hjb(p, g, m, post, V)
        br = stationary_mass(Q)
        fp = float(np.max(np.abs(br - p)))
        p = (1 - m.stat_relax) * p + m.stat_relax * br
        p /= p.sum()
        if fp < m.stat_tol:
            break
    else:
        raise RuntimeError(f"The stationary equilibrium did not converge (fixed-point gap {fp:.3e}).")
    V, a, Q, hjb = stationary_hjb(p, g, m, post, V)
    br = stationary_mass(Q)
    return Stat(V, p, a, Q, float(np.max(np.abs(br - p))), hjb, float(np.max(np.abs(Q.T @ p))))


# ---------------------------------------------------------------------------
# Transition paths
# ---------------------------------------------------------------------------

def solve_path(name, m, g, p0, post_flags, terminal):
    """Solve a transition from ``p0`` to the ``terminal`` continuation value.

    ``post_flags[n]`` says whether the project pays at time step n. Each round
    solves the HJB backward in time (workers' choices, given where everyone
    else will be) and then the KFE forward (where everyone ends up, given those
    choices). Rounds continue until the population path stops changing.
    """
    I = eye(g.n, format="csr")
    p0 = np.asarray(p0, float)

    def backward(pp):
        """HJB backward in time for a given population path ``pp``."""
        V = np.zeros_like(pp)
        V[-1] = terminal
        Apath = np.zeros((len(g.t) - 1, *m.drift(terminal, g).shape))
        maxh = 0.0
        for n in range(len(g.t) - 2, -1, -1):
            u = m.utility(pp[n], g, bool(post_flags[n]))
            Vn = V[n + 1].copy()
            for _ in range(m.policy_max):
                a = m.drift(Vn, g)
                Q = m.Q(a, g)
                z = np.asarray(spsolve((1 / m.dt + m.rho) * I - Q, V[n + 1] / m.dt + u - m.cost(a)))
                if np.max(np.abs(z - Vn)) < m.policy_tol:
                    Vn = z
                    break
                Vn = z
            else:
                raise RuntimeError(f"The dynamic HJB did not converge at time step {n}.")
            a = m.drift(Vn, g)
            Q = m.Q(a, g)
            rr = (Vn - V[n + 1]) / m.dt + m.rho * Vn - Q @ Vn - u + m.cost(a)
            maxh = max(maxh, float(np.max(np.abs(rr))))
            V[n] = Vn
            Apath[n] = a
        return V, Apath, maxh

    def forward(Apath):
        """KFE forward in time for a given path of moves ``Apath`` (implicit Euler)."""
        pp = np.zeros((len(g.t), g.n))
        pp[0] = p0
        maxk = 0.0
        for n, a in enumerate(Apath):
            Q = m.Q(a, g)
            z = np.asarray(spsolve(I - m.dt * Q.T, pp[n]))
            if z.min() < -1e-10:
                raise RuntimeError("The dynamic KFE produced negative mass.")
            z = np.maximum(z, 0)
            rr = z - pp[n] - m.dt * (Q.T @ z)
            maxk = max(maxk, float(np.max(np.abs(rr))))
            pp[n + 1] = z
        return pp, maxk

    pp = np.repeat(p0[None, :], len(g.t), 0)
    fp = np.inf
    # The 2D model's fine grids barely move with a fixed step, so there the step
    # adapts each round (Aitken acceleration). That changes how fast the loop
    # gets there, not where it ends up.
    big = m.adaptive and g.n > 1500
    relax = min(m.dyn_relax, 0.20) if big else m.dyn_relax
    max_iter = max(m.dyn_max, 2500) if big else m.dyn_max
    prev_r, prev_fp = None, np.inf
    for it in range(1, max_iter + 1):
        V, A, maxh = backward(pp)   # choices, given where everyone will be
        br, maxk = forward(A)       # where everyone ends up, given those choices
        r = br - pp
        fp = float(np.max(np.abs(r)))
        if fp < m.dyn_tol:
            break
        if not m.adaptive:
            pp = (1 - m.dyn_relax) * pp + m.dyn_relax * br
        else:
            if prev_r is not None:
                dr = (r - prev_r).ravel()
                denom = float(dr @ dr)
                if denom > 1e-30:
                    # Aitken step, clipped hard to keep mass positive and the loop stable.
                    candidate = -relax * float(prev_r.ravel() @ dr) / denom
                    if np.isfinite(candidate):
                        relax = float(np.clip(candidate, 0.01, m.dyn_relax))
                if fp > 1.05 * prev_fp:
                    relax = max(0.01, 0.5 * relax)
            pp = pp + relax * r
            prev_r, prev_fp = r.copy(), fp
            if it % 100 == 0:
                print(f"  {name}: iter={it}, fp={fp:.3e}, relax={relax:.3f}", flush=True)
        pp[0] = p0
    else:
        raise RuntimeError(f"The {name} transition did not converge after {max_iter} rounds (fixed-point gap {fp:.3e}).")

    # Solve once more on the converged path and report the gap that actually remains.
    V, A, maxh = backward(pp)
    br, maxk = forward(A)
    fp = float(np.max(np.abs(br - pp)))
    return PathSol(name, V, br, A, fp, maxh, maxk)


def subgrid(g, start):
    """The same grid, with the clock restarted at time step ``start``."""
    return replace(g, t=g.t[start:] - g.t[start])


def time_index(g, t):
    """Index of the grid time closest to ``t``."""
    return int(np.argmin(np.abs(g.t - t)))


def surprise_path(m, g, no, post):
    """Follow the no-project path until opening, then restart from that population with the project in place."""
    i = time_index(g, m.opening)
    sg = subgrid(g, i)
    seg = solve_path("surprise_post", m, sg, no.mass[i], np.ones(len(sg.t), bool), post.value)
    return PathSol(
        "surprise",
        np.vstack([no.value[:i], seg.value]),
        np.vstack([no.mass[:i], seg.mass]),
        np.vstack([no.drift[:i], seg.drift]),
        max(no.fp, seg.fp), max(no.hjb, seg.hjb), max(no.kfe, seg.kfe),
    )


def revision_path(m, g, initial, post, revisions):
    """Apply each delay announcement in turn to a forward-looking path.

    At each news date the population carries over unchanged, and workers
    re-solve from there with the new expected opening. Returns the combined
    path and per-revision diagnostics.
    """
    mass, value = initial.mass.copy(), initial.value.copy()
    details = []
    for label, news, new_open in revisions:
        i = int(np.searchsorted(g.t, news - 1e-12, side="left"))
        i = min(i, len(g.t) - 1)
        sg = subgrid(g, i)
        seg = solve_path(label, m, sg, mass[i], sg.t >= (new_open - g.t[i]), post.value)
        before = mass[i].copy()
        mass[i:] = seg.mass
        value[i:] = seg.value
        details.append({
            "label": label,
            "news_grid": float(g.t[i]),
            "opening": new_open,
            "inheritance_error": float(np.max(np.abs(mass[i] - before))),
            "fp": float(seg.fp),
            "hjb": float(seg.hjb),
            "kfe": float(seg.kfe),
            "min_mass": float(seg.mass.min()),
        })
    return mass, details


def sequential_revisions(cal, m, g, known, post, local, key, note):
    """Layer the 2024 and 2025 delays onto the forward-looking path and check every segment.

    ``local`` maps a population to the local outcome; ``key`` names its gap in the output.
    """
    news = lambda d: grid_at_or_after(years(cal.announcement, d), m.dt)
    rm, details = revision_path(m, g, known, post, [
        ("2024_revision", news(cal.revision1_date), grid_at_or_after(cal.revision1_opening_lag, m.dt)),
        ("2025_revision", news(cal.revision2_date), grid_at_or_after(cal.revision2_opening_lag, m.dt)),
    ])
    i0 = time_index(g, m.opening)
    rev = {
        "events": details,
        "max_mass_error": float(np.max(np.abs(rm.sum(1) - 1))),
        "min_mass": float(rm.min()),
        key: 100 * (local(known.mass[i0]) - local(rm[i0])),
        "tv_at_initial_expected_opening": tv(known.mass[i0], rm[i0]),
        "note": note,
    }
    rev["all_passed"] = bool(
        rev["max_mass_error"] < 1e-10
        and rev["min_mass"] >= -1e-12
        and all(
            e["inheritance_error"] < 1e-12 and e["fp"] < 2e-6 and e["hjb"] < 1e-7
            and e["kfe"] < 1e-10 and e["min_mass"] >= -1e-12
            for e in details
        )
    )
    if not rev["all_passed"]:
        raise RuntimeError(f"Revision-path checks failed: {rev}")
    return rm, rev


# ---------------------------------------------------------------------------
# Outcome measures
# ---------------------------------------------------------------------------

def interval_weights(g, halfwidth):
    """Share of each cell that lies inside [-halfwidth, halfwidth]."""
    left = g.x - .5 * g.dx
    right = g.x + .5 * g.dx
    overlap = np.maximum(0.0, np.minimum(right, halfwidth) - np.maximum(left, -halfwidth))
    return overlap / g.dx


def metrics(p, g, m):
    """Mass in the central five-mile band, plus the mean and variance of location."""
    p = np.asarray(p, float)
    mean = float(p @ g.x)
    return {
        "band_mass": float(p @ interval_weights(g, m.band_h)),
        "mean_x": mean,
        "variance_x": float(p @ ((g.x - mean) ** 2)),
    }


def tv(a, b):
    """Total variation distance between two distributions."""
    return float(.5 * np.sum(np.abs(np.asarray(a) - np.asarray(b))))


def headline(m, g, pre, post, known, surprise):
    """The paper's headline numbers, measured at the original expected opening.

    The two ratios divide the forward-looking/surprise gap by the distance the
    surprise economy still has to travel to the long-run equilibrium. They are
    normalizations, not shares, so nothing keeps them inside [0, 1].
    """
    i = time_index(g, m.opening)
    kp, up = known.mass[i], surprise.mass[i]
    bu = metrics(up, g, m)["band_mass"]
    bk = metrics(kp, g, m)["band_mass"]
    bpost = metrics(post.mass, g, m)["band_mass"]
    den = bpost - bu
    ratio = (bk - bu) / den if abs(den) > 1e-12 else np.nan
    longtv = tv(post.mass, up)
    return {
        "opening_time": m.opening,
        "projected_band_gap_pp": 100 * (bk - bu),
        "opening_tv": tv(kp, up),
        "long_run_band_gap_pp": 100 * den,
        "band_anticipation_ratio": float(ratio),
        "long_run_tv_from_surprise_state": longtv,
        "tv_anticipation_ratio": tv(kp, up) / longtv if longtv > 1e-12 else np.nan,
        "ratio_warning": "Ratios are normalizations, not guaranteed fractions in [0,1].",
    }


# ---------------------------------------------------------------------------
# Assembling and solving the benchmark
# ---------------------------------------------------------------------------

def rebin_mass(p, n):
    """Move cell probabilities onto ``n`` cells by interpolating the CDF."""
    p = np.asarray(p, float)
    p = p / p.sum()
    old = np.linspace(-1, 1, len(p) + 1)
    new = np.linspace(-1, 1, n + 1)
    cdf = np.r_[0, np.cumsum(p)]
    out = np.diff(np.interp(new, old, cdf))
    out = np.maximum(out, 0)
    return out / out.sum()


def build_model(cal: Calibration, raw_mass, qcew, opening=None, nx=None, dt=None,
                smoothing=None, c=None, sigma2=None, rho=None, gamma=None, kappa_mult=1.0, omega_mult=1.0,
                shock_scale_miles=None, tail=None):
    """Turn the calibration and data into a solvable ``Model``.

    Any keyword argument overrides the matching calibration value; the
    robustness suite uses these to run sensitivity checks.
    """
    r1_news = years(cal.announcement, cal.revision1_date)
    r2_news = years(cal.announcement, cal.revision2_date)
    if not (0 < r1_news < r2_news):
        raise ValueError("The revision dates must come after the announcement, in order.")
    if not (cal.initial_opening_lag > 0 and cal.revision1_opening_lag > r1_news and cal.revision2_opening_lag > r2_news):
        raise ValueError("Each expected opening must come after the news date that announces it.")
    if not (cal.initial_opening_lag < cal.revision1_opening_lag < cal.revision2_opening_lag):
        raise ValueError("The delay experiment needs each revision to push the opening later.")

    nx = nx or cal.nx
    dt = dt or cal.dt
    smoothing = cal.smoothing_miles if smoothing is None else smoothing
    gam = cal.gamma if gamma is None else gamma
    cval = cal.c if c is None else c
    sig = cal.sigma2 if sigma2 is None else sigma2
    rh = cal.rho if rho is None else rho
    hmi = cal.project_scale_miles if shock_scale_miles is None else shock_scale_miles
    tailv = cal.tail_years if tail is None else tail
    if nx < 21 or nx % 2 == 0 or min(dt, smoothing, cval, sig, rh, gam, kappa_mult, hmi, tailv) <= 0 or omega_mult < 0:
        raise ValueError("nx must be odd and at least 21, every scale must be positive, and omega_mult must be nonnegative.")
    if qcew["employment"] <= 0 or qcew["pay"] <= 0:
        raise ValueError("QCEW employment and pay must both be positive.")

    # Smooth the LODES residence shares. Reflecting at the edges matches the
    # zero-flux boundary the inverse baseline assumes.
    raw = rebin_mass(raw_mass, nx)
    cell_miles = 2 * cal.half_length_miles / nx
    p = gaussian_filter1d(raw, smoothing / cell_miles, mode="reflect")
    p = np.maximum(p, 1e-15)
    p /= p.sum()

    # The inverse baseline can absorb any level of congestion through b(x), so
    # kappa is not identified; it is a normalization with its own sensitivity check.
    kappa = kappa_mult * cal.kappa

    # Premium size: Intel's direct jobs as a share of county employment, times
    # how far Intel's announced pay sits above the county average.
    local_share = INTEL_JOBS / qcew["employment"]
    pay_gap = max(INTEL_AVG_PAY - qcew["pay"], 0) / qcew["pay"]
    omega = local_share * pay_gap * omega_mult

    open_raw = cal.initial_opening_lag if opening is None else opening
    open_grid = grid_at_or_after(open_raw, dt)
    latest = max(cal.initial_opening_lag, cal.revision1_opening_lag, cal.revision2_opening_lag)
    horizon = grid_at_or_after(latest + tailv, dt)

    m0 = Model(
        nx, dt, horizon, open_grid, rh, cval, sig, kappa, gam, omega,
        hmi / cal.half_length_miles,
        cal.central_band_miles / cal.half_length_miles, tuple(np.ones(nx)),
        baseline_fit_tv_tol=cal.baseline_fit_tv_tol,
    )
    m, V0 = baseline_from_density(p, m0)
    return m, p, V0, {
        "local_job_share": local_share,
        "pay_gap_ratio": pay_gap,
        "omega": omega,
        "kappa": kappa,
        "smoothing_miles": smoothing,
        "kappa_multiplier": kappa_mult,
        "project_scale_miles": m.shock_h * cal.half_length_miles,
        "shock_information": "Jan-2022 public announcement: 3,000 direct jobs and $135,000 average salary; omega remains a reduced-form intensity, not a wage elasticity.",
    }


def validate_equilibrium(m, pre, post, known, surprise, diag):
    """Stop the run if any equilibrium check fails, and say which ones."""
    mass_err = max(
        float(np.max(np.abs(known.mass.sum(1) - 1))),
        float(np.max(np.abs(surprise.mass.sum(1) - 1))),
    )
    failures = []
    if pre.hjb >= 1e-7 or post.hjb >= 1e-7:
        failures.append("stationary HJB residual")
    if pre.kfe >= 1e-6 or post.kfe >= 1e-6:
        failures.append("stationary KFE residual")
    if known.fp >= 2e-6 or surprise.fp >= 2e-6 or diag.get("no_project_fp", 0) >= 2e-6:
        failures.append("dynamic fixed-point error")
    if (mass_err >= 1e-10
            or diag.get("no_project_mass_error", 0) >= 1e-10
            or known.mass.min() < -1e-12
            or surprise.mass.min() < -1e-12
            or diag.get("no_project_min_mass", 0) < -1e-12):
        failures.append("population accounting")
    tail = max(diag["no_project_terminal_tv_to_pre"], diag["known_terminal_tv_to_post"], diag["surprise_terminal_tv_to_post"])
    if diag.get("no_project_hjb", 0) >= 1e-7 or diag.get("no_project_kfe", 0) >= 1e-10:
        failures.append("no-project residual")
    if diag.get("baseline_target_vs_stationary_tv", np.inf) > m.baseline_fit_tv_tol:
        failures.append(f"baseline fit TV={diag['baseline_target_vs_stationary_tv']:.3e}")
    if diag.get("no_project_tv_at_opening", 0.0) > 1e-5:
        failures.append(f"no-project drift at opening={diag['no_project_tv_at_opening']:.3e}")
    if tail > m.terminal_tv_tol:
        failures.append(f"terminal continuation TV={tail:.3e}")
    if failures:
        raise RuntimeError("Equilibrium checks failed: " + ", ".join(failures))


def solve_all(cal, raw, qcew, **variant):
    """Solve the full benchmark: both stationary states and all three paths.

    The three paths are forward-looking, surprise (no project until opening,
    then the project arrives unannounced), and no-project. The no-project path
    should stay flat; if it drifts, the baseline is wrong.
    """
    m, p0, V0, diag = build_model(cal, raw, qcew, **variant)
    g = grid(m)
    pre = stationary_mfg(m, g, False, p0, V0)

    # The calibration uses the same discrete generator as the transition
    # solver, so the smoothed LODES target already is the pre-project
    # equilibrium, up to numerical tolerance.
    initial = pre.mass.copy()
    post = stationary_mfg(m, g, True, initial, pre.value)
    known = solve_path("known", m, g, initial, g.t >= m.opening, post.value)
    no = solve_path("no_project", m, g, initial, np.zeros(len(g.t), bool), pre.value)
    surprise = surprise_path(m, g, no, post)
    i = time_index(g, m.opening)

    diag.update({
        "baseline_target_vs_stationary_tv": tv(p0, pre.mass),
        "no_project_fp": no.fp,
        "no_project_hjb": no.hjb,
        "no_project_kfe": no.kfe,
        "no_project_mass_error": float(np.max(np.abs(no.mass.sum(1) - 1))),
        "no_project_min_mass": float(no.mass.min()),
        "no_project_tv_at_opening": tv(no.mass[i], pre.mass),
        "no_project_terminal_tv_to_pre": tv(no.mass[-1], pre.mass),
        "known_terminal_tv_to_post": tv(known.mass[-1], post.mass),
        "surprise_terminal_tv_to_post": tv(surprise.mass[-1], post.mass),
    })
    validate_equilibrium(m, pre, post, known, surprise, diag)
    return m, g, p0, pre, post, known, surprise, diag


def demo_inputs(cal):
    """Synthetic two-hump residence pattern and made-up QCEW values, for testing only."""
    x = -1 + (np.arange(cal.nx) + .5) * (2 / cal.nx)
    p = np.exp(-.5 * ((x - .35) / .24) ** 2) + .38 * np.exp(-.5 * ((x + .42) / .32) ** 2)
    p /= p.sum()
    return p, {"employment": 66325.0, "pay": 46448.0}


def live_inputs(cal, refresh=False):
    """Download and prepare the real LODES and QCEW inputs."""
    raw, workers, geo5 = lodes_mass(cal, refresh=refresh)
    q = qcew_summary(cal, refresh)
    return raw, q, {"lodes_workers": workers, "raw_geographic_5mi_share": geo5}


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def write_results(out, cal, m, g, p0, pre, post, known, surprise, diag, meta, revision):
    """Write every output file, after a final round of validation checks.

    Files: calibration, headline metrics, and validation JSON; path and
    baseline CSVs; Figure 2; and a .tex file of macros the manuscript reads
    its numbers from.
    """
    out.mkdir(parents=True, exist_ok=True)
    head = headline(m, g, pre, post, known, surprise)

    val = {
        "version": VERSION,
        "baseline_hjb": pre.hjb,
        "baseline_kfe": pre.kfe,
        "post_hjb": post.hjb,
        "post_kfe": post.kfe,
        "known_fp": known.fp,
        "surprise_fp": surprise.fp,
        "known_mass_error": float(np.max(np.abs(known.mass.sum(1) - 1))),
        "surprise_mass_error": float(np.max(np.abs(surprise.mass.sum(1) - 1))),
        "baseline_target_vs_stationary_tv": float(diag["baseline_target_vs_stationary_tv"]),
        "no_project_fp": float(diag["no_project_fp"]),
        "no_project_hjb": float(diag["no_project_hjb"]),
        "no_project_kfe": float(diag["no_project_kfe"]),
        "no_project_mass_error": float(diag["no_project_mass_error"]),
        "baseline_fit_warning": bool(diag["baseline_target_vs_stationary_tv"] > m.baseline_fit_tv_tol),
        "no_project_tv_at_opening": float(diag["no_project_tv_at_opening"]),
        "no_project_terminal_tv_to_pre": float(diag["no_project_terminal_tv_to_pre"]),
        "known_terminal_tv_to_post": float(diag["known_terminal_tv_to_post"]),
        "surprise_terminal_tv_to_post": float(diag["surprise_terminal_tv_to_post"]),
        "all_passed": True,  # solve_all's validate_equilibrium already stopped on any failure
    }

    # How far grid rounding moved each date, in days.
    r1, r2 = years(cal.announcement, cal.revision1_date), years(cal.announcement, cal.revision2_date)
    info = {
        "calibration": asdict(cal),
        "model": asdict(m),
        "diagnostics": diag,
        "data": meta,
        "information_sources": INFO_SOURCES,
        "timing_rounding_days": {
            "initial_opening_assumption": 365.2425 * (m.opening - cal.initial_opening_lag),
            "rev1_news": 365.2425 * (grid_at_or_after(r1, m.dt) - r1),
            "rev2_news": 365.2425 * (grid_at_or_after(r2, m.dt) - r2),
        },
    }
    (out / "Benchmark_1D_Calibration.json").write_text(json.dumps(info, indent=2))
    (out / "Benchmark_1D_Headline_Metrics.json").write_text(json.dumps(head, indent=2))
    (out / "Benchmark_1D_Validation.json").write_text(json.dumps(val, indent=2))
    (out / "Benchmark_1D_Revision_Validation.json").write_text(json.dumps(revision, indent=2))

    rows = []
    for name, sol in (("forward_looking", known), ("surprise_opening", surprise)):
        for i, t in enumerate(g.t):
            rows.append({"scenario": name, "time": t, **metrics(sol.mass[i], g, m)})
    pd.DataFrame(rows).to_csv(out / "Benchmark_1D_Paths.csv", index=False)
    pd.DataFrame({
        "x": g.x,
        "smoothed_target_mass": p0,
        "discrete_pre_stationary_mass": pre.mass,
        "baseline_b": m.b,
        "project_premium": premium(g.x, m),
    }).to_csv(out / "Benchmark_1D_Baseline.csv", index=False)

    # Figure 2: mass in the central band over time, for both economies.
    plt.figure(figsize=(8, 5))
    for name, z in pd.DataFrame(rows).groupby("scenario"):
        plt.plot(z.time, z.band_mass, label=name)
    plt.axvline(m.opening, ls="--", lw=1)
    plt.xlabel("Years since Jan. 2022")
    plt.ylabel("Mass in central projected band")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "Figure_2_Anticipatory_Adjustment_1D.png", dpi=220)
    plt.close()

    tex = ("% generated by Before the Factory Opens benchmark 1D model (r11)\n" +
           f"\\newcommand{{\\OpeningTV}}{{{head['opening_tv']:.4f}}}\n" +
           f"\\newcommand{{\\OpeningTVPct}}{{{100*head['opening_tv']:.2f}}}\n" +
           f"\\newcommand{{\\TVAnticipationRatio}}{{{head['tv_anticipation_ratio']:.3f}}}\n" +
           f"\\newcommand{{\\TVAnticipationRatioPct}}{{{100*head['tv_anticipation_ratio']:.1f}}}\n" +
           f"\\newcommand{{\\BandAnticipationRatioPct}}{{{100*head['band_anticipation_ratio']:.1f}}}\n" +
           f"\\newcommand{{\\ProjectedBandGapPP}}{{{head['projected_band_gap_pp']:.3f}}}\n" +
           f"\\newcommand{{\\NoProjectOpeningTVPct}}{{{100*diag['no_project_tv_at_opening']:.2f}}}\n" +
           f"\\newcommand{{\\ProjectScaleMiles}}{{{m.shock_h*cal.half_length_miles:.1f}}}\n" +
           f"\\newcommand{{\\CongestionGamma}}{{{m.gamma:.2f}}}\n" +
           f"\\newcommand{{\\SequentialGapAtInitialExpectedOpeningPP}}{{{revision.get('gap_at_initial_expected_opening_pp',float('nan')):.3f}}}\n" +
           f"\\newcommand{{\\SequentialTVAtInitialExpectedOpeningPct}}{{{100*revision.get('tv_at_initial_expected_opening',float('nan')):.2f}}}\n")
    (out / "Benchmark_1D_Manuscript_Results.tex").write_text(tex)
    return head, val


def run(cal, raw, q, meta, out, note):
    """Solve the benchmark and the sequential revisions, write every output, and print the headline."""
    m, g, p0, pre, post, known, surprise, diag = solve_all(cal, raw, q)
    rm, rev = sequential_revisions(cal, m, g, known, post, lambda p: metrics(p, g, m)["band_mass"],
                                   "gap_at_initial_expected_opening_pp", note)
    head, val = write_results(out, cal, m, g, p0, pre, post, known, surprise, diag, meta, rev)
    pd.DataFrame(
        [{"time": t, **metrics(rm[i], g, m)} for i, t in enumerate(g.t)]
    ).to_csv(out / "Benchmark_1D_Sequential_Revision_Path.csv", index=False)
    print(json.dumps({"version": VERSION, "headline": head, "validation": val}, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Run the one-dimensional benchmark for Before the Factory Opens.")
    ap.add_argument("--output-dir", default="Before_the_Factory_Opens_Benchmark_1D_Output",
                    help="Where to write results. An existing folder is replaced.")
    ap.add_argument("--demo", action="store_true",
                    help="Use synthetic inputs on a coarse grid. For testing code only.")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-download the Census and BLS files instead of using the cache.")
    args = ap.parse_args()

    cal = replace(Calibration(), nx=31, dt=.20, tail_years=4.0) if args.demo else Calibration()
    out = Path(args.output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if args.demo:
        raw, q = demo_inputs(cal)
        meta = {"data_mode": "synthetic_demo_FOR_CODE_VALIDATION_ONLY"}
    else:
        raw, q, meta = live_inputs(cal, args.refresh)
        meta["data_mode"] = "live_public_data"
    meta.update({"raw_mass": [float(v) for v in raw], "qcew": q})
    run(cal, raw, q, meta, out, "Information dates are representative public revision dates. Expected opening "
        "times are central timing assumptions, not claims of a uniquely correct operating day.")


if __name__ == "__main__":
    main()
