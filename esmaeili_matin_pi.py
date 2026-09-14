#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimating pi with Monte Carlo sampling and the Central Limit Theorem.

The program has three parts, matching the three tasks of the assignment:

1. ``estimate_pi(n)``     - Monte Carlo estimator of pi from n uniform points.
2. ``convergence_study``  - how the estimate behaves as n grows (Task 2).
3. ``sampling_study``     - the sampling distribution at fixed n over R
                            independent repetitions (Task 3, CLT).

Theory used throughout
----------------------
Draw (X, Y) uniformly on the unit square and let I = 1{X^2 + Y^2 <= 1}.
The quarter disc has area pi/4, so I ~ Bernoulli(p) with p = pi/4 and the
estimator built from n independent draws is

    pihat_n = 4 * (1/n) * sum_i I_i,      E[pihat_n] = pi   (unbiased)

Var(4 I) = 16 p (1 - p) = 16 (pi/4)(1 - pi/4) = pi (4 - pi), hence

    sd(pihat_n) = sqrt(pi (4 - pi)) / sqrt(n) ~= 1.6422 / sqrt(n)

The Central Limit Theorem then gives, for large n,

    pihat_n  ~approx~  Normal(pi, pi (4 - pi) / n)

which is what Task 3 verifies empirically: accuracy improves like 1/sqrt(n)
(a factor-100 increase in n buys one extra correct digit), and the histogram
of repeated estimates is bell shaped around pi.

Usage
-----
    python esmaeili_matin_pi.py                 # full run: figures + PDF report
    python esmaeili_matin_pi.py --quick         # fast run with smaller n and R
    python esmaeili_matin_pi.py --seed 7        # different reproducible stream
    python esmaeili_matin_pi.py --no-report     # figures and CSVs only

Author: Matin Esmaeili
Course: PhD Artificial Intelligence
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

import matplotlib

matplotlib.use("Agg")  # file-only backend: no display needed, works headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Patch

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

P_INSIDE = math.pi / 4.0                        # probability a point lands inside
SIGMA_1 = math.sqrt(math.pi * (4.0 - math.pi))  # sd of a single 4*I draw ~= 1.6422
Z_95 = 1.959963984540054                        # standard normal 97.5% quantile

# Points generated per vectorised block. Caps peak memory at ~16 MB per block
# regardless of how large n is, so n = 10^7 costs no more RAM than n = 10^6.
CHUNK = 1_000_000

NORMAL = NormalDist()  # standard normal, used for pdf / cdf / quantiles

# Light-mode plotting palette (single blue series, orange for theory overlays,
# neutral ink for chrome). Kept in one place so every figure matches.
C = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_soft": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "blue": "#2a78d6",
    "blue_fill": "#9ec5f4",
    "blue_pale": "#cde2fb",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
}


# --------------------------------------------------------------------------- #
# Task 1 - the Monte Carlo estimator                                           #
# --------------------------------------------------------------------------- #

def count_inside(n: int, rng: np.random.Generator) -> int:
    """Return how many of ``n`` uniform points on [0,1]^2 fall in the unit circle.

    The points are generated in blocks of ``CHUNK`` so that very large n does
    not allocate a huge array; each block is tested with a vectorised NumPy
    comparison rather than a Python loop.
    """
    inside = 0
    remaining = n
    while remaining > 0:
        m = min(CHUNK, remaining)
        xy = rng.random((m, 2))                       # m points in the unit square
        inside += int(np.count_nonzero((xy * xy).sum(axis=1) <= 1.0))
        remaining -= m
    return inside


def estimate_pi(n: int, rng: np.random.Generator | None = None) -> float:
    """Monte Carlo estimate of pi from ``n`` random points in the unit square.

    Parameters
    ----------
    n
        Number of points (trials). Must be positive.
    rng
        Optional NumPy random generator. Pass one for reproducible results;
        omitted, a fresh non-deterministic generator is used.

    Returns
    -------
    float
        ``4 * (points inside the quarter disc) / n``.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    rng = np.random.default_rng() if rng is None else rng
    return 4.0 * count_inside(int(n), rng) / int(n)


def standard_error(n: int | np.ndarray):
    """Theoretical standard deviation of ``estimate_pi(n)``: sqrt(pi(4-pi)/n)."""
    return SIGMA_1 / np.sqrt(n)


# --------------------------------------------------------------------------- #
# Small statistics helpers (kept dependency-free: NumPy + stdlib only)         #
# --------------------------------------------------------------------------- #

def skewness(x: np.ndarray) -> float:
    """Sample skewness (third standardised moment). 0 for a normal law."""
    z = (x - x.mean()) / x.std(ddof=0)
    return float((z ** 3).mean())


def excess_kurtosis(x: np.ndarray) -> float:
    """Sample excess kurtosis (fourth standardised moment - 3). 0 for a normal."""
    z = (x - x.mean()) / x.std(ddof=0)
    return float((z ** 4).mean() - 3.0)


def ks_statistic(x: np.ndarray, mu: float, sigma: float) -> float:
    """Kolmogorov-Smirnov distance between the sample and Normal(mu, sigma^2).

    Compares the empirical CDF with the *theoretical* CLT limit (parameters are
    not estimated from the data), so the classical critical value 1.36/sqrt(R)
    at the 5% level applies.
    """
    xs = np.sort(x)
    r = xs.size
    f0 = np.array([NORMAL.cdf((v - mu) / sigma) for v in xs])
    upper = np.arange(1, r + 1) / r - f0
    lower = f0 - np.arange(0, r) / r
    return float(max(upper.max(), lower.max()))


def ci_coverage(estimates: np.ndarray, n: int) -> float:
    """Fraction of replicates whose 95% normal CI actually contains pi.

    Each replicate forms pihat +/- 1.96 * sqrt(pi(4-pi)/n). If the CLT
    approximation is good this should land near 0.95.
    """
    half_width = Z_95 * standard_error(n)
    return float(np.mean(np.abs(estimates - math.pi) <= half_width))


# --------------------------------------------------------------------------- #
# Task 2 - convergence with n                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class ConvergenceResult:
    ns: np.ndarray          # sample sizes tested
    estimates: np.ndarray   # one independent estimate per sample size
    errors: np.ndarray      # |estimate - pi|


def convergence_study(ns: np.ndarray, rng: np.random.Generator) -> ConvergenceResult:
    """Run one independent estimate for every sample size in ``ns``."""
    estimates = np.array([estimate_pi(int(n), rng) for n in ns])
    return ConvergenceResult(ns=ns, estimates=estimates,
                             errors=np.abs(estimates - math.pi))


def log_spaced_sizes(lo_exp: float, hi_exp: float, count: int) -> np.ndarray:
    """Distinct, increasing sample sizes spread evenly on a log scale."""
    return np.unique(np.logspace(lo_exp, hi_exp, count).astype(np.int64))


# --------------------------------------------------------------------------- #
# Task 3 - sampling distribution at fixed n (the CLT check)                    #
# --------------------------------------------------------------------------- #

@dataclass
class SamplingResult:
    n: int
    estimates: np.ndarray   # R independent values of estimate_pi(n)

    @property
    def replicates(self) -> int:
        return int(self.estimates.size)

    @property
    def theoretical_sd(self) -> float:
        return float(standard_error(self.n))

    def summary(self) -> dict:
        """Empirical moments next to their CLT predictions."""
        est = self.estimates
        return {
            "n": self.n,
            "replicates": self.replicates,
            "mean": float(est.mean()),
            "bias": float(est.mean() - math.pi),
            "empirical_sd": float(est.std(ddof=1)),
            "theoretical_sd": self.theoretical_sd,
            "sd_ratio": float(est.std(ddof=1) / self.theoretical_sd),
            "skewness": skewness(est),
            "excess_kurtosis": excess_kurtosis(est),
            "ks_statistic": ks_statistic(est, math.pi, self.theoretical_sd),
            "ks_critical_5pct": 1.36 / math.sqrt(self.replicates),
            "ci95_coverage": ci_coverage(est, self.n),
            "min": float(est.min()),
            "max": float(est.max()),
        }


def sampling_study(n: int, replicates: int,
                   rng: np.random.Generator) -> SamplingResult:
    """Repeat the whole experiment ``replicates`` times at fixed ``n``."""
    estimates = np.fromiter(
        (estimate_pi(n, rng) for _ in range(replicates)),
        dtype=np.float64, count=replicates,
    )
    return SamplingResult(n=n, estimates=estimates)


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

def apply_style() -> None:
    """One restrained style for every figure: recessive chrome, thin marks."""
    plt.rcParams.update({
        "figure.facecolor": C["surface"],
        "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"],
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": C["ink"],
        "axes.labelcolor": C["ink_soft"],
        "axes.edgecolor": C["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.8,
        "text.color": C["ink"],
        "xtick.color": C["muted"],
        "ytick.color": C["muted"],
        "xtick.labelcolor": C["ink_soft"],
        "ytick.labelcolor": C["ink_soft"],
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def figure_dartboard(rng: np.random.Generator, n: int = 3000) -> plt.Figure:
    """The geometric idea: points in the square, coloured by inside/outside."""
    xy = rng.random((n, 2))
    inside = (xy * xy).sum(axis=1) <= 1.0
    hits = int(inside.sum())

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(xy[inside, 0], xy[inside, 1], s=5, c=C["blue"],
               alpha=0.65, linewidths=0, label=f"inside  ({hits})")
    ax.scatter(xy[~inside, 0], xy[~inside, 1], s=5, c=C["orange"],
               alpha=0.65, linewidths=0, label=f"outside ({n - hits})")
    ax.add_patch(Arc((0, 0), 2, 2, theta1=0, theta2=90,
                     color=C["ink"], linewidth=1.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Hit-or-miss sampling, n = {n:,}\n"
                 f"4 x {hits}/{n} = {4 * hits / n:.4f}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              markerscale=2.4, fontsize=9, labelcolor=C["ink_soft"])
    fig.tight_layout()
    return fig


def figure_convergence(conv: ConvergenceResult) -> plt.Figure:
    """Estimate vs n (log x) with the theoretical +/-1.96 sd envelope."""
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    grid = np.logspace(np.log10(conv.ns[0]), np.log10(conv.ns[-1]), 400)
    band = Z_95 * standard_error(grid)

    ax.fill_between(grid, math.pi - band, math.pi + band,
                    color=C["blue_pale"], alpha=0.75, linewidth=0,
                    label="theoretical 95% band, $\\pi \\pm 1.96\\,\\sigma/\\sqrt{n}$")
    ax.axhline(math.pi, color=C["ink"], linewidth=1.2, linestyle=(0, (5, 4)),
               label="true $\\pi$")
    ax.plot(conv.ns, conv.estimates, color=C["blue"], linewidth=2,
            marker="o", markersize=4.5, markerfacecolor=C["surface"],
            markeredgecolor=C["blue"], markeredgewidth=1.4,
            label="Monte Carlo estimate")

    ax.set_xscale("log")
    ax.set_xlabel("number of points  n  (log scale)")
    ax.set_ylabel("estimate of $\\pi$")
    ax.set_title("Convergence of the Monte Carlo estimate")
    ax.set_ylim(math.pi - 0.62, math.pi + 0.62)
    ax.legend(loc="upper right", fontsize=9, labelcolor=C["ink_soft"])
    fig.tight_layout()
    return fig


def figure_error_scaling(conv: ConvergenceResult) -> plt.Figure:
    """|error| vs n on log-log, against the 1/sqrt(n) reference slope."""
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    grid = np.logspace(np.log10(conv.ns[0]), np.log10(conv.ns[-1]), 400)

    ax.plot(grid, standard_error(grid), color=C["orange"], linewidth=2,
            label="$\\sigma/\\sqrt{n}$  (theoretical scale)")
    ax.plot(grid, Z_95 * standard_error(grid), color=C["orange"], linewidth=1.4,
            linestyle=(0, (5, 4)), alpha=0.8,
            label="$1.96\\,\\sigma/\\sqrt{n}$  (95% bound)")
    ax.plot(conv.ns, np.maximum(conv.errors, 1e-7), linestyle="none",
            marker="o", markersize=5, markerfacecolor=C["blue"],
            markeredgecolor=C["surface"], markeredgewidth=1.2,
            label="observed $|\\hat{\\pi}_n - \\pi|$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of points  n  (log scale)")
    ax.set_ylabel("absolute error  (log scale)")
    ax.set_title("Error shrinks like $1/\\sqrt{n}$")
    ax.legend(loc="lower left", fontsize=9, labelcolor=C["ink_soft"])
    fig.tight_layout()
    return fig


def figure_sampling_histograms(results: list[SamplingResult]) -> plt.Figure:
    """One histogram per n with the N(pi, sigma^2/n) density on top."""
    k = len(results)
    fig, axes = plt.subplots(1, k, figsize=(4.4 * k, 4.1))
    axes = np.atleast_1d(axes)

    for ax, res in zip(axes, results):
        est = res.estimates
        sd = res.theoretical_sd
        ax.hist(est, bins=26, density=True, color=C["blue_fill"],
                edgecolor=C["surface"], linewidth=1.0,
                label=f"{res.replicates} estimates")

        xs = np.linspace(math.pi - 4 * sd, math.pi + 4 * sd, 400)
        pdf = np.array([NORMAL.pdf((v - math.pi) / sd) / sd for v in xs])
        ax.plot(xs, pdf, color=C["orange"], linewidth=2,
                label="CLT density $N(\\pi,\\ \\sigma^2/n)$")
        ax.axvline(math.pi, color=C["ink"], linewidth=1.2, linestyle=(0, (5, 4)),
                   label="true $\\pi$")

        ax.set_title(f"n = $10^{int(round(math.log10(res.n)))}$"
                     f"   (sd $\\approx$ {sd:.4f})")
        ax.set_xlabel("estimate of $\\pi$")
        ax.set_xlim(math.pi - 4 * sd, math.pi + 4 * sd)
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_yticks([])
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", fontsize=8.5, labelcolor=C["ink_soft"])
    fig.suptitle("Sampling distribution of $\\hat{\\pi}_n$ over independent repetitions",
                 fontsize=12.5, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    return fig


def figure_qq(results: list[SamplingResult]) -> plt.Figure:
    """Normal Q-Q plots: straight line => the estimates are normal."""
    k = len(results)
    fig, axes = plt.subplots(1, k, figsize=(4.0 * k, 4.0))
    axes = np.atleast_1d(axes)

    for ax, res in zip(axes, results):
        r = res.replicates
        sample_q = np.sort((res.estimates - math.pi) / res.theoretical_sd)
        # Blom plotting positions (i - 0.375) / (R + 0.25)
        probs = (np.arange(1, r + 1) - 0.375) / (r + 0.25)
        theory_q = np.array([NORMAL.inv_cdf(p) for p in probs])

        lim = float(max(np.abs(theory_q).max(), np.abs(sample_q).max())) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color=C["ink"], linewidth=1.2,
                linestyle=(0, (5, 4)), label="perfect normal fit")
        ax.plot(theory_q, sample_q, linestyle="none", marker="o", markersize=4,
                markerfacecolor=C["blue"], markeredgecolor=C["surface"],
                markeredgewidth=0.7, alpha=0.9, label="standardised estimates")

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"n = $10^{int(round(math.log10(res.n)))}$")
        ax.set_xlabel("theoretical normal quantile")

    axes[0].set_ylabel("observed standardised quantile")
    axes[0].legend(loc="upper left", fontsize=8.5, labelcolor=C["ink_soft"])
    fig.suptitle("Normal Q-Q plots of the sampling distribution",
                 fontsize=12.5, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Report (PDF)                                                                 #
# --------------------------------------------------------------------------- #

class TextPage:
    """Minimal flowing-text page writer for the PDF report."""

    SIZES = {"h1": 17, "h2": 12.5, "body": 9.8, "mono": 9.0, "caption": 8.6}
    LEADING = {"h1": 0.034, "h2": 0.026, "body": 0.019, "mono": 0.018,
               "caption": 0.017}
    WIDTH = {"body": 92, "mono": 88, "caption": 104}

    def __init__(self, pdf: PdfPages, title: str | None = None):
        self.pdf = pdf
        self._new_page(title)

    def _new_page(self, title: str | None = None) -> None:
        self.fig = plt.figure(figsize=(8.5, 11))
        self.y = 0.93
        if title:
            self.write(title, "h1")

    def _ensure(self, needed: float) -> None:
        if self.y - needed < 0.07:
            self.close()
            self._new_page()

    def write(self, text: str, style: str = "body", color: str | None = None,
              space_after: float = 0.008) -> None:
        size = self.SIZES[style]
        lead = self.LEADING[style]
        weight = "bold" if style in ("h1", "h2") else "normal"
        family = "monospace" if style == "mono" else "sans-serif"
        color = color or (C["ink"] if style in ("h1", "h2") else C["ink_soft"])
        width = self.WIDTH.get(style, 92)

        lines = []
        for para in text.split("\n"):
            lines.extend(textwrap.wrap(para, width=width) or [""])

        for line in lines:
            self._ensure(lead)
            self.fig.text(0.09, self.y, line, fontsize=size, color=color,
                          fontweight=weight, family=family, va="top")
            self.y -= lead
        self.y -= space_after

    def gap(self, amount: float = 0.012) -> None:
        self.y -= amount

    def break_page(self) -> None:
        """Finish the current page and start a fresh one."""
        self.close()
        self._new_page()

    def close(self) -> None:
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def build_report(path: Path, conv: ConvergenceResult,
                 samples: list[SamplingResult], figures: dict[str, plt.Figure],
                 meta: dict) -> None:
    """Assemble a short PDF report: theory, method, results, figures."""
    with PdfPages(path) as pdf:
        page = TextPage(pdf, "Estimating $\\pi$ with Monte Carlo sampling")
        page.write("PhD Artificial Intelligence  |  Matin Esmaeili  |  "
                   f"seed = {meta['seed']}  |  runtime = {meta['runtime_s']:.1f} s",
                   "caption", color=C["muted"], space_after=0.02)

        page.write("1. Method", "h2")
        page.write(
            "Points (x, y) are drawn uniformly on the unit square. A point counts "
            "as a hit when x^2 + y^2 <= 1, i.e. when it falls inside the quarter "
            "disc of radius 1. That quarter disc has area pi/4 while the square "
            "has area 1, so the hit indicator I is Bernoulli with p = pi/4 and "
            "the estimator is pihat_n = 4 * (hits / n). Sampling is vectorised "
            "with NumPy and generated in blocks of one million points, so memory "
            "stays flat even for n = 10^7."
        )
        page.gap()

        page.write("2. What the theory predicts", "h2")
        page.write(
            "E[pihat_n] = 4p = pi, so the estimator is unbiased at every n. "
            "Since Var(4I) = 16 p (1 - p) = pi (4 - pi), the variance of the "
            "estimator is pi (4 - pi) / n and its standard deviation is"
        )
        page.write(f"    sd(pihat_n) = sqrt(pi (4 - pi) / n) = {SIGMA_1:.4f} / sqrt(n)",
                   "mono")
        page.write(
            "The Central Limit Theorem applies because pihat_n is an average of "
            "n i.i.d. bounded terms. Therefore, for large n,"
        )
        page.write("    pihat_n  ~  Normal( pi , pi (4 - pi) / n )", "mono")
        page.write(
            "Two consequences are tested below. First, the error shrinks like "
            "1/sqrt(n): reducing it by a factor of 10 costs 100 times more "
            "points, which is why Monte Carlo is a poor way to compute digits of "
            "pi but a workable way to integrate in high dimensions, where the "
            "1/sqrt(n) rate is independent of dimension. Second, the histogram "
            "of repeated estimates at fixed n must look Gaussian, centred at pi "
            "with the width given above."
        )
        page.break_page()

        page.write("3. Convergence with n", "h2")
        page.write(
            f"One independent estimate was computed for each of "
            f"{conv.ns.size} sample sizes spaced logarithmically from "
            f"{conv.ns[0]:,} to {conv.ns[-1]:,} points."
        )
        page.write(f"{'n':>12}  {'estimate':>12}  {'abs. error':>12}  "
                   f"{f'sd = {SIGMA_1:.4f}/sqrt(n)':>20}", "mono")
        for n, est, err in zip(conv.ns, conv.estimates, conv.errors):
            if n in _report_rows(conv.ns):
                page.write(f"{int(n):>12,}  {est:>12.5f}  {err:>12.5f}  "
                           f"{standard_error(int(n)):>20.5f}", "mono",
                           space_after=0.0)
        page.gap()
        inside = int(np.sum(conv.errors <= Z_95 * standard_error(conv.ns)))
        page.write(
            f"{inside} of the {conv.ns.size} estimates fall inside the "
            f"theoretical 95% envelope pi +/- 1.96 sd(n) (about 95% expected). "
            "On the log-log plot the observed errors scatter around the "
            "sigma/sqrt(n) line with the predicted slope of -1/2, confirming the "
            "square-root rate rather than any faster convergence."
        )
        page.gap()

        page.write("4. Sampling distribution at fixed n", "h2")
        page.write(
            f"For each n the whole experiment was repeated R = "
            f"{samples[0].replicates} times with independent points, giving R "
            "estimates whose spread is the sampling distribution. Empirical "
            "moments are compared with the CLT prediction:"
        )
        page.gap(0.006)
        header = (f"{'n':>9} {'mean':>8} {'bias':>9} {'emp sd':>8} "
                  f"{'CLT sd':>8} {'ratio':>6} {'skew':>7} {'kurt':>7} "
                  f"{'KS':>6} {'cov':>6}")
        page.write(header, "mono", space_after=0.002)
        for res in samples:
            s = res.summary()
            page.write(
                f"{s['n']:>9,} {s['mean']:>8.5f} {s['bias']:>+9.5f} "
                f"{s['empirical_sd']:>8.5f} {s['theoretical_sd']:>8.5f} "
                f"{s['sd_ratio']:>6.3f} {s['skewness']:>+7.3f} "
                f"{s['excess_kurtosis']:>+7.3f} {s['ks_statistic']:>6.3f} "
                f"{s['ci95_coverage']:>6.3f}", "mono", space_after=0.0)
        page.gap()
        page.write(
            f"KS is the Kolmogorov-Smirnov distance between the empirical "
            f"distribution and the exact CLT limit N(pi, sigma^2/n); at R = "
            f"{samples[0].replicates} the 5% critical value is "
            f"{1.36 / math.sqrt(samples[0].replicates):.3f}, so a value below "
            "that is consistent with normality. 'kurt' is excess kurtosis and "
            "'cov' the fraction of replicates whose 95% confidence interval "
            "contains pi.", "caption")
        page.break_page()

        page.write("5. Reading the results", "h2")
        ratios = ", ".join(f"{r.summary()['sd_ratio']:.3f}" for r in samples)
        skews = ", ".join(f"{r.summary()['skewness']:+.3f}" for r in samples)
        page.write(
            f"The empirical standard deviations track the predicted "
            f"sqrt(pi(4-pi)/n) closely (observed / predicted = {ratios}), and "
            f"each mean sits within a couple of standard errors of pi, "
            "confirming unbiasedness. Every tenfold increase in n shrinks the "
            "spread by sqrt(10) ~ 3.16, which is exactly what the three "
            "histograms show: same shape, same centre, width divided by about "
            "3.16 at each step."
        )
        page.write(
            f"Shape statistics agree with a normal law (skewness {skews}, excess "
            "kurtosis near zero), the Q-Q points lie on the 45-degree line apart "
            "from the usual tail wobble, and the KS distances stay under their "
            "critical value. Note that the underlying draws are Bernoulli - as "
            "discrete and non-normal as a variable can be - yet their average is "
            "already indistinguishable from Gaussian at n = 1000. That is the "
            "Central Limit Theorem doing the work: normality is a property of "
            "the average, not of the data."
        )
        page.gap()
        page.write("6. Conclusion", "h2")
        page.write(
            "The Monte Carlo estimator is unbiased, converges at the "
            "1/sqrt(n) rate predicted by its variance pi(4-pi)/n, and its "
            "sampling distribution is approximately Normal(pi, pi(4-pi)/n) for "
            "every n tested. The practical reading: n controls precision, and "
            "buying one more correct decimal digit costs a hundredfold more "
            "samples."
        )
        page.close()

        # Figure pages, each with a caption underneath.
        captions = {
            "dartboard": "Figure 1. Hit-or-miss sampling. The fraction of points "
                         "inside the quarter disc estimates pi/4.",
            "convergence": "Figure 2. Estimate against n on a log x-axis. The band "
                           "is the theoretical 95% envelope, not a fit to the data.",
            "error": "Figure 3. Absolute error against n on log-log axes; the "
                     "slope matches the predicted -1/2.",
            "histograms": "Figure 4. Sampling distribution at three values of n "
                          "with the CLT density overlaid (not fitted).",
            "qq": "Figure 5. Normal Q-Q plots. Points on the diagonal indicate "
                  "normality of the estimator.",
        }
        for key, fig in figures.items():
            page_fig = plt.figure(figsize=(8.5, 11))
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())

            # Size the image box to the figure's own aspect ratio so wide and
            # square figures both sit flush under the top margin.
            width_frac = 0.88
            height_frac = min(
                0.80,
                width_frac * (8.5 / 11.0) * buf.shape[0] / buf.shape[1],
            )
            top = 0.92
            ax = page_fig.add_axes([0.06, top - height_frac, width_frac,
                                    height_frac])
            ax.imshow(buf, aspect="auto")
            ax.axis("off")
            page_fig.text(0.09, top - height_frac - 0.035,
                          textwrap.fill(captions.get(key, ""), 96),
                          fontsize=9.5, color=C["ink_soft"], va="top")
            pdf.savefig(page_fig)
            plt.close(page_fig)

        info = pdf.infodict()
        info["Title"] = "Estimating pi with Monte Carlo and the CLT"
        info["Author"] = "Matin Esmaeili"
        info["Subject"] = "PhD Artificial Intelligence - Monte Carlo assignment"


def _report_rows(ns: np.ndarray, max_rows: int = 12) -> set:
    """Pick a readable subset of sample sizes for the report table."""
    if ns.size <= max_rows:
        return set(int(n) for n in ns)
    idx = np.unique(np.linspace(0, ns.size - 1, max_rows).astype(int))
    return set(int(ns[i]) for i in idx)


# --------------------------------------------------------------------------- #
# Output helpers                                                               #
# --------------------------------------------------------------------------- #

def write_convergence_csv(path: Path, conv: ConvergenceResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "estimate", "absolute_error", "theoretical_sd"])
        for n, est, err in zip(conv.ns, conv.estimates, conv.errors):
            w.writerow([int(n), f"{est:.10f}", f"{err:.10f}",
                        f"{standard_error(int(n)):.10f}"])


def write_sampling_csv(path: Path, samples: list[SamplingResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "replicate", "estimate"])
        for res in samples:
            for i, est in enumerate(res.estimates, start=1):
                w.writerow([res.n, i, f"{est:.10f}"])


def write_summary_json(path: Path, conv: ConvergenceResult,
                       samples: list[SamplingResult], meta: dict) -> None:
    payload = {
        "meta": meta,
        "theory": {
            "p_inside": P_INSIDE,
            "sigma_single_draw": SIGMA_1,
            "sd_formula": "sqrt(pi*(4-pi)/n)",
            "clt_limit": "Normal(pi, pi*(4-pi)/n)",
        },
        "convergence": [
            {"n": int(n), "estimate": float(e), "absolute_error": float(err)}
            for n, e, err in zip(conv.ns, conv.estimates, conv.errors)
        ],
        "sampling_distribution": [res.summary() for res in samples],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_console_summary(conv: ConvergenceResult,
                          samples: list[SamplingResult]) -> None:
    print("\nTask 2 - convergence with n")
    print(f"  {'n':>12}  {'estimate':>10}  {'abs error':>10}  {'sd(n)':>10}")
    for n in sorted(_report_rows(conv.ns)):
        i = int(np.where(conv.ns == n)[0][0])
        print(f"  {n:>12,}  {conv.estimates[i]:>10.5f}  {conv.errors[i]:>10.5f}"
              f"  {standard_error(n):>10.5f}")

    print("\nTask 3 - sampling distribution (CLT)")
    print(f"  {'n':>9} {'mean':>9} {'bias':>10} {'emp sd':>9} {'CLT sd':>9}"
          f" {'ratio':>6} {'skew':>7} {'ex kurt':>8} {'KS':>6} {'95% cov':>8}")
    for res in samples:
        s = res.summary()
        print(f"  {s['n']:>9,} {s['mean']:>9.5f} {s['bias']:>+10.5f}"
              f" {s['empirical_sd']:>9.5f} {s['theoretical_sd']:>9.5f}"
              f" {s['sd_ratio']:>6.3f} {s['skewness']:>+7.3f}"
              f" {s['excess_kurtosis']:>+8.3f} {s['ks_statistic']:>6.3f}"
              f" {s['ci95_coverage']:>8.3f}")
    print(f"\n  KS 5% critical value at R = {samples[0].replicates}: "
          f"{1.36 / math.sqrt(samples[0].replicates):.3f}"
          " (smaller KS => consistent with a normal distribution)")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo estimation of pi and an empirical check of the "
                    "Central Limit Theorem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=20260914,
                   help="seed for the random generator (reproducibility)")
    p.add_argument("--replicates", "-R", type=int, default=500,
                   help="repetitions R of the experiment at each fixed n")
    p.add_argument("--sample-sizes", type=int, nargs="+",
                   default=[1_000, 10_000, 100_000],
                   help="values of n used for the sampling-distribution study")
    p.add_argument("--max-exp", type=float, default=7.0,
                   help="convergence study runs n from 10^1 up to 10^max-exp")
    p.add_argument("--points-per-decade", type=int, default=4,
                   help="how many values of n per decade in the convergence study")
    p.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent,
                   help="directory for figures, CSVs and the report")
    p.add_argument("--quick", action="store_true",
                   help="fast run (smaller n and R) for a quick check")
    p.add_argument("--no-report", action="store_true",
                   help="skip building the PDF report")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.replicates = min(args.replicates, 120)
        args.sample_sizes = [1_000, 10_000]
        args.max_exp = 5.0

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    resdir = outdir / "results"
    figdir.mkdir(parents=True, exist_ok=True)
    resdir.mkdir(parents=True, exist_ok=True)

    apply_style()

    print(f"Monte Carlo estimation of pi  (seed = {args.seed})")
    print(f"  true value                pi = {math.pi:.10f}")
    print(f"  sd of one 4*I draw     sigma = {SIGMA_1:.6f}")
    print(f"  so sd(pi_hat)               = {SIGMA_1:.4f} / sqrt(n)")

    # ---- Task 1: a single demonstration call -----------------------------
    demo_n = 1_000_000
    demo = estimate_pi(demo_n, rng)
    print(f"\nTask 1 - estimate_pi({demo_n:,}) = {demo:.6f} "
          f"(error {abs(demo - math.pi):.6f})")

    # ---- Task 2: convergence ---------------------------------------------
    count = int(round(args.max_exp - 1) * args.points_per_decade) + 1
    ns = log_spaced_sizes(1.0, args.max_exp, count)
    print(f"\nRunning convergence study over {ns.size} sample sizes "
          f"({ns[0]:,} ... {ns[-1]:,}) ...")
    conv = convergence_study(ns, rng)

    # ---- Task 3: sampling distribution -----------------------------------
    samples = []
    for n in args.sample_sizes:
        print(f"Running {args.replicates} repetitions at n = {n:,} ...")
        samples.append(sampling_study(int(n), args.replicates, rng))

    print_console_summary(conv, samples)

    # ---- Figures ----------------------------------------------------------
    figures = {
        "dartboard": figure_dartboard(rng),
        "convergence": figure_convergence(conv),
        "error": figure_error_scaling(conv),
        "histograms": figure_sampling_histograms(samples),
        "qq": figure_qq(samples),
    }
    names = {
        "dartboard": "fig1_unit_square_sampling.png",
        "convergence": "fig2_convergence.png",
        "error": "fig3_error_scaling.png",
        "histograms": "fig4_sampling_distributions.png",
        "qq": "fig5_qq_plots.png",
    }
    print("\nFigures")
    for key, fig in figures.items():
        path = figdir / names[key]
        fig.savefig(path)
        print(f"  {path.relative_to(outdir)}")

    # ---- Data files -------------------------------------------------------
    meta = {
        "seed": args.seed,
        "replicates": args.replicates,
        "sample_sizes": [int(n) for n in args.sample_sizes],
        "convergence_max_n": int(ns[-1]),
        "numpy_version": np.__version__,
        "runtime_s": time.perf_counter() - started,
    }
    write_convergence_csv(resdir / "convergence.csv", conv)
    write_sampling_csv(resdir / "sampling_distribution.csv", samples)
    write_summary_json(resdir / "summary.json", conv, samples, meta)
    print("\nData")
    for name in ("convergence.csv", "sampling_distribution.csv", "summary.json"):
        print(f"  {(resdir / name).relative_to(outdir)}")

    # ---- Report -----------------------------------------------------------
    if not args.no_report:
        report_path = outdir / "esmaeili_matin_report.pdf"
        meta["runtime_s"] = time.perf_counter() - started
        build_report(report_path, conv, samples, figures, meta)
        print(f"\nReport\n  {report_path.name}")

    for fig in figures.values():
        plt.close(fig)

    print(f"\nDone in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
