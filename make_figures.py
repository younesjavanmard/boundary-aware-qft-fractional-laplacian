#!/usr/bin/env python3
"""
make_figures.py
===============

Generates the four numerical figures of

    "Boundary-Aware QFT Block-Encoding of Fractional Laplacians"

Default run reproduces the figures cited in the manuscript:

    fig_heatmaps.pdf            (matrix structure, alpha = 1.5)
    fig_embedding_scaling.pdf   (operator-norm scaling, several alpha)
    fig_states_embedded.pdf     (state action on two Gaussians)
    fig_boundary_sweep.pdf      (relative error vs Gaussian center j0)
    fig_functional_benchmarks.pdf
                                (operator action on bounded-domain functions)

Quick start
-----------
    python make_figures.py                  # all four figures, defaults
    python make_figures.py --figure all     # same
    python make_figures.py --figure heatmaps --alpha 1.0
    python make_figures.py --figure scaling --alphas 0.3 0.7 1.5
    python make_figures.py --N 128 --alpha 1.2 --sigma 5
    python make_figures.py --no-validate    # skip the unit-test pre-checks

CLI options
-----------
    --N N           grid size (default 64; must be power of 2 for QFT)
    --h H           mesh size (default 1.0)
    --alpha A       fractional order for the single-alpha figures (default 1.5)
    --alphas ...    list of orders for the scaling figure (default 0.5 1.0 1.5 2.0)
    --sigma S       Gaussian width for the state figures (default 4.0)
    --M-factor F    padding factor for fig_states_embedded (default 2; M = F*N)
    --pads ...      padding factors for boundary sweep (default 2 4 8)
    --figure NAME   one of  all heatmaps scaling states sweep functions
                    (default all)
    --outdir DIR    where to write PDFs (default .)
    --no-validate   skip the validation checks before plotting

Dependencies
------------
    numpy, scipy (for quad), matplotlib
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import quad

# REVTeX-friendly figure defaults: readable after manuscript inclusion.
plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 10.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})


def savefig(fig, path: Path) -> Path:
    """Save a figure as the manuscript PDF and a high-resolution PNG preview."""
    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return pdf_path


def format_matrix_axes(ax, N: int) -> None:
    ticks = [0, N // 4, N // 2, 3 * N // 4, N - 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$i$")
    ax.set_aspect("equal")


def clean_grid(ax) -> None:
    ax.grid(True, which="major", alpha=0.25, lw=0.5)
    ax.grid(True, which="minor", alpha=0.12, lw=0.35)


# =====================================================================
#  Core operators
# =====================================================================

@lru_cache(maxsize=None)
def cm_coefficient(m: int, alpha: float, h: float) -> float:
    """Semi-discrete Toeplitz coefficient

        c_m = (h / pi) * integral_0^{pi/h} xi^alpha * cos(m h xi) dxi.

    For alpha = 1 or 2 with h = 1 the result agrees with closed forms
    to machine precision; see validate().
    """
    integrand = lambda xi: xi ** alpha * np.cos(m * h * xi)
    val, _ = quad(integrand, 0.0, np.pi / h, limit=400)
    return (h / np.pi) * val


def open_toeplitz(N: int, alpha: float, h: float) -> np.ndarray:
    """Open-boundary Toeplitz target A^{(N)}_{alpha,h}, entries c_{i-j}."""
    cs = np.array([cm_coefficient(m, alpha, h)
                   for m in range(-(N - 1), N)])
    offset = N - 1
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            A[i, j] = cs[(i - j) + offset]
    return A


def qft_circulant(M: int, alpha: float, h: float) -> np.ndarray:
    """QFT-realisable size-M circulant tilde A^{(M)}_{alpha,h}."""
    k = np.arange(M)
    xi = np.where(k < M // 2,
                  2.0 * np.pi * k / (M * h),
                  2.0 * np.pi * k / (M * h) - 2.0 * np.pi / h)
    D = np.abs(xi) ** alpha
    c_circ_complex = np.fft.ifft(D)
    if np.max(np.abs(c_circ_complex.imag)) > 1e-10:
        raise RuntimeError("Imaginary leak in circulant kernel; "
                           "check the symbol or grid.")
    c_circ = c_circ_complex.real
    A = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            A[i, j] = c_circ[(i - j) % M]
    return A


def apply_embedded(u: np.ndarray, alpha: float, h: float, M: int) -> np.ndarray:
    """Compute (P^dag tilde A^{(M)} P) u via padded FFT, without
       materialising the M x M circulant explicitly. Returns an N-vector."""
    N = len(u)
    k = np.arange(M)
    xi = np.where(k < M // 2,
                  2.0 * np.pi * k / (M * h),
                  2.0 * np.pi * k / (M * h) - 2.0 * np.pi / h)
    D = np.abs(xi) ** alpha
    u_pad = np.concatenate([u, np.zeros(M - N)])
    out = np.fft.ifft(D * np.fft.fft(u_pad)).real
    return out[:N]


def gaussian_state(N: int, mu: float, sigma: float) -> np.ndarray:
    x = np.arange(N, dtype=float)
    psi = np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))
    return psi / np.linalg.norm(psi)


# =====================================================================
#  Validation checks (run before plotting unless --no-validate)
# =====================================================================

def validate() -> None:
    print("Validation checks ...")
    # (i) Closed form for alpha = 1, h = 1: c_m = ((-1)^m - 1)/(pi m^2)
    for m in [1, 2, 3, 5, 10, 21]:
        analyt = ((-1) ** m - 1) / (np.pi * m * m)
        num = cm_coefficient(m, 1.0, 1.0)
        assert abs(num - analyt) < 1e-12, \
            f"  alpha=1 closed-form mismatch at m={m}: {num} vs {analyt}"
    # (ii) Closed form for alpha = 2, h = 1: c_m = 2(-1)^m/m^2
    for m in [1, 2, 3, 5, 10, 21]:
        analyt = 2.0 * (-1) ** m / (m * m)
        num = cm_coefficient(m, 2.0, 1.0)
        assert abs(num - analyt) < 1e-12, \
            f"  alpha=2 closed-form mismatch at m={m}: {num} vs {analyt}"
    # (iii) Symmetry c_m = c_{-m}.
    for alpha in [0.5, 1.0, 1.5]:
        for m in [1, 3, 7]:
            assert abs(cm_coefficient(m, alpha, 1.0)
                       - cm_coefficient(-m, alpha, 1.0)) < 1e-12, \
                f"  symmetry failure at alpha={alpha}, m={m}"
    # (iv) Two definitions of tilde A^{(N)} agree:
    #      (a) F^{-1} diag F  (qft_circulant) and
    #      (b) periodisation of the Toeplitz sequence.
    #     The agreement is limited by the periodisation cutoff n_per.
    N = 32
    alpha = 1.5
    A_C = qft_circulant(N, alpha, 1.0)
    n_per = 16
    A_C_alt = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            m0 = i - j
            s = 0.0
            for ell in range(-n_per, n_per + 1):
                s += cm_coefficient(m0 + ell * N, alpha, 1.0)
            A_C_alt[i, j] = s
    diff_max = np.max(np.abs(A_C - A_C_alt))
    assert diff_max < 1e-3, \
        f"  QFT vs periodisation circulants disagree by {diff_max}"
    # (v) Embedding identity at the matrix level:
    #     top-left N x N block of tilde A^{(2N)} - A^{(N)} = E^{(M)}.
    M = 2 * N
    A_T = open_toeplitz(N, alpha, 1.0)
    A_M = qft_circulant(M, alpha, 1.0)
    E = A_M[:N, :N] - A_T
    # E should match  sum_{ell != 0} c_{i-j + ell M}.
    E_alt = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            s = 0.0
            for ell in range(-n_per, n_per + 1):
                if ell == 0:
                    continue
                s += cm_coefficient(i - j + ell * M, alpha, 1.0)
            E_alt[i, j] = s
    e_diff = np.max(np.abs(E - E_alt))
    assert e_diff < 1e-3, \
        f"  embedding-identity mismatch by {e_diff}"
    print("  all checks passed.")


# =====================================================================
#  Figure 1: matrix-structure heatmaps
# =====================================================================

def figure_heatmaps(N: int, h: float, alpha: float, outdir: Path) -> Path:
    print(f"\n[heatmaps] N={N}, h={h}, alpha={alpha}")
    A_T = open_toeplitz(N, alpha, h)
    A_C = qft_circulant(N, alpha, h)
    D = A_C - A_T

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)

    # Robust signed scaling keeps the diagonal from washing out off-diagonal structure.
    vmax = np.percentile(np.abs(np.r_[A_T.ravel(), A_C.ravel()]), 99.5)
    im0 = axes[0].imshow(A_T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Open Toeplitz")
    format_matrix_axes(axes[0], N)

    im1 = axes[1].imshow(A_C, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("QFT circulant")
    format_matrix_axes(axes[1], N)

    err = np.abs(D)
    positive = err[err > 0]
    floor = max(np.percentile(positive, 1), 1e-14) if positive.size else 1e-14
    im2 = axes[2].imshow(err + floor, origin="lower", cmap="viridis",
                         norm=LogNorm(vmin=floor, vmax=err.max() + floor))
    axes[2].set_title("Aliasing error")
    format_matrix_axes(axes[2], N)

    # Subtle corner boxes mark the wrap-around regions where aliasing is largest.
    corner = max(8, N // 8)
    for xy in [(N - corner - 0.5, -0.5), (-0.5, N - corner - 0.5)]:
        axes[2].add_patch(plt.Rectangle(xy, corner, corner, fill=False,
                                        edgecolor="white", lw=0.8, alpha=0.8))

    cbar0 = fig.colorbar(im0, ax=axes[:2], shrink=0.82, pad=0.015)
    cbar0.set_label("entry value")
    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.82, pad=0.018)
    cbar2.set_label("absolute error")

    fname = savefig(fig, outdir / "fig_heatmaps")
    plt.close(fig)
    print(f"  saved {fname}")
    return fname

def figure_scaling(N: int, h: float, alphas: list[float],
                   pads: tuple = (2, 3, 4, 6, 8, 12, 16, 24, 32),
                   sigma: float = 4.0,
                   outdir: Path = Path('.')) -> Path:
    print(f"\n[scaling] N={N}, h={h}, alphas={alphas}")
    u_b = gaussian_state(N, 3.0, sigma)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=False)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    markers = ['o', 's', '^', 'D']
    handles, labels = [], []

    for idx, alpha in enumerate(alphas):
        A_T = open_toeplitz(N, alpha, h)
        Ms = [p * N for p in pads]
        op_norms, state_errs = [], []
        for M in Ms:
            A_M_circ = qft_circulant(M, alpha, h)
            E = A_M_circ[:N, :N] - A_T
            op_norms.append(np.linalg.norm(E, ord=2))
            state_errs.append(
                np.linalg.norm(apply_embedded(u_b, alpha, h, M) - A_T @ u_b))
        Kvals = np.array([M - N + 1 for M in Ms])
        lam_max = (np.pi / h) ** alpha
        op_norms = np.array(op_norms) / lam_max
        state_errs = np.array(state_errs) / lam_max

        r_alpha = min(2.0, 1.0 + alpha)
        col = colors[idx % len(colors)]
        mk = markers[idx % len(markers)]
        label = rf'$\alpha={alpha:g}$'

        h0, = axes[0].loglog(np.array(Ms, dtype=float), op_norms, color=col, marker=mk, label=label)
        # Guide slope from the finite-window bound ||E||/lambda_max = O(N M^{-r_alpha})
        Mvals = np.array(Ms, dtype=float)
        ref = op_norms[0] * (Mvals[0] / Mvals) ** r_alpha
        axes[0].loglog(Mvals, ref, color=col, ls=':', lw=0.9, alpha=0.45)
        axes[1].loglog(Mvals, state_errs, color=col, marker=mk, label=label)
        handles.append(h0)
        labels.append(label)

        print(f"  alpha={alpha:.1f}: ||E||_2/lambda_max from {op_norms[0]:.2e} "
              f"(M={Ms[0]}) down to {op_norms[-1]:.2e} (M={Ms[-1]})")

    axes[0].set_title("Operator residual")
    axes[0].set_xlabel(r'$M$')
    axes[0].set_ylabel(r'$\|E^{(M)}\|_2/\lambda_{\max}$')
    clean_grid(axes[0])

    axes[1].set_title("Boundary-state residual")
    axes[1].set_xlabel(r'$M$')
    axes[1].set_ylabel(r'$\|E^{(M)}u_b\|_2/\lambda_{\max}$')
    clean_grid(axes[1])

    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    m_ticks = [N * p for p in (2, 4, 8, 16, 32)]
    for ax in axes:
        ax.xaxis.set_major_locator(FixedLocator(m_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in m_ticks]))
        ax.xaxis.set_minor_formatter(NullFormatter())

    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.24,
                        wspace=0.30)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, 0.02), columnspacing=1.6)

    fname = savefig(fig, outdir / "fig_embedding_scaling")
    plt.close(fig)
    print(f"  saved {fname}")
    return fname

def figure_states_embedded(N: int, h: float, alpha: float,
                           sigma: float, M_factor: int,
                           outdir: Path) -> Path:
    print(f"\n[states] N={N}, h={h}, alpha={alpha}, sigma={sigma}, "
          f"M={M_factor}N")
    M = M_factor * N
    A_T = open_toeplitz(N, alpha, h)
    A_C = qft_circulant(N, alpha, h)

    u_c = gaussian_state(N, (N - 1) / 2.0, sigma)
    u_b = gaussian_state(N, 3.0, sigma)

    j = np.arange(N)

    # Two-column "test panel" layout: each column tells one story (input
    # state shown as a thin strip, operator action drawn below). The styling
    # is inspired by Zhou & Zhang (arXiv:2311.07814) Fig. 1, which presents
    # each test as a self-contained target-vs-surrogate comparison.
    fig = plt.figure(figsize=(7.2, 4.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 3.0],
                          hspace=0.10, wspace=0.22,
                          left=0.085, right=0.985, top=0.92, bottom=0.13)

    columns = [
        ("Bulk-localized Gaussian", u_c, r"$j_0=N/2$"),
        ("Boundary-localized Gaussian", u_b, r"$j_0=3$"),
    ]
    line_handles = None
    out_axes = []

    for col, (title, u, sub) in enumerate(columns):
        ax_in = fig.add_subplot(gs[0, col])
        ax_out = fig.add_subplot(gs[1, col], sharex=ax_in)
        out_axes.append(ax_out)

        # ---- input strip: bare profile, no axis decoration -------------
        ax_in.fill_between(j, 0, u, color="0.15", alpha=0.12, lw=0)
        ax_in.plot(j, u, color="0.15", lw=1.2)
        ax_in.set_title(title, pad=4)
        ax_in.set_xlim(0, N - 1)
        ax_in.set_ylim(0, u.max() * 1.18)
        ax_in.set_yticks([])
        ax_in.tick_params(labelbottom=False, bottom=False)
        for side in ("top", "right", "left"):
            ax_in.spines[side].set_visible(False)
        ax_in.text(0.985, 0.86, sub, transform=ax_in.transAxes,
                   ha="right", va="top", fontsize=8, color="0.35")

        # ---- operator-action panel -------------------------------------
        out_T = A_T @ u
        out_C = A_C @ u
        out_E = apply_embedded(u, alpha, h, M)
        err_C = np.linalg.norm(out_C - out_T)
        err_E = np.linalg.norm(out_E - out_T)

        ax_out.axhline(0, color="0.85", lw=0.6, zorder=0)
        h_T, = ax_out.plot(j, out_T, color="0.12", ls="-", lw=1.9,
                           label="open target")
        h_C, = ax_out.plot(j, out_C, color="#D55E00", ls="--", lw=1.3,
                           label="native QFT")
        h_E, = ax_out.plot(j, out_E, color="#0072B2", ls=(0, (4, 1.5, 1, 1.5)),
                           lw=1.4, label=f"padded {M_factor}N")
        if line_handles is None:
            line_handles = [h_T, h_C, h_E]

        ax_out.set_xlim(0, N - 1)
        ax_out.set_xlabel(r"$j$")
        if col == 0:
            ax_out.set_ylabel(r"$(Au)_j$")
        ax_out.tick_params(direction="out")

        # Compact error annotation. Native diverges for the boundary state, so
        # the "boundary" panel reads best with the box in the upper-left.
        ann_loc = (0.04, 0.96) if col == 1 else (0.04, 0.96)
        ax_out.text(ann_loc[0], ann_loc[1],
                    f"native:  {err_C:.1e}\npadded: {err_E:.1e}",
                    transform=ax_out.transAxes, ha="left", va="top",
                    fontsize=7.5, family="monospace",
                    bbox=dict(facecolor="white", edgecolor="0.85",
                              alpha=0.85, pad=2, lw=0.5))

        print(f"  {'u_c' if col == 0 else 'u_b'}: native err = {err_C:.3e}, "
              f"embedded err = {err_E:.3e}")

    # Single shared legend on the bulk panel (curves overlap there, so the
    # top-right region is empty); inspired by the per-panel legends of
    # Zhou & Zhang Fig. 1, but lighter (frameless, one panel only).
    out_axes[0].legend(line_handles, [h.get_label() for h in line_handles],
                       loc="upper right", frameon=False, handlelength=2.2,
                       labelspacing=0.25, borderaxespad=0.4)

    fname = savefig(fig, outdir / "fig_states_embedded")
    plt.close(fig)
    print(f"  saved {fname}")
    return fname


def figure_functional_benchmarks(N: int, h: float, alpha: float,
                                 M_factor: int, outdir: Path) -> Path:
    """Compare open, native periodic, and padded actions on explicit functions.

    The test functions are generated on their physical intervals and then
    normalised in the Euclidean vector norm before applying the discrete
    operators. This keeps the plotted operator actions comparable without
    changing the operator definitions.
    """
    N_fun = 128
    alpha_fun = 1.5
    M_fun = 2 * N_fun
    print(f"\n[functions] N={N_fun}, alpha={alpha_fun}, M=2N")

    # Bounded-domain tests in the visual spirit of Zhou & Zhang
    # (arXiv:2311.07814) Fig. 1, but using this repository's operators:
    # open Toeplitz target, native QFT-circulant surrogate, and padded
    # compressed action.
    cases = [
        {
            "title": r"$u(x)=\exp(-x^{2})$",
            "domain": (-2.0, 2.0),
            "u": lambda x: np.exp(-x ** 2),
        },
        {
            "title": r"$u(x)=1/(1+x^{2})^{7}$",
            "domain": (-1.0, 1.0),
            "u": lambda x: 1.0 / (1.0 + x ** 2) ** 7,
        },
        {
            "title": r"$u(x)=(1-x^{2})_{+}^{2}$",
            "domain": (-1.0, 1.0),
            "u": lambda x: np.maximum(1.0 - x ** 2, 0.0) ** 2,
        },
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.4),
                             constrained_layout=False)
    handles = None
    labels = None
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for idx, case in enumerate(cases):
        xmin, xmax = case["domain"]
        x = np.linspace(xmin, xmax, N_fun)
        h_fun = (xmax - xmin) / (N_fun - 1)
        u = case["u"](x)
        u = u / np.linalg.norm(u)

        A_T = open_toeplitz(N_fun, alpha_fun, h_fun)
        A_C = qft_circulant(N_fun, alpha_fun, h_fun)
        out_T = A_T @ u
        out_C = A_C @ u
        out_P = apply_embedded(u, alpha_fun, h_fun, M_fun)

        err_native = np.linalg.norm(out_C - out_T) / np.linalg.norm(out_T)
        err_padded = np.linalg.norm(out_P - out_T) / np.linalg.norm(out_T)
        point_native = np.abs(out_C - out_T)
        point_padded = np.abs(out_P - out_T)

        ax_top = axes[0, idx]
        ax_err = axes[1, idx]

        h_open, = ax_top.plot(x, out_T, color="black", ls="-", lw=1.8,
                              label="open target", zorder=3)
        h_native, = ax_top.plot(x, out_C, color="#1f77b4", ls="--", lw=2.2,
                                label="native QFT", zorder=5,
                                dash_capstyle="round")
        h_padded, = ax_top.plot(x, out_P, color="#d62728", ls="-.", lw=1.8,
                                label="padded method", zorder=4)
        ax_top.axhline(0.0, color="0.85", lw=0.6, zorder=0)
        ax_top.set_title(case["title"], fontsize=10)
        ax_top.set_xlim(xmin, xmax)
        ax_top.set_ylabel(r"$Au$" if idx == 0 else "")
        ax_top.tick_params(labelbottom=False, direction="out")

        ax_err.semilogy(x, point_native + 1e-16, color="#1f77b4", ls="--",
                        lw=2.2, zorder=5)
        ax_err.semilogy(x, point_padded + 1e-16, color="#d62728", ls="-.",
                        lw=1.8, zorder=4)
        ax_err.set_xlim(xmin, xmax)
        ax_err.set_xlabel(r"$x$")
        ax_err.set_ylabel("pointwise error" if idx == 0 else "")
        ax_err.text(0.50, 0.95,
                    f"native {err_native:.1e}\npadded {err_padded:.1e}",
                    transform=ax_err.transAxes, ha="center", va="top",
                    fontsize=6.5,
                    bbox=dict(facecolor="white", edgecolor="0.8",
                              alpha=0.85, pad=1.5, lw=0.5))

        if handles is None:
            handles = [h_open, h_native, h_padded]
            labels = [line.get_label() for line in handles]

        for ax, label in ((ax_top, panel_labels[idx]),
                          (ax_err, panel_labels[idx + 3])):
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction="out")
            ax.text(0.02, 0.95, label, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, fontweight="normal")
        clean_grid(ax_err)

        print(f"  {case['title']}: h={h_fun:.4g}, "
              f"native rel = {err_native:.3e}, padded rel = {err_padded:.3e}")

    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.12,
                        wspace=0.32, hspace=0.42)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=8,
               handlelength=2.7, columnspacing=1.6)

    fname = savefig(fig, outdir / "fig_functional_benchmarks")
    plt.close(fig)
    print(f"  saved {fname}")
    return fname


def figure_boundary_sweep(N: int, h: float, alpha: float,
                          sigma: float, pads: tuple,
                          j0_values: range,
                          outdir: Path) -> Path:
    print(f"\n[sweep] N={N}, h={h}, alpha={alpha}, sigma={sigma}, pads={pads}")
    A_T = open_toeplitz(N, alpha, h)
    A_C = qft_circulant(N, alpha, h)
    j0_arr = np.array(list(j0_values))

    rel_native = np.zeros(len(j0_arr))
    rel_padded = {p: np.zeros(len(j0_arr)) for p in pads}
    for k, j0 in enumerate(j0_arr):
        u = gaussian_state(N, float(j0), sigma)
        target = A_T @ u
        norm_t = np.linalg.norm(target)
        rel_native[k] = np.linalg.norm(A_C @ u - target) / norm_t
        for p in pads:
            rel_padded[p][k] = (
                np.linalg.norm(apply_embedded(u, alpha, h, p * N) - target)
                / norm_t)

    fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=False)
    ax.axvspan(j0_arr[0], 8, color='0.92', alpha=0.45, zorder=0)
    ax.semilogy(j0_arr, rel_native, 'o-', color="#D55E00", label='native')
    styles = [('s--', '#0072B2'), ('^-.', '#009E73'), ('d:', '#CC79A7')]
    for (mk, col), p in zip(styles, pads):
        ax.semilogy(j0_arr, rel_padded[p], mk, color=col,
                    label=f'padded {p}N')
    ax.axhline(1.0, color='0.35', ls=':', lw=1.0)
    ax.text(j0_arr[-1] - 0.5, 1.05, '100%', color='0.35', fontsize=8,
            ha='right', va='bottom')
    ax.set_xlabel(r'Gaussian center $j_0$')
    ax.set_ylabel('relative error')
    ax.set_title('Relative state error')
    clean_grid(ax)
    ax.set_xlim(j0_arr[0] - 1, j0_arr[-1] + 1)
    fig.subplots_adjust(left=0.10, right=0.82, top=0.88, bottom=0.22)
    ax.legend(frameon=False, loc='center left',
              bbox_to_anchor=(1.01, 0.5),
              borderaxespad=0.0, handlelength=1.8)

    fname = savefig(fig, outdir / "fig_boundary_sweep")
    plt.close(fig)

    print(f"  j0  rel(native)  " +
          "  ".join(f"rel(M={p}N)" for p in pads))
    for k, j0 in enumerate(j0_arr):
        if j0 in (j0_arr[0], j0_arr[5], j0_arr[-1]):
            print(f"  {j0:2d}  {rel_native[k]:.3e}  " +
                  "  ".join(f"{rel_padded[p][k]:.3e}" for p in pads))
    print(f"  saved {fname}")
    return fname

def main():
    p = argparse.ArgumentParser(
        description='Numerical figures for the boundary-aware QFT '
                    'block-encoding paper.')
    p.add_argument('--N', type=int, default=64,
                   help='grid size (default 64; power of 2 for QFT)')
    p.add_argument('--h', type=float, default=1.0,
                   help='mesh size (default 1.0)')
    p.add_argument('--alpha', type=float, default=1.5,
                   help='fractional order for single-alpha figures '
                        '(default 1.5)')
    p.add_argument('--alphas', type=float, nargs='+',
                   default=[0.5, 1.0, 1.5, 2.0],
                   help='alpha list for the scaling figure '
                        '(default 0.5 1.0 1.5 2.0)')
    p.add_argument('--sigma', type=float, default=4.0,
                   help='Gaussian width for the state figures (default 4.0)')
    p.add_argument('--M-factor', type=int, default=2,
                   help='M = M_factor * N for fig_states_embedded (default 2)')
    p.add_argument('--pads', type=int, nargs='+', default=[2, 4, 8],
                   help='padding factors for the sweep (default 2 4 8)')
    p.add_argument('--figure',
                   choices=['all', 'heatmaps', 'scaling', 'states', 'sweep',
                            'functions', 'functional'],
                   default='all', help='which figure to generate')
    p.add_argument('--outdir', type=str, default='.',
                   help='output directory (default .)')
    p.add_argument('--no-validate', action='store_true',
                   help='skip pre-plot validation checks')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.no_validate:
        validate()

    if args.figure in ('all', 'heatmaps'):
        figure_heatmaps(args.N, args.h, args.alpha, outdir)
    if args.figure in ('all', 'scaling'):
        figure_scaling(args.N, args.h, args.alphas,
                       sigma=args.sigma, outdir=outdir)
    if args.figure in ('all', 'states'):
        figure_states_embedded(args.N, args.h, args.alpha,
                               args.sigma, args.M_factor, outdir)
    if args.figure in ('all', 'functions', 'functional'):
        figure_functional_benchmarks(args.N, args.h, args.alpha,
                                     args.M_factor, outdir)
    if args.figure in ('all', 'sweep'):
        figure_boundary_sweep(args.N, args.h, args.alpha, args.sigma,
                              tuple(args.pads),
                              range(2, args.N // 2 + 1),
                              outdir)

    print("\nDone.")


if __name__ == '__main__':
    main()
