"""Summary figure for the README: boundary error of the zero-padded QFT block.

Panels (1D, h = 1, symbol |xi|^alpha):
  (a) normalized boundary error ||E||_2 / lambda_max versus padding ratio M/N;
  (b) the same error at minimal padding M = 2N versus grid size N;
  (c) inversion perturbation ||A^{-1} E||_2 versus M/N for several N.

Usage:  python make_summary_figure.py [--out summary.png]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from solver_and_normalization import kernel, toeplitz_and_residual

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {0.5: "#2a78d6", 1.0: "#eb6834", 1.5: "#1baf7a"}   # validated categorical slots 1-3
MARKERS = {0.5: "o", 1.0: "s", 1.5: "^"}
LABEL = {0.5: r"$\alpha=1/2$", 1.0: r"$\alpha=1$", 1.5: r"$\alpha=3/2$"}
N_STYLE = {64: ("o", 0.88), 256: ("s", 1.0), 1024: ("D", 1.135)}   # marker, x-offset factor


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, which="both", labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(7, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def clean_log_x(ax, ticks):
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="summary.png")
    args = p.parse_args()

    alphas = [0.5, 1.0, 1.5]
    ratios = np.array([2, 4, 8, 16, 32])
    Ns = [64, 128, 256, 512, 1024]
    kernels = {a: kernel(a, 1024) for a in alphas}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.27, top=0.83, wspace=0.42)

    # (a) error vs padding ratio, N = 256
    ax = axes[0]
    N = 256
    for a in alphas:
        lm = np.pi**a
        err = [np.linalg.norm(toeplitz_and_residual(N, f * N, a, kernels[a])[1], 2) / lm
               for f in ratios]
        ax.loglog(ratios, err, color=SERIES[a], lw=2, marker=MARKERS[a], ms=6,
                  mec=SURFACE, mew=1.2, label=LABEL[a])
        r = min(2, 1 + a)
        ax.loglog(ratios, err[0] * (ratios / ratios[0]) ** (-r), color=SERIES[a],
                  lw=1, ls=":", alpha=0.7)
        end_label(ax, ratios[-1], err[-1], LABEL[a], dy={0.5: 0, 1.0: -7, 1.5: 7}[a])
    clean_log_x(ax, ratios)
    ax.set_xlim(1.7, 75)
    ax.set_ylim(bottom=5e-7)
    style(ax, "(a) Padding removes the boundary error",
          r"padding ratio $M/N$", r"$\|E^{(M)}\|_2/\lambda_{\max}$")
    ax.text(0.97, 0.97, "$N=256$\ndotted: $(M/N)^{-\\min(2,1+\\alpha)}$ (Theorem 2)",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, ha="right", va="top")

    # (b) error at M = 2N vs N
    ax = axes[1]
    Ns_arr = np.array(Ns)
    for a in alphas:
        lm = np.pi**a
        err = [np.linalg.norm(toeplitz_and_residual(n, 2 * n, a, kernels[a])[1], 2) / lm
               for n in Ns]
        ax.loglog(Ns_arr, err, color=SERIES[a], lw=2, marker=MARKERS[a], ms=6,
                  mec=SURFACE, mew=1.2)
        end_label(ax, Ns_arr[-1], err[-1], LABEL[a], dy={0.5: 0, 1.0: -5, 1.5: 5}[a])
    clean_log_x(ax, Ns_arr)
    ax.set_ylim(bottom=4e-5)
    ax.set_xlim(50, 2600)
    style(ax, "(b) Minimal padding $M=2N$ already converges",
          r"grid size $N$", r"$\|E^{(2N)}\|_2/\lambda_{\max}$")
    ax.text(0.97, 0.97, "$\\propto N^{-1}$ for $\\alpha\\geq1$\n$\\propto N^{-1/2}$ for $\\alpha=1/2$",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, ha="right", va="top")

    # (c) inversion perturbation vs M/N for several N
    ax = axes[2]
    rr = np.array([2, 4, 8, 16])
    for a in alphas:
        last = None
        for n, (mk, dx) in N_STYLE.items():
            A, _ = toeplitz_and_residual(n, 2 * n, a, kernels[a])
            vals = [np.linalg.norm(np.linalg.solve(A, toeplitz_and_residual(n, f * n, a, kernels[a])[1]), 2)
                    for f in rr]
            ax.loglog(rr * dx, vals, color=SERIES[a], lw=2 if n == 256 else 0, marker=mk, ms=6,
                      mfc="none" if n != 256 else SERIES[a], mec=SERIES[a], mew=1.4)
            last = vals
        end_label(ax, rr[-1] * 1.135, last[-1], LABEL[a])
    clean_log_x(ax, rr)
    ax.set_xlim(1.6, 40)
    style(ax, "(c) Solver error does not grow with $N$",
          r"padding ratio $M/N$", r"$\|A^{-1}E^{(M)}\|_2$")
    handles = [plt.Line2D([], [], color=INK_2, lw=0, marker=mk, ms=6,
                          mfc="none" if n != 256 else INK_2, mec=INK_2, mew=1.4,
                          label=f"$N={n}$") for n, (mk, _) in N_STYLE.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=False,
              labelcolor=INK_2, title="markers side by side:\nsame value for all $N$",
              title_fontsize=8.5, alignment="left")

    series_handles = [plt.Line2D([], [], color=SERIES[a], lw=2, marker=MARKERS[a], ms=6,
                                 mec=SURFACE, mew=1.2, label=LABEL[a]) for a in alphas]
    fig.legend(handles=series_handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Zero-padded QFT block-encoding of the open-boundary fractional Laplacian (1D, $h=1$)",
                 x=0.06, ha="left", fontsize=12.5, color=INK, y=0.98)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
