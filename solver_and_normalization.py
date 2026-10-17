"""Inversion perturbation (Table IV) and exact-circulant normalization (Sec. IV.A).

For the 1D semi-discrete fractional Laplacian with symbol |xi|^alpha (h = 1):
  * the open-boundary Toeplitz target A has entries c_{i-j}, with c_r from
    adaptive quadrature;
  * the padded block P^dag A_circ^(M) P is built from an M-point FFT of the
    sampled symbol, and E = P^dag A_circ^(M) P - A;
  * Table IV reports ||A^{-1} E||_2;
  * Gamma = max_k |gamma_k| is the normalization of the exact 2N-circulant
    embedding, reported as Gamma / lambda_max with lambda_max = pi^alpha.

Usage:  python solver_and_normalization.py [--Ns 64 256 1024]
"""
import argparse

import numpy as np
from scipy.integrate import quad


def kernel(alpha, R):
    """c_r = (1/pi) int_0^pi s^alpha cos(r s) ds for r = 0..R-1."""
    c = np.empty(R)
    c[0] = np.pi**alpha / (alpha + 1)
    for r in range(1, R):
        v, _ = quad(lambda s: s**alpha, 0, np.pi, weight='cos', wvar=r,
                    limit=500, epsabs=1e-13, epsrel=1e-13)
        c[r] = v / np.pi
    return c


def toeplitz_and_residual(N, M, alpha, c):
    k = np.arange(M)
    xi = 2 * np.pi * np.minimum(k, M - k) / M
    col = np.real(np.fft.ifft(xi**alpha))
    d = np.arange(N)
    D = d[:, None] - d[None, :]
    A = c[np.abs(D)]
    return A, col[D % M] - A


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--Ns', type=int, nargs='+', default=[64, 256, 1024])
    p.add_argument('--alphas', type=float, nargs='+', default=[0.5, 1.0, 1.5])
    p.add_argument('--ratios', type=int, nargs='+', default=[2, 4, 8, 16])
    args = p.parse_args()

    for alpha in args.alphas:
        c = kernel(alpha, max(args.Ns))
        lmax = np.pi**alpha
        for N in args.Ns:
            A, _ = toeplitz_and_residual(N, 2 * N, alpha, c)
            g = np.concatenate([c[:N], [0.0], c[N - 1:0:-1]])
            gamma = np.real(np.fft.fft(g))
            row = []
            for f in args.ratios:
                if N * f > 16384:
                    row.append('   --   ')
                    continue
                _, E = toeplitz_and_residual(N, f * N, alpha, c)
                row.append(f"{np.linalg.norm(np.linalg.solve(A, E), 2):.2e}")
            print(f"alpha={alpha:<4} N={N:<5} Gamma/lmax={np.max(np.abs(gamma)) / lmax:.5f} "
                  f"min(gamma)>0: {gamma.min() > 0}  ||A^-1 E|| at M/N={args.ratios}: "
                  + "  ".join(row), flush=True)


if __name__ == '__main__':
    main()
