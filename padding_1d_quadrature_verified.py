"""Reproducible 1D padding study for Fourier symbol |xi|**alpha on [-pi,pi].

The physical Toeplitz kernel c_r is evaluated by independent quadrature,
NOT by cutting off aliases of an L-point discrete Fourier transform.
The padded symbol is evaluated by its finite M-point DFT. Therefore
  e_r(M) = (1/M) sum_k |xi_k|**alpha exp(i r xi_k) - c_r
is exactly the desired residual up to the stated floating-point/quadrature errors.
Spectral norms are computed using Hermitian Lanczos (`eigsh`) with an FFT
Toeplitz matvec and checked against dense eigvalsh for N<=64.

No finite-L infinite-reference claim or certified quadrature bound is made.
"""
import argparse, csv, math, time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import toeplitz, eigvalsh


def target_kernel(alpha, N, quad_tol=2e-12):
    c = np.empty(N, dtype=np.float64)
    reported = np.empty(N, dtype=np.float64)
    c[0] = np.pi**alpha/(alpha + 1)
    reported[0] = np.finfo(float).eps * abs(c[0])
    for r in range(1, N):
        v, e = quad(lambda x: x**alpha, 0, np.pi, weight='cos',
                    wvar=r, limit=500, epsabs=quad_tol, epsrel=quad_tol)
        c[r] = v / np.pi
        reported[r] = e / np.pi
    return c, reported


def periodic_coefficients(alpha, M, N):
    xi = 2*np.pi*np.fft.fftfreq(M)
    symbol = np.abs(xi)**alpha
    return np.fft.ifft(symbol).real[:N]


def toeplitz_norm_lanczos(e, check_dense=False):
    N = len(e)
    if N == 1: return abs(float(e[0])), 0.
    embed = np.r_[e, 0., e[:0:-1]]
    spec = np.fft.fft(embed)
    def mv(v):
        pad = np.r_[v, np.zeros(N)]
        return np.fft.ifft(spec*np.fft.fft(pad)).real[:N]
    mat = LinearOperator((N,N), matvec=mv, rmatvec=mv, dtype=np.float64)
    val, vec = eigsh(mat, k=1, which='LM', return_eigenvectors=True,
                     tol=1e-11, maxiter=2000, v0=np.ones(N))
    norm = abs(float(val[0]))
    residual = np.linalg.norm(mv(vec[:,0]) - val[0]*vec[:,0])
    if check_dense:
        evs=eigvalsh(toeplitz(e), check_finite=False)
        dense=max(abs(evs[0]),abs(evs[-1]))
        if not np.isclose(norm,dense,rtol=2e-8,atol=2e-10):
            raise RuntimeError(f'Dense/Lanczos mismatch {norm} vs {dense}')
    return norm, residual


def run(alphas, ns, eps_list, max_m, quad_tol, outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    table=[];curves=[]
    for alpha in alphas:
        for N in ns:
            t0=time.time()
            target,reported=target_kernel(alpha,N,quad_tol)
            target_tight,_=target_kernel(alpha,N,quad_tol/5)
            quadrature_delta=float(np.max(np.abs(target-target_tight)))
            fft_ref_deltas=[]
            for Q in [2**14,2**16]:
                if Q>2*N:
                    fft_ref_deltas.append(float(np.max(np.abs(periodic_coefficients(alpha,Q,N)-target))))
                else: fft_ref_deltas.append(float('nan'))
            vals={}
            for m in range(math.ceil(math.log2(2*N)),max_m+1):
                M=2**m
                e=periodic_coefficients(alpha,M,N)-target
                norm,eigres=toeplitz_norm_lanczos(e,check_dense=N<=64)
                eta=norm/np.pi**alpha
                vals[m]=eta
                curves.append(dict(alpha=alpha,N=N,M=M,m=m,eta=eta,
                      eigensolver_residual=eigres/np.pi**alpha,
                      quadrature_maxdelta=quadrature_delta,
                      numerical_schur_upper=np.sum(np.abs(e))*2/np.pi**alpha))
                if eta < min(eps_list)/30: break
            row=dict(alpha=alpha,N=N,eta_M2N=vals[min(vals)],
                quad_maxdelta=quadrature_delta,
                quad_maxreported=float(np.max(reported)),
                fft_Q16384_maxdelta=fft_ref_deltas[0],fft_Q65536_maxdelta=fft_ref_deltas[1],
                runtime_seconds=time.time()-t0)
            for eps in eps_list:
                available=[m for m,eta in vals.items() if eta<=eps]
                row[f'm_for_{eps:g}'] = min(available) if available else 'not reached'
            table.append(row)
            print(f'alpha={alpha:g} N={N} eta(2N)={row["eta_M2N"]:.9g} '
                  f'min m: '+str([(eps,row[f'm_for_{eps:g}']) for eps in eps_list])+ 
                  f'  Q65536 difference={fft_ref_deltas[1]:.3g}',flush=True)
    for name,rows in [('1d_quadrature_results.csv',table),('1d_quadrature_curves.csv',curves)]:
        with (outdir/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return table,curves


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--quick',action='store_true')
    p.add_argument('--outdir',default='results')
    p.add_argument('--max-m',type=int,default=18)
    a=p.parse_args()
    run([.5,1.,1.5], [32,64] if a.quick else [64,256,1024],
        [1e-2,1e-3,1e-4], a.max_m,2e-12,a.outdir)
if __name__=='__main__':main()
