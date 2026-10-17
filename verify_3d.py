"""Matrix-free, independently reproducible 3D fractional-Laplacian aliasing tests.

Only uses NumPy and SciPy; Python 3.10+.
Physical open-domain N^3 Toeplitz operator is approximated by a high-resolution
Q-point grid Fourier coefficient reference. Q-grid is NOT the mathematical limit;
Q-sensitivity is explicitly reported rather than silently called 'exact'.
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from scipy.fft import ifftn, fftn
from scipy.sparse.linalg import LinearOperator, eigsh


def kernel_grid(Q, alpha, d=3):
    # FFT frequencies in [-pi,pi); h=1. Dimension d=3 in benchmark.
    f = 2*np.pi*np.fft.fftfreq(Q)
    symbol = np.zeros((Q,)*d, dtype=np.float64)
    for ax in range(d):
        shape = [1]*d
        shape[ax] = Q
        symbol += f.reshape(shape)**2
    np.sqrt(symbol, out=symbol)
    np.power(symbol, alpha, out=symbol)
    symbol /= (np.sqrt(d)*np.pi)**alpha
    return ifftn(symbol, workers=2).real


def coeff_window(C, N):
    """Coefficients c_r for r in [-(N-1), ..., N-1]^3."""
    index = np.concatenate((np.arange(N),np.arange(C.shape[0]-(N-1), C.shape[0])))
    return C[np.ix_(index,index,index)]


def make_toeplitz_matvec(err_coeff, N):
    """3-level Toeplitz matvec via 2N-period circulant embedding."""
    L=2*N
    # coeff_window index maps r=0,...,N-1,-N+1,...,-1,
    # circulant layout is r=0,...,N-1,0,-N+1,...,-1.
    idx = np.r_[np.arange(N), -1, np.arange(N,2*N-1)]
    kernel = np.zeros((L,L,L), dtype=np.float64)
    good = np.r_[np.arange(N),np.arange(N+1,L)]
    refs = np.r_[np.arange(N),np.arange(N,2*N-1)]
    kernel[np.ix_(good,good,good)] = err_coeff[np.ix_(refs,refs,refs)]
    spectrum = fftn(kernel, workers=2)
    del kernel

    def matvec(x):
        x = np.asarray(x).reshape((N,)*3)
        padded = np.zeros((L,L,L), dtype=np.float64)
        padded[:N,:N,:N]=x
        y = ifftn(fftn(padded,workers=2)*spectrum,workers=2).real[:N,:N,:N]
        return np.ascontiguousarray(y).ravel()
    return LinearOperator((N**3,N**3),matvec=matvec,rmatvec=matvec,dtype=np.float64)


def spectral_norm(C_M,C_Q,N,alpha,eig_tol=1e-5):
    E_coeff = coeff_window(C_M,N)-coeff_window(C_Q,N)
    op=make_toeplitz_matvec(E_coeff,N)
    eig=eigsh(op,k=1,which='LM',return_eigenvectors=False,
              tol=eig_tol,maxiter=250,v0=np.ones(N**3,dtype=np.float64))[0]
    return abs(float(eig)) # kernels already normalized


def experiment(N,alpha,M_values,Q_lo,Q_hi):
    assert Q_hi > max(M_values) and Q_lo > max(M_values)
    start=time.time()
    C_lo=kernel_grid(Q_lo,alpha)
    C_hi=kernel_grid(Q_hi,alpha)
    results=[]
    for M in M_values:
        C_M=kernel_grid(M,alpha)
        e_lo=spectral_norm(C_M,C_lo,N,alpha)
        e_hi=spectral_norm(C_M,C_hi,N,alpha)
        results.append(dict(alpha=alpha,N=N,M=M,Q_lo=Q_lo,Q_hi=Q_hi,
                            error_lowref=e_lo,error_highref=e_hi,
                            reference_sensitivity=abs(e_hi-e_lo),
                            relative_ref_sensitivity=(abs(e_hi-e_lo)/e_hi if e_hi else 0)))
        print(f'alpha={alpha} N={N} M={M} refQ {Q_lo}->{Q_hi}: '
              f'{e_lo:.7g} -> {e_hi:.7g}, refdiff={abs(e_hi-e_lo):.3g}',flush=True)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['quick','extended'],default='quick')
    parser.add_argument('--output',default='results/3d_reference_convergence.csv')
    args=parser.parse_args()
    configs=([(8,[16,32],64,128),(16,[32,64],128,256),
             (32,[64,128],192,256)] if args.mode=='quick' else
             [(8,[16,32],64,128),(16,[32,64],128,256),
              (32,[64,128],192,256),(64,[128],256,384)])
    rows=[]
    for alpha in [0.5,1.,1.5]:
        for N,Mvals,Qlo,Qhi in configs:
            rows.extend(experiment(N,alpha,Mvals,Qlo,Qhi))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Saved {out}')
if __name__=='__main__':main()
