# Boundary-aware QFT block-encoding of fractional Laplacians: code

Scripts that reproduce the numerical results of

> Y. Javanmard and S. Kazemian, *Boundary-Aware QFT Block-Encoding of Fractional
> Laplacians*, arXiv:2605.16749.

All computations are classical linear algebra in double precision (Python 3.10+,
see `requirements.txt`). Install with `pip install -r requirements.txt`.

| Script | Reproduces | Typical run |
|---|---|---|
| `make_figures.py` | Figs. 4–8 (heatmaps, functional benchmarks, padding decay, Gaussian diagnostics, boundary sweep) | `python make_figures.py --figure all --outdir figs` |
| `padding_1d_quadrature_verified.py` | Table III: 1D residual at M = 2N and register size m for ε = 1e-2, 1e-3, 1e-4 | `python padding_1d_quadrature_verified.py --outdir results` (`--quick` for a short run) |
| `solver_and_normalization.py` | Table IV: inversion perturbation ‖A⁻¹E‖₂; normalization ratio Γ/λ_max of the exact 2N circulant (Sec. IV.A) | `python solver_and_normalization.py` |
| `verify_3d.py` | Table V: 3D residuals with two reference grids | `python verify_3d.py --mode quick --output results/3d.csv` |
| `test_comparator_lcu.py` | Appendix F: exact comparator/LCU block-encoding of the 1D α = 1 symbol | `python test_comparator_lcu.py` |

## Notes

* The 1D Toeplitz coefficients c_r are computed by adaptive quadrature; the padded
  operator independently by an M-point FFT of the sampled symbol.
* The 3D target kernel is computed on a finite reference Fourier grid Q³; the
  script reports the change between two reference grids. These values are
  numerical support, not certified bounds. The quick mode needs a few GB of RAM.
* Gate and qubit counts in Sec. IV.A are analytic model estimates and are not
  produced by these scripts.
* Some scripts were written with the assistance of AI language models; all
  reported values were checked by independent implementations or closed-form
  results (see the Acknowledgments of the paper).
