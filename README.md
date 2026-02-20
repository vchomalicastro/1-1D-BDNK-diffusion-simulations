# (1+1)D BDNK Diffusion Simulations
SA-PINN-ACTO (physics-informed neural network) and KT (Kurganov-Tadmor) simulations of BDNK diffusion in (1+1)D, as used in [**Solving BDNK diffusion using physics-informed neural networks**](https://doi.org/10.48550/arXiv.2602.16117) (Chomalí-Castro, Clarisse, Mullins, Noronha, 2026).

---

## Overview

This repository contains Python code for simulating BDNK diffusion both with the SA-PINN-ACTO technique and with a finite-volume Kurganov-Tadmor scheme. The SA-PINN-ACTO is a variant of the vanilla physics-informed neural network, and in particular, a variant of the self-adaptive PINN (SA-PINN). These codes generate all the results in section **IV. Numerical results for BDNK diffusion** of the paper, as well as of **Appendix B: First- vs. second-order formulation of the BDNK diffusion problem for the SA-PINN-ACTO** and **Appendix D: Results of convergence tests**.

## Paper

- **Title:** Solving BDNK diffusion using physics-informed neural networks
- **Authors:** Vicente Chomalí-Castro, Nick Clarisse, Nicki Mullins, Jorge Noronha
- **Year:** 2026
- **DOI:** [10.48550/arXiv.2602.16117](https://doi.org/10.48550/arXiv.2602.16117)

## Repository Structure

### `(1+1)D BDNK Diffusion - Kurganov-Tadmor/`

- **`BDNKProblem - KT.ipynb`**  
  Jupyter Notebook to run KT (Kurganov–Tadmor) finite-volume solver for (1+1)D BDNK diffusion and visualize results.

- **`Kurganov-Tadmor - Data and figures/`**  
  Precomputed KT outputs and plots (these are the figures in the paper):
  - `test1a/`, `test1b/`, `test2a/`, `test2b/`, `test3a/`, `test3b/`  
    Each test folder contains `kt_data.npz` and plots (e.g., `*_n.png`, `*_J0.png`) plus `*_convergence.png`.
    `a` and `b` correspond to the small (`c_ch=0.5`) and large (`c_ch=0.9`) characteristic velocities.

- **`BDNK Background Simulations/`**  
  Background-field data used for the BDNK background (third) setup in the paper, stored as NumPy arrays:
  - `ep(t,x).npy`, `v(t,x).npy`  
  Folder names encode the simulation parameters. See [Phys. Rev. D (DOI: 10.1103/f8y1-3yck)](https://doi.org/10.1103/f8y1-3yck).

---

### `(1+1)D BDNK Diffusion - SA-PINN-ACTO/`

- **`BDNKProblem - SA-PINN-ACTO.ipynb`**  
  Jupyter Notebook to run SA-PINN-ACTO for the first-order (coupled-ODE) formulation.

- **`SA_PINN_ACTO.py`, `BDNK_Functions.py`, `IC_1D.py`, `Plotting.py`**  
  Core Python modules used by the SA-PINN-ACTO notebooks (network setup, initial conditions, BDNK-relevant functions, plotting).

- **`SA-PINN-ACTO - Data and figures/`**  
  Precomputed SA-PINN-ACTO outputs and plots (these are the figures in the paper):
  - `test1a/`, `test1b/`, `test2a/`, `test2b/`, `test3a/`, `test3b/`  
    Each test folder contains `pinn_data.npz`, `L2_results.txt`, and plots:
    `*_n.png`, `*_J0.png`, `*_loss.png`, `*_residuals.png`, `*_collocation-points.png`, `*_charge-conservation.png`.
    `a` and `b` correspond to the small (`c_ch=0.5`) and large (`c_ch=0.9`) characteristic velocities.

  Also includes:
  - `test1a (Appendix B - Second-Order)/`  
    Script and precomputed outputs/plots for the second-order PINN formulation (Appendix B).

- **`Appendix B - Second-Order SA-PINN-ACTO/`**  
  Jupyter Notebook and core Python modules for the Second-order SA-PINN-ACTO implementation (Appendix B):
  - `BDNKProblem - SA-PINN-ACTO.ipynb`
  - `SA_PINN_ACTO.py`, `BDNK_Functions.py`, `IC_1D.py`, `Plotting.py`

- **`BDNK Background Simulations/`**  
  Background-field data used for the BDNK background (third) setup in the paper, stored as NumPy arrays:
  - `ep(t,x).npy`, `v(t,x).npy`
  Folder names encode the simulation parameters. See [Phys. Rev. D (DOI: 10.1103/f8y1-3yck)](https://doi.org/10.1103/f8y1-3yck).

---

### `Relative L2 Error - SA-PINN-ACTO vs. KT/`

- **`L2.ipynb`**  
  Jupyter Notebook to compute the relative L2 error between KT and SA-PINN-ACTO solutions from saved `.npz` outputs.  
  This includes:
  - The spacetime-integrated L2 norms reported in Table I of the paper;
  - The full pointwise spacetime field differences between KT and PINN solutions.

- **`test1a/`, `test1b/`, `test2a/`, `test2b/`, `test3a/`, `test3b/`**  
  For each test case, the folder contains:
  - `kt_data.npz`, `pinn_data.npz`
  - `L2_results.txt` — relative L2 norms (Table I values)
  - `Field_differences.png` — pointwise spacetime difference heatmaps
  - `PINN_vs_KT.png` — direct solution comparisons

- **`test1a (Appendix B - Second-Order)/`**  
  Same as above, but for the second-order PINN formulation (Appendix B).

---

### Root

- **`README.md`** — This file.  
- **`LICENSE`** — MIT license.

---

## Reproducibility

All figures in the paper can be reproduced by running the corresponding Jupyter notebooks in each directory. Precomputed outputs are provided for convenience and to ensure exact reproducibility of published results.

---

## Citation

If you use this code in your research, please cite:

Chomalí-Castro, V., Clarisse, N., Mullins, N., & Noronha, J. (2026).  
*Solving BDNK diffusion using physics-informed neural networks*.  
DOI: [10.48550/arXiv.2602.16117](https://doi.org/10.48550/arXiv.2602.16117)

BibTeX:
```bibtex
@misc{chomali2026bdnk,
  title        = {Solving BDNK diffusion using physics-informed neural networks},
  author       = {Chomal{\'i}-Castro, Vicente and Clarisse, Nick and Mullins, Nicki and Noronha, Jorge},
  year         = {2026},
  eprint       = {2602.16117},
  archivePrefix= {arXiv},
  primaryClass = {nucl-th},
  doi          = {10.48550/arXiv.2602.16117}
}
```
