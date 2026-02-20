# (1+1)D BDNK Diffusion Simulations
SA-PINN-ACTO (physics-informed neural network) and KT (Kurganov-Tadmor) simulations to BDNK diffusion in (1+1)D, as used in "Solving BDNK diffusion using physics-informed neural networks" (Chomalí-Castro, Clarisse, Mullins, Noronha, 2026).

---

## Overview

This repository contains Python code for simulating BDNK diffusion both with the SA-PINN-ACTO technique and with a finite-volume Kurganov-Tadmor scheme. The SA-PINN-ACTO is a variant of the vanilla physics-informed neural network, and in particular, a variant of the self-adaptive PINN (SA-PINN). These codes generate all the results in section **IV. Numerical results for BDNK diffusion** of the paper, as well as of **Appendix B: First- vs. second-order formulation of the BDNK diffusion problem for the SA-PINN-ACTO** and **Appendix D: Results of convergence tests**.

## Paper

- **Title:** Solving BDNK diffusion using physics-informed neural networks
- **Authors:** Vicente Chomalí-Castro, Nick Clarisse, Nicki Mullins, Jorge Noronha
- **Year:** 2026
- **DOI:** N/A

## Repository Structure

### `(1+1)D BDNK Diffusion - Kurganov-Tadmor/`

- **`BDNKProblem - KT.ipynb`**  
  Jupyter notebook to run/visualize the KT (Kurganov–Tadmor) finite-volume solver.

- **`Kurganov-Tadmor - Data and figures/`**  
  Precomputed KT outputs and plots for the paper test cases:
  - `test1a/`, `test1b/`, `test2a/`, `test2b/`, `test3a/`, `test3b/`  
    Each test folder contains `kt_data.npz` and plots (e.g., `*_n.png`, `*_J0.png`) plus `*_convergence.png`.

- **`BDNK Background Simulations/`**  
  Background-field data used for the “BDNK background” setup(s), stored as NumPy arrays:
  - `ep(t,x).npy`, `v(t,x).npy`  
  *(Folder names encode the simulation parameters.)*

---

### `(1+1)D BDNK Diffusion - SA-PINN-ACTO/`

- **`BDNKProblem - SA-PINN-ACTO.ipynb`**  
  Main notebook to run SA-PINN-ACTO for the first-order (coupled-ODE) formulation.

- **`SA_PINN_ACTO.py`, `BDNK_Functions.py`, `IC_1D.py`, `Plotting.py`**  
  Core Python modules used by the SA-PINN-ACTO notebooks (training, PDE definitions, initial conditions, plotting).

- **`SA-PINN-ACTO - Data and figures/`**  
  Precomputed SA-PINN-ACTO outputs and plots for the paper test cases:
  - `test1a/`, `test1b/`, `test2a/`, `test2b/`, `test3a/`, `test3b/`  
    Each test folder contains `pinn_data.npz`, `L2_results.txt`, and plots:
    `*_n.png`, `*_J0.png`, `*_loss.png`, `*_residuals.png`, `*_collocation-points.png`, `*_charge-conservation.png`.

  Also includes:
  - `test1a (Appendix B - Second-Order)/`  
    Precomputed outputs/plots for the second-order PINN formulation (Appendix B).

- **`Appendix B - Second-Order SA-PINN-ACTO/`**  
  Second-order PINN implementation + notebook:
  - `BDNKProblem - SA-PINN-ACTO.ipynb`
  - `SA_PINN_ACTO.py`, `BDNK_Functions.py`, `IC_1D.py`, `Plotting.py`

- **`BDNK Background Simulations/`**  
  Background-field NumPy arrays used in the background setups:
  - `ep(t,x).npy`, `v(t,x).npy`

---

### `Relative L2 Error - SA-PINN-ACTO vs. KT/`

- **`L2.ipynb`**  
  Notebook to compute and plot KT vs SA-PINN-ACTO errors from saved `.npz` outputs.

- **`test1a/`, `test1b/`, `test2a/`, `test2b/`, `test3a/`, `test3b/`**  
  Pointwise spacetime KT–PINN comparisons for `n` and `J0`:
  - `kt_data.npz`, `pinn_data.npz`, `L2_results.txt`
  - `Field_differences.png` *(spacetime difference/heatmaps)*
  - `PINN_vs_KT.png` *(direct solution comparisons)*

- **`test1a (Appendix B - Second-Order)/`**  
  Same as above, but for the second-order PINN formulation (Appendix B).

---

### Root

- **`README.md`** — This file.  
- **`LICENSE`** — MIT license.
