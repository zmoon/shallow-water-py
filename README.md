# shallow-water-py
1-D version for PSU class METEO 526 – Numerical Weather Prediction

Features:
* Solvers include:
  * Centered-in-time, centered-in-space (CTCS; leapfrog) 2nd, 4th, 6th order
  * Forward-in-time, centered-in-space (FTCS) 2nd order
  * Forward-in-time, backward-in-space (FTBS) 1st order
  * Runge–Kutta (RK) 3rd and 4th order with 4th order centered-in-space (CS4), + 4th order RK with CS6
  * Naive spectral model (no rotation; only for h)
* Rotation/Coriolis term and diffusion term options

