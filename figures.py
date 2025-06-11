#!/usr/bin/env python3
"""Generate all PDF figures for the Week 2 presentation.

This script creates the following files inside a local `Figures/` folder:

    * LogisticIllustration.pdf   – SI model logistic growth
    * DecayIllustration.pdf     – IR model exponential decay
    * SI_SIS_Timeseries.pdf     – comparison of SI vs SIS models
    * PhasePlane_R0_0.8.pdf     – SIR phase‑plane portrait for R₀=0.8
    * PhasePlane_R0_2.0.pdf     – SIR phase‑plane portrait for R₀=2.0
    * FinalSizeSketch.pdf       – Kermack–McKendrick final‑size implicit curve
    * SIR_Timeseries.pdf        – full SIR time‑series

Dependencies
------------
- numpy ≥ 1.20
- matplotlib ≥ 3.6
- scipy ≥ 1.10

Usage
-----
$ python generate_week2_figures.py

All output is written as vector PDF so the graphics stay crisp when
included in LaTeX.
"""
from __future__ import annotations
import pathlib
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------------
#  Configuration
# ----------------------------------------------------------------------------
OUTPUT_DIR = pathlib.Path("Figures")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 14,
    "figure.figsize": (6.4, 4.0),
    "axes.labelpad": 8,
    "axes.titlesize": 16
})

# ----------------------------------------------------------------------------
#  Figure 1 – Logistic growth (SI model)
# ----------------------------------------------------------------------------

def make_logistic():
    N, I0, beta = 100, 1, 0.42
    t = np.linspace(0, 40, 400)
    I = N / (1 + (N/I0 - 1) * np.exp(-beta * t))

    fig, ax = plt.subplots()
    ax.plot(t, I, lw=2)
    ax.set_xlabel('time (days)')
    ax.set_ylabel(r'$I(t)$')
    ax.set_title('SI model: logistic growth')
    fig.savefig(OUTPUT_DIR / 'LogisticIllustration.pdf', bbox_inches='tight')
    plt.close(fig)

# ----------------------------------------------------------------------------
#  Figure 2 – Exponential decay (IR model)
# ----------------------------------------------------------------------------

def make_decay():
    I0, gamma = 100, 0.25
    t = np.linspace(0, 30, 300)
    I = I0 * np.exp(-gamma * t)

    fig, ax = plt.subplots()
    ax.plot(t, I, lw=2)
    ax.set_xlabel('time (days)')
    ax.set_ylabel(r'$I(t)$')
    ax.set_title('IR model: exponential decay')
    fig.savefig(OUTPUT_DIR / 'DecayIllustration.pdf', bbox_inches='tight')
    plt.close(fig)

# ----------------------------------------------------------------------------
#  Figure 3 – SI vs SIS time‑series comparison
# ----------------------------------------------------------------------------

def make_si_sis_timeseries():
    beta, gamma = 0.42, 0.25
    N = 1000
    I0 = 1
    i0 = I0 / N
    t_end, dt = 60.0, 0.1
    t = np.arange(0, t_end + dt, dt)

    i_SI = np.empty_like(t)
    i_SIS = np.empty_like(t)
    i_SI[0] = i0
    i_SIS[0] = i0

    for k in range(1, len(t)):
        prev = i_SI[k-1]
        di = beta * prev * (1 - prev)
        i_SI[k] = prev + di * dt

        prev = i_SIS[k-1]
        di = beta * prev * (1 - prev) - gamma * prev
        i_SIS[k] = prev + di * dt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, i_SI, label='SI model')
    ax.plot(t, i_SIS, label='SIS model')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Infectious proportion $i(t)$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'SI_SIS_Timeseries.pdf', bbox_inches='tight')
    plt.close(fig)

# ----------------------------------------------------------------------------
#  Helpers for SIR model
# ----------------------------------------------------------------------------

def sir_rhs(t: float, y: Sequence[float], r0: float) -> list[float]:
    s, i = y
    ds_dt = -r0 * s * i
    di_dt = r0 * s * i - i
    return [ds_dt, di_dt]

# ----------------------------------------------------------------------------
#  Figure 4 – Phase‑plane portraits
# ----------------------------------------------------------------------------

def plot_phase_plane(r0: float, fname: str, t_max: float = 60.0) -> None:
    s = np.linspace(0, 1, 25)
    i = np.linspace(0, 1, 25)
    S, I = np.meshgrid(s, i)
    dS = -r0 * S * I
    dI = r0 * S * I - I

    # normalize for visualization
    L = np.hypot(dS, dI)
    L[L == 0] = 1.0
    dS, dI = dS / L, dI / L

    fig, ax = plt.subplots()
    ax.streamplot(S, I, dS, dI, density=1.2, linewidth=0.5, arrowsize=0.8)

    sol = solve_ivp(lambda t, y: sir_rhs(t, y, r0), (0.0, t_max), [0.999, 0.001], rtol=1e-8, atol=1e-10)
    ax.plot(sol.y[0], sol.y[1], 'k--', lw=1.4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('susceptible proportion $s$')
    ax.set_ylabel('infectious proportion $i$')
    ax.set_title(f'Phase‑plane portrait (R₀ = {r0})')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, bbox_inches='tight', dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------------
#  Figure 5 – Final‑size implicit curves
# ----------------------------------------------------------------------------

def plot_final_size(s0: float, r0_values: Sequence[float], fname: str) -> None:
    s_grid = np.linspace(0.05, s0, 500)
    fig, ax = plt.subplots()
    for r0 in r0_values:
        f = np.log(s_grid) + r0 * (1.0 - s_grid) - np.log(s0)
        ax.plot(s_grid, f, label=f'R₀ = {r0}')

    ax.axhline(0.0, color='k', lw=0.8)
    ax.set_xlabel('susceptible proportion $s$')
    ax.set_ylabel('$f(s)$ – root gives $s_\infty$')
    ax.set_title('Kermack–McKendrick final‑size equation')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, bbox_inches='tight', dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------------
#  Figure 6 – Full SIR time‑series
# ----------------------------------------------------------------------------

def plot_sir_timeseries(r0: float, beta: float, gamma: float, fname: str) -> None:
    def rhs(t: float, y: Sequence[float]) -> list[float]:
        s, i, r = y
        return [-beta * s * i, beta * s * i - gamma * i, gamma * i]

    t_eval = np.linspace(0.0, 180.0, 1000)
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [0.999, 0.001, 0.0], t_eval=t_eval, rtol=1e-8, atol=1e-10)

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], label='$s(t)$')
    ax.plot(sol.t, sol.y[1], label='$i(t)$')
    ax.plot(sol.t, sol.y[2], label='$r(t)$')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('proportion of population')
    ax.set_title(f'SIR time‑series (R₀ = {r0:.2f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, bbox_inches='tight', dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------------

def main():
    make_logistic()
    make_decay()
    make_si_sis_timeseries()

    plot_phase_plane(0.8, "PhasePlane_R0_0.8.pdf")
    plot_phase_plane(2.0, "PhasePlane_R0_2.0.pdf")

    plot_final_size(0.999, [1.2, 2.0, 3.0], "FinalSizeSketch.pdf")

    BETA = 0.42  # day⁻¹
    GAMMA = 0.25  # day⁻¹
    R0 = BETA / GAMMA
    plot_sir_timeseries(R0, BETA, GAMMA, "SIR_Timeseries.pdf")

    print(f"✔ Generated figures in '{OUTPUT_DIR.resolve()}'")

if __name__ == '__main__':
    main()
