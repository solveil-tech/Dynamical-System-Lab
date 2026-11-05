# Dynamical System Lab

A native desktop workspace for exploring differential equations: 2D phase
portraits, 3D trajectories, equilibrium stability, nullclines and scalar
bifurcation diagrams.

## Run on Windows

Install Python and the packages in `requirements.txt`, then double-click
`run_dynamical_system_lab.bat`. The Python entry point is
`dynamical_system_lab.py`.

Dark mode is beside the application title. The upper-right Flow / Bifurcation
tabs switch workspaces. Bifurcation hides the flow action buttons and disables
the left flow controls under a gray mask. Flow restores those controls and
opens Phase portrait, keeping the previous equations and parameters.
See [BIFURCATION.md](BIFURCATION.md) for
the scalar analysis and parameter-sweep controls.

New experiment files use the Dynamical System Lab name. Experiment files saved
by earlier ODE Flow Lab versions remain supported.
