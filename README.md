# Dynamical System Lab

A native desktop workspace for exploring differential equations: 2D phase
portraits, 3D trajectories, equilibrium stability, nullclines and scalar
bifurcation diagrams.

## Run on Windows

Install Python and the packages in `requirements.txt`, then double-click
`run_dynamical_system_lab.bat`. The Python entry point is
`dynamical_system_lab.py`.

Dark mode is beside the application title. The upper-right Flow / Bifurcation
tabs switch workspaces. Bifurcation hides the Flow workspace's three navigation
tabs and its action buttons, and disables the left flow controls at 50% opacity.
Flow restores those controls and
opens Phase portrait, keeping the previous equations and parameters.
Phase-plot zoom buttons have a separate control row. A fixed-height speed-spectrum
slot is reserved even when Speed colors is off, so toggling it does not move or
resize the plot. During window resizing the cached background retains its aspect
ratio and alignment with particles until the native-resolution redraw arrives.
The phase axes are horizontally centered with equal left/right margins. The
window's outer aspect ratio (including its title bar) is locked to the minimum
Flow workspace size measured at startup. Edge/corner resizing preserves this
ratio; maximize fits the same proportion inside the screen's available area.
See [BIFURCATION.md](BIFURCATION.md) for
the scalar analysis and parameter-sweep controls.

New experiment files use the Dynamical System Lab name. Experiment files saved
by earlier ODE Flow Lab versions remain supported.
