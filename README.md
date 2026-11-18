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
Hidden actions retain their layout space, so switching workspaces does not move
the sidebar controls or change their spacing. Both workspaces share the same
minimum window size.
Flow restores those controls and
opens Phase portrait, keeping the previous equations and parameters.
The phase plot has a borderless palette wheel at its upper left (grayscale when
off, colorful when on) and zoom buttons at its upper right. A fixed-height speed-spectrum
slot is reserved even when speed colors are off, so toggling it does not move or
resize the plot. During window resizing the cached background retains its aspect
ratio and alignment with particles until the native-resolution redraw arrives.
The phase axes are horizontally centered with equal left/right margins. The
window opens at its minimum layout size and can be resized freely in either
direction. Compact, fixed plot margins keep the title, zoom controls and speed
spectrum close to the frame; extra height stays below the spectrum instead of
opening a gap between it and the plot.
Extra sidebar height is distributed evenly between the tool groups. Inputs,
buttons and the checkbox rows retain their normal heights; shrinking the window
restores the original compact inter-group gaps.
Greek letters restores the previous window height after collapsing an expansion
that automatically enlarged the window. Params shows only its title when no
parameters occur in the equations; detected parameters appear up to four per row,
defaulting to 1. Edit their values directly in those fields.
See [BIFURCATION.md](BIFURCATION.md) for
the scalar analysis and parameter-sweep controls.

New experiment files use the Dynamical System Lab name. Experiment files saved
by earlier ODE Flow Lab versions remain supported.

Travelling-wave analysis is available in Others settings and files, and beside
Trace Fisher front in the Fisher model's Flow page. Enter a smooth reaction
f(u), diffusion D > 0 and trial speed v; other parameters are detected automatically.
The tool shows the transformed phase plane, detected equilibria, signed reaction
areas between selected left/right states, and the minimum admissible Fisher
speed when the reaction is logistic. Use in Flow exports x = u, y = w; the ODE
independent variable then represents the travelling coordinate ξ. Analysis runs
in a cancellable worker. Area-based propagation direction assumes a connecting
front exists; it does not solve the boundary-value problem or determine its speed.
The Budworm bifurcation preset varies R = μ with fixed B = 10.

**Planar criteria** in Phase portrait checks the displayed rectangle or a
region selected on the divergence map. It reports Bendixson–Dulac and curl
tests separately from a sampled estimate of the boundary's Poincaré index.
The equilibrium inspector can compare linearised and nonlinear trajectories
from the same perturbation, and overlays quadratic approximations in its saddle
manifold view. **Scan for Hopf** follows the selected equilibrium of a planar
polynomial system while varying a parameter. The eigenvalues are sampled and the
reported amplitude curves are local normal-form approximations.

In Bifurcation, **Experiments** opens either an additive perturbation δ of the
scalar equation or the planar cubic Hopf normal form. The scalar perturbation
applies to the diagram, phase line and parameter sweep together.
