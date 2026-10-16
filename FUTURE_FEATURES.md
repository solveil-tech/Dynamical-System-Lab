# Deferred course features

Recorded at the user's request on 2026-10-16. These are a backlog, not implemented
features or authorization to begin implementing them. The current task covers bug
fixes and interface improvements only.

## Suggested order

1. Nullclines: draw dx/dt = 0 and dy/dt = 0, optionally show signs in each region.
   Nullclines must not be described as trajectories or invariant sets in general.
2. Extend the initial-value/time display with trajectory comparison and linked
   selection between phase-space and time plots.
3. Genuine 1D mode: phase line, f(x), equilibria and scalar stability. dy/dt = 0
   means y is constant; y = 0 is only a selected slice, not an automatic reduction.
4. Equilibrium bifurcation diagrams: parameter on the horizontal axis, equilibrium
   coordinate on the vertical axis, stable/unstable branches distinguished.
   Use continuation/root solving to include unstable branches; parameter sweeps of
   long-time trajectories alone do not produce all equilibrium branches.
5. Equilibrium inspection: analytic Jacobian, eigenvectors, stable/unstable
   directions, and a 2D trace-determinant diagram.
6. Periodic-orbit analysis: periods, amplitudes, oriented Poincare sections and
   return maps with interpolated event crossings.

## Mathematical biology presets and tools

- Ecology: logistic growth with harvesting, Allee effect, competition, and
  Rosenzweig-MacArthur predator-prey dynamics with saturating predation.
- Epidemiology: SIS and SIR, peak infections, model-specific R0, and checks of
  population conservation where the equations imply it. Full SEIR has four state
  variables unless a justified conservation reduction is applied.
- Biochemistry/neuroscience: Michaelis-Menten, Hill-type switches and
  FitzHugh-Nagumo; examine fast/slow dynamics, bistability and excitability.
- Allow biological state labels and meaningful nonnegative domains. Check boundary
  invariance and explain negative populations; do not silently clamp them to zero.

## Later computational modes

- Discrete maps (e.g. logistic map).
- Delay differential equations.
- Reaction-diffusion systems, travelling waves and Turing patterns.
- Stochastic population/reaction models.

These require distinct numerical methods, not additional 2D/3D ODE presets.
Do not revive the removed automatic Lyapunov-function, basin-of-attraction or
"all invariant sets" features without a new explicit request and a precise scope.
The user's course syllabi have not been supplied; priorities can be adjusted when
those become available.
