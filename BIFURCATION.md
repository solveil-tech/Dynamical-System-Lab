# Dynamical System Lab — scalar bifurcation page

The pre-feature version is saved on `main` and GitHub as commit `9240613`.
The new Bifurcation tab has independent equations and settings; the phase-flow
equations, calculators, saved experiment format and 2D/3D controls retain their
existing behavior. Existing experiment files do not include this tab's settings.

## Usage

1. Enter `f(x, μ)` for `dx/dt = f(x, μ)`. `mu` is an alias for `μ`.
   Numeric constants, implicit multiplication, powers and the existing parser's
   mathematical functions are supported. Additional free parameters must be
   replaced by numeric values in this first version.
   Nonsmooth absolute-value, sign, floor and piecewise systems are explicitly
   excluded from this first classical bifurcation analysis; the original phase
   portrait's absolute-value support is unaffected.
2. Set the μ and x ranges. Analyze draws `f = 0`, an optional `f_x = 0`
   locus, and the isolated intersections. Manual zoom synchronizes the range
   controls and analyzes the visible region after a short pause. Views within
   completed or pending coverage reuse that search; expansion searches the union
   of the previous coverage and the new view. New results preserve the zoom.
   Reset view returns to the last manually entered range.
3. Blue solid branches are attracting; red dashed branches are repelling.
   Orange marks semistable or inconclusive branches. Orange points mark
   nonhyperbolic candidates. A nonhyperbolic point need not be a bifurcation.
4. Click the diagram or enter `(μ₀, x₀)`, choose increasing/decreasing μ and
   Trace sweep. Round trip reverses at the chosen range endpoint and returns
   to μ₀. Purple denotes the outward leg; green denotes the return leg.

The color key, mathematical conclusions and status messages share a compact,
fixed-height scroll area with the same scrollbar styling as Flow. Long analysis
text scrolls there without reducing the graph. The hidden Bifurcation page does
not increase the Flow workspace's minimum window size.

## Classification and stability

The solver uses the existing restricted AST, not evaluation of user Python.
For polynomial/rational functions, it solves the numerator equations `f=f_x=0`
algebraically where supported and checks the original denominator constraints.
Nonpolynomial models also use a bounded multistart Newton search. Such searches
are incomplete; numerical candidates without exact verification remain labelled
unconfirmed. Non-isolated nonhyperbolic sets are reported separately.

At a critical point:

- Saddle-node requires `f_μ != 0` and `f_xx != 0`.
- Transcritical requires `f_μ = 0`, `f_xx != 0` and negative determinant of
  the Hessian in `(x, μ)` (two transverse equilibrium branches).
- Pitchfork classification additionally requires a certified exact smooth
  primary equilibrium branch, `f_xx = 0`, `f_xμ != 0` and `f_xxx != 0`.
  Negative cubic coefficient gives supercritical; positive gives subcritical,
  with the parameter oriented toward loss of stability of the primary branch.
- Other cases remain Other / degenerate; nonhyperbolicity alone does not
  establish bifurcation or its type.

At fixed μ, the first nonzero x derivative through order eight determines local
scalar stability: odd negative is attracting, odd positive repelling, even is
semistable. Unresolved higher-order or nonsmooth cases remain inconclusive.
Branches are contour samples, not exact global continuation. Dense oscillations,
closely spaced roots and nonpolynomial tangencies can be missed.

## Sweeps, jumps and reversibility

The page uses **Quasistatic** scanning only; there is no mode selector.
It is a phase-line, adiabatic branch-following approximation, not an
integration at a specified nonzero speed. At each sampled μ, it follows the
scalar flow direction to the reachable equilibrium. Dashed vertical connectors
represent initial fast relaxation and loss-of-branch jumps. They do not assert
that a finite-rate ODE solution is discontinuous.

The sweep uses a fixed positive perturbation of 0.001 when the state is exactly
on a repelling equilibrium, retaining the previous default. This setting has no
input field. At symmetric pitchforks its sign selects the positive arm.

Round-trip reversibility compares the entire sampled outward and return
branches, with tolerance 1% of the x window (minimum 0.0001). Different
attracting branches are reported as hysteresis **on the selected sweep**.
Escape, domain boundaries or non-isolated equilibrium slices give undetermined,
not an irreversible conclusion. This is not a proof of global reversibility.

Analysis and sweeps run in cancellable processes with 20-second calculation
budgets. New formulas cancel old jobs; late results cannot overwrite new input.

## References

- Stephen Wiggins, *Ordinary Differential Equations*, section 8.1:
  https://math.libretexts.org/Bookshelves/Differential_Equations/Ordinary_Differential_Equations_(Wiggins)/08%3A_8._Bifurcation_of_Equilibria_I/8.01%3A_Bifurcation_of_Equilibria_I
- Jeffrey R. Chasnov, *Applied Linear Algebra and Differential Equations*, section 11.2:
  https://math.libretexts.org/Bookshelves/Differential_Equations/Applied_Linear_Algebra_and_Differential_Equations_(Chasnov)/03%3A_III._Differential_Equations/11%3A_Nonlinear_Differential_Equations/11.02%3A_Bifurcation_Theory
