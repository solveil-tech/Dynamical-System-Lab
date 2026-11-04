"""Scalar autonomous bifurcations and explicitly specified parameter sweeps.

User expressions pass through the existing restricted AST parser. All symbolic
work and sampling run in an isolated, cancellable analysis process.
"""
import ast
import re
import numpy as np
import sympy as sp

X, MU = sp.symbols("x mu", real=True)


def scalar_model(source):
    from ode_flow import compile_formula, FUNCTIONS
    from equilibrium_algebra import symbolic_expression
    source = re.sub(r"\bmu\b", "μ", source)
    tree = compile_formula(source, {"μ"})
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in {"x", "μ", "pi", "e"} | set(FUNCTIONS):
            raise ValueError("Use only x and μ (or mu); enter other constants as numbers.")
        if isinstance(node, ast.Name) and node.id == "μ":
            node.id = "y"
    expression = symbolic_expression(tree, {}, (X, MU))
    guards = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            guards.append(symbolic_expression(node.right, {}, (X, MU)))
    if expression.has(sp.Derivative, sp.Integral):
        raise ValueError("This function is outside the supported scalar model subset.")
    return expression, guards


def numeric(expression):
    return sp.lambdify((X, MU), expression, modules="numpy", cse=True)


def grid_value(fn, x, mu):
    with np.errstate(all="ignore"):
        return np.array(np.broadcast_to(np.asarray(fn(x, mu), dtype=float), np.broadcast(x, mu).shape), copy=True)


def pretty(value):
    number = float(value)
    return "0" if abs(number) < 1e-9 else f"{number:.6g}"


def stability(expression, point):
    for order in range(1, 9):
        coefficient = sp.simplify(sp.diff(expression, X, order).subs(point))
        if coefficient.is_zero is True:
            continue
        if coefficient.is_real is not True or coefficient.is_finite is not True:
            return "Inconclusive (not smooth at this point)"
        if coefficient.is_positive is True or coefficient.is_negative is True:
            if order % 2 == 0:
                return "Semistable (attracting from one side)"
            return "Asymptotically stable" if coefficient.is_negative else "Unstable"
        break
    return "Inconclusive (higher-order or non-isolated equilibrium)"


def classify(expression, xx, mm, factors):
    point = {X: xx, MU: mm}
    d = lambda nx, nm: sp.simplify(sp.diff(expression, X, nx, MU, nm).subs(point))
    fm, fxx, fxm, fmm, fxxx = d(0, 1), d(2, 0), d(1, 1), d(0, 2), d(3, 0)
    nonzero = lambda v: v.is_real is True and v.is_finite is True and v.is_zero is False
    if nonzero(fm) and nonzero(fxx):
        return "Saddle-node"
    determinant = sp.simplify(fxx*fmm-fxm**2)
    if fm == 0 and nonzero(fxx) and determinant.is_negative is True:
        return "Transcritical"
    if fm == 0 and fxx == 0 and nonzero(fxm) and nonzero(fxxx):
        # A primary equilibrium branch is necessary: a cubic jet alone does
        # not establish a pitchfork. Certify an exact smooth factor branch.
        for factor in factors:
            if factor.is_polynomial(X) and sp.degree(factor, X) == 1:
                poly = sp.Poly(factor, X)
                branch = -poly.nth(0)/poly.nth(1)
                if (sp.simplify(branch.subs(MU, mm)-xx) == 0
                        and branch.subs(MU, mm).is_finite is True
                        and sp.simplify(expression.subs(X, branch)) == 0):
                    return "Supercritical pitchfork" if fxxx.is_negative else "Subcritical pitchfork"
    return "Other / degenerate (type not certified)"


def real_point(values, bounds, expression, derivative, guards):
    try:
        xx, mm = values
        if xx.free_symbols or mm.free_symbols:
            return None
        x, mu = complex(xx.evalf()), complex(mm.evalf())
        if abs(x.imag) > 1e-9 or abs(mu.imag) > 1e-9:
            return None
        x, mu = float(x.real), float(mu.real)
        m0, m1, x0, x1 = bounds
        if not (x0-1e-9 <= x <= x1+1e-9 and m0-1e-9 <= mu <= m1+1e-9):
            return None
        point = {X: xx, MU: mm}
        if any(sp.simplify(g.subs(point)).is_zero is not False for g in guards):
            return None
        for f in (expression, derivative):
            value = complex(f.subs(point).evalf())
            if not np.isfinite(value) or abs(value) > 1e-7:
                return None
        return xx, mm
    except (ValueError, TypeError, OverflowError):
        return None


def factors_of(expression):
    numerator = sp.fraction(sp.together(expression))[0]
    try:
        return [factor for factor, power in sp.factor_list(numerator, X, MU)[1]] or [numerator]
    except sp.PolynomialError:
        return [numerator]


def numerical_critical_points(expression, derivative, bounds):
    """Bounded multistart Newton search; candidates are verified afterwards."""
    m0, m1, x0, x1 = bounds
    ms, xs = np.meshgrid(np.linspace(m0, m1, 29), np.linspace(x0, x1, 23))
    xs, ms = xs.ravel(), ms.ravel()
    f, fx = numeric(expression), numeric(derivative)
    fm = numeric(sp.diff(expression, MU))
    fxx, fxm = numeric(sp.diff(derivative, X)), numeric(sp.diff(derivative, MU))
    for _ in range(55):
        with np.errstate(all="ignore"):
            a, b = grid_value(f, xs, ms), grid_value(fx, xs, ms)
            u, v, w = grid_value(fm, xs, ms), grid_value(fxx, xs, ms), grid_value(fxm, xs, ms)
            det = b*w-u*v
            dx, dm = (a*w-u*b)/det, (b*b-v*a)/det
            valid = np.isfinite(dx) & np.isfinite(dm)
            scale = np.maximum(1, np.maximum(abs(dx)/(0.2*(x1-x0)), abs(dm)/(0.2*(m1-m0))))
            xs[valid] -= dx[valid]/scale[valid]
            ms[valid] -= dm[valid]/scale[valid]
            xs = np.clip(xs, x0, x1)
            ms = np.clip(ms, m0, m1)
    a, b = grid_value(f, xs, ms), grid_value(fx, xs, ms)
    good = np.isfinite(a) & np.isfinite(b) & (abs(a) < 1e-9) & (abs(b) < 1e-9)
    points = []
    for xx, mm in zip(xs[good], ms[good]):
        if any(abs(xx-x) < (x1-x0)*1e-5 and abs(mm-m) < (m1-m0)*1e-5 for x, m in points):
            continue
        points.append((xx, mm))
    result = []
    for xx, mm in points:
        # Recover familiar exact values only after verifying both identities.
        exact = (sp.nsimplify(xx, [sp.pi], tolerance=1e-10), sp.nsimplify(mm, [sp.pi], tolerance=1e-10))
        sub = {X: exact[0], MU: exact[1]}
        if sp.simplify(expression.subs(sub)) == 0 and sp.simplify(derivative.subs(sub)) == 0:
            result.append(exact)
        else:
            result.append((sp.Float(xx), sp.Float(mm)))
    return result


def contours(expression, bounds, guards, resolution=501):
    from matplotlib.figure import Figure
    from matplotlib.path import Path
    m0, m1, x0, x1 = bounds
    ms, xs = np.meshgrid(np.linspace(m0, m1, resolution), np.linspace(x0, x1, resolution))
    figure = Figure()
    ax = figure.add_subplot(111)
    paths = []
    guard_values = [grid_value(numeric(g), xs, ms) for g in guards]
    for factor in factors_of(expression):
        if not factor.has(X, MU):
            continue
        values = grid_value(numeric(factor), xs, ms)
        mask = ~np.isfinite(values)
        for g in guard_values:
            mask |= ~np.isfinite(g) | (g == 0)
        values = np.ma.array(values, mask=mask)
        if values.count() == 0 or values.min() > 0 or values.max() < 0 or values.min() == values.max():
            continue
        plot = ax.contour(ms, xs, values, levels=[0], corner_mask=False)
        for path in plot.get_paths():
            starts = np.flatnonzero(path.codes == Path.MOVETO) if path.codes is not None else [0]
            for start, end in zip(starts, list(starts[1:])+[len(path.vertices)]):
                if end-start > 1:
                    points = path.vertices[start:end].copy()
                    # Never connect a contour across a pole/domain boundary.
                    good = np.ones(len(points)-1, dtype=bool)
                    for guard in guards:
                        a = grid_value(numeric(guard), points[:, 1], points[:, 0])
                        good &= np.isfinite(a[:-1]) & np.isfinite(a[1:]) & (a[:-1]*a[1:] > 0)
                    segments = np.stack((points[:-1], points[1:]), axis=1)
                    paths.extend(segments[good])
    figure.clear()
    return np.asarray(paths, dtype=float).reshape(-1, 2, 2)


def analyze(data):
    expression, guards = scalar_model(data["formula"])
    if expression.has(sp.Abs, sp.sign, sp.floor, sp.ceiling, sp.Min, sp.Max, sp.Piecewise):
        raise ValueError("This first version classifies smooth scalar models. Nonsmooth absolute-value, sign, floor and piecewise models require separate bifurcation criteria.")
    derivative = sp.diff(expression, X)
    if derivative.has(sp.Derivative, sp.DiracDelta):
        raise ValueError("Bifurcation classification requires a differentiable f(x, μ).")
    bounds = data["bounds"]
    factors = factors_of(expression)
    candidates, notes = [], []
    numerator = sp.fraction(sp.together(expression))[0]
    dn = sp.fraction(sp.together(derivative))[0]
    exact = False
    if expression == 0:
        notes.append("Every x is an equilibrium for every μ; there is no isolated bifurcation point.")
        exact = True
    elif numerator.is_polynomial(X, MU) and dn.is_polynomial(X, MU):
        common = sp.gcd(numerator, dn)
        if common.has(X, MU):
            notes.append("Non-isolated nonhyperbolic set present; f = fₓ = 0 alone does not certify a bifurcation.")
        else:
            try:
                solutions = sp.solve_poly_system([numerator, dn], X, MU)
                candidates = solutions or []
                exact = True
            except (NotImplementedError, sp.PolynomialError):
                candidates = numerical_critical_points(expression, derivative, bounds)
                notes.append("Algebraic solution incomplete; bounded numerical candidates shown.")
    else:
        try:
            solutions = sp.solve([expression, derivative], (X, MU), dict=True)
            candidates = [(s[X], s[MU]) for s in solutions if X in s and MU in s]
        except (NotImplementedError, ValueError, TypeError):
            pass
        candidates += numerical_critical_points(expression, derivative, bounds)
        notes.append("Non-polynomial search: detected points are not an exhaustive classification.")
    critical, seen = [], []
    for values in candidates:
        point = real_point(values, bounds, expression, derivative, guards)
        if point is None:
            continue
        xx, mm = point
        number = np.array([float(mm), float(xx)])
        if any(np.linalg.norm(number-p) < 1e-8 for p in seen):
            continue
        seen.append(number)
        certified = (sp.simplify(expression.subs({X: xx, MU: mm})) == 0
                     and sp.simplify(derivative.subs({X: xx, MU: mm})) == 0)
        critical.append(dict(mu=float(mm), x=float(xx),
                             mu_text=sp.latex(mm) if not mm.has(sp.Float) else pretty(mm),
                             x_text=sp.latex(xx) if not xx.has(sp.Float) else pretty(xx),
                             kind=classify(expression, xx, mm, factors) if certified else "Numerical nonhyperbolic candidate (type unconfirmed)",
                             stability=stability(expression, {X: xx, MU: mm})))
    if not critical and expression != 0:
        notes.append("No isolated critical point detected inside the selected window.")
    if exact and expression != 0:
        notes.append("Isolated critical points solved algebraically; branch curves are sampled in this window.")
    segments = contours(expression, bounds, guards)
    if len(segments):
        middle = segments.mean(axis=1)
        slopes = grid_value(numeric(derivative), middle[:, 1], middle[:, 0])
        # Classify multiple-root branches via the first nonzero x derivative.
        tolerance = max(float(np.nanmax(abs(slopes)))*1e-9, 1e-300)
        kinds = np.where(slopes < -tolerance, -1, np.where(slopes > tolerance, 1, 0))
        unresolved = kinds == 0
        for order in range(2, 9):
            if not np.any(unresolved):
                break
            d = grid_value(numeric(sp.diff(expression, X, order)), middle[:, 1], middle[:, 0])
            tol = max(float(np.nanmax(abs(d)))*1e-9, 1e-300)
            active = unresolved & np.isfinite(d) & (abs(d) > tol)
            kinds[active] = np.sign(d[active]).astype(int) if order % 2 else 2
            unresolved &= ~active
    else:
        kinds = np.empty(0, dtype=int)
    return dict(segments=segments, stability=kinds, derivative=contours(derivative, bounds, guards),
                critical=critical, notes=notes, formula=data["formula"], bounds=bounds,
                expression_latex=sp.latex(expression).replace("mu", r"\mu"))


def roots_at(expression, mu, low, high):
    """Real roots at a parameter slice, with domain validation."""
    substituted = expression.subs(MU, float(mu))
    if substituted == 0:
        return None
    numerator, denominator = sp.fraction(sp.together(expression))
    try:
        poly = sp.Poly(numerator, X)
        # Factor first so repeated roots are not lost to complex roundoff.
        factors = sp.factor_list(numerator, X)[1]
        result = []
        for factor, _power in factors:
            coefficients = sp.Poly(factor, X).all_coeffs()
            coefficients = [complex(c.subs(MU, float(mu))) for c in coefficients]
            for root in np.roots(np.trim_zeros(coefficients, "f")):
                if abs(root.imag) < 1e-7 and low-1e-8 <= root.real <= high+1e-8:
                    result.append(float(root.real))
    except (sp.PolynomialError, TypeError, ValueError):
        fn = numeric(expression)
        xs = np.linspace(low, high, 1201)
        ys = grid_value(fn, xs, mu)
        result = list(xs[np.isfinite(ys) & (abs(ys) < 1e-10)])
        for i in np.flatnonzero(ys[:-1]*ys[1:] < 0):
            a, b, fa = xs[i], xs[i+1], ys[i]
            for _ in range(42):
                mid = (a+b)/2
                fm = float(fn(mid, mu))
                if fa*fm <= 0:
                    b = mid
                else:
                    a, fa = mid, fm
            result.append((a+b)/2)
    fn, guard = numeric(expression), numeric(denominator)
    output = []
    for root in sorted(result):
        with np.errstate(all="ignore"):
            residual, domain = float(fn(root, mu)), float(guard(root, mu))
        if np.isfinite(residual) and abs(residual) < 1e-6 and np.isfinite(domain) and domain != 0:
            if not output or abs(root-output[-1]) > 1e-7:
                output.append(root)
    return output


def root_sampler(expression, low, high):
    """Compile polynomial coefficients once, rather than on each sweep step."""
    numerator, denominator = sp.fraction(sp.together(expression))
    try:
        prefactor, factors = sp.factor_list(numerator, X)
        coefficients = [sp.lambdify(MU, sp.Poly(f, X).all_coeffs(), "numpy") for f, _ in factors]
    except sp.PolynomialError:
        return lambda mu: roots_at(expression, mu, low, high)
    fn, domain, cache = numeric(expression), numeric(denominator), {}
    prefactor_fn = sp.lambdify(MU, prefactor, "numpy")
    def sample(mu):
        key = round(float(mu), 14)
        if key in cache:
            return cache[key]
        if prefactor_fn(mu) == 0:
            cache[key] = None
            return None
        result = []
        for coeff in coefficients:
            values = np.trim_zeros(np.asarray(coeff(mu), dtype=complex), "f")
            if len(values) == 0:
                cache[key] = None
                return None
            for r in np.roots(values):
                if abs(r.imag) < 1e-7 and low-1e-8 <= r.real <= high+1e-8:
                    r = float(r.real)
                    with np.errstate(all="ignore"):
                        v, d = float(fn(r, mu)), float(domain(r, mu))
                    if np.isfinite(v) and abs(v) < 1e-6 and np.isfinite(d) and d != 0:
                        result.append(r)
        result = sorted(set(round(r, 12) for r in result))
        cache[key] = result
        return result
    return sample


def trace(data):
    expression, guards = scalar_model(data["formula"])
    m0, m1, x0, x1 = data["bounds"]
    start_mu, start_x = data["initial"]
    endpoint = m1 if data["direction"] > 0 else m0
    if abs(endpoint-start_mu) < 1e-9:
        raise ValueError("The initial μ is already at the selected sweep endpoint. Choose the other direction.")
    if data["mode"] == "finite":
        return finite_trace(expression, guards, data, endpoint)
    fn, derivative = numeric(expression), numeric(sp.diff(expression, X))
    # Positive rescaling leaves a quasistatic phase line unchanged. Normalize
    # its amplitude so a small physical time scale is not mistaken for f = 0.
    ms, xs = np.meshgrid(np.linspace(m0, m1, 17), np.linspace(x0, x1, 17))
    amplitude = np.nanmax(abs(grid_value(fn, xs, ms)))
    if np.isfinite(amplitude) and amplitude > 0:
        fn, derivative = numeric(expression/amplitude), numeric(sp.diff(expression, X)/amplitude)
    sample_roots = root_sampler(expression, x0, x1)
    mus = np.linspace(start_mu, endpoint, 401)
    routes = [mus]
    if data["roundtrip"]:
        routes.append(mus[::-1])
    segments, jumps, legs, segment_legs = [], [], [], []
    current, initial_settled = float(start_x), None
    status, invariant = "Completed quasistatic sweep", False
    step_mu = abs(endpoint-start_mu)/400
    for leg_index, route in enumerate(routes):
        path, complete_leg = [], []
        for mm in route:
            roots = sample_roots(mm)
            if roots is None:
                status = "Undetermined: a whole equilibrium slice was encountered"
                break
            value = float(fn(current, mm))
            if not np.isfinite(value) or any(not np.isfinite(float(numeric(g)(current, mm))) or float(numeric(g)(current, mm)) == 0 for g in guards):
                status = "Undetermined: the path reached a domain boundary"
                break
            probe = current
            slope = float(derivative(current, mm))
            if abs(value) < 1e-10 and slope > 1e-8:
                if data["perturbation"] == 0:
                    invariant = True
                else:
                    probe += data["perturbation"]
                    value = float(fn(probe, mm))
            if abs(value) < 1e-10:
                target = current
            elif value > 0:
                reachable = [r for r in roots if r > probe-1e-9]
                target = min(reachable) if reachable else None
            else:
                reachable = [r for r in roots if r < probe+1e-9]
                target = max(reachable) if reachable else None
            if target is None:
                status = "Undetermined: no attracting landing state inside the x window (escape)"
                break
            blocked = False
            for guard in guards:
                values = grid_value(numeric(guard), np.linspace(current, target, 33), mm)
                if (not np.all(np.isfinite(values)) or np.any(values == 0)
                        or np.any(np.signbit(values[:-1]) != np.signbit(values[1:]))):
                    blocked = True
                    break
            if blocked:
                status = "Undetermined: relaxation would cross a domain boundary"
                break
            if initial_settled is None:
                initial_settled = target
                if abs(target-current) > 1e-7:
                    jumps.append([[float(mm), current], [float(mm), target]])
            elif path:
                previous_mu, previous_x = path[-1]
                # A quasistatic branch jump is a finite state change over one
                # small μ increment, not a discontinuity of a finite-rate ODE.
                if abs(target-previous_x) > max((x1-x0)*0.06, 6*np.sqrt(step_mu)*(x1-x0)*0.05):
                    jumps.append([[float(mm), previous_x], [float(mm), target]])
                    segments.append(np.asarray(path))
                    segment_legs.append(leg_index)
                    path = []
            current = float(target)
            path.append((float(mm), current))
            complete_leg.append(current)
        if path:
            segments.append(np.asarray(path))
            segment_legs.append(leg_index)
        legs.append(np.asarray(complete_leg))
        if status.startswith("Undetermined"):
            break
    if status.startswith("Undetermined"):
        reversible = status
    elif invariant:
        reversible = "Reversibility undetermined: the exact unstable invariant branch was retained (perturbation = 0)."
    elif len(routes) < 2:
        reversible = "Reversibility not assessed: enable Round trip."
    else:
        difference = float(np.max(np.abs(legs[0]-legs[1][::-1])))
        if difference > max(1e-4, (x1-x0)*0.01):
            reversible = "Irreversible on this sweep: different attracting branches on outward/return paths (hysteresis)."
        else:
            reversible = "Reversible on this quasistatic sweep: outward and return branches coincide within sampling tolerance."
    return dict(segments=segments, jumps=jumps, initial=(start_mu, start_x), status=status,
                reversibility=reversible, mode="quasistatic", segment_legs=segment_legs)


def finite_trace(expression, guards, data, endpoint):
    from flow_tools import integrate_trajectory
    start_mu, start_x = data["initial"]
    rate = data["rate"]
    duration = abs(endpoint-start_mu)/rate
    fn = numeric(expression)
    paths, current, reason = [], np.array([start_x, start_mu]), "Completed"
    for direction in ([data["direction"], -data["direction"]] if data["roundtrip"] else [data["direction"]]):
        def field(point):
            x, mu = point
            if not data["bounds"][2] <= x <= data["bounds"][3]:
                raise ValueError("Path left the selected x window")
            for guard in guards:
                value = float(numeric(guard)(x, mu))
                if not np.isfinite(value) or value == 0:
                    raise ValueError("Path reached a domain boundary")
            return np.array([float(fn(x, mu)), direction*rate])
        times, points, reason = integrate_trajectory(field, current, duration,
                        max_step=min(0.1, duration/400), samples=1201, time_limit=8,
                        retain_steps=True)
        paths.append(points[:, [1, 0]])
        if times[-1] < duration*(1-1e-7):
            break
        current = points[-1].copy()
    return dict(segments=paths, jumps=[], initial=(start_mu, start_x), status=reason,
                segment_legs=list(range(len(paths))),
                reversibility="Finite-rate paths include rate-dependent lag; they do not establish quasistatic reversibility. Compare using Quasistatic mode.",
                mode="finite")
