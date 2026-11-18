"""Chapter 2–4 calculations for planar vector fields."""
import ast
import copy
from itertools import product
from math import comb

import numpy as np
import sympy as sp

from equilibrium_algebra import VARIABLES, symbolic_expression
from flow_tools import integrate_trajectory


def _polynomial_sign(expression, bounds):
    symbols = expression.free_symbols
    # The common univariate quadratic case admits exact extrema, even
    # when a Bernstein bound straddles the minimum on a wide interval.
    if len(symbols) == 1 and (variable := next(iter(symbols))) in VARIABLES:
        try:
            poly = sp.Poly(expression, variable)
            if poly.degree() <= 2:
                axis = 2*VARIABLES.index(variable)
                a, b = (sp.Rational(str(x)) for x in bounds[axis:axis+2])
                values = [poly.eval(a), poly.eval(b)]
                if poly.degree() == 2:
                    extremum = -poly.nth(1)/(2*poly.nth(2))
                    if (extremum-a).is_positive and (b-extremum).is_positive:
                        values.append(poly.eval(extremum))
                if all(value.is_nonnegative is True for value in values):
                    return 1
                if all(value.is_nonpositive is True for value in values):
                    return -1
        except sp.PolynomialError:
            pass
    return _bernstein_sign(expression, bounds)


def _bernstein_sign(expression, bounds):
    """Certify a weak sign on a rectangle using exact Bernstein coefficients."""
    x, y = VARIABLES
    a, b, c, d = map(lambda value: sp.Rational(str(value)), bounds)
    try:
        poly = sp.Poly(expression.subs({x: a+(b-a)*x, y: c+(d-c)*y}, simultaneous=True), x, y)
    except sp.PolynomialError:
        return None
    if poly.is_zero:
        return 0
    m, n = poly.degree(x), poly.degree(y)
    if m+n > 16:
        return None
    coefficients = []
    for i in range(m+1):
        for j in range(n+1):
            coefficients.append(sum(value*sp.Rational(comb(i, k)*comb(j, l), comb(m, k)*comb(n, l))
                                    for (k, l), value in poly.terms() if k <= i and l <= j))
    if all(value.is_nonnegative is True for value in coefficients):
        return 1
    if all(value.is_nonpositive is True for value in coefficients):
        return -1
    # A translated square can have negative Bernstein coefficients across
    # its minimum. Even multiplicities have a known sign in factored form.
    scalar, factors = sp.factor_list(poly.as_expr(), x, y)
    if factors and all(power % 2 == 0 for _, power in factors):
        return 1 if scalar.is_positive else -1 if scalar.is_negative else None
    return None


def _rational_smooth(tree, params, bounds):
    """Check the original expression, including denominators before cancellation."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            return False
        if isinstance(node, ast.BinOp):
            guard = None
            if isinstance(node.op, ast.Div):
                guard = node.right
            elif isinstance(node.op, ast.Pow):
                exponent = symbolic_expression(node.right, params)
                if exponent.is_integer is not True or exponent.free_symbols:
                    return False
                if exponent < 0:
                    guard = node.left
            if guard is not None:
                numerator = sp.fraction(sp.cancel(symbolic_expression(guard, params)))[0]
                # A weak sign after a tiny exact shift implies a strict sign.
                epsilon = sp.Rational(1, 10**12)
                if (_polynomial_sign(numerator-epsilon, bounds) != 1
                        and _polynomial_sign(numerator+epsilon, bounds) != -1):
                    return False
    return True


def _boundary_index(fields, params, bounds):
    from dynamical_system_lab import evaluate, _equilibrium_domain_guards
    a, b, c, d = bounds
    results = []
    guards = _equilibrium_domain_guards(fields, params)
    for count in (256, 1024, 4096):
        t = np.linspace(0, 1, count, endpoint=False)
        x = np.concatenate((a+(b-a)*t, np.full_like(t, b), b-(b-a)*t, np.full_like(t, a)))
        y = np.concatenate((np.full_like(t, c), c+(d-c)*t, np.full_like(t, d), d-(d-c)*t))
        for guard, positive in guards:
            values = evaluate(guard, x, y, params)
            if (not np.isfinite(values).all() or np.any(values == 0)
                    or np.any(np.signbit(values) != np.signbit(np.roll(values, -1)))
                    or positive and np.any(values < 0)):
                return None
        u, v = (np.broadcast_to(evaluate(field, x, y, params), x.shape) for field in fields)
        length = np.hypot(u, v)
        if not np.isfinite(length).all() or length.min() <= 1e-9*length.max():
            return None
        unit = (u+1j*v)/length
        increments = np.angle(np.roll(unit, -1)*unit.conj())
        if np.max(abs(increments)) > np.pi/2:
            results.clear()
        else:
            results.append(round(np.sum(increments)/(2*np.pi)))
    return results[-1] if len(results) >= 2 and results[-1] == results[-2] else None


def planar_criteria(data):
    from dynamical_system_lab import compile_formula, evaluate
    fields, params, bounds = data["fields"], data["params"], data["bounds"]
    a, b, c, d = bounds
    if not a < b or not c < d:
        raise ValueError("Each minimum must be smaller than its maximum.")
    x, y = VARIABLES
    weight = compile_formula(data["weight"], params)
    f, g, rho = (symbolic_expression(tree, params) for tree in (*fields, weight))
    divergence = sp.factor(sp.diff(rho*f, x)+sp.diff(rho*g, y))
    if divergence.has(sp.DiracDelta, sp.Derivative):
        raise ValueError("Use a differentiable field and Dulac function.")
    xx, yy = np.meshgrid(np.linspace(a, b, 181), np.linspace(c, d, 181))
    with np.errstate(all="ignore"):
        values = np.broadcast_to(np.asarray(sp.lambdify((x, y), divergence, 'numpy')(xx, yy),
                                            dtype=float), xx.shape).copy()
    values[~np.isfinite(values)] = np.nan
    for tree in (*fields, weight):
        values[~np.broadcast_to(np.isfinite(evaluate(tree, xx, yy, params)), xx.shape)] = np.nan
    verified = None
    field_smooth = all(_rational_smooth(tree, params, bounds) for tree in fields)
    if field_smooth and _rational_smooth(weight, params, bounds):
        num, den = sp.fraction(sp.cancel(divergence))
        # The denominator is nonzero on the rectangle by the domain check.
        verified = _polynomial_sign(num*den, bounds)
    index = _boundary_index(fields, params, bounds)
    finite = values[np.isfinite(values)]
    if verified in (-1, 1):
        sign = "nonnegative" if verified == 1 else "nonpositive"
        summary = f"Verified: weighted divergence is {sign} and not identically zero. No periodic orbit lies entirely inside this rectangle."
    elif verified == 0:
        summary = "Weighted divergence is identically zero. The Bendixson–Dulac test is inconclusive."
    elif not finite.size:
        summary = "Weighted divergence is undefined throughout the sampled grid."
    elif finite.min() < 0 < finite.max():
        summary = "The sampled weighted divergence changes sign. The Bendixson–Dulac test does not exclude periodic orbits here."
    else:
        summary = "No sign change on the sampled grid. A constant sign and the domain conditions were not verified."
    curl = sp.factor(sp.diff(g, x)-sp.diff(f, y))
    if curl == 0 and field_smooth:
        summary += "\nVerified: curl is zero on this simply connected region. The field is a gradient and has no periodic orbit there."
    return dict(values=values, formula=sp.latex(divergence), curl=sp.latex(curl),
                index=index, summary=summary, divergence_verified=verified)


def linear_comparison(data):
    from dynamical_system_lab import _field_vector
    fields, params = data["fields"], data["params"]
    point, perturbation, matrix = map(np.asarray, (data["point"], data["offset"], data["jacobian"]))
    paths = []
    for direction in ((1, -1) if data["backward"] else (1,)):
        for name, rhs in (("Nonlinear", lambda state: _field_vector(fields, point+state, params)),
                          ("Linearised", lambda state: matrix @ state)):
            t, offsets, reason = integrate_trajectory(lambda state: direction*rhs(state), perturbation,
                                                       data["duration"], max_step=.03, samples=501)
            paths.append((name, direction*t, offsets, reason))
    return paths


def saddle_series(fields, params, point, directions):
    """Quadratic coefficient in the real eigenbasis of a smooth planar saddle."""
    x, y = VARIABLES
    try:
        field = [symbolic_expression(tree, params) for tree in fields]
        v = np.column_stack([direction for direction, _stable in directions])
        if v.shape != (2, 2):
            return []
        inv = np.linalg.inv(v)
        sub = {x: float(point[0]), y: float(point[1])}
        jacobian = np.array(sp.Matrix(field).jacobian([x, y]).subs(sub), dtype=float)
        eigenvalues = np.diag(inv @ jacobian @ v)
        hessian = [np.array(sp.hessian(f, (x, y)).subs(sub), dtype=float) for f in field]
        series = []
        for i, (axis, stable) in enumerate(directions):
            j = 1-i
            curvature = .5*inv[j] @ [axis @ h @ axis for h in hessian]
            coefficient = curvature/(2*eigenvalues[i]-eigenvalues[j])
            if np.isfinite(coefficient):
                series.append((stable, axis, v[:, j], float(coefficient)))
        return series
    except (KeyError, ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError):
        return []


def _hopf_cubic(expressions, state, parameter, point, mu, matrix, omega):
    """Cubic radial coefficient after reducing the linear part to rotation."""
    rotation = matrix - np.trace(matrix)*np.eye(2)/2
    axis = np.eye(2)[int(np.linalg.norm(rotation[:, 1]) > np.linalg.norm(rotation[:, 0]))]
    basis = np.column_stack((axis, rotation @ axis / omega))
    xx, yy = state
    local = sp.Matrix(np.linalg.inv(basis)) @ sp.Matrix(expressions).subs(
        {xx: point[0]+basis[0, 0]*xx+basis[0, 1]*yy,
         yy: point[1]+basis[1, 0]*xx+basis[1, 1]*yy, parameter: mu}, simultaneous=True)
    origin = {xx: 0, yy: 0}

    def derivatives(order):
        return [dict((indices, float(sp.diff(field, *(state[i] for i in indices)).subs(origin)))
                     for indices in product(range(2), repeat=order)) for field in local]

    f, g = derivatives(2)
    f3, g3 = derivatives(3)
    fxx, fxy, fyy = f[0, 0], f[0, 1], f[1, 1]
    gxx, gxy, gyy = g[0, 0], g[0, 1], g[1, 1]
    cubic = f3[0, 0, 0]+f3[0, 1, 1]+g3[0, 0, 1]+g3[1, 1, 1]
    quadratic = (fxy*(fxx+fyy)-gxy*(gxx+gyy)-fxx*gxx+fyy*gyy)/omega
    return (cubic+quadratic)/16, basis, (abs(cubic)+abs(quadratic))/16


def hopf_scan(data):
    """Sample one local planar equilibrium branch and bracket Hopf crossings."""
    x, y = state = VARIABLES
    parameter = sp.Symbol('mu', real=True)
    name = data["parameter"]
    params = dict(data["params"])
    reference = params.pop(name)
    a, b = data["bounds"]
    if not a < b or not a <= reference <= b:
        raise ValueError("Choose a range that contains the current parameter value.")
    expressions = []
    for original in data["fields"]:
        tree = copy.deepcopy(original)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == name:
                node.id = 'z'
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                raise ValueError("This local scan supports polynomial planar systems.")
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Pow)):
                value = symbolic_expression(node.right, params, (x, y, parameter))
                if (value.free_symbols or (isinstance(node.op, ast.Div) and value.is_zero is not False)
                        or (isinstance(node.op, ast.Pow)
                            and (value.is_integer is not True or value.is_nonnegative is not True))):
                    raise ValueError("Use constant nonzero denominators and constant nonnegative integer powers.")
        expressions.append(symbolic_expression(tree, params, (x, y, parameter)))
        if not expressions[-1].is_polynomial(x, y, parameter):
            raise ValueError("This local scan supports polynomial planar systems.")
    if not any(expr.has(parameter) for expr in expressions):
        raise ValueError("The selected parameter does not affect the vector field.")
    field = sp.lambdify((x, y, parameter), expressions, 'numpy', cse=True)
    jacobian = sp.Matrix(expressions).jacobian(state)
    derivative = sp.lambdify((x, y, parameter), jacobian, 'numpy', cse=True)
    parameter_derivative = sp.lambdify((x, y, parameter), sp.Matrix(expressions).diff(parameter), 'numpy')
    trace_gradient = sp.lambdify((x, y, parameter), [sp.diff(sp.trace(jacobian)/2, s)
                                                   for s in (x, y, parameter)], 'numpy')

    def solve(mu, seed):
        point = np.asarray(seed, dtype=float).copy()
        with np.errstate(all='ignore'):
            for _ in range(18):
                f = np.asarray(field(*point, mu), dtype=float)
                matrix = np.asarray(derivative(*point, mu), dtype=float)
                if not np.isfinite(f).all() or not np.isfinite(matrix).all():
                    return None
                try:
                    correction = np.linalg.solve(matrix, f)
                except np.linalg.LinAlgError:
                    return None
                point -= correction
                if np.linalg.norm(correction, ord=np.inf) <= 1e-11*(1+np.linalg.norm(point, ord=np.inf)):
                    return point, matrix, np.trace(matrix)/2, np.linalg.det(matrix)
        return None

    samples = np.unique(np.r_[np.linspace(a, b, 81), reference])
    centre = int(np.searchsorted(samples, reference))
    branch = [None]*len(samples)
    branch[centre] = solve(reference, data["point"])
    if branch[centre] is None:
        raise ValueError("The selected equilibrium has a singular or unresolved Jacobian at the current parameter.")
    for step in (-1, 1):
        for i in range(centre+step, -1 if step < 0 else len(samples), step):
            branch[i] = solve(samples[i], branch[i-step][0])
            if branch[i] is None:
                break

    candidates = []
    for i in range(len(samples)-1):
        left, right = branch[i:i+2]
        if left is None or right is None or left[2]*right[2] > 0 or left[2] == right[2]:
            continue
        low, high = samples[i:i+2]
        initial_width = high-low
        sampled_slope = (right[2]-left[2])/initial_width
        for _ in range(36):
            middle = (low+high)/2
            solved = solve(middle, (left[0]+right[0])/2)
            if solved is None:
                break
            if np.sign(solved[2]) == np.sign(left[2]) and left[2] != 0:
                low, left = middle, solved
            else:
                high, right = middle, solved
        mu, item = (low, left) if abs(left[2]) < abs(right[2]) else (high, right)
        point, matrix, real, determinant = item
        if (determinant <= 0 or abs(real) > 1e-8*max(1, abs(sampled_slope)*initial_width)
                or high-low > initial_width*1e-5):
            continue
        if candidates and abs(candidates[-1]['mu']-mu) < initial_width*1e-5:
            continue
        if real == 0 and (abs(mu-a) < initial_width*1e-5 or abs(mu-b) < initial_width*1e-5):
            continue
        radius = min(initial_width*.05, mu-a, b-mu)
        if radius <= 0:
            continue
        before, after = solve(mu-radius, point), solve(mu+radius, point)
        if before is None or after is None or before[2]*after[2] >= 0:
            continue
        equilibrium_slope = -np.linalg.solve(matrix, np.ravel(parameter_derivative(*point, mu)))
        gradient = np.asarray(trace_gradient(*point, mu), dtype=float)
        slope = float(gradient @ np.r_[equilibrium_slope, 1.])
        if abs(slope) <= 1e-7*max(abs(sampled_slope), np.finfo(float).tiny):
            continue
        omega = np.sqrt(determinant)
        alpha, basis, uncertainty_scale = _hopf_cubic(expressions, state, parameter, point, mu, matrix, omega)
        kind = ('Unresolved cubic coefficient' if abs(alpha) <= 1e-8*max(1., uncertainty_scale)
                else 'Supercritical' if alpha < 0 else 'Subcritical')
        candidates.append(dict(mu=mu, point=point, omega=omega, alpha=alpha, basis=basis,
                               slope=slope, kind=kind))

    real_parts = np.array([np.sort(np.linalg.eigvals(item[1]).real) if item is not None
                           else [np.nan, np.nan] for item in branch])
    points = np.array([item[0] if item is not None else [np.nan, np.nan] for item in branch])
    return dict(samples=samples, real_parts=real_parts, points=points, candidates=candidates,
                complete=all(item is not None for item in branch))
