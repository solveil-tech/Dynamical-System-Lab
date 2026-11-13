"""Reaction-diffusion travelling-wave analysis using the application's safe parser."""
import ast
import math

import numpy as np
import sympy as sp


def reaction_model(source, values):
    """Interpret u as the state and infer other symbols with existing notation rules."""
    from dynamical_system_lab import compile_formula, parameters_for_equations, display_parameter_name
    definitions = ", ".join(f"{display_parameter_name(name)}={value:.17g}"
                            for name, value in values.items())
    params = parameters_for_equations([source], definitions)
    params.pop("u", None)
    if {"D", "v"}.intersection(params):
        raise ValueError("D and v are wave controls. Use a different name for a reaction parameter.")
    tree = compile_formula(source, set(params) | {"u"})
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in {"x", "y", "z"}:
                raise ValueError("Use u for the concentration in f(u).")
            if node.id == "u":
                node.id = "x"
    return ast.fix_missing_locations(tree), params


def flow_equations(tree):
    return "y", f"(-v*y-({ast.unparse(tree)}))/D"


def _number(value):
    return f"{float(value):.7g}"


def analyze(data):
    import dynamical_system_lab as core
    from bifurcation import X, roots_at
    from equilibrium_algebra import symbolic_expression, clean_numerical_point

    diffusion, speed = float(data["D"]), float(data["v"])
    low, high = map(float, data["bounds"])
    if not np.isfinite([diffusion, speed, low, high]).all() or diffusion <= 0 or low >= high:
        raise ValueError("Use D > 0 and u min < u max, with finite values.")
    tree, params = reaction_model(data["reaction"], data["params"])
    expression = symbolic_expression(tree, params, (X,))
    if expression.has(sp.Abs, sp.sign, sp.floor, sp.ceiling, sp.Min, sp.Max):
        raise ValueError("Travelling-wave linearisation requires a smooth reaction function.")
    guards = core._equilibrium_domain_guards((tree,), params)
    roots = roots_at(expression, 0, low, high)
    stationary = roots is None
    if stationary:
        roots = []
    roots = [float(clean_numerical_point((tree,), params, (root,))[0]) for root in roots]
    roots = [root for root in roots if all(
        np.isfinite(value := float(core.evaluate(guard, root, 0, params)))
        and (value > 0 if positive else value != 0) for guard, positive in guards)]

    u = np.linspace(low, high, 1001)
    values = np.broadcast_to(core.evaluate(tree, u, np.zeros_like(u), params), u.shape).copy()
    values[~np.isfinite(values)] = np.nan
    for guard, positive in guards:
        g = np.broadcast_to(core.evaluate(guard, u, 0, params), u.shape)
        values[(g <= 0 if positive else g == 0) | ~np.isfinite(g)] = np.nan
        crosses = np.flatnonzero(np.signbit(g[:-1]) != np.signbit(g[1:]))
        values[crosses] = values[crosses+1] = np.nan
    if not np.isfinite(values).any():
        raise ValueError("f(u) is undefined throughout the displayed interval.")

    u_symbol = sp.Symbol("u", real=True)
    reaction = expression.subs(X, u_symbol)
    rows = [r"$C_t=D C_{xx}+f(C),\quad \xi=x-vt,\quad U'=W,\quad "
            r"W'=\frac{-vW-f(U)}{D}$",
            f"$f(u)={sp.latex(reaction)}$",
            "Equilibrium types below refer to the travelling coordinate ξ."]
    derivative = sp.diff(expression, X)
    fx_text, fy_text = flow_equations(tree)
    wave_params = dict(params, D=diffusion, v=speed)
    fields = tuple(core.compile_formula(text, wave_params) for text in (fx_text, fy_text))
    slopes = []
    for root in roots:
        slope = float(derivative.subs(X, root))
        slopes.append(slope)
        jacobian = np.array([[0, 1], [-slope/diffusion, -speed/diffusion]])
        details = core.classify_equilibrium(fields, np.array([root, 0.]), wave_params, jacobian)
        rows.append(f"$(U,W)=({_number(root)},0),\quad "
                    f"\\lambda=({details['eigenvalues']})$    {details['type']}")
    if not roots:
        rows.append("Every (U,0) is stationary on this reaction's domain." if stationary else
                    "No isolated equilibrium detected in the displayed u interval.")

    left_index = data.get("left_index")
    right_index = data.get("right_index")
    if roots:
        left_index = len(roots)-1 if left_index is None or not 0 <= left_index < len(roots) else left_index
        if right_index is None or not 0 <= right_index < len(roots):
            stable = [i for i, slope in enumerate(slopes) if slope < 0 and i != left_index]
            right_index = stable[0] if stable else 0
    else:
        left_index = right_index = -1
    left = roots[left_index] if roots else None
    right = roots[right_index] if roots else None
    area, direction, area_error = None, None, None
    if left is not None and left != right:
        interval = sp.Interval(min(left, right), max(left, right))
        continuous = sp.calculus.util.continuous_domain(expression, X, interval)
        defined = continuous.is_superset(interval) is True
        for guard, _positive in guards:
            symbolic_guard = symbolic_expression(guard, params, (X,))
            defined &= sp.solveset(symbolic_guard, X, domain=interval).is_empty is True
        if defined:
            # Use two quadrature orders to bound sensitivity before deciding a sign.
            estimates, magnitudes = [], []
            for order in (64, 128):
                nodes, weights = np.polynomial.legendre.leggauss(order)
                samples = (right+left)/2+(left-right)*nodes/2
                f = np.broadcast_to(core.evaluate(tree, samples, 0, params), samples.shape)
                estimates.append(float((left-right)/2*np.dot(weights, f)))
                magnitudes.append(float(abs(left-right)/2*np.dot(weights, abs(f))))
            area = estimates[-1]
            if not np.isfinite(estimates + magnitudes).all():
                raise ValueError("The area integral overflowed. Reduce the u interval or reaction scale.")
            area_error = max(abs(estimates[-1]-estimates[0])*4,
                             64*np.finfo(float).eps*max(magnitudes))
            if abs(area) <= area_error:
                rows.append(r"$A=\int_{U_+}^{U_-}f(u)\,du\approx0$. "
                            "Direction unresolved at integration accuracy; stationary-front candidate.")
                direction = 0
            else:
                direction = 1 if area > 0 else -1
                rows.append(r"$A=\int_{U_+}^{U_-}f(u)\,du\approx " + _number(area) + "$    "
                            + ("Right-moving" if direction > 0 else "Left-moving")
                            + " connecting front predicted.")
            rows.append("Direction assumes a connecting front exists. "
                        "An area sign does not determine its speed.")
        else:
            rows.append("Area unavailable: the reaction is not verified continuous between the selected states.")
    else:
        rows.append("Select two distinct equilibria to compare propagation direction.")

    minimum_speed = None
    try:
        polynomial = sp.Poly(expression, X)
        if polynomial.degree() == 2 and polynomial.nth(0) == 0:
            growth, quadratic = float(polynomial.nth(1)), float(polynomial.nth(2))
            if growth > 0 and quadratic < 0:
                capacity = -growth/quadratic
                minimum_speed = 2*math.sqrt(diffusion*growth)
                rows.append(f"Fisher logistic reaction    $K={_number(capacity)},\quad "
                            f"|v|_{{\\min}}=2\\sqrt{{D r}}={_number(minimum_speed)}$")
                if left is not None and right is not None and sorted((left, right)) == sorted(roots):
                    oriented_speed = speed*np.sign(left-right)
                    if oriented_speed >= minimum_speed*(1-1e-12):
                        rows.append("The selected speed permits a nonnegative monotone Fisher front.")
                    else:
                        rows.append("The selected speed does not permit this nonnegative monotone Fisher front.")
                rows.append("This is the minimum admissible speed; initial data can select faster waves.")
    except sp.PolynomialError:
        pass

    finite = abs(values[np.isfinite(values)])
    scale = max(float(np.percentile(finite, 90))/max(abs(speed), math.sqrt(diffusion)),
                (high-low)*.1, 1e-6)
    grid_u, grid_w = np.meshgrid(np.linspace(low, high, 19), np.linspace(-scale, scale, 15))
    du = grid_w
    dw = (-speed*grid_w-core.evaluate(tree, grid_u, grid_w, params))/diffusion
    length = np.hypot(du/(high-low), dw/(2*scale))
    valid = np.isfinite(length) & (length > 0)
    arrows = (grid_u[valid], grid_w[valid], du[valid]/length[valid], dw[valid]/length[valid])
    return dict(rows=rows, u=u, reaction=values, roots=roots,
                left_index=left_index, right_index=right_index, left=left, right=right,
                area=area, area_error=area_error, direction=direction, minimum_speed=minimum_speed,
                nullcline=-values/speed if speed != 0 else None,
                w_scale=scale, arrows=arrows, params=params, fields=(fx_text, fy_text))
