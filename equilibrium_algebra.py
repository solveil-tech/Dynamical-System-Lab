"""Exact algebraic equilibrium sets, built from the application's safe AST."""
import ast
from functools import lru_cache

import sympy as sp


VARIABLES = sp.symbols("x y", real=True)
_FUNCTIONS = {name: getattr(sp, name) for name in (
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh",
    "tanh", "exp", "log", "sqrt", "floor", "ceiling", "sign")}
_FUNCTIONS.update(ln=sp.log, abs=sp.Abs, ceil=sp.ceiling,
                  minimum=sp.Min, maximum=sp.Max,
                  log10=lambda value: sp.log(value, 10))


def symbolic_expression(node, params, variables=VARIABLES):
    """Do not pass user strings to sympify/parse_expr, which can evaluate code."""
    names = dict(zip(("x", "y", "z"), variables))
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return sp.Rational(str(node.value))
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in params:
            return sp.Rational(str(params[node.id]))
        return {"pi": sp.pi, "e": sp.E}[node.id]
    if isinstance(node, ast.UnaryOp):
        value = symbolic_expression(node.operand, params, variables)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        a = symbolic_expression(node.left, params, variables)
        b = symbolic_expression(node.right, params, variables)
        operations = {ast.Add: lambda: a+b, ast.Sub: lambda: a-b,
                      ast.Mult: lambda: a*b, ast.Div: lambda: a/b,
                      ast.Pow: lambda: a**b}
        if type(node.op) in operations:
            return operations[type(node.op)]()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return _FUNCTIONS[node.func.id](
            *(symbolic_expression(arg, params, variables) for arg in node.args))
    raise ValueError("Expression is outside the symbolic algebra subset")


def _valid_point(point, expressions, guards):
    substitution = dict(zip(VARIABLES, point))
    if any(value.is_real is not True for value in point):
        return False
    for guard in guards:
        value = sp.simplify(guard.subs(substitution))
        if value.is_zero is not False:
            return False
    return all(sp.simplify(field.subs(substitution)) == 0 for field in expressions)


def _quadratic_real_point(factor):
    """A definite quadratic can have an isolated real zero, not a curve."""
    poly = sp.Poly(factor, *VARIABLES)
    if poly.total_degree() != 2:
        return None
    matrix = sp.hessian(factor, VARIABLES) / 2
    if matrix.det().is_positive is not True:
        return None
    gradient0 = sp.Matrix([sp.diff(factor, v).subs(dict.fromkeys(VARIABLES, 0))
                           for v in VARIABLES])
    center = tuple(-matrix.inv() * gradient0 / 2)
    level = sp.simplify(factor.subs(dict(zip(VARIABLES, center))))
    if level == 0:
        return [center]
    if (level * matrix[0, 0]).is_positive:
        return []
    return None


@lru_cache(maxsize=32)
def _analyze(expressions, guards, axis_only):
    x, y = VARIABLES
    equations = (expressions[0].subs(y, 0), y) if axis_only else expressions
    if not all(field.is_rational_function(x, y) for field in equations):
        return None
    numerators = tuple(sp.fraction(sp.together(field))[0] for field in equations)
    polynomials = [sp.Poly(field, x, y) for field in numerators]
    if any(poly.total_degree() > 6 or len(poly.terms()) > 40 for poly in polynomials):
        return None
    common = sp.gcd(*numerators)
    sets, isolated = [], []
    if common == 0:
        sets.append({"factor": sp.S.Zero, "mapping": {}, "whole_plane": True})
        reduced = (sp.S.One, sp.S.One)
    else:
        for factor, _ in sp.factor_list(common, x, y)[1]:
            # An invariant/closed curve is an equilibrium set only if BOTH
            # components vanish on it. Tangency (dH/dt=0) is not sufficient.
            if not all(sp.rem(numerator, factor, x, y) == 0 for numerator in numerators):
                continue
            # Cancellation must not restore points excluded by the original AST.
            if any(guard.is_polynomial(x, y) and sp.rem(guard, factor, x, y) == 0
                   for guard in guards):
                continue
            free = factor.free_symbols
            if len(free) == 1:
                variable = next(iter(free))
                for root in sp.Poly(factor, variable).real_roots():
                    if any(sp.simplify(guard.subs(variable, root)) == 0 for guard in guards):
                        continue
                    sets.append({"factor": variable-root, "plot_factor": factor,
                                 "mapping": {variable: root}})
                continue
            real_points = _quadratic_real_point(factor)
            if real_points is not None:
                isolated.extend(real_points)
                continue
            mapping = {}
            if sp.Poly(factor, x, y).total_degree() == 1:
                variable = x if sp.diff(factor, x) != 0 else y
                mapping[variable] = sp.simplify(-factor.subs(variable, 0) / sp.diff(factor, variable))
            sets.append({"factor": factor, "mapping": mapping})
        reduced = tuple(sp.cancel(field/common) for field in numerators)
    unresolved = False
    if all(field != 0 and field.free_symbols for field in reduced):
        try:
            if (all(len(field.free_symbols) == 1 for field in reduced)
                    and reduced[0].free_symbols != reduced[1].free_symbols):
                from itertools import product
                first, second = (next(iter(field.free_symbols)) for field in reduced)
                for a, b in product(sp.Poly(reduced[0], first).real_roots(),
                                    sp.Poly(reduced[1], second).real_roots()):
                    mapping = {first: a, second: b}
                    isolated.append((mapping[x], mapping[y]))
            elif max(sp.Poly(field, x, y).total_degree() for field in reduced) > 4:
                unresolved = True
            else:
                isolated.extend(sp.solve_poly_system(reduced, x, y, strict=True) or [])
        except (NotImplementedError, sp.polys.polyerrors.UnsolvableFactorError):
            unresolved = True
    points = []
    for point in isolated:
        point = tuple(sp.simplify(value) for value in point)
        if any(value.is_real is None for value in point):
            unresolved = True
            continue
        if not _valid_point(point, expressions, guards):
            continue
        substitution = dict(zip(VARIABLES, point))
        if any(sp.simplify(item["factor"].subs(substitution)) == 0 for item in sets):
            continue
        if point not in points:
            points.append(point)
    return {"points": tuple(points), "sets": tuple(sets), "guards": guards,
            "expressions": expressions, "unresolved": reduced if unresolved else ()}


def analyze_algebraic_equilibria(fields, params, axis_only=False):
    if len(fields) != 2:
        return None
    try:
        expressions = tuple(symbolic_expression(field, params) for field in fields)
        guards = []
        for field in fields:
            for node in ast.walk(field):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    guards.append(symbolic_expression(node.right, params))
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                    exponent = symbolic_expression(node.right, params)
                    if exponent.is_negative:
                        guards.append(symbolic_expression(node.left, params))
        return _analyze(expressions, tuple(dict.fromkeys(guards)), bool(axis_only))
    except (ValueError, KeyError, TypeError, NotImplementedError, sp.PolynomialError):
        return None


def exact_point_text(analysis, point):
    matrix = sp.Matrix(analysis["expressions"]).jacobian(VARIABLES)
    matrix = matrix.subs(dict(zip(VARIABLES, point)))
    trace, determinant = sp.simplify(matrix.trace()), sp.simplify(matrix.det())
    discriminant = sp.simplify(trace**2-4*determinant)
    eigenvalues = tuple(sp.simplify((trace+sign*sp.sqrt(discriminant))/2) for sign in (-1, 1))
    return (", ".join(sp.latex(value) for value in point),
            ", ".join(sp.latex(sp.together(value)) for value in eigenvalues))


def exact_point_jacobian(analysis, point):
    matrix = sp.Matrix(analysis["expressions"]).jacobian(VARIABLES)
    return matrix.subs(dict(zip(VARIABLES, point))).applyfunc(sp.simplify)


def is_hamiltonian_center(analysis, point):
    """A regular local integrating factor and det(J)>0 certify a local centre."""
    if analysis is None:
        return False
    x, y = VARIABLES
    f, g = analysis["expressions"]
    matrix = exact_point_jacobian(analysis, point)
    if matrix.det().is_positive is not True:
        return False
    substitution = dict(zip(VARIABLES, point))
    for density in (sp.S.One, 1/x, 1/y, 1/(x*y), 1/x**2, 1/y**2):
        value = density.subs(substitution)
        if value.is_finite is not True or value.is_zero is not False:
            continue
        if sp.cancel(sp.diff(density*f, x) + sp.diff(density*g, y)) == 0:
            return True
    return False


def clean_numerical_point(fields, params, point):
    """Snap only when a simple exact candidate satisfies the original equations."""
    variables = sp.symbols("x y z", real=True)[:len(fields)]
    try:
        expressions = [symbolic_expression(field, params, variables) for field in fields]
        candidate = tuple(sp.Rational(str(float(value))).limit_denominator(1000000)
                          for value in point)
        if all(abs(float(a)-float(b)) <= 1e-8 * max(1, abs(float(b)))
               for a, b in zip(candidate, point)):
            substitution = dict(zip(variables, candidate))
            if all(sp.simplify(field.subs(substitution)) == 0 for field in expressions):
                # Original divisions are checked too, even if symbolic algebra
                # cancelled a denominator while simplifying an expression.
                for field in fields:
                    for node in ast.walk(field):
                        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                            guard = symbolic_expression(node.right, params, variables)
                            if sp.simplify(guard.subs(substitution)) == 0:
                                return point
                return tuple(float(value) for value in candidate)
    except (ValueError, KeyError, TypeError, NotImplementedError):
        pass
    return point


def equilibrium_set_text(analysis, item):
    x, y = VARIABLES
    mapping = item["mapping"]
    if item.get("whole_plane"):
        equation = r"(x,y)\in\mathbb{R}^{2}"
    elif mapping:
        variable, value = next(iter(mapping.items()))
        free = y if variable == x else x
        equation = sp.latex(variable) + "=" + sp.latex(value) + r",\quad " + sp.latex(free) + r"\in\mathbb{R}"
    else:
        equation = sp.latex(item["factor"]) + "=0"
    restrictions = [sp.simplify(guard.subs(mapping)) for guard in analysis["guards"]]
    restrictions = [guard for guard in restrictions if guard.free_symbols or guard == 0]
    if restrictions:
        equation += r",\quad " + r",\ ".join(sp.latex(guard) + r"\ne0" for guard in restrictions)
    row = "Equilibrium set: $" + equation + "$"
    matrix = sp.Matrix(analysis["expressions"]).jacobian(VARIABLES).subs(mapping)
    trace = sp.simplify(matrix.trace())
    if not mapping and item["factor"] != 0 and trace.is_polynomial(x, y):
        trace = sp.rem(trace, item["factor"], x, y)
    row += r"    $\lambda=(0," + sp.latex(trace) + ")$    Non-hyperbolic"
    if trace.is_positive:
        row += "    Linear result: Unstable"
    elif mapping and len(trace.free_symbols) == 1:
        try:
            condition = sp.solve_univariate_inequality(trace > 0, next(iter(trace.free_symbols)))
            if condition is sp.S.false:
                row += "    Linear result: Inconclusive"
            else:
                row += "    Linear result: Unstable for $" + sp.latex(condition) + "$; Inconclusive otherwise"
        except (NotImplementedError, ValueError):
            row += "    Linear result: Depends on position along the set"
    elif trace.is_nonpositive:
        row += "    Linear result: Inconclusive"
    else:
        row += "    Linear result: Depends on position along the set"
    return row


def unresolved_equilibrium_text(analysis):
    equations = r",\quad ".join(sp.latex(field) + "=0" for field in analysis["unresolved"])
    return "Additional equilibrium conditions: $" + equations + "$"
