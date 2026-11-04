"""Bounded planar Poincare-return search, separate from equilibrium solving.

Detected isolated cycles are numerical candidates, not an exhaustive proof.
Closed-orbit families are labelled only when supplied an analytic centre
certificate; a nearly closed numerical path alone cannot certify a centre.
"""
import numpy as np


def _radial_orbits(expressions, center, bounds, guards=()):
    """Exact radial reduction also detects nonhyperbolic/semistable circles."""
    import sympy as sp
    from equilibrium_algebra import VARIABLES
    x, y = VARIABLES
    cx, cy = (sp.Rational(str(float(value))) for value in center)
    dx, dy = x-cx, y-cy
    square = dx**2+dy**2
    s = sp.Symbol("radius_squared", real=True)
    f, g = expressions
    radial = sp.cancel((dx*f+dy*g)/square)
    angular = sp.cancel((dx*g-dy*f)/square)
    pr = sp.simplify(radial.subs({x: cx+sp.sqrt(s), y: cy}))
    pa = sp.simplify(angular.subs({x: cx+sp.sqrt(s), y: cy}))
    if not pr.is_polynomial(s) or not pa.is_polynomial(s):
        return None
    if sp.cancel(radial-pr.subs(s, square)) != 0 or sp.cancel(angular-pa.subs(s, square)) != 0:
        return None
    room = min(float(cx)-bounds[0], bounds[1]-float(cx),
               float(cy)-bounds[2], bounds[3]-float(cy))
    if room <= 0:
        return [], []
    angles = np.linspace(0, 2*np.pi, 721)
    def in_domain(square_radius):
        u, v = sp.symbols("orbit_u orbit_v", real=True)
        circle_equation = u*u+v*v-square_radius
        for guard in guards:
            shifted = sp.together(guard.subs({x: cx+u, y: cy+v}))
            for part in sp.fraction(shifted):
                if part.is_positive is True or part.is_negative is True:
                    continue
                if not part.is_polynomial(u, v):
                    return False
                reduced = sp.rem(part, circle_equation, u, v)
                if not reduced.free_symbols:
                    if reduced.is_zero is not False:
                        return False
                    continue
                polynomial = sp.Poly(reduced, u, v)
                constant = polynomial.coeff_monomial(1)
                variation = sum(abs(coefficient)*square_radius**sp.Rational(sum(powers), 2)
                                for powers, coefficient in polynomial.terms() if sum(powers))
                if (abs(constant)-variation).is_positive is not True:
                    return False
        return True

    def circle(radius, family):
        return {"points": np.column_stack((float(cx)+radius*np.cos(angles),
                                          float(cy)+radius*np.sin(angles))), "family": family}
    if pr == 0:
        for fraction in (0.35, 0.25, 0.5, 0.7, 0.15, 0.07, 0.03):
            radius = room*fraction
            square_radius = sp.Rational(str(radius))**2
            if pa.subs(s, square_radius).is_zero is False and in_domain(square_radius):
                return [circle(radius, True)], ["Closed-orbit family: one representative shown."]
        return [], []
    roots = list(dict.fromkeys(sp.Poly(pr, s).real_roots()))
    roots = [root for root in roots if root.is_positive is True]
    orbits, rows = [], []
    for i, root in enumerate(roots):
        radius = float(sp.sqrt(root))
        if radius >= room or pa.subs(s, root).is_zero is not False or not in_domain(root):
            continue
        left = pr.subs(s, (root+(roots[i-1] if i else 0))/2)
        right = pr.subs(s, (root+(roots[i+1] if i+1 < len(roots) else root+1))/2)
        if left.is_positive and right.is_negative:
            kind = "attracting"
        elif left.is_negative and right.is_positive:
            kind = "repelling"
        elif (left.is_positive and right.is_positive) or (left.is_negative and right.is_negative):
            kind = "semistable"
        else:
            kind = "stability undetermined"
        orbits.append(circle(radius, False))
        rows.append(f"Limit cycle: {kind}.")
    return orbits, rows


def _returns(field, center, radii, bounds, cancelled, paths=False):
    radii = np.asarray(radii, dtype=float)
    count = len(radii)
    scale = min(bounds[1]-bounds[0], bounds[3]-bounds[2])
    step = scale / 320
    positions = np.tile(center, (count, 1))
    positions[:, 0] += radii
    start = positions.copy()
    answer = np.full(count, np.nan)
    history = [[] for _ in radii]

    def velocity(points):
        with np.errstate(all="ignore"):
            values = np.asarray(field(points), dtype=float)
            lengths = np.linalg.norm(values, axis=1)
            direction = values / np.maximum(lengths[:, None], 1e-300)
        valid = np.isfinite(direction).all(axis=1) & (lengths > 1e-14)
        return np.where(valid[:, None], direction, 0), valid

    first, live = velocity(positions)
    orientation = np.sign(first[:, 1])
    live &= orientation != 0
    travelled = np.zeros(count)
    for iteration in range(7680):
        if iteration % 32 == 0 and cancelled():
            return answer, history
        if not live.any():
            break
        # Prevent tiny trial circles from being traversed in one coarse step.
        h = np.minimum(step, np.maximum(radii, scale*1e-6)/32)
        k1, ok1 = velocity(positions)
        k2, ok2 = velocity(positions + h[:, None]*k1/2)
        k3, ok3 = velocity(positions + h[:, None]*k2/2)
        k4, ok4 = velocity(positions + h[:, None]*k3)
        nxt = positions + h[:, None]*(k1+2*k2+2*k3+k4)/6
        live &= ok1 & ok2 & ok3 & ok4 & np.isfinite(nxt).all(axis=1)
        if hasattr(field, "valid_step"):
            live &= field.valid_step(positions, nxt)
        live &= ((nxt[:, 0] >= bounds[0]) & (nxt[:, 0] <= bounds[1])
                 & (nxt[:, 1] >= bounds[2]) & (nxt[:, 1] <= bounds[3]))
        if paths:
            for i in np.flatnonzero(live):
                history[i].append(positions[i].copy())
        before = (positions[:, 1]-center[1])*orientation
        after = (nxt[:, 1]-center[1])*orientation
        crossed = live & (before < 0) & (after >= 0) & (nxt[:, 0] > center[0])
        crossed &= travelled > radii
        if crossed.any():
            # Hermite event interpolation avoids the radial bias of intersecting
            # a coarse straight chord with the Poincare section.
            end_velocity, _ = velocity(nxt)
            for i in np.flatnonzero(crossed):
                t = np.clip(-before[i]/max(after[i]-before[i], 1e-300), 0, 1)
                a, b = h[i]*k1[i], h[i]*end_velocity[i]
                delta = nxt[i]-positions[i]
                c2, c3 = 3*delta-2*a-b, -2*delta+a+b
                for _ in range(4):
                    value = positions[i, 1]-center[1] + t*a[1]+t*t*c2[1]+t**3*c3[1]
                    derivative = a[1]+2*t*c2[1]+3*t*t*c3[1]
                    if abs(derivative) > 1e-15:
                        t = np.clip(t-value/derivative, 0, 1)
                point = positions[i]+t*a+t*t*c2+t**3*c3
                answer[i] = point[0]-center[0]
                if paths:
                    history[i].append(point)
            live[crossed] = False
        travelled += h
        positions = nxt
    return answer, history


def find_periodic_orbits(field, bounds, anchors, cancelled=lambda: False):
    """Search positive-ray return maps around equilibria in the visible box."""
    scale = min(bounds[1]-bounds[0], bounds[3]-bounds[2])
    orbits, rows = [], []
    for center, certified_family in anchors[:8]:
        if cancelled():
            return [], []
        center = np.asarray(center, dtype=float)
        if getattr(field, "symbolic_fields", None) is not None:
            try:
                radial = _radial_orbits(field.symbolic_fields, center, bounds,
                                       getattr(field, "symbolic_guards", ()))
            except (ArithmeticError, ValueError, TypeError, NotImplementedError):
                radial = None
            if radial is not None:
                exact_orbits, exact_rows = radial
                orbits.extend(exact_orbits)
                rows.extend(exact_rows)
                continue
        room = min(center[0]-bounds[0], bounds[1]-center[0],
                   center[1]-bounds[2], bounds[3]-center[1])
        if room <= scale*0.005:
            continue
        radii = np.geomspace(max(scale*0.006, room*0.025), room*0.88, 18)
        returned, _ = _returns(field, center, radii, bounds, cancelled)
        drift = returned-radii
        if certified_family:
            # The certificate is local. Only show a returning orbit which stays
            # away from the equilibrium and whose computed closure is accurate.
            candidates = np.flatnonzero(np.isfinite(drift) & (np.abs(drift) < scale*1e-7))
            if len(candidates):
                i = candidates[min(len(candidates)-1, len(candidates)//2)]
                _, paths = _returns(field, center, [radii[i]], bounds, cancelled, True)
                points = np.asarray(paths[0])
                if len(points) > 24 and np.linalg.norm(points[-1]-points[0]) < scale*1e-7:
                    points[-1] = points[0]
                    orbits.append({"points": points, "family": True})
                    rows.append("Closed-orbit family: one representative shown.")
            continue
        brackets = [(radii[i], radii[i+1], drift[i], drift[i+1])
                    for i in range(len(radii)-1)
                    if np.isfinite(drift[i:i+2]).all() and drift[i]*drift[i+1] < 0
                    and min(abs(drift[i]), abs(drift[i+1])) > scale*1e-8]
        # Also examine a sample almost exactly on a cycle; verify opposite
        # return displacements on either side before accepting it as isolated.
        for i in np.flatnonzero(np.isfinite(drift) & (np.abs(drift) < scale*1e-7)):
            eps = max(radii[i]*0.005, scale*1e-5)
            pair = np.array([radii[i]-eps, radii[i]+eps])
            result, _ = _returns(field, center, pair, bounds, cancelled)
            delta = result-pair
            if np.isfinite(delta).all() and delta[0]*delta[1] < 0 and max(abs(delta)) > scale*1e-8:
                brackets.append((*pair, *delta))
        for low, high, flow, fhigh in brackets[:8]:
            attracting = flow > 0 and fhigh < 0
            for _ in range(16):
                if cancelled():
                    return [], []
                mid = (low+high)/2
                result, _ = _returns(field, center, [mid], bounds, cancelled)
                delta = result[0]-mid
                if not np.isfinite(delta):
                    break
                if flow*delta <= 0:
                    high = mid
                else:
                    low, flow = mid, delta
            else:
                radius = (low+high)/2
                returned, paths = _returns(field, center, [radius], bounds, cancelled, True)
                if not np.isfinite(returned[0]) or abs(returned[0]-radius) > scale*2e-6:
                    continue
                points = np.asarray(paths[0])
                if len(points) < 24:
                    continue
                # Deduplicate a cycle found from multiple enclosed equilibria.
                if any(not item["family"] and np.linalg.norm(
                        points.mean(axis=0)-item["points"].mean(axis=0)) < scale*0.02
                       and abs(np.ptp(points, axis=0).max()
                               -np.ptp(item["points"], axis=0).max()) < scale*0.02
                       for item in orbits):
                    continue
                points[-1] = points[0]
                orbits.append({"points": points, "family": False})
                rows.append("Numerically detected limit cycle: "
                            + ("attracting." if attracting else "repelling."))
    rows.append("Periodic-orbit search is limited to the searched 2D region and may miss cycles.")
    return orbits, rows
