"""Trajectory integration and portable settings for Dynamical System Lab."""
import math
import time

import numpy as np
from matplotlib.colors import FuncNorm, LinearSegmentedColormap


def speed_colormap():
    """Match the spectrum used by the OpenGL trail shader."""
    values = np.linspace(0.0, 1.0, 257)
    colors = np.clip(1.5 - np.abs(4 * values[:, None] - [3, 2, 1]), 0, 1)
    return LinearSegmentedColormap.from_list("flow_speed", colors)


def speed_normalization(maximum):
    return FuncNorm((np.log1p, np.expm1), vmin=0, vmax=max(float(maximum), 1e-9), clip=True)


def trajectory_curve_points(times, points, velocities, spatial_scale, pixel_tolerance=0.10):
    """Refine ODE Hermite dense output at the current display scale.

    This preserves stored solution times and states. Zooming only changes the
    number of plotted vertices, using actual endpoint ODE tangents rather than a
    free spline fit. Most already-smooth intervals need no additional vertices.
    """
    if len(points) < 2:
        return points
    chord = np.diff(points, axis=0)
    h = np.diff(times)[:, None]
    ta, tb = h * velocities[:-1], h * velocities[1:]
    c2 = (3*chord - 2*ta - tb) * spatial_scale
    c3 = (-2*chord + ta + tb) * spatial_scale
    second = np.stack((2*c2, 6*c3 + 2*c2), axis=1)
    pixel_chord = chord * spatial_scale
    length = np.linalg.norm(pixel_chord, axis=1)
    direction = pixel_chord / np.maximum(length[:, None], 1e-300)
    second -= np.sum(second * direction[:, None, :], axis=2)[:, :, None] * direction[:, None, :]
    bound = np.max(np.linalg.norm(second, axis=2), axis=1) / 8
    pieces = np.clip(np.ceil(np.sqrt(bound / pixel_tolerance)), 1, 256).astype(np.int64)
    # Bound work for extreme zooms/off-screen escapes. No segment crosses a
    # discontinuity: all intervals come from accepted steps of this same solution.
    total = int(np.sum(pieces))
    if total > 200000:
        pieces = np.maximum(1, np.floor(pieces * (200000 / total))).astype(np.int64)
    if np.all(pieces == 1):
        return points
    intervals = np.repeat(np.arange(len(pieces)), pieces)
    starts = np.repeat(np.cumsum(pieces) - pieces, pieces)
    fraction = (np.arange(len(intervals)) - starts + 1) / pieces[intervals]
    t = fraction[:, None]
    refined = ((2*t**3 - 3*t**2 + 1) * points[intervals]
               + (t**3 - 2*t**2 + t) * ta[intervals]
               + (-2*t**3 + 3*t**2) * points[intervals+1]
               + (t**3 - t**2) * tb[intervals])
    return np.vstack((points[0], refined))


def integrate_trajectory(field, initial, duration, max_step=0.01, samples=1201,
                         rtol=1e-7, atol=1e-9, time_limit=6.0,
                         retain_steps=False, spatial_scale=None, curve_tolerance=0.12):
    """Dormand-Prince 5(4), sampled at actual equation times, without normalization.

    Return a finite partial solution and an explicit reason if integration cannot
    reach the requested end time. No escaping or invalid segment is connected.
    retain_steps keeps every accepted step in addition to the requested output
    times. spatial_scale (pixels per coordinate unit) also limits the curvature
    error of the displayed chords; numerical accuracy alone does not ensure a
    smooth polyline when most internal integration points are discarded.
    """
    point = np.asarray(initial, dtype=float).copy()
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise ValueError("Initial coordinates must be finite numbers.")
    if not math.isfinite(duration) or duration <= 0 or max_step <= 0:
        raise ValueError("Duration and maximum step must be positive.")
    metric = None if spatial_scale is None else np.asarray(spatial_scale, dtype=float)
    if metric is not None and (metric.shape != point.shape or not np.all(np.isfinite(metric)) or np.any(metric <= 0)):
        raise ValueError("Plot scales must be finite positive values for each coordinate.")
    if not math.isfinite(curve_tolerance) or curve_tolerance <= 0:
        raise ValueError("Curve tolerance must be positive.")
    output_times = np.linspace(0.0, duration, int(samples))
    times, points = [0.0], [point.copy()]
    t, h = 0.0, min(max_step, duration / 20)
    started = time.monotonic()
    attempts = 0
    reason = "Complete"
    # Dormand-Prince tableau; the final stage is also the next step's first stage.
    rows = (
        (1/5,), (3/40, 9/40), (44/45, -56/15, 32/9),
        (19372/6561, -25360/2187, 64448/6561, -212/729),
        (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
        (35/384, 0, 500/1113, 125/192, -2187/6784, 11/84),
    )
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            first = np.asarray(field(point), dtype=float)
            if first.shape != point.shape or not np.all(np.isfinite(first)):
                raise ValueError("The field is undefined at the initial point.")
            for target in output_times[1:]:
                while t < target:
                    if target - t <= 16 * np.finfo(float).eps * max(1.0, abs(target)):
                        t = float(target)
                        break
                    attempts += 1
                    if attempts > 150000 or time.monotonic() - started > time_limit:
                        reason = "Stopped at the calculation limit; partial solution"
                        return np.asarray(times), np.asarray(points), reason
                    step = min(h, max_step, target - t)
                    if step < 16 * np.finfo(float).eps * max(1.0, abs(t)):
                        reason = "Stopped: required step is too small; partial solution"
                        return np.asarray(times), np.asarray(points), reason
                    stages = [first]
                    for coefficients in rows:
                        offset = np.asarray(coefficients) @ np.asarray(stages)
                        stages.append(np.asarray(field(point + step * offset), dtype=float))
                    stages = np.asarray(stages)
                    candidate = point + step * (b5 @ stages)
                    if not np.all(np.isfinite(candidate)) or np.max(np.abs(candidate)) > 1e12:
                        raise ValueError("Trajectory escaped the finite calculation range")
                    scale = atol + rtol * np.maximum(np.abs(point), np.abs(candidate))
                    error = np.max(np.abs(step * ((b5 - b4) @ stages)) / scale)
                    if metric is not None:
                        # ODE endpoint tangents define Hermite dense output.
                        # Bound its deviation from the endpoint chord in pixels;
                        # remove tangential acceleration, which cannot bend a line.
                        chord = (candidate - point) * metric
                        tangent_a = step * first * metric
                        tangent_b = step * stages[-1] * metric
                        c2 = 3 * chord - 2 * tangent_a - tangent_b
                        c3 = -2 * chord + tangent_a + tangent_b
                        second = np.array([2*c2, 6*c3 + 2*c2])
                        length = np.linalg.norm(chord)
                        if length > 1e-12:
                            direction = chord / length
                            second -= (second @ direction)[:, None] * direction
                        bend_error = np.max(np.linalg.norm(second, axis=1)) / 8
                        error = max(error, bend_error / curve_tolerance)
                    if not np.isfinite(error):
                        raise ValueError("Non-finite integration error")
                    h = step * (5.0 if error == 0 else np.clip(0.9 * error ** -0.2, 0.1, 5.0))
                    if error <= 1:
                        point, first = candidate, stages[-1]
                        t = float(target) if step == target - t else t + step
                        if retain_steps:
                            times.append(t)
                            points.append(point.copy())
                if times[-1] < target:
                    times.append(float(target))
                    points.append(point.copy())
    except (ArithmeticError, ValueError, TypeError) as exc:
        if len(times) == 1:
            raise ValueError(f"Cannot integrate from this initial point: {exc}") from exc
        reason = f"Stopped: {exc}; partial solution"
    return np.asarray(times), np.asarray(points), reason


SETTING_RANGES = {
    "particles": (20, 3000), "trail_frames": (3, 300), "respawn": (1, 60),
    "step": (0.001, 0.5), "playback": (0.01, 10), "color_max": (0.001, 1e9),
    "duration": (0.001, 1000),
}


def validate_experiment(data):
    """Validate the whole document before the application changes any state."""
    if (not isinstance(data, dict)
            or data.get("format") not in ("Dynamical System Lab", "ODE Flow Lab")
            or data.get("version") != 1):
        raise ValueError("Unsupported experiment file.")
    if data.get("dimension") not in ("2D", "3D") or data.get("coordinates") not in ("Cartesian", "Polar"):
        raise ValueError("Invalid dimension or coordinate mode.")
    modes = data.get("modes")
    if not isinstance(modes, dict) or set(modes) != {"2D", "3D", "Polar"}:
        raise ValueError("The experiment must contain the three editor modes.")
    for state in modes.values():
        if not isinstance(state, list) or len(state) != 5 or not all(isinstance(s, str) and len(s) <= 20000 for s in state):
            raise ValueError("Invalid equation editor state.")
    settings = data.get("settings", {})
    for name, (lower, upper) in SETTING_RANGES.items():
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"Invalid setting: {name}")
        if name in ("particles", "trail_frames") and int(value) != value:
            raise ValueError(f"{name} must be an integer.")
    for name in ("normalized", "speed_colors", "dark", "vectors", "equilibria", "fixed_colors", "show_selected"):
        if not isinstance(settings.get(name), bool):
            raise ValueError(f"Invalid switch: {name}")
    if not isinstance(settings.get("nullclines", False), bool):
        raise ValueError("Invalid switch: nullclines")
    if not isinstance(settings.get("first_quadrant", False), bool):
        raise ValueError("Invalid switch: first_quadrant")
    bounds = data.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) <= 1e9 for v in bounds):
        raise ValueError("Invalid view bounds.")
    if bounds[1] - bounds[0] < 0.001 or bounds[3] - bounds[2] < 0.001:
        raise ValueError("View width and height must be at least 0.001.")
    initial = data.get("initial")
    if not isinstance(initial, list) or len(initial) != 3 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) <= 1e6 for v in initial):
        raise ValueError("Invalid initial point.")
    if not isinstance(data.get("solution_computed", False), bool):
        raise ValueError("Invalid solution switch.")
    if "view3d" in data:
        view = data["view3d"]
        if not isinstance(view, dict) or not all(isinstance(view.get(key), (int, float)) and math.isfinite(view[key]) for key in ("elevation", "azimuth")):
            raise ValueError("Invalid 3D view angles.")
        bounds = np.asarray(view.get("bounds"), dtype=float)
        if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Invalid 3D view bounds.")
    return data
