"""Cancellable, time-bounded analysis in an isolated Python process."""
import base64
import pickle
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


def _pack(value):
    return b"ODE:" + base64.b64encode(pickle.dumps(value, protocol=5)) + b"\n"


class AnalysisProcess(QObject):
    completed = Signal(object, object)
    failed = Signal(object, str)

    def __init__(self, parent=None, timeout_ms=8000):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(["-u", str(Path(__file__).resolve()), "--worker"])
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.readyReadStandardError.connect(self._discard_stderr)
        self.process.finished.connect(self._exited)
        self.process.errorOccurred.connect(self._error)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._timeout)
        self.timeout_ms = timeout_ms
        self.pending = self.busy = None
        self.ready = False
        self.expected_exit = False
        self.buffer = bytearray()
        self.sequence = 0
        self.closed = False

    def submit(self, key, payload, kind="equilibrium"):
        self.pending = (key, payload, kind)
        if self.busy is not None:
            self._kill()
        else:
            self._dispatch()

    def _dispatch(self):
        if self.closed or self.pending is None:
            return
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.ready = False
            self.expected_exit = False
            self.buffer.clear()
            self.process.start()
            self.timer.start(30000)  # Cold imports have a separate startup budget.
        elif self.ready and self.busy is None:
            key, payload, kind = self.pending
            self.pending = None
            self.sequence += 1
            self.busy = (self.sequence, key)
            self.process.write(_pack({"id": self.sequence, "kind": kind, "payload": payload}))
            self.timer.start(self.timeout_ms)

    def _read(self):
        if self.expected_exit:
            self.process.readAllStandardOutput()
            return
        self.buffer.extend(bytes(self.process.readAllStandardOutput()))
        while b"\n" in self.buffer:
            line, _, tail = self.buffer.partition(b"\n")
            self.buffer = bytearray(tail)
            if not line.startswith(b"ODE:"):
                continue
            # This pipe belongs exclusively to the trusted child process.
            message = pickle.loads(base64.b64decode(line[4:]))
            if message.get("ready"):
                self.ready = True
                self.timer.stop()
                self._dispatch()
            elif self.busy is not None and message.get("id") == self.busy[0]:
                _, key = self.busy
                self.busy = None
                self.timer.stop()
                if "error" in message:
                    self.failed.emit(key, message["error"])
                else:
                    self.completed.emit(key, message["result"])
                self._dispatch()

    def _discard_stderr(self):
        self.process.readAllStandardError()

    def _kill(self):
        self.timer.stop()
        self.expected_exit = True
        self.ready = False
        self.busy = None
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def cancel(self):
        self.pending = None
        if self.busy is not None:
            self._kill()

    def _timeout(self):
        key = self.busy[1] if self.busy else self.pending[0] if self.pending else None
        self.pending = None
        self._kill()
        if key is not None:
            self.failed.emit(key, "Calculation time limit reached; analysis is incomplete.")

    def _exited(self, *_args):
        key = self.busy[1] if self.busy else None
        self.busy = None
        self.ready = False
        self.timer.stop()
        if key is not None and not self.expected_exit:
            self.failed.emit(key, "Analysis process stopped before completing.")
        self._dispatch()

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            key = self.pending[0] if self.pending else None
            self.pending = None
            self.timer.stop()
            if key is not None:
                self.failed.emit(key, "Could not start the analysis process.")

    def close(self):
        self.closed = True
        self.pending = None
        self._kill()
        self.process.waitForFinished(1000)


def compute_equilibria(data):
    import ast
    import numpy as np
    import ode_flow as core
    fields, params, bounds = data["fields"], data["params"], data["bounds"]
    two_d = len(fields) == 2
    note = data["note"]
    axis = False
    if two_d:
        try:
            names = {node.id for node in ast.walk(fields[1]) if isinstance(node, ast.Name)}
            axis = not names.intersection({"x", "y"}) and float(core.eval_node(fields[1], params)) == 0
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            pass
    rows, sets = [], []
    if axis:
        rows.append("Search restricted to y = 0; the full system may also have non-isolated equilibria away from the x-axis.")
    analysis = core.analyze_algebraic_equilibria(fields, params, axis)
    if analysis is None:
        results = core.find_equilibria(fields, params, bounds, x_axis_only=axis)
    else:
        results = []
        for item in analysis["sets"]:
            mapping = item["mapping"]
            if mapping and not next(iter(mapping.values())).free_symbols:
                variable, value = next(iter(mapping.items()))
                source = f"{variable}-({float(value):.17g})"
            else:
                source = str(item["factor"])
            sets.append(dict(item, field=core.compile_formula(source)))
            rows.append(core.equilibrium_set_text(analysis, item))
        for exact in analysis["points"]:
            point = np.asarray([float(value) for value in exact])
            if any(value < low or value > high for value, (low, high) in zip(point, bounds)):
                continue
            details = core.classify_equilibrium(fields, point, params, core.exact_point_jacobian(analysis, exact))
            details["coordinate_latex"], details["eigenvalue_latex"] = core.exact_point_text(analysis, exact)
            details["certified_center"] = core.is_hamiltonian_center(analysis, exact)
            results.append((point, details))
        if analysis["unresolved"]:
            reduced = tuple(core.compile_formula(str(field)) for field in analysis["unresolved"])
            results.extend((point, core.classify_equilibrium(fields, point, params))
                           for point, _ in core.find_equilibria(reduced, params, bounds))
            rows.append(core.unresolved_equilibrium_text(analysis))
    if sets:
        rows.append("Every point in an equilibrium set is stationary; it is not a limit cycle.")
    for point, details in results:
        if analysis is not None and analysis["unresolved"] and "coordinate_latex" not in details:
            continue
        coordinates = details.get("coordinate_latex", ", ".join(f"{value:.8g}" for value in point))
        eigenvalues = details.get("eigenvalue_latex", details["eigenvalues"])
        hyperbolic = ("Hyperbolic" if details["hyperbolic"] else "Non-hyperbolic") if details["hyperbolic"] is not None else "Hyperbolicity unavailable"
        if len(coordinates)+len(eigenvalues) > 4000:
            rows.append("Equilibrium found; its exact expression exceeds the display size.")
        else:
            rows.append(f"Equilibrium point: $({coordinates})$    $\\lambda=({eigenvalues})$    "
                        f"{hyperbolic}    Linear result: {details['type']}; {details['linearization']}")
    if not rows:
        rows.append(f"{note}. No isolated equilibrium detected.")
    anchors = [(point.copy(), details.get("certified_center", False)) for point, details in results] if two_d else []
    if two_d and not anchors:
        anchors = [(np.mean(bounds, axis=1), False)]
    return dict(results=results, sets=sets, analysis=analysis, anchors=anchors, rows=rows)


def compute_periodic(data):
    import numpy as np
    import ode_flow as core
    fields, params = data["fields"], data["params"]
    def field(points):
        return np.column_stack([np.broadcast_to(
            core.evaluate(component, points[:, 0], points[:, 1], params), len(points))
            for component in fields])
    guards = core._equilibrium_domain_guards(fields, params)
    def valid_step(before, after):
        valid = np.ones(len(before), dtype=bool)
        with np.errstate(all="ignore"):
            for guard, positive in guards:
                a = core.evaluate(guard, before[:, 0], before[:, 1], params)
                b = core.evaluate(guard, after[:, 0], after[:, 1], params)
                valid &= (np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
                          & (np.signbit(a) == np.signbit(b)))
                if positive:
                    valid &= (a > 0) & (b > 0)
        return valid
    field.valid_step = valid_step
    if data["analysis"] is not None:
        field.symbolic_fields = data["analysis"]["expressions"]
        field.symbolic_guards = data["analysis"]["guards"]
    return core.find_periodic_orbits(field, data["bounds"], data["anchors"])


def compute_nullclines(data):
    import numpy as np
    import ode_flow as core
    from matplotlib.path import Path as MathPath
    figure = core.Figure()
    ax = figure.add_subplot(111)
    xx, yy, components = core.sample_nullclines(
        data["fields"], data["params"], data["bounds"], data["resolution"])
    result = []
    for component in components:
        lines, fills = [], []
        if component.count():
            minimum, maximum = component.min(), component.max()
            if minimum == maximum == 0:
                plot = ax.contourf(xx, yy, np.ma.array(np.ones(component.shape), mask=component.mask),
                                   levels=[0.5, 1.5])
                fills = [(path.vertices.copy(), path.codes.copy() if path.codes is not None else None)
                         for path in plot.get_paths()]
            elif minimum <= 0 <= maximum and minimum != maximum:
                plot = ax.contour(xx, yy, component, levels=[0], corner_mask=False)
                for path in plot.get_paths():
                    starts = np.flatnonzero(path.codes == MathPath.MOVETO) if path.codes is not None else [0]
                    for start, end in zip(starts, list(starts[1:])+[len(path.vertices)]):
                        if end-start >= 2:
                            lines.append(path.vertices[start:end].copy())
        result.append((lines, fills))
    figure.clear()
    return result


def compute_trajectory3d(data):
    import ode_flow as core
    return core.Flow3DCanvas._integrate(
        data["fields"], data["params"], data["initial"], data["dt"])


def _serve():
    import ode_flow  # Warm imports once, without constructing a window.
    sys.stdout.buffer.write(_pack({"ready": True}))
    sys.stdout.buffer.flush()
    for line in sys.stdin.buffer:
        if not line.startswith(b"ODE:"):
            continue
        request = pickle.loads(base64.b64decode(line[4:]))
        try:
            handler = {"equilibrium": compute_equilibria, "periodic": compute_periodic,
                       "nullclines": compute_nullclines,
                       "trajectory3d": compute_trajectory3d}[request["kind"]]
            result = handler(request["payload"])
            message = {"id": request["id"], "result": result}
        except Exception as exc:
            message = {"id": request["id"], "error": str(exc)}
        sys.stdout.buffer.write(_pack(message))
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    _serve()
