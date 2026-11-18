"""Interactive views for the planar dynamics coursework tools."""
import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PySide6.QtCore import Qt, QSignalBlocker, QTimer
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QGridLayout,
                              QHBoxLayout, QLabel, QSlider, QVBoxLayout)

from background_analysis import AnalysisProcess


def _colors(host):
    return ("#000000", "#eeeeee", "#80cfff", "#ff8f74") if host.dark_mode_check.isChecked() else (
        "#ffffff", "#222222", "#2453a5", "#c04b25")


def _axes(figure, host, rows=1, cols=1):
    figure.clear()
    bg, fg, *_ = _colors(host)
    figure.set_facecolor(bg)
    axes = figure.subplots(rows, cols, squeeze=False).flat
    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#555555" if bg == "#000000" else "#cccccc")
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        ax.grid(color=fg, alpha=.12)
    return list(figure.axes)


class _AnalysisDialog(QDialog):
    def __init__(self, host, parent, title, kind):
        super().__init__(parent)
        self.host, self.kind = host, kind
        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(830, 570)
        self.layout_box = QVBoxLayout(self)
        self.result = None
        self.worker = AnalysisProcess(self, timeout_ms=12000)
        self.worker.completed.connect(self._complete)
        self.worker.failed.connect(self._failed)
        self.finished.connect(lambda _: self.worker.close())

    def _build_plot(self):
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(190)
        self.layout_box.addWidget(self.canvas, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.layout_box.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.run_button = buttons.addButton("Recalculate", QDialogButtonBox.ButtonRole.ActionRole)
        self.run_button.clicked.connect(self.run)
        self.layout_box.addWidget(buttons)
        self.host.dark_mode_check.toggled.connect(self._redraw)

    def _complete(self, _key, result):
        self.result = result
        self._redraw()

    def _failed(self, _key, message):
        self.result = None
        self._redraw()
        self.status.setText(message)

    def _submit(self, data):
        self.result = None
        self._redraw()
        self.status.setText("Calculating…")
        self.worker.submit(self.kind, data, self.kind)


class PlanarCriteriaDialog(_AnalysisDialog):
    def __init__(self, host, fields, params, bounds):
        super().__init__(host, host, "Planar criteria", "planar_criteria")
        from dynamical_system_lab import EquilibriumInfo, NotationLineEdit
        self.fields, self.params = fields, params
        grid = QGridLayout()
        self.bounds = []
        for i, (label, value) in enumerate(zip(("x min", "x max", "y min", "y max"), bounds)):
            spin = host.make_spin(value, -1e9, 1e9, .1, compact_decimals=True)
            spin.setDecimals(6)
            spin.setValue(value)
            self.bounds.append(spin)
            grid.addWidget(QLabel(label), 0, 2*i)
            grid.addWidget(spin, 0, 2*i+1)
        self.weight = NotationLineEdit("1")
        self.weight.setToolTip("Use x, y and parameters of the current system. 1 gives the ordinary divergence test.")
        grid.addWidget(QLabel("Dulac function B(x, y)"), 1, 0, 1, 3)
        grid.addWidget(self.weight, 1, 3, 1, 5)
        self.layout_box.addLayout(grid)
        guide = QLabel("Drag a rectangle on the map or edit the bounds above and recalculate.")
        guide.setWordWrap(True)
        self.layout_box.addWidget(guide)
        self._build_plot()
        self.formula = EquilibriumInfo()
        self.formula.setMinimumHeight(44)
        self.formula.setMaximumHeight(70)
        self.layout_box.insertWidget(3, self.formula)
        self.run()

    def run(self):
        self.worker.cancel()
        self.plot_bounds = tuple(spin.value() for spin in self.bounds)
        self._submit(dict(fields=self.fields, params=self.params, bounds=self.plot_bounds,
                          weight=self.weight.text()))

    def _redraw(self, *_):
        if getattr(self, "selector", None) is not None:
            self.selector.disconnect_events()
            self.selector = None
        ax, = _axes(self.figure, self.host)
        ax.set(xlabel="x", ylabel="y", title="Divergence of B·F")
        if self.result is None:
            self.formula.set_rows([], _colors(self.host)[1])
        else:
            r = self.result
            a, b, c, d = self.plot_bounds
            values = r["values"]
            if np.isfinite(values).any():
                vmax = max(np.nanmax(abs(values)), 1e-15)
                plot = ax.imshow(values, extent=(a, b, c, d), origin="lower", cmap="coolwarm",
                                 vmin=-vmax, vmax=vmax, aspect="auto")
                bar = self.figure.colorbar(plot, ax=ax)
                bar.ax.tick_params(colors=_colors(self.host)[1])
            self.formula.set_rows([r"$\nabla\cdot(BF)=" + r["formula"] + r"\qquad "
                                   r"\partial_x g-\partial_y f=" + r["curl"] + "$"], _colors(self.host)[1])
            self.selector = RectangleSelector(ax, self._select_region, button=[1], minspanx=8, minspany=8,
                                              spancoords="pixels", useblit=False)
            index = (f"Estimated Poincaré index of the counterclockwise boundary: {r['index']}."
                     if r["index"] is not None else "Boundary index unavailable: zero, undefined field or unresolved winding.")
            self.status.setText(r["summary"] + "\n" + index)
        self.canvas.draw_idle()

    def _select_region(self, start, end):
        if None in (start.xdata, end.xdata, start.ydata, end.ydata):
            return
        limits = (*sorted((start.xdata, end.xdata)), *sorted((start.ydata, end.ydata)))
        for spin, value in zip(self.bounds, limits):
            spin.setValue(value)
        # Let Matplotlib finish the selection event before replacing its axes.
        QTimer.singleShot(0, self, self.run)


class LinearComparisonDialog(_AnalysisDialog):
    def __init__(self, host, parent, fields, params, point, jacobian, radius):
        super().__init__(host, parent, "Linearisation comparison", "linear_comparison")
        self.problem = dict(fields=fields, params=params, point=point, jacobian=jacobian)
        row = QHBoxLayout()
        self.dx = host.make_spin(.15*radius, -1e9, 1e9, .01, compact_decimals=True)
        self.dy = host.make_spin(.08*radius, -1e9, 1e9, .01, compact_decimals=True)
        for spin, initial in ((self.dx, .15*radius), (self.dy, .08*radius)):
            spin.setDecimals(8)
            spin.setValue(initial)
        self.duration = host.make_spin(3, .001, 100, .1)
        self.backward = QCheckBox("Include reverse time")
        for label, widget in (("Δx₀", self.dx), ("Δy₀", self.dy), ("Duration", self.duration)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addWidget(self.backward)
        self.layout_box.addLayout(row)
        self._build_plot()
        self.run()

    def run(self):
        self.worker.cancel()
        self._submit(dict(self.problem, offset=(self.dx.value(), self.dy.value()),
                          duration=self.duration.value(), backward=self.backward.isChecked()))

    def _redraw(self, *_):
        phase, times = _axes(self.figure, self.host, cols=2)
        phase.set(xlabel="x − x*", ylabel="y − y*", title="Perturbation phase plane", aspect="equal")
        times.set(xlabel="Time t", ylabel="Distance from equilibrium", title="Perturbation magnitude")
        if self.result:
            _, fg, blue, orange = _colors(self.host)
            phase.scatter(0, 0, color=fg, s=20, zorder=5)
            seen, notes = set(), []
            for name, t, points, reason in self.result:
                color, style = (blue, "-") if name == "Nonlinear" else (orange, "--")
                label = name if name not in seen else None
                phase.plot(*points.T, color=color, linestyle=style, label=label)
                times.plot(t, np.linalg.norm(points, axis=1), color=color, linestyle=style)
                seen.add(name)
                if reason != "Complete":
                    notes.append(f"{name}: {reason} (t = {t[-1]:.5g}).")
            for label in phase.legend(facecolor=self.figure.get_facecolor()).get_texts():
                label.set_color(fg)
            self.status.setText("Both systems start from the same perturbation. " + " ".join(notes))
        self.canvas.draw_idle()


class HopfDialog(QDialog):
    """Exact branch and phase portrait of the planar cubic normal form."""
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.setWindowTitle("Planar Hopf normal form")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(870, 570)
        layout = QVBoxLayout(self)
        formula = QLabel("ṙ = μr + αr³     θ̇ = ω")
        layout.addWidget(formula)
        row = QHBoxLayout()
        self.alpha = host.make_spin(-1, -100, 100, .1)
        self.omega = host.make_spin(1, -100, 100, .1)
        self.slider = host.make_spin(.5, -2, 2, .05)
        for label, widget in (("Cubic coefficient α", self.alpha), ("Angular frequency ω", self.omega),
                              ("Selected μ", self.slider)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            widget.valueChanged.connect(self._redraw)
        layout.addLayout(row)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(230)
        layout.addWidget(self.canvas, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.host.dark_mode_check.toggled.connect(self._redraw)
        self._redraw()

    def _redraw(self, *_):
        branch, phase = _axes(self.figure, self.host, cols=2)
        _, fg, stable, unstable = _colors(self.host)
        alpha, omega, mu = self.alpha.value(), self.omega.value(), self.slider.value()
        stable_x, unstable_x = np.linspace(-2, 0, 201), np.linspace(0, 2, 201)
        branch.plot(stable_x, np.zeros(201), color=stable)
        branch.plot(unstable_x, np.zeros(201), '--', color=unstable)
        radius = None
        if alpha and omega:
            side = unstable_x if alpha < 0 else stable_x
            branch.plot(side, np.sqrt(-side/alpha), '-' if alpha < 0 else '--',
                        color=stable if alpha < 0 else unstable)
            branch.scatter(0, 0, s=24, color=fg, zorder=4)
            if -mu/alpha > 0:
                radius = math.sqrt(-mu/alpha)
        branch.axvline(mu, color=fg, alpha=.45, linewidth=1)
        branch.set(xlabel="μ", ylabel="Radius r", title="Equilibrium and periodic-orbit branches", xlim=(-2, 2))
        extent = max(1., 1.3*(radius or 0))
        x, y = np.meshgrid(np.linspace(-extent, extent, 21), np.linspace(-extent, extent, 21))
        local = mu+alpha*(x*x+y*y)
        u, v = local*x-omega*y, omega*x+local*y
        magnitude = np.hypot(u, v)
        safe = np.where(magnitude > 0, magnitude, 1)
        phase.quiver(x, y, u/safe, v/safe, color=fg, alpha=.38, pivot="mid")
        phase.scatter(0, 0, color=fg if mu == 0 else stable if mu < 0 else unstable, s=24, zorder=4)
        if radius is not None:
            theta = np.linspace(0, 2*np.pi, 401)
            phase.plot(radius*np.cos(theta), radius*np.sin(theta), '-' if alpha < 0 else '--',
                       color=stable if alpha < 0 else unstable)
        phase.set(xlabel="x", ylabel="y", title=f"Phase portrait at μ = {mu:g}", aspect="equal",
                  xlim=(-extent, extent), ylim=(-extent, extent))
        if not omega:
            summary = "ω = 0: real eigenvalues; circular equilibria can occur instead of periodic orbits."
        elif not alpha:
            summary = "α = 0: the cubic Hopf nondegeneracy condition fails."
        else:
            summary = ("Supercritical: stable cycles for μ > 0." if alpha < 0 else
                       "Subcritical: unstable cycles for μ < 0.")
        cycle = f"   Cycle radius = {radius:.5g}; period = {2*math.pi/abs(omega):.5g}." if radius is not None else ""
        self.status.setText(f"λ = {mu:g} ± {abs(omega):g}i   ·   {summary}{cycle}\n"
                            "Blue solid: attracting. Orange dashed: repelling.")
        self.canvas.draw_idle()


class HopfScanDialog(_AnalysisDialog):
    def __init__(self, host, parent, fields, params, point, radius):
        super().__init__(host, parent, "Scan for planar Hopf bifurcations", "hopf_scan")
        from dynamical_system_lab import PopupComboBox, display_parameter_name
        self.resize(1000, 570)
        self.problem = dict(fields=fields, params=params, point=point)
        self.local_radius = radius
        controls = QHBoxLayout()
        self.parameter = PopupComboBox()
        for name in params:
            self.parameter.addItem(display_parameter_name(name), name)
        controls.addWidget(QLabel("Vary"))
        controls.addWidget(self.parameter, 1)
        self.low, self.high = (host.make_spin(v, -1e12, 1e12, .1, compact_decimals=True) for v in (-1, 1))
        for label, spin in (("Minimum", self.low), ("Maximum", self.high)):
            spin.setDecimals(8)
            controls.addWidget(QLabel(label))
            controls.addWidget(spin)
        self.layout_box.addLayout(controls)
        note = QLabel("Polynomial planar systems. The scan starts at the inspected equilibrium and follows one branch.")
        note.setWordWrap(True)
        self.layout_box.addWidget(note)
        controls = QHBoxLayout()
        self.selected = QLabel("Phase portrait parameter")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        controls.addWidget(self.selected)
        controls.addWidget(self.slider, 1)
        self.layout_box.addLayout(controls)
        self._build_plot()
        self.parameter.currentIndexChanged.connect(self._parameter_changed)
        self.slider.valueChanged.connect(self._redraw)
        self.canvas.mpl_connect("button_press_event", self._pick_parameter)
        self._parameter_changed()
        self.run()

    def _parameter_changed(self, *_):
        name = self.parameter.currentData()
        current = self.problem["params"][name]
        span = max(1, abs(current)*1.25)
        self.low.setValue(current-span)
        self.high.setValue(current+span)

    def run(self):
        self.worker.cancel()
        self.scan_parameter = self.parameter.currentData()
        self.slider.setEnabled(False)
        self._submit(dict(self.problem, parameter=self.scan_parameter,
                          bounds=(self.low.value(), self.high.value())))

    def _pick_parameter(self, event):
        if self.result is not None and event.button == 1 and event.inaxes in self.figure.axes[:2]:
            self.slider.setValue(int(np.argmin(abs(self.result['samples']-event.xdata))))

    def _complete(self, key, result):
        with QSignalBlocker(self.slider):
            samples = result["samples"]
            self.slider.setRange(0, len(samples)-1)
            reference = self.problem["params"][self.scan_parameter]
            self.slider.setValue(int(np.argmin(abs(samples-reference))))
        self.slider.setEnabled(True)
        super()._complete(key, result)

    def _redraw(self, *_):
        from dynamical_system_lab import evaluate, display_parameter_name
        eigen, amplitude, phase = _axes(self.figure, self.host, cols=3)
        _, fg, blue, orange = _colors(self.host)
        for ax, title, ylabel in ((eigen, "Eigenvalues", "Real part"),
                                  (amplitude, "Local Hopf approximation", "Radius in normal form"),
                                  (phase, "Selected phase portrait", "y")):
            ax.set(title=title, ylabel=ylabel)
        phase.set(xlabel="x", aspect="equal")
        if r := self.result:
            samples = r['samples']
            name = self.scan_parameter
            parameter_label = display_parameter_name(name)
            index = self.slider.value()
            mu, point = samples[index], r['points'][index]
            self.selected.setText(f"{parameter_label} = {mu:.6g}")
            for ax in (eigen, amplitude):
                ax.set(xlabel=parameter_label, xlim=(samples[0], samples[-1]))
                ax.axvline(mu, color=fg, alpha=.35)
            eigen.axhline(0, color=fg, alpha=.45)
            for values in r['real_parts'].T:
                eigen.plot(samples, values, color=blue, linewidth=1.3)
            notes, nearby = [], []
            for crit in r['candidates']:
                location = crit['mu']
                eigen.scatter(location, 0, color=orange, s=23, zorder=5)
                notes.append(f"{parameter_label} ≈ {location:.6g}: {crit['kind']}  (ω ≈ {crit['omega']:.5g})")
                if crit['kind'].startswith('Unresolved'):
                    continue
                side = -np.sign(crit['alpha']/crit['slope'])
                width = .1*(samples[-1]-samples[0])
                offsets = np.linspace(0, side*width, 51)
                inside = (location+offsets >= samples[0]) & (location+offsets <= samples[-1])
                radius = np.sqrt(-crit['slope']*offsets[inside]/crit['alpha'])
                amplitude.plot(location+offsets[inside], radius, '-' if crit['alpha'] < 0 else '--',
                               color=blue if crit['alpha'] < 0 else orange)
                distance = mu-location
                if 0 < side*distance <= width:
                    nearby.append((abs(distance), crit))
            if np.isfinite(point).all():
                params = dict(self.problem['params'])
                params[name] = mu
                extent = self.local_radius
                if nearby:
                    _, crit = min(nearby, key=lambda item: item[0])
                    radius = math.sqrt(-crit['slope']*(mu-crit['mu'])/crit['alpha'])
                    theta = np.linspace(0, 2*np.pi, 241)
                    offsets = radius*crit['basis'] @ np.array([np.cos(theta), np.sin(theta)])
                    phase.plot(*(point[:, None]+offsets), ':', color=blue if crit['alpha'] < 0 else orange,
                               linewidth=1.8)
                    extent = max(extent, 1.2*np.max(abs(offsets)))
                s = np.linspace(-extent, extent, 19)
                x, y = np.meshgrid(point[0]+s, point[1]+s)
                u, v = (np.broadcast_to(evaluate(f, x, y, params), x.shape) for f in self.problem['fields'])
                speed = np.hypot(u, v)
                keep = np.isfinite(speed) & (speed > 0)
                phase.quiver(x[keep], y[keep], u[keep]/speed[keep], v[keep]/speed[keep], color=fg, alpha=.45)
                phase.scatter(*point, color=fg, s=23, zorder=5)
            else:
                phase.text(.5, .5, "Branch unresolved here", transform=phase.transAxes,
                           color=fg, ha="center")
            if not notes:
                notes.append("No nondegenerate trace crossing detected on the sampled branch.")
            if not r['complete']:
                notes.append("The local equilibrium branch could not be followed over the full interval.")
            self.status.setText("\n".join(notes) + "\nRadius curves and dotted phase-plane outlines are local approximations.")
        else:
            self.selected.setText("Phase portrait parameter")
        self.canvas.draw_idle()
