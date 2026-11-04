"""Independent scalar bifurcation tab using the application's native styles."""
import re
import numpy as np
from PySide6.QtCore import QTimer, QSignalBlocker, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QCheckBox, QScrollArea, QFrame, QFileDialog)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from background_analysis import AnalysisProcess


PRESETS = {
    "Saddle-node": "mu-x^2",
    "Transcritical": "mu*x-x^2",
    "Supercritical pitchfork": "mu*x-x^3",
    "Subcritical pitchfork": "mu*x+x^3",
    "Hysteresis (two folds)": "mu+x-x^3",
    "Subcritical with saturation": "mu*x+x^3-x^5",
}


def continuous_paths(segments):
    """Keep dash phase continuous across a sampled contour's short segments."""
    paths = []
    for segment in segments:
        if paths and np.allclose(paths[-1][-1], segment[0], rtol=0, atol=1e-10):
            paths[-1].append(segment[1])
        else:
            paths.append(list(segment))
    return [np.asarray(path) for path in paths]


class BifurcationPanel(QWidget):
    def __init__(self, host, editor_class, combo_class, info_class, formatter):
        super().__init__(host)
        self.host, self.formatter = host, formatter
        self.result = self.trace_result = self.analysis_key = self.trace_key = None
        self.dark, self.initialized = False, False
        self._press = None
        self.analysis = AnalysisProcess(self, timeout_ms=20000)
        self.analysis.completed.connect(self._complete)
        self.analysis.failed.connect(self._failed)
        self.tracer = AnalysisProcess(self, timeout_ms=20000)
        self.tracer.completed.connect(self._traced)
        self.tracer.failed.connect(self._trace_failed)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(650)
        self.debounce.timeout.connect(self.analyze)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        box = QGroupBox("Scalar system  dx/dt = f(x, μ)")
        grid = QGridLayout(box)
        self.preset = combo_class()
        self.preset.addItems(["Custom"]+list(PRESETS))
        self.preset.setCurrentText("Hysteresis (two folds)")
        self.formula = editor_class()
        self.formula.setText(formatter("μ+x-x^3", {"μ"}))
        self.formula.setToolTip("Use x and μ (or mu), implicit products and powers. Classical bifurcation classification requires a smooth scalar model.")
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze)
        grid.addWidget(QLabel("Model"), 0, 0)
        grid.addWidget(self.preset, 0, 1, 1, 3)
        grid.addWidget(self.analyze_btn, 0, 4, 1, 2)
        grid.addWidget(QLabel("f(x, μ) ="), 1, 0)
        grid.addWidget(self.formula, 1, 1, 1, 5)
        self.bounds = []
        for i, (label, value) in enumerate((("μ min", -2), ("μ max", 2), ("x min", -2.5), ("x max", 2.5))):
            spin = host.make_spin(value, -10000, 10000, 0.1)
            spin.setMinimumWidth(60)
            self.bounds.append(spin)
            grid.addWidget(QLabel(label), 2+i//2, 3*(i%2))
            grid.addWidget(spin, 2+i//2, 3*(i%2)+1, 1, 2)
            spin.valueChanged.connect(self._changed)
        layout.addWidget(box)
        box = QGroupBox("Parameter sweep (click the diagram to select μ₀, x₀)")
        grid = QGridLayout(box)
        self.mode, self.direction = combo_class(), combo_class()
        self.mode.addItems(["Quasistatic", "Finite-rate"])
        self.direction.addItems(["Increasing μ", "Decreasing μ"])
        self.roundtrip = QCheckBox("Round trip")
        self.roundtrip.setChecked(True)
        self.rate = host.make_spin(0.1, 0.001, 100, 0.01, compact_decimals=True)
        self.rate.setEnabled(False)
        self.perturbation = host.make_spin(0.001, -1, 1, 0.001)
        self.perturbation.setToolTip("Explicit signed perturbation when an exact equilibrium loses stability. Zero preserves invariant branches.")
        self.mu0 = host.make_spin(-1, -10000, 10000, 0.1)
        self.x0 = host.make_spin(-1.3, -10000, 10000, 0.1)
        for col, widget in enumerate((self.mode, self.direction, self.roundtrip)):
            grid.addWidget(widget, 0, col*2, 1, 2)
        for col, (label, widget) in enumerate((("μ₀", self.mu0), ("x₀", self.x0), ("|dμ/dt|", self.rate))):
            grid.addWidget(QLabel(label), 1, col*2)
            grid.addWidget(widget, 1, col*2+1)
        grid.addWidget(QLabel("Perturbation"), 2, 0)
        grid.addWidget(self.perturbation, 2, 1)
        self.trace_btn = QPushButton("Trace sweep")
        self.trace_btn.clicked.connect(self.trace)
        grid.addWidget(self.trace_btn, 2, 2, 1, 2)
        clear = QPushButton("Clear trace")
        clear.clicked.connect(self.clear_trace)
        grid.addWidget(clear, 2, 4, 1, 2)
        layout.addWidget(box)
        row = QHBoxLayout()
        self.show_derivative = QCheckBox("Show fₓ = 0")
        self.show_derivative.setChecked(True)
        self.show_derivative.toggled.connect(lambda _: self.draw())
        row.addWidget(self.show_derivative)
        row.addStretch()
        for label, callback in (("+", lambda: self.zoom(.8)), ("−", lambda: self.zoom(1.25)),
                                ("Reset view", lambda: self.draw(reset=True)), ("Export PNG…", self.export_png)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.figure = Figure(figsize=(7, 5), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)
        self.key_label = QLabel("Blue solid: stable   Red dashed: unstable   Orange: semistable / inconclusive   Gray dotted: fₓ = 0\nPurple: outward sweep   Green: return sweep   Dashed sweep connectors: jumps / fast relaxation")
        self.key_label.setWordWrap(True)
        layout.addWidget(self.key_label)
        self.info = info_class()
        scroll = QScrollArea()
        self.info_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedHeight(114)
        scroll.setWidget(self.info)
        layout.addWidget(scroll)
        self.status = QLabel("Enter f(x, μ), then Analyze. This tab has its own equations and settings.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.preset.currentTextChanged.connect(self._preset)
        self.formula.textChanged.connect(self._formula_changed)
        for spin in (self.mu0, self.x0, self.rate, self.perturbation):
            spin.valueChanged.connect(self.clear_trace)
        self.mode.currentTextChanged.connect(self._mode_changed)
        self.direction.currentTextChanged.connect(self.clear_trace)
        self.roundtrip.toggled.connect(self.clear_trace)
        self.canvas.mpl_connect("button_press_event", self._mouse_down)
        self.canvas.mpl_connect("button_release_event", self._mouse_up)
        self.canvas.mpl_connect("scroll_event", lambda event: self.zoom(.8 if event.button == "up" else 1.25))
        self.draw(reset=True)

    def activate(self):
        if not self.initialized:
            self.initialized = True
            self.analyze()

    def _preset(self, name):
        if name in PRESETS:
            self.formula.setText(PRESETS[name])

    def _formula_changed(self, _text):
        text = re.sub(r"\bmu\b", "μ", self.formula.text())
        shown = self.formatter(text, {"μ"})
        if shown != self.formula.text():
            cursor = len(self.formatter(self.formula.text()[:self.formula.cursorPosition()], {"μ", "mu"}))
            blocker = QSignalBlocker(self.formula)
            self.formula.setText(shown)
            self.formula.setCursorPosition(min(cursor, len(shown)))
            del blocker
        name = self.preset.currentText()
        if name in PRESETS and shown != self.formatter(re.sub(r"\bmu\b", "μ", PRESETS[name]), {"μ"}):
            blocker = QSignalBlocker(self.preset)
            self.preset.setCurrentText("Custom")
            del blocker
        self._changed()

    def _changed(self, *_args):
        self.analysis_key = self.trace_key = None
        self.analysis.cancel()
        self.tracer.cancel()
        self.result = self.trace_result = None
        self.status.setText("Settings changed. Updating bifurcation diagram…")
        self.draw(reset=True)
        if self.initialized:
            self.debounce.start()

    def configuration(self):
        bounds = tuple(spin.value() for spin in self.bounds)
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("Each minimum must be smaller than its maximum.")
        return dict(formula=self.formula.text(), bounds=bounds)

    def analyze(self, *_args):
        self.debounce.stop()
        self.initialized = True
        try:
            data = self.configuration()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        key = (data["formula"], data["bounds"])
        if key == self.analysis_key:
            return
        self.analysis_key = key
        self.trace_key = None
        self.tracer.cancel()
        self.result = self.trace_result = None
        self.info.set_rows(["Computing equilibrium branches and critical points…"], self.fg)
        self.status.setText("Analyzing f = 0 and fₓ = 0…")
        self.analysis.submit(key, data, "bifurcation")

    @property
    def fg(self):
        return "#ffffff" if self.dark else "#111111"

    def _complete(self, key, result):
        if key != self.analysis_key:
            return
        self.result = result
        self.status.setText("Ready. Click inside the diagram to trace a parameter sweep.")
        self._info()
        self.draw(reset=True)

    def _failed(self, key, message):
        if key == self.analysis_key:
            self.status.setText("Analysis incomplete: "+message)
            self.info.set_rows(["Analysis incomplete: "+message], self.fg)
            self.analysis_key = None

    def _mode_changed(self, _text):
        finite = self.mode.currentIndex() == 1
        self.rate.setEnabled(finite)
        self.perturbation.setEnabled(not finite)
        self.clear_trace()

    def clear_trace(self, *_args):
        self.tracer.cancel()
        self.trace_key = None
        self.trace_result = None
        self._info()
        self.draw()

    def trace(self, *_args):
        if self.result is None:
            self.status.setText("Analyze the current formula successfully before tracing.")
            return
        data = self.configuration()
        initial = (self.mu0.value(), self.x0.value())
        m0, m1, x0, x1 = data["bounds"]
        if not (m0 <= initial[0] <= m1 and x0 <= initial[1] <= x1):
            self.status.setText("The initial point must lie inside the analysis window.")
            return
        data.update(initial=initial, direction=1 if self.direction.currentIndex() == 0 else -1,
                    roundtrip=self.roundtrip.isChecked(), rate=self.rate.value(),
                    perturbation=self.perturbation.value(), mode="finite" if self.mode.currentIndex() else "quasistatic")
        self.trace_key = (self.analysis_key, tuple(sorted((k, str(v)) for k, v in data.items())))
        self.trace_result = None
        self.draw()
        self.status.setText("Computing parameter sweep…")
        self.tracer.submit(self.trace_key, data, "bifurcation_trace")

    def _traced(self, key, result):
        if key == self.trace_key:
            self.trace_result = result
            self.status.setText(result["status"])
            self._info()
            self.draw()

    def _trace_failed(self, key, message):
        if key == self.trace_key:
            self.status.setText("Sweep incomplete: "+message)

    def _info(self):
        if self.result is None:
            self.info.set_rows(["Enter a scalar system and select the analysis window."], self.fg)
            return
        rows = []
        for p in self.result["critical"]:
            rows.append(f"$(\\mu,x)=({p['mu_text']},{p['x_text']})$    {p['kind']}    {p['stability']} at the critical parameter")
        rows += self.result["notes"]
        if self.trace_result:
            rows.append(self.trace_result["reversibility"])
            if self.trace_result["mode"] == "quasistatic":
                rows.append("Quasistatic branch following; dashed connectors represent fast relaxation, not finite-time discontinuities. Perturbation applies only at loss of stability.")
        self.info.set_rows(rows, self.fg)

    def set_dark(self, dark):
        self.dark = bool(dark)
        self.info_scroll.setStyleSheet(self.host.equilibrium_scroll.styleSheet())
        self._info()
        self.draw()

    def draw(self, reset=False):
        limits = (self.ax.get_xlim(), self.ax.get_ylim())
        self.ax.clear()
        bg, fg = ("#000000", "#ffffff") if self.dark else ("#ffffff", "#222222")
        self.figure.set_facecolor(bg)
        self.ax.set_facecolor(bg)
        self.ax.tick_params(colors=fg, labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#555555" if self.dark else "#cccccc")
        self.ax.set_xlabel("μ", color=fg)
        self.ax.set_ylabel("x", color=fg)
        self.ax.set_title("BIFURCATION DIAGRAM", color=fg, fontsize=11, fontweight="bold")
        self.ax.grid(alpha=.1, color=fg)
        if self.result:
            segments, kind = self.result["segments"], self.result["stability"]
            for code, color, style in ((-1, "#80cfff" if self.dark else "#2453a5", "solid"),
                                         (1, "#ff7272" if self.dark else "#ca4040", "dashed"),
                                         (0, "#e29834", "solid"), (2, "#e29834", "dashdot")):
                if np.any(kind == code):
                    self.ax.add_collection(LineCollection(continuous_paths(segments[kind == code]), colors=color,
                                            linewidths=1.7, linestyles=style))
            if self.show_derivative.isChecked() and len(self.result["derivative"]):
                self.ax.add_collection(LineCollection(continuous_paths(self.result["derivative"]), colors="#999999",
                                        linewidths=1, linestyles="dotted", alpha=.75))
            for p in self.result["critical"]:
                self.ax.scatter(p["mu"], p["x"], c="#e35b25", s=30, edgecolors="none", zorder=6)
                self.ax.annotate(p["kind"], (p["mu"], p["x"]), xytext=(8, 9),
                                 textcoords="offset points", fontsize=8, color=fg)
        if self.trace_result:
            for i, path in enumerate(self.trace_result["segments"]):
                if len(path):
                    color = "#c0429b" if self.trace_result["segment_legs"][i] == 0 else "#219c79"
                    self.ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.1)
                    if len(path) > 5:
                        j = len(path)//2
                        self.ax.annotate("", xy=path[j+1], xytext=path[j-2],
                                         arrowprops=dict(arrowstyle="->", color=color))
            if self.trace_result["jumps"]:
                self.ax.add_collection(LineCollection(self.trace_result["jumps"], colors="#c0429b",
                                        linewidths=1.7, linestyles="dashed", zorder=5))
            self.ax.scatter(*self.trace_result["initial"], s=28, color="#c0429b", zorder=7, edgecolors="none")
        if reset:
            m0, m1, x0, x1 = (spin.value() for spin in self.bounds)
            if m0 < m1 and x0 < x1:
                self.ax.set_xlim(m0, m1)
                self.ax.set_ylim(x0, x1)
        else:
            self.ax.set_xlim(limits[0])
            self.ax.set_ylim(limits[1])
        self.canvas.draw_idle()

    def zoom(self, factor):
        for get, set_ in ((self.ax.get_xlim, self.ax.set_xlim), (self.ax.get_ylim, self.ax.set_ylim)):
            low, high = get()
            center = (low+high)/2
            set_(center+(low-center)*factor, center+(high-center)*factor)
        self.canvas.draw_idle()

    def _mouse_down(self, event):
        self._press = (event.x, event.y) if event.inaxes is self.ax and event.button == 1 else None

    def _mouse_up(self, event):
        pressed, self._press = self._press, None
        if pressed is None or event.inaxes is not self.ax or event.button != 1:
            return
        if np.hypot(event.x-pressed[0], event.y-pressed[1]) > 5:
            return
        for spin, value in ((self.mu0, event.xdata), (self.x0, event.ydata)):
            blocker = QSignalBlocker(spin)
            spin.setValue(value)
            del blocker
        self.trace()

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export bifurcation diagram", "bifurcation.png", "PNG (*.png)")
        if path:
            self.figure.savefig(path, dpi=200, facecolor=self.figure.get_facecolor())

    def close_workers(self):
        self.debounce.stop()
        self.analysis.close()
        self.tracer.close()
