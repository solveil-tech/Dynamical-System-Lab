"""Compact travelling-wave tools, independent of the main Flow workspace."""
import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, QSignalBlocker
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QGroupBox, QLabel, QLineEdit, QPushButton, QScrollArea, QFrame)

from background_analysis import AnalysisProcess
from travelling_waves import reaction_model


PRESETS = {
    "Fisher logistic": ("r*u*(1-u/K)", {"r": 1., "K": 1.}, 1., 2., (-.1, 1.1)),
    "Budworm": ("R*u*(1-u/B)-u^2/(1+u^2)", {"R": .5, "B": 10.}, 1., 0., (-.2, 10.5)),
}


class TravellingWaveDialog(QDialog):
    def __init__(self, host):
        super().__init__(host)
        from dynamical_system_lab import NotationLineEdit, PopupComboBox, EquilibriumInfo
        self.host = host
        self.setWindowTitle("Travelling waves")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setMinimumSize(720, 600)
        self.resize(900, 650)
        self.values, self.editors = {}, {}
        self.result, self.key = None, None
        self._loading = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(450)
        self.timer.timeout.connect(self._analyze)
        self.worker = AnalysisProcess(self, timeout_ms=15000)
        self.worker.completed.connect(self._complete)
        self.worker.failed.connect(self._failed)
        self.finished.connect(self._close)

        layout = QVBoxLayout(self)
        box = QGroupBox("Reaction–diffusion  Cₜ = D Cₓₓ + f(C)")
        form = QVBoxLayout(box)
        row = QHBoxLayout()
        self.model = PopupComboBox()
        self.model.addItems([*PRESETS, "Custom"])
        self.model.setFixedWidth(150)
        self.reaction = NotationLineEdit()
        row.addWidget(QLabel("Model"))
        row.addWidget(self.model)
        row.addWidget(QLabel("f(u) ="))
        row.addWidget(self.reaction, 1)
        form.addLayout(row)
        row = QHBoxLayout()
        self.diffusion = host.make_spin(1, .001, 1e6, .1)
        self.speed = host.make_spin(2, -1e6, 1e6, .1)
        self.low = host.make_spin(-.1, -1e6, 1e6, .1)
        self.high = host.make_spin(1.1, -1e6, 1e6, .1)
        for label, widget in (("D", self.diffusion), ("v", self.speed),
                              ("u<sub>min</sub>", self.low), ("u<sub>max</sub>", self.high)):
            row.addWidget(QLabel(label))
            row.addWidget(widget, 1)
            widget.valueChanged.connect(self._changed)
        form.addLayout(row)
        self.parameter_box = QGroupBox("Params")
        self.parameter_grid = QGridLayout(self.parameter_box)
        form.addWidget(self.parameter_box)
        row = QHBoxLayout()
        self.left_state, self.right_state = PopupComboBox(), PopupComboBox()
        self.left_state.setMinimumWidth(0)
        self.right_state.setMinimumWidth(0)
        row.addWidget(QLabel("U(−∞)"))
        row.addWidget(self.left_state, 1)
        row.addWidget(QLabel("U(+∞)"))
        row.addWidget(self.right_state, 1)
        self.use_flow = QPushButton("Use in Flow")
        self.use_flow.setToolTip("Export x = u, y = w to the phase portrait; its ODE time represents ξ.")
        self.use_flow.setEnabled(False)
        self.use_flow.clicked.connect(self._use_in_flow)
        row.addWidget(self.use_flow)
        form.addLayout(row)
        layout.addWidget(box)

        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        self.info = EquilibriumInfo(line_spacing=1.1)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(140)
        scroll.setStyleSheet(host.equilibrium_scroll.styleSheet())
        scroll.setWidget(self.info)
        layout.addWidget(scroll)
        self.status = QLabel("Equations and parameters update automatically.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.model.currentTextChanged.connect(self._preset)
        self.reaction.textChanged.connect(self._reaction_changed)
        self.left_state.currentIndexChanged.connect(self._states_changed)
        self.right_state.currentIndexChanged.connect(self._states_changed)
        self._left_index = self._right_index = None
        self._preset("Fisher logistic")
        host.dark_mode_check.toggled.connect(self._theme_changed)
        if host._flow_biology_preset() == "Fisher travelling front":
            self.speed.setValue(host.params.get("v", 2))

    def _close(self, *_args):
        self.timer.stop()
        self.worker.close()

    def _preset(self, name):
        if name not in PRESETS:
            return
        source, self.values, diffusion, speed, bounds = PRESETS[name]
        self.values = dict(self.values)
        self._loading = True
        try:
            self.reaction.setText(source)
            for widget, value in zip((self.diffusion, self.speed, self.low, self.high),
                                      (diffusion, speed, *bounds)):
                widget.setValue(value)
            self._sync_parameters()
        finally:
            self._loading = False
        self._changed()

    def _reaction_changed(self, *_args):
        if self._loading:
            return
        with QSignalBlocker(self.model):
            self.model.setCurrentText("Custom")
        self._changed()

    def _sync_parameters(self):
        from dynamical_system_lab import format_formula_input, display_parameter_name
        tree, params = reaction_model(self.reaction.text(), self.values)
        if not self._loading:
            for name in params.keys() & self.editors.keys():
                value = float(self.editors[name].text())
                if not math.isfinite(value):
                    raise ValueError("Parameters must be finite numbers.")
                params[name] = value
        text = self.reaction.text()
        shown = format_formula_input(text, set(params) | {"u"})
        if shown != text:
            cursor = len(format_formula_input(text[:self.reaction.cursorPosition()], set(params) | {"u"}))
            with QSignalBlocker(self.reaction):
                self.reaction.setText(shown)
                self.reaction.setCursorPosition(min(cursor, len(shown)))
        self.values = params
        if set(self.editors) != set(params):
            while self.parameter_grid.count():
                self.parameter_grid.takeAt(0).widget().deleteLater()
            self.editors = {}
            for i, (name, value) in enumerate(params.items()):
                label = QLabel(display_parameter_name(name))
                editor = QLineEdit(f"{value:.7g}")
                editor.setMinimumWidth(0)
                editor.editingFinished.connect(lambda symbol=name: self._parameter_changed(symbol))
                self.parameter_grid.addWidget(label, i//3, (i%3)*2)
                self.parameter_grid.addWidget(editor, i//3, (i%3)*2+1)
                self.parameter_grid.setColumnStretch((i%3)*2+1, 1)
                self.editors[name] = editor
        for name, editor in self.editors.items():
            if not editor.hasFocus():
                editor.setText(f"{params[name]:.7g}")
        self.parameter_box.setVisible(bool(params))
        return tree

    def _parameter_changed(self, name):
        try:
            value = float(self.editors[name].text())
            if not math.isfinite(value):
                raise ValueError("Parameters must be finite numbers.")
            self.values[name] = round(value, 3)
            self.editors[name].setText(f"{self.values[name]:.7g}")
            if self.model.currentText() in PRESETS:
                capacity = self.values.get("K", self.values.get("B", 1))
                if capacity > 0:
                    self._loading = True
                    self.low.setValue(-.1*capacity)
                    self.high.setValue(1.1*capacity)
                    self._loading = False
            self._changed()
        except ValueError as exc:
            self._invalidate(str(exc))

    def _invalidate(self, message):
        self.timer.stop()
        self.worker.cancel()
        self.result = self.key = None
        self.use_flow.setEnabled(False)
        self.status.setText(message)
        self.info.set_rows([message], self._colors()[1])
        self._draw()

    def _changed(self, *_args):
        if self._loading:
            return
        if self.sender() not in (self.diffusion, self.speed):
            self._left_index = self._right_index = None
        try:
            self._sync_parameters()
        except (ValueError, SyntaxError, KeyError) as exc:
            self._invalidate(f"Reaction input: {exc}")
            return
        self._schedule()

    def _states_changed(self, *_args):
        self._left_index, self._right_index = self.left_state.currentIndex(), self.right_state.currentIndex()
        self._schedule()

    def _schedule(self):
        self.worker.cancel()
        self.key = self.result = None
        self.use_flow.setEnabled(False)
        self.info.set_rows(["Updating travelling-wave analysis…"], self._colors()[1])
        self.status.setText("Updating…")
        self._draw()
        self.timer.start()

    def _analyze(self):
        self.key = object()
        self.worker.submit(self.key, dict(reaction=self.reaction.text(), params=dict(self.values),
                           D=self.diffusion.value(), v=self.speed.value(),
                           bounds=(self.low.value(), self.high.value()),
                           left_index=self._left_index, right_index=self._right_index), "travelling_wave")

    def _complete(self, key, result):
        if key is not self.key:
            return
        self.result = result
        for combo, selected in ((self.left_state, result["left_index"]),
                                 (self.right_state, result["right_index"])):
            with QSignalBlocker(combo):
                combo.clear()
                combo.addItems([f"{root:.8g}" for root in result["roots"]])
                combo.setCurrentIndex(selected)
        self.info.set_rows(result["rows"], self._colors()[1])
        self.use_flow.setEnabled(True)
        self.status.setText("U(−∞) is the left spatial state; U(+∞) is the right spatial state.")
        self._draw()

    def _failed(self, key, message):
        if key is self.key:
            self._invalidate("Analysis unavailable: "+message)

    def _colors(self):
        return ("#000000", "#eeeeee") if self.host.dark_mode_check.isChecked() else ("#ffffff", "#111111")

    def _theme_changed(self, *_args):
        if self.result is not None:
            self.info.set_rows(self.result["rows"], self._colors()[1])
        self._draw()

    def _draw(self):
        self.figure.clear()
        bg, fg = self._colors()
        self.figure.set_facecolor(bg)
        reaction, phase = self.figure.subplots(1, 2)
        for ax in (reaction, phase):
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg, labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#555555" if bg == "#000000" else "#cccccc")
            ax.grid(alpha=.12, color=fg)
            ax.set_xlabel("u", color=fg)
        reaction.set(title="REACTION AND SIGNED AREA", ylabel="f(u)")
        phase.set(title="TRAVELLING-WAVE PHASE PLANE", ylabel="w = U′")
        for ax in (reaction, phase):
            ax.title.set(color=fg, fontsize=10)
            ax.yaxis.label.set_color(fg)
        result = self.result
        if result is not None:
            u, f = result["u"], result["reaction"]
            blue = "#80cfff" if bg == "#000000" else "#2453a5"
            reaction.axhline(0, color=fg, linewidth=.6, alpha=.5)
            reaction.plot(u, f, color=blue, linewidth=1.3)
            if result["area"] is not None:
                low, high = sorted((result["left"], result["right"]))
                interval = (u >= low) & (u <= high)
                reaction.fill_between(u, 0, f, where=interval & (f >= 0), interpolate=True,
                                      color="#ee9b32", alpha=.4)
                reaction.fill_between(u, 0, f, where=interval & (f < 0), interpolate=True,
                                      color=blue, alpha=.3)
            reaction.scatter(result["roots"], np.zeros(len(result["roots"])),
                             color="#e34c45", edgecolors="none", s=24, zorder=4)
            phase.quiver(*result["arrows"], color=fg, alpha=.38, angles="xy", scale_units="xy", scale=28)
            phase.axhline(0, color="#ee9b32", linewidth=1)
            if result["nullcline"] is not None:
                phase.plot(u, result["nullcline"], color="#27a36a", linewidth=1)
            else:
                for root in result["roots"]:
                    phase.axvline(root, color="#27a36a", linewidth=1)
            phase.scatter(result["roots"], np.zeros(len(result["roots"])),
                          color="#e34c45", edgecolors="none", s=24, zorder=4)
            phase.set(xlim=(u[0], u[-1]), ylim=(-result["w_scale"], result["w_scale"]))
        self.canvas.draw_idle()

    def _use_in_flow(self):
        if self.result is None:
            return
        self.host._load_travelling_wave(self.result, self.diffusion.value(), self.speed.value())
        self.accept()
