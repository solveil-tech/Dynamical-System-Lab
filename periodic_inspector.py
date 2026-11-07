"""Inspection of a detected planar periodic orbit."""

import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from background_analysis import AnalysisProcess


class PeriodicOrbitInspector(QDialog):
    def __init__(self, parent, fields, params, bounds, orbit, dark):
        super().__init__(parent)
        self.setWindowTitle("Periodic orbit analysis")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.orbit = orbit
        self.bg, self.fg = ("#000000", "#ededed") if dark else ("#ffffff", "#222222")
        self.color = "#ff92df" if dark else "#ba277f"
        layout = QVBoxLayout(self)
        self.summary = QLabel("Calculating the first return…")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        guide = QLabel(
            "Return map: start at r on the section and read P(r) at the next same-direction crossing. "
            "Where P(r) = r, the orbit returns to its starting point.\n"
            "Transverse multiplier: the factor by which a small offset from the orbit changes after one return."
        )
        guide.setWordWrap(True)
        layout.addWidget(guide)
        self.figure = Figure(layout="constrained", facecolor=self.bg)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(850, 480)

        self.worker = AnalysisProcess(self, timeout_ms=15000)
        self.worker.completed.connect(self._complete)
        self.worker.failed.connect(self._failed)
        self.finished.connect(lambda _: self.worker.close())
        self._draw()
        self.worker.submit("orbit", dict(fields=fields, params=params, bounds=bounds, orbit=orbit),
                           "periodic_inspection")

    def _complete(self, _key, result):
        points = result["points"]
        ranges = np.ptp(points, axis=0)
        period = result["period"]
        log_multiplier = result["log_multiplier"]
        stability = ("Closed-orbit family" if self.orbit["family"]
                     else self.orbit.get("stability", "undetermined").capitalize())
        if log_multiplier is None or not math.isfinite(log_multiplier):
            multiplier = "unavailable"
        elif abs(log_multiplier) > 700:
            multiplier = f"exp({log_multiplier:.5g})"
        else:
            multiplier = f"{math.exp(log_multiplier):.6g}"
        if period is None:
            stability = f"{stability} (from detection)"
        closure = result["closure"]
        closure_label = "Section closure error (backward check)" if result["backward"] else "Section closure error"
        period_text = f"{period:.6g}" if period is not None else "unverified"
        closure_text = f"{closure:.3g}" if math.isfinite(closure) else "no return"
        self.summary.setText(
            f"{stability}   ·   Period T ≈ {period_text}   ·   Transverse multiplier ≈ {multiplier}\n"
            f"Amplitudes (half peak-to-peak)   x ≈ {ranges[0]/2:.4g}   y ≈ {ranges[1]/2:.4g}"
            f"   ·   {closure_label} ≈ {closure_text}"
        )
        self._draw(result)

    def _failed(self, _key, message):
        self.summary.setText(f"Orbit measurements unavailable. {message}")
        self._draw(empty_message="Return map unavailable")

    def _draw(self, result=None, empty_message="Waiting for returns…"):
        self.figure.clear()
        phase, mapping = self.figure.subplots(1, 2)
        for ax in (phase, mapping):
            ax.set_facecolor(self.bg)
            ax.tick_params(colors=self.fg, labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(self.fg)
                spine.set_alpha(.4)
            ax.grid(color=self.fg, alpha=.12)
        points = self.orbit["points"] if result is None else result["points"]
        center = self.orbit["center"]
        radius = self.orbit["points"][0, 0] - center[0]
        phase.plot(*points.T, color=self.color)
        phase.plot([center[0], center[0]+1.12*radius], [center[1]]*2,
                   "--", color=self.fg, alpha=.7, label="Poincaré section")
        phase.scatter(center[0]+radius, center[1], color=self.color, edgecolors=self.fg, zorder=3)
        phase.set_aspect("equal", adjustable="box")
        phase.legend(loc="upper center", bbox_to_anchor=(.5, -.19),
                     frameon=False, labelcolor=self.fg, fontsize=8)
        phase.set_title("Orbit and section", color=self.fg)
        phase.set_xlabel("x", color=self.fg)
        phase.set_ylabel("y", color=self.fg)
        mapping.set_title("Poincaré return map", color=self.fg)
        mapping.set_xlabel("Starting position r on section", color=self.fg)
        mapping.set_ylabel("Return position P(r) on section", color=self.fg)
        if result is not None and np.isfinite(result["returns"]).any():
            radii, returned = result["radii"], result["returns"]
            mapping.plot(radii, radii, "--", color=self.fg, alpha=.65, label="P(r) = r")
            mapping.plot(radii, returned, ".-", color=self.color, label="Same-direction returns")
            if result["period"] is not None:
                mapping.scatter(radius, radius, color=self.color, edgecolors=self.fg, zorder=3)
            mapping.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncols=2,
                           frameon=False, labelcolor=self.fg, fontsize=8)
        else:
            mapping.text(.5, .5, empty_message if result is None else "No return measured",
                         ha="center", va="center", color=self.fg, transform=mapping.transAxes)
        self.canvas.draw_idle()
