"""Details and local plots for a selected Flow equilibrium."""

import ctypes
import math
from ctypes import wintypes

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
from PySide6.QtCore import QPoint, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLayout,
    QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from background_analysis import AnalysisProcess


class _WindowPosition(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND), ("insert_after", wintypes.HWND),
                ("x", ctypes.c_int), ("y", ctypes.c_int),
                ("width", ctypes.c_int), ("height", ctypes.c_int), ("flags", wintypes.UINT)]


def _real_eigendirections(details):
    vectors = details.get("eigenvectors")
    if vectors is None:
        return []
    directions = []
    for value, vector in zip(details["eigenvalues_numeric"], vectors.T):
        if value.imag or np.any(vector.imag):
            continue
        direction = vector.real.copy()
        direction *= np.sign(direction[np.argmax(np.abs(direction))])
        if not any(value.real == previous and np.allclose(direction, other) for previous, other in directions):
            directions.append((value.real, direction))
    return directions


class EquilibriumMap(FigureCanvasQTAgg):
    point_selected = Signal(int)

    def __init__(self, results, radius, background, foreground, stable_color, unstable_color):
        figure = Figure(layout="constrained", facecolor=background)
        super().__init__(figure)
        self.setMinimumSize(275, 270)
        self.coordinates = np.asarray([point for point, _ in results], dtype=float)
        self.two_d = self.coordinates.shape[1] == 2
        self.ax = figure.add_subplot(projection=None if self.two_d else "3d", facecolor=background)
        self.ax.set_title("Equilibrium distribution", color=foreground, fontsize=11)
        self.ax.tick_params(colors=foreground, labelsize=8)
        axes = [(self.ax.xaxis, self.ax.set_xlim, "x"), (self.ax.yaxis, self.ax.set_ylim, "y")]
        if self.two_d:
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.grid(color=foreground, alpha=.13)
            self.ax.xaxis.set_major_locator(MaxNLocator(6))
            self.ax.yaxis.set_major_locator(MaxNLocator(6))
            for spine in self.ax.spines.values():
                spine.set_color(foreground)
                spine.set_alpha(.35)
        else:
            axes.append((self.ax.zaxis, self.ax.set_zlim, "z"))
            self.ax.mouse_init(rotate_btn=MouseButton.RIGHT, zoom_btn=[])
            self.ax.set_box_aspect((1, 1, 1))

        minimum, maximum = self.coordinates.min(axis=0), self.coordinates.max(axis=0)
        half_span = max(float((maximum - minimum).max()) * .62, radius)
        self._reference_span = 2 * half_span
        for (axis, set_limits, name), center in zip(axes, (minimum + maximum) / 2):
            axis.set_label_text(name, color=foreground)
            axis.get_offset_text().set_color(foreground)
            if not self.two_d:
                axis.set_pane_color((0, 0, 0, 0))
                axis.set_major_locator(MaxNLocator(4))
            set_limits(center - half_span, center + half_span)

        kinds = []
        for _, details in results:
            if details["type"].startswith("Stable "):
                kinds.append("Stable")
            elif details["linearization"].startswith("Unstable"):
                kinds.append("Unstable")
            else:
                kinds.append("Other")
        for label, color in (("Stable", stable_color), ("Unstable", unstable_color), ("Other", "#858585")):
            selected = [kind == label for kind in kinds]
            if any(selected):
                self.ax.scatter(*self.coordinates[selected].T, color=color, marker="o", s=42,
                                linewidths=0, label=label, zorder=4)
        self.ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), ncols=2,
                       frameon=False, fontsize=8, labelcolor=foreground)
        self.highlight, = self.ax.plot(*([[]] * self.coordinates.shape[1]), linestyle="none", marker="o",
                                       markersize=13, markerfacecolor="none", markeredgecolor=foreground,
                                       markeredgewidth=1.3, zorder=6)
        self._press = None
        self._dragging = False
        for name, handler in (("button_press_event", self._mouse_down), ("motion_notify_event", self._mouse_move),
                              ("button_release_event", self._mouse_up), ("scroll_event", self._scroll)):
            self.mpl_connect(name, handler)

    def mark_selected(self, index):
        point = self.coordinates[index, :, None]
        if self.two_d:
            self.highlight.set_data(*point)
        else:
            self.highlight.set_data_3d(*point)
        self.draw_idle()

    def _mouse_down(self, event):
        if event.inaxes is self.ax and event.button == MouseButton.LEFT:
            self._press = (event.x, event.y)

    def _mouse_move(self, event):
        if self._press is None:
            return
        if not self._dragging:
            dx, dy = event.x - self._press[0], event.y - self._press[1]
            if math.hypot(dx, dy) < QApplication.startDragDistance() * self.device_pixel_ratio:
                return
            self._dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.ax.start_pan(*self._press, MouseButton.LEFT)
        self.ax.drag_pan(MouseButton.LEFT, None, event.x, event.y)
        self.draw_idle()

    def _mouse_up(self, event):
        if self._press is None or event.button != MouseButton.LEFT:
            return
        self._mouse_move(event)
        self._press = None
        if self._dragging:
            self._dragging = False
            self.ax.end_pan()
            self.unsetCursor()
            return
        if event.inaxes is not self.ax:
            return
        points = self.coordinates
        if not self.two_d:
            x, y, _ = proj3d.proj_transform(*points.T, self.ax.get_proj())
            points = np.column_stack((x, y))
        pixels = self.ax.transData.transform(points)
        distance = np.hypot(*(pixels - [event.x, event.y]).T)
        x, y = pixels.T
        box = self.ax.bbox
        visible = (box.x0 <= x) & (x <= box.x1) & (box.y0 <= y) & (y <= box.y1)
        distance[~visible] = np.inf
        index = int(np.argmin(distance))
        if distance[index] <= 10 * self.device_pixel_ratio:
            self.point_selected.emit(index)

    def _scroll(self, event):
        if event.inaxes is not self.ax or self._press is not None or event.step == 0:
            return
        factor = .8 if event.step > 0 else 1.25
        axes = [(self.ax.get_xlim(), self.ax.set_xlim), (self.ax.get_ylim(), self.ax.set_ylim)]
        if not self.two_d:
            axes.append((self.ax.get_zlim(), self.ax.set_zlim))
        anchors = (event.xdata, event.ydata) if self.two_d else [sum(bounds) / 2 for bounds, _ in axes]
        bounds = [(anchor + (low - anchor) * factor, anchor + (high - anchor) * factor)
                  for anchor, ((low, high), _) in zip(anchors, axes)]
        if all(self._reference_span * 1e-8 < high - low < self._reference_span * 1e6 for low, high in bounds):
            for limits, (_, set_limits) in zip(bounds, axes):
                set_limits(limits)
            self.draw_idle()


class EquilibriumInspector(QDialog):
    def __init__(self, parent, results, fields, params, radius, dark):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.MSWindowsFixedSizeDialogHint)
        self.setWindowTitle("Equilibrium inspector")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.results, self.fields, self.params = results, fields, params
        self.radius = radius
        self.background, self.foreground = ("#000000", "#ededed") if dark else ("#ffffff", "#222222")
        self.stable_color, self.unstable_color = ("#80cfff", "#ff8080") if dark else ("#2453a5", "#bd3434")
        self.two_d = len(fields) == 2
        self.selected_index = None
        self._changing_native_dpi = False

        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        content = QHBoxLayout()
        layout.addLayout(content, 1)
        overview = QVBoxLayout()
        self.point_map = EquilibriumMap(results, radius, self.background, self.foreground,
                                        self.stable_color, self.unstable_color)
        overview.addWidget(self.point_map, 1)
        hint = "Click to inspect · Drag to pan · Scroll to zoom"
        if not self.two_d:
            hint += " · Right-drag to rotate"
        navigation_hint = QLabel(hint)
        navigation_hint.setWordWrap(True)
        overview.addWidget(navigation_hint)
        point_type = QFormLayout()
        point_type.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        point_type.setVerticalSpacing(3)
        self.summary = self._detail(point_type, "Equilibrium type and stability")
        self.summary.setText("No point selected.")
        self.summary.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.summary.setMinimumHeight(4 * self.summary.fontMetrics().lineSpacing())
        overview.addLayout(point_type)
        content.addLayout(overview, 4)

        self.analysis_stack = QStackedWidget()
        prompt = QLabel("Click an equilibrium on the left to open its analysis.")
        prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt.setWordWrap(True)
        self.analysis_stack.addWidget(prompt)
        self.analysis_page = QWidget()
        analysis = QVBoxLayout(self.analysis_page)
        analysis.setContentsMargins(0, 0, 0, 0)
        self.selected_point = QLabel()
        self.selected_point.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        analysis.addWidget(self.selected_point)
        details = QHBoxLayout()
        classification, spectrum = QFormLayout(), QFormLayout()
        for form in (classification, spectrum):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
            form.setVerticalSpacing(3)
            details.addLayout(form, 1)
        self.matrix = self._detail(spectrum, "Jacobian J")
        self.matrix.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.spectrum = self._detail(spectrum, "Eigenvalues")
        self.directions = self._detail(spectrum, "Real eigenvectors (unit length)")
        analysis.addLayout(details)
        self.analysis_stack.addWidget(self.analysis_page)
        content.addWidget(self.analysis_stack, 5)

        if self.two_d:
            self.invariants = self._detail(classification, "Planar invariants")
            self.plots = QTabWidget()
            self.trace_canvas = self._canvas()
            self.saddle_canvas = self._canvas()
            self.plots.addTab(self.trace_canvas, "Trace–determinant")
            saddle_page = QWidget()
            saddle_layout = QVBoxLayout(saddle_page)
            saddle_layout.addWidget(self.saddle_canvas, 1)
            self.saddle_status = QLabel()
            self.saddle_status.setWordWrap(True)
            saddle_layout.addWidget(self.saddle_status)
            self.plots.addTab(saddle_page, "Saddle manifolds")
            self.plots.setTabEnabled(1, False)
            rgb = ", ".join(str(value) for value in QColor(self.foreground).getRgb()[:3])
            self.plots.tabBar().setStyleSheet(f"""
                QTabBar::tab:disabled {{ color: rgba({rgb}, 128);
                    background-color: rgba({rgb}, 10); border: 1px solid rgba({rgb}, 38); }}
            """)
            analysis.insertWidget(1, self.plots, 1)
            self.worker = AnalysisProcess(self)
            self.worker.completed.connect(self._manifolds_ready)
            self.worker.failed.connect(self._manifolds_failed)
            self.finished.connect(lambda _: self.worker.close())
            self.manifold_cache = {}
            self.pending_manifold = None
            self.plot_layout_timer = QTimer(self)
            self.plot_layout_timer.setSingleShot(True)
            self.plot_layout_timer.timeout.connect(self._align_plot_frames)
            for canvas in (self.point_map, self.trace_canvas, self.saddle_canvas):
                canvas.mpl_connect("resize_event", self._schedule_plot_alignment)
            self.plots.currentChanged.connect(self._show_manifolds)
            self.plots.currentChanged.connect(self._schedule_plot_alignment)
        else:
            analysis.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        width, height = (850, 590) if self.two_d else (740, 470)
        self.setFixedSize(width, height)
        self.point_map.point_selected.connect(self._select)

    def nativeEvent(self, event_type, message):
        if event_type == b"windows_generic_MSG" and self.isVisible():
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == 0x02E0:  # WM_DPICHANGED
                self._changing_native_dpi = True
                QTimer.singleShot(0, self._finish_native_dpi_change)
            elif msg.message == 0x0046 and not self._changing_native_dpi:  # WM_WINDOWPOSCHANGING
                # Windows can stretch even a fixed-frame dialog during a
                # title-bar drag. Keep its move, but reject native resizing.
                _WindowPosition.from_address(msg.lParam).flags |= 0x0001  # SWP_NOSIZE
        return super().nativeEvent(event_type, message)

    def _finish_native_dpi_change(self):
        self._changing_native_dpi = False

    @staticmethod
    def _detail(form, title):
        label = QLabel()
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        form.addRow(title, label)
        return label

    @staticmethod
    def _canvas():
        canvas = FigureCanvasQTAgg(Figure(layout="none"))
        canvas.setMinimumSize(370, 290)
        return canvas

    def _schedule_plot_alignment(self, *_):
        self.plot_layout_timer.start(0)

    def _align_plot_frames(self):
        if self.selected_index is None:
            return
        map_canvas = self.point_map
        right_canvas = self.trace_canvas if self.plots.currentIndex() == 0 else self.saddle_canvas
        if not right_canvas.figure.axes or not right_canvas.isVisible():
            return

        # Use dialog coordinates for the axes frames, since the two canvases
        # have different space above and below them for tabs, legends and text.
        map_y = map_canvas.mapTo(self, QPoint()).y()
        right_y = right_canvas.mapTo(self, QPoint()).y()
        top = max(map_y + 27, right_y + 27)
        bottom = min(map_y + map_canvas.height() - 80, right_y + right_canvas.height() - 48)
        left_pad, right_pad = 62, 21
        map_width = map_canvas.width() - left_pad - right_pad
        right_width = right_canvas.width() - left_pad - right_pad
        map_axes = map_canvas.ax
        right_axes = right_canvas.figure.axes[0]
        map_ratio = map_axes.get_data_ratio()
        right_ratio = right_axes.get_data_ratio() if right_canvas is self.saddle_canvas else None
        height = min(bottom - top, map_width * map_ratio)
        if right_ratio is not None:
            height = min(height, right_width * right_ratio)
        if height <= 0 or right_width <= 0:
            return
        top += (bottom - top - height) / 2

        map_canvas.figure.set_layout_engine("none")
        for canvas, axes, canvas_y, available_width, ratio in (
                (map_canvas, map_axes, map_y, map_width, map_ratio),
                (right_canvas, right_axes, right_y, right_width, right_ratio)):
            width = height / ratio if ratio is not None else available_width
            left = left_pad + (available_width - width) / 2
            axes.set_position((left / canvas.width(), (canvas_y + canvas.height() - top - height) / canvas.height(),
                               width / canvas.width(), height / canvas.height()))
            canvas.draw_idle()

        # Keep the legend below the x label when the plot frame moves.
        map_axes.get_legend().set_bbox_to_anchor((.5, -38 / height), transform=map_axes.transAxes)

    def _axes(self, canvas):
        canvas.figure.clear()
        canvas.figure.set_facecolor(self.background)
        ax = canvas.figure.add_subplot()
        ax.set_facecolor(self.background)
        ax.tick_params(colors=self.foreground, labelsize=9)
        for axis in (ax.xaxis, ax.yaxis):
            axis.get_offset_text().set_color(self.foreground)
        for spine in ax.spines.values():
            spine.set_color(self.foreground)
            spine.set_alpha(.35)
        return ax

    def _select(self, index):
        if index == self.selected_index:
            return
        self.selected_index = index
        point, details = self.results[index]
        coordinates = ", ".join(f"{coordinate:.8g}" for coordinate in point)
        self.selected_point.setText(f"Equilibrium at ({coordinates})")
        self.point_map.mark_selected(index)
        matrix = details["jacobian"]
        summary = details["type"] + "\n" + details["linearization"]
        if details["hyperbolic"] is not None:
            summary += "\nHyperbolic" if details["hyperbolic"] else "\nNon-hyperbolic"
        if details.get("certified_center"):
            summary += "\nNonlinear center verified by a local integrating factor."
        self.summary.setText(summary)
        available = matrix is not None and np.isfinite(matrix).all()
        self.matrix.setText("\n".join(" ".join(f"{value:>11.5g}" for value in row) for row in matrix)
                            if available else "Unavailable")
        self.spectrum.setText(details["eigenvalues"])
        directions = _real_eigendirections(details)
        self.directions.setText("\n".join(
            f"λ = {value:.5g}   ({', '.join(f'{component:.5g}' for component in direction)})"
            for value, direction in directions
        ) or ("No real eigendirections" if available else "Unavailable"))
        if self.two_d:
            if self.pending_manifold is not None and self.pending_manifold != index:
                self.worker.cancel()
                self.pending_manifold = None
            saddle = details["type"] == "Saddle" and len(directions) == 2
            if not saddle:
                self.plots.setCurrentIndex(0)
            self.plots.setTabEnabled(1, saddle)
            self._draw_trace_determinant(matrix if available else None)
            self._show_manifolds(self.plots.currentIndex())
        self.analysis_stack.setCurrentWidget(self.analysis_page)
        if self.two_d:
            self._schedule_plot_alignment()

    def _draw_trace_determinant(self, matrix):
        ax = self._axes(self.trace_canvas)
        trace = determinant = math.nan
        if matrix is not None:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                trace, determinant = float(np.trace(matrix)), float(np.linalg.det(matrix))
                if determinant == 0 and np.linalg.slogdet(matrix)[0] != 0:
                    determinant = math.nan
        span = max(abs(trace), 2 * math.sqrt(abs(determinant)))
        if not math.isfinite(trace) or not math.isfinite(determinant) or span > 1e100 or 0 < span < 1e-100:
            self.invariants.setText("\n".join(
                f"{name} = {value:.6g}" if math.isfinite(value) else f"{name} unavailable"
                for name, value in (("tr J", trace), ("det J", determinant))
            ))
            ax.set_axis_off()
            ax.text(.5, .5, "Trace–determinant plot unavailable", transform=ax.transAxes,
                    ha="center", va="center", color=self.foreground)
        else:
            discriminant = trace**2 - 4 * determinant
            self.invariants.setText(f"tr J = {trace:.6g}\ndet J = {determinant:.6g}\nΔ = {discriminant:.6g}")
            x_span = 1.35 * span if span else 2.0
            y_span = x_span**2 / 4
            for sign, color, label in ((-1, self.stable_color, "Stable"), (1, self.unstable_color, "Unstable")):
                x = np.linspace(0, sign * x_span, 151)
                boundary = x**2 / 4
                ax.fill_between(x, 0, boundary, color=color, alpha=.13)
                ax.fill_between(x, boundary, y_span, color=color, alpha=.07)
                ax.plot(x, boundary, color=self.foreground, linewidth=.9, alpha=.65)
                ax.text(sign * .70 * x_span, .17 * y_span, f"{label} node", ha="center", color=self.foreground, fontsize=9)
                ax.text(sign * .52 * x_span, .75 * y_span, f"{label} focus", ha="center", color=self.foreground, fontsize=9)
            ax.axhline(0, color=self.foreground, linewidth=.8, alpha=.55)
            ax.axvline(0, color=self.foreground, linewidth=.8, alpha=.55)
            bottom = min(-.42 * y_span, 1.25 * determinant)
            ax.text(0, .65 * bottom, "Saddle", ha="center", color=self.foreground, fontsize=9)
            ax.scatter(trace, determinant, s=55, color="#e4782e", edgecolors=self.background, zorder=5)
            ax.set(xlim=(-x_span, x_span), ylim=(bottom, y_span))
            ax.set_xlabel("tr J", color=self.foreground)
            ax.set_ylabel("det J", color=self.foreground)
            ax.set_title("Linear type  ·  Δ = (tr J)² − 4 det J", color=self.foreground, fontsize=10)
        self.trace_canvas.draw_idle()

    def _show_manifolds(self, tab):
        if tab != 1 or not self.plots.isTabEnabled(1):
            return
        index = self.selected_index
        if index in self.manifold_cache:
            self._draw_manifolds(index, *self.manifold_cache[index])
        elif self.pending_manifold != index:
            self._draw_manifolds(index, [], 0)
            self.saddle_status.setText("Tracing four local branches…")
            point, details = self.results[index]
            self.pending_manifold = index
            self.worker.submit(index, {
                "fields": self.fields, "params": self.params, "point": point, "radius": self.radius,
                "directions": [(vector, value < 0) for value, vector in _real_eigendirections(details)],
            }, "saddle_manifolds")

    def _manifolds_ready(self, index, result):
        if index == self.pending_manifold:
            self.pending_manifold = None
            self.manifold_cache[index] = result
            self._draw_manifolds(index, *result)

    def _manifolds_failed(self, index, message):
        if index == self.pending_manifold:
            self.pending_manifold = None
            self.saddle_status.setText(f"Manifold analysis incomplete. {message}")

    def _draw_manifolds(self, index, paths, incomplete, arrows=None):
        ax = self._axes(self.saddle_canvas)
        point, details = self.results[index]
        if arrows is not None and len(arrows[0]):
            ax.quiver(*arrows, color=self.foreground, alpha=.25, angles="xy", scale_units="xy",
                      scale=18 / self.radius, pivot="middle")
        for value, direction in _real_eigendirections(details):
            stable = value < 0
            color = self.stable_color if stable else self.unstable_color
            tangent = point + np.array([-1.5, 1.5])[:, None] * self.radius * direction
            ax.plot(*tangent.T, "--", color=color, linewidth=1.1, alpha=.65)
            for is_stable, path in paths:
                if is_stable == stable:
                    ax.plot(*path.T, color=color, linewidth=1.8)
        ax.scatter(*point, color=self.foreground, s=23, zorder=5)
        ax.set(xlim=point[0] + np.array([-self.radius, self.radius]),
               ylim=point[1] + np.array([-self.radius, self.radius]), aspect="equal")
        ax.grid(color=self.foreground, alpha=.12)
        ax.set_xlabel("x", color=self.foreground)
        ax.set_ylabel("y", color=self.foreground)
        ax.set_title("Stable (blue) and unstable (red)", color=self.foreground, fontsize=10)
        message = "Dashed lines are eigenvector tangents. Solid lines are numerical local branches."
        if incomplete:
            message += f" {incomplete} of 4 branches stopped early."
        self.saddle_status.setText(message)
        self.saddle_canvas.draw_idle()
        self._schedule_plot_alignment()
