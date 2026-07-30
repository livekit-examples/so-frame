"""Qt pieces shared by the two operator windows: policy/viz.py and rl/calibrate/window.py.

PySide6 is an optional dependency group of this project (`viz`), so only those two windows may
import this module, never anything on the robot's import path.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# The rules both windows share. Each appends its own selectors, and a later rule of the same
# specificity wins on the properties it repeats.
STYLE = """
QWidget { background:#14161a; color:#e4e7ec; font-size:13px; }
QDoubleSpinBox, QComboBox { background:#1f232b; border:1px solid #2d333d;
    border-radius:3px; padding:2px 4px; }
QPushButton { background:#242932; border:1px solid #333a45; border-radius:3px; padding:5px 9px; }
QPushButton:hover { background:#2e3440; }
QSlider::groove:horizontal { height:3px; background:#2d333d; border-radius:2px; }
QSlider::handle:horizontal { width:10px; margin:-6px 0; border-radius:2px; background:#8a93a3; }
QSlider::handle:horizontal:hover { background:#c3cad6; }
QScrollArea { border:0; }
QScrollBar:vertical, QScrollBar:horizontal { background:#14161a; width:10px; height:10px; }
QScrollBar::handle { background:#333a45; border-radius:5px; min-height:30px; min-width:30px; }
QScrollBar::add-line, QScrollBar::sub-line { height:0; width:0; }
"""


class Value(QtWidgets.QWidget):
    """One number as a slider and a spin box: drag to search, type to land on an exact value.

    lo/hi/step/decimals go to the spin box, `value` is the initial value (default: wherever the
    range leaves the spin box). `ticks` is the slider resolution over lo..hi, and the two widths
    are the spin box's fixed width and the slider's minimum, in pixels.
    """

    def __init__(self, lo, hi, step, decimals, value=None, *,
                 ticks=400, spin_width=76, slider_width=110):
        super().__init__()
        self.lo, self.hi = float(lo), float(hi)
        self.ticks = int(ticks)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(self.lo, self.hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)     # commit on Enter/focus-out, not per keystroke
        self.spin.setFixedWidth(spin_width)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.ticks)
        self.slider.setMinimumWidth(slider_width)
        box = QtWidgets.QHBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(self.slider, 1)
        box.addWidget(self.spin)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._to_slider)
        if value is not None:
            self.spin.setValue(float(value))
        self._to_slider(self.spin.value())

    def _from_slider(self, i):
        self.spin.blockSignals(True)
        self.spin.setValue(self.lo + (self.hi - self.lo) * i / self.ticks)
        self.spin.blockSignals(False)

    def _to_slider(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((v - self.lo) / (self.hi - self.lo) * self.ticks)))
        self.slider.blockSignals(False)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v) -> None:      # noqa: N802 - matches the Qt spelling it stands in for
        self.spin.setValue(float(v))    # valueChanged moves the slider


class Panel(QtWidgets.QLabel):
    """An image that fills whatever the layout gives it, keeping its aspect ratio.

    `placeholder` is the text shown until a frame arrives, `minimum` the square minimum size and
    `maximum` an optional height cap, both in pixels; `colour` and `border` are the border.
    Both size policies are Ignored so the pixmap's own size never feeds back into the layout.
    """

    def __init__(self, placeholder, minimum, *, colour="#2d333d", border=1, maximum=None):
        super().__init__()
        self._rgb = None
        self.placeholder = placeholder
        self._inset = 2 * border + 2      # the border on both sides, plus a pixel either way
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet(f"border:{border}px solid {colour}; background:#0f1115; color:#7e8797;")
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setMinimumSize(minimum, minimum)
        if maximum:
            self.setMaximumHeight(maximum)
        self.setText(placeholder)

    def show_rgb(self, rgb):
        self._rgb = rgb
        self._render()

    def resizeEvent(self, ev):         # noqa: N802 - Qt naming
        super().resizeEvent(ev)
        self._render()

    def _render(self):
        if self._rgb is None:
            self.setText(self.placeholder)
            return
        rgb = np.ascontiguousarray(self._rgb)
        h, w = rgb.shape[:2]
        img = QtGui.QImage(rgb.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
        side = max(min(self.width(), self.height()) - self._inset, 16)
        self.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            side, side, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))


class Driven:
    """Mixin for a window the caller's own loop drives: no Qt event loop and no threads.

    Mix in before the QWidget base so closeEvent overrides Qt's.
    """

    _closed = False

    def closeEvent(self, ev):          # noqa: N802 - Qt naming
        self._closed = True
        ev.accept()

    @property
    def closed(self) -> bool:
        return self._closed

    def step(self) -> None:
        """Service Qt from the caller's loop. Nothing else drives this window."""
        QtWidgets.QApplication.instance().processEvents()
