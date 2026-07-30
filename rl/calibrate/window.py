"""The calibrate window: the images on top, the numbers that move them underneath. Plain Qt.

Real is yellow, sim is blue, on the panel borders and nowhere else, so you can tell the two apart
without reading labels. That is the only styling here.

Every value is a slider AND a spin box: drag to search, type to land on an exact value. Resizable
throughout, and both halves scroll rather than clip.

The caller owns the loop: build a Window, set the images and numbers each tick, call step(). No
Qt event loop, no threads, so it drops into the existing `async for _ in pace(fps)`.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

REAL, SIM, ALARM, MUTED = "#d9a521", "#4a6cf7", "#e5484d", "#7e8797"

# (key, label, min, max, step, decimals). Decimals are what you can actually type: the mapping
# stores whatever is here, so anything rounded off in this table is not reachable at all.
MAP_FIELDS = [
    ("rot90", "rot90", 0, 3, 1, 0),
    ("angle_deg", "angle (deg)", -20, 20, 0.1, 2),
    ("k1", "k1", -0.6, 0.6, 0.005, 4),
    ("k2", "k2", -0.3, 0.3, 0.005, 4),
    ("focal_px", "focal (px)", 100, 1200, 1, 2),
    ("zoom", "zoom", 0.2, 2.0, 0.005, 4),
    ("offset_x", "offset x", -400, 400, 0.5, 2),
    ("offset_y", "offset y", -400, 400, 0.5, 2),
    ("gain_r", "gain R", 0.5, 2.0, 0.005, 4),
    ("gain_g", "gain G", 0.5, 2.0, 0.005, 4),
    ("gain_b", "gain B", 0.5, 2.0, 0.005, 4),
    ("gamma", "gamma", 0.4, 2.5, 0.005, 4),
]

BUTTONS = [("go to rest", "rest"), ("go to park", "park"),
           ("hold here", "hold"), ("clear speed peaks", "clear_speed"),
           ("match colour to sim", "match_colour"), ("reset zoom + offset", "recentre"),
           ("save mapping", "save")]

STYLE = """
QWidget { background:#14161a; color:#e4e7ec; font-size:13px; }
QGroupBox { border:1px solid #262b34; border-radius:3px; margin-top:14px; padding-top:10px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; color:#7e8797; }
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
QLabel#mono { font-family:Menlo,monospace; }
"""

_TICKS = 2000     # slider resolution; finer than the eye on a 300px groove


class _Field(QtWidgets.QWidget):
    """One value, as a slider and a spin box over the same number.

    The slider is for searching (drag and watch the overlay), the spin box for landing on a value
    you can write down. They are two views of one number, so each mirrors the other.
    """

    def __init__(self, lo, hi, step, decimals):
        super().__init__()
        self.lo, self.hi = float(lo), float(hi)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(self.lo, self.hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)     # commit on Enter/focus-out, not per keystroke
        self.spin.setFixedWidth(84 if decimals < 4 else 96)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, _TICKS)
        self.slider.setMinimumWidth(70)
        box = QtWidgets.QHBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(self.slider, 1)
        box.addWidget(self.spin)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._to_slider)
        self._to_slider(self.spin.value())

    def _from_slider(self, i):
        v = self.lo + (self.hi - self.lo) * i / _TICKS
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)

    def _to_slider(self, v):
        i = round((v - self.lo) / (self.hi - self.lo) * _TICKS)
        self.slider.blockSignals(True)
        self.slider.setValue(int(i))
        self.slider.blockSignals(False)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v) -> None:      # noqa: N802 - matches the Qt spelling it stands in for
        self.spin.setValue(float(v))    # valueChanged moves the slider


class _Panel(QtWidgets.QLabel):
    """An image that fills whatever the layout gives it, keeping its aspect ratio.

    Both size policies are Ignored so the pixmap's own size never feeds back into the layout;
    without that, setting a large frame grows the panel, which grows the window.
    """

    def __init__(self, title, colour, minimum, maximum=None):
        super().__init__()
        self._rgb = None
        self.title = title
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet(f"border:2px solid {colour}; background:#0f1115; color:#7e8797;")
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setMinimumSize(minimum, minimum)
        if maximum:
            self.setMaximumHeight(maximum)
        self.setText("no frame")

    def show_rgb(self, rgb):
        self._rgb = rgb
        self._render()

    def resizeEvent(self, ev):         # noqa: N802 - Qt naming
        super().resizeEvent(ev)
        self._render()

    def _render(self):
        if self._rgb is None:
            self.setText("no frame")
            return
        rgb = np.ascontiguousarray(self._rgb)
        h, w = rgb.shape[:2]
        img = QtGui.QImage(rgb.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
        side = max(min(self.width(), self.height()) - 6, 16)
        self.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            side, side, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))


def _scroll(inner: QtWidgets.QWidget, minimum: int) -> QtWidgets.QScrollArea:
    """Wrap a widget so it scrolls instead of clipping when the window gets small.

    `minimum` is deliberately below what the contents need. Without it the pane refuses to shrink
    past its contents and takes the space from its neighbour instead of scrolling, which is how the
    control rail ends up crushing the images on a small screen.
    """
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    area.setMinimumHeight(minimum)
    return area


def _pairs(rows, cols: int) -> QtWidgets.QGridLayout:
    """(label, widget) pairs in `cols` label/field columns, filling column by column."""
    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(12)
    grid.setVerticalSpacing(3)
    per = -(-len(rows) // cols)
    for i, (text, widget) in enumerate(rows):
        c, r = divmod(i, per)
        lab = QtWidgets.QLabel(text)
        lab.setStyleSheet(f"color:{MUTED};")
        grid.addWidget(lab, r, 2 * c, alignment=QtCore.Qt.AlignRight)
        grid.addWidget(widget, r, 2 * c + 1)
    for c in range(cols):
        grid.setColumnStretch(2 * c + 1, 1)
    grid.setRowStretch(per, 1)     # keep the rows at their natural height, packed to the top
    return grid


class Window(QtWidgets.QWidget):
    def __init__(self, cameras, sim_names, joint_limits):
        super().__init__()
        self.setWindowTitle("so-frame calibrate")
        self.setStyleSheet(STYLE)
        self.requests: set[str] = set()          # drained by the caller each tick
        self.sim_names = sim_names

        # -- images ---------------------------------------------------------------------------
        self.panels = {k: _Panel(k, c, 140) for k, c in
                       (("real", REAL), ("sim", SIM), ("blend", MUTED))}
        # The other camera is a reference strip, not a working view: capped so it cannot take
        # height from the camera actually being fitted.
        self.others = {k: _Panel(k, c, 80, maximum=150) for k, c in
                       (("real", REAL), ("sim", SIM), ("blend", MUTED))}

        grid = QtWidgets.QGridLayout()
        for col, (key, text, colour) in enumerate((("real", "REAL", REAL), ("sim", "SIM", SIM),
                                                   ("blend", "OVERLAY", MUTED))):
            lab = QtWidgets.QLabel(text)
            lab.setStyleSheet(f"color:{colour};")
            grid.addWidget(lab, 0, col)
            grid.addWidget(self.panels[key], 1, col)
            grid.addWidget(self.others[key], 4, col)
            grid.setColumnStretch(col, 1)
        # The mix slider sits between the two rows, spanning them, since it drives both overlays.
        self.blend = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blend.setRange(0, 100)
        self.blend.setValue(50)
        mix = QtWidgets.QHBoxLayout()
        for text, colour in (("real", REAL), (None, None), ("sim", SIM)):
            if text is None:
                mix.addWidget(self.blend, 1)
                continue
            lab = QtWidgets.QLabel(text)
            lab.setStyleSheet(f"color:{colour};")
            mix.addWidget(lab)
        grid.addLayout(mix, 2, 0, 1, 3)
        other = QtWidgets.QLabel("the other camera")
        other.setStyleSheet(f"color:{MUTED};")
        grid.addWidget(other, 3, 0, 1, 3)
        grid.setRowStretch(1, 1)          # the big row absorbs the spare height
        grid.setRowMinimumHeight(4, 110)

        self.heading = QtWidgets.QLabel("")
        self.heading.setStyleSheet(f"color:{MUTED};")
        top = QtWidgets.QWidget()
        top_box = QtWidgets.QVBoxLayout(top)
        top_box.setContentsMargins(6, 4, 6, 0)
        top_box.addWidget(self.heading)
        top_box.addLayout(grid, 1)

        # -- controls, across the bottom so the sliders have room to be worth dragging --------
        self.camera = QtWidgets.QComboBox()
        self.camera.addItems(cameras)

        self.joints: dict[str, _Field] = {}
        for k, (lo, hi) in joint_limits.items():
            self.joints[k] = _Field(lo, hi, (hi - lo) / 200, 4)
        arm = QtWidgets.QGroupBox("arm  (sim units, ranged to the urdf limits)")
        arm.setLayout(_pairs([(k.split(".")[0], f) for k, f in self.joints.items()], 1))

        self.fields: dict[str, _Field] = {}
        for key, label, lo, hi, step, dec in MAP_FIELDS:
            self.fields[key] = _Field(lo, hi, step, dec)
        cam = QtWidgets.QGroupBox("camera mapping")
        cam.setLayout(_pairs([(lab, self.fields[key]) for key, lab, *_ in MAP_FIELDS], 2))

        acts = QtWidgets.QGridLayout()
        for i, (text, tag) in enumerate(BUTTONS):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, t=tag: self.requests.add(t))
            acts.addWidget(b, *divmod(i, 2))

        self.status = QtWidgets.QLabel("")
        self.telemetry = QtWidgets.QLabel("")
        for lab in (self.status, self.telemetry):
            lab.setObjectName("mono")
            lab.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        side = QtWidgets.QVBoxLayout()
        cam_row = QtWidgets.QHBoxLayout()
        cam_row.addWidget(QtWidgets.QLabel("camera"))
        cam_row.addWidget(self.camera, 1)
        side.addLayout(cam_row)
        side.addLayout(acts)
        side.addWidget(self.status)
        side.addWidget(self.telemetry)
        side.addStretch()

        bottom = QtWidgets.QWidget()
        bottom_box = QtWidgets.QHBoxLayout(bottom)
        bottom_box.setContentsMargins(6, 0, 6, 6)
        bottom_box.addWidget(arm, 2)
        bottom_box.addWidget(cam, 4)
        bottom_box.addLayout(side, 2)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(_scroll(top, 240))
        split.addWidget(_scroll(bottom, 180))
        split.setStretchFactor(0, 1)      # extra height goes to the images, not the controls
        split.setStretchFactor(1, 0)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)
        # setSizes only after the resize: on a splitter that is still at its default height the
        # sizes get clamped and then rescaled, and the images end up with less than the controls.
        self.resize(1480, 1000)
        split.setSizes([660, 340])

        self._closed = False

    def closeEvent(self, ev):          # noqa: N802 - Qt naming
        self._closed = True
        ev.accept()

    @property
    def closed(self) -> bool:
        return self._closed

    def step(self) -> None:
        """Service Qt from the caller's loop. Nothing else drives this window."""
        QtWidgets.QApplication.instance().processEvents()

    # -- called by the loop ---------------------------------------------------------------
    def track(self) -> str:
        return self.camera.currentText()

    def blend_value(self) -> float:
        return self.blend.value() / 100.0

    def joint_values(self) -> dict:
        return {k: f.value() for k, f in self.joints.items()}

    def set_joints(self, pose: dict) -> None:
        for k, v in pose.items():
            if k in self.joints:
                self.joints[k].setValue(v)

    def mapping_values(self) -> dict:
        return {k: f.value() for k, f in self.fields.items()}

    def set_mapping(self, m: dict) -> None:
        for k, f in self.fields.items():
            if k in m:
                f.setValue(m[k])

    def show_frames(self, panels: dict, others: dict) -> None:
        for k, p in self.panels.items():
            p.show_rgb(panels.get(k))
        for k, p in self.others.items():
            p.show_rgb(others.get(k))

    def set_text(self, heading: str, status: str, telemetry: str) -> None:
        self.heading.setText(heading)
        self.status.setText(status)
        self.telemetry.setText(telemetry)

    def take_requests(self) -> set:
        r = set(self.requests)
        self.requests.clear()
        return r
