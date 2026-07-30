"""The calibrate window: the images on top, the numbers that move them underneath. Plain Qt.

Real is yellow, sim is blue, on the panel borders and nowhere else, so you can tell the two apart
without reading labels. That is the only styling here.

Every value is a slider AND a spin box: drag to search, type to land on an exact value. Resizable
throughout, and both halves scroll rather than clip.

The caller owns the loop: build a Window, set the images and numbers each tick, call step(). No
Qt event loop, no threads, so it drops into the existing `async for _ in pace(fps)`.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from utils import qt

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

STYLE = qt.STYLE + """
QGroupBox { border:1px solid #262b34; border-radius:3px; margin-top:14px; padding-top:10px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; color:#7e8797; }
QLabel#mono { font-family:Menlo,monospace; }
"""

_TICKS = 2000     # slider resolution; finer than the eye on a 300px groove


def _field(lo, hi, step, decimals) -> qt.Value:
    """One value as a slider and a spin box, at this window's resolution and column widths."""
    return qt.Value(lo, hi, step, decimals, ticks=_TICKS,
                    spin_width=84 if decimals < 4 else 96, slider_width=70)


def _scroll(inner: QtWidgets.QWidget, minimum: int) -> QtWidgets.QScrollArea:
    """Wrap a widget so it scrolls instead of clipping when the window gets small.

    `minimum` is deliberately below what the contents need, or the pane takes space from its
    neighbour instead of scrolling.
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


class Window(qt.Driven, QtWidgets.QWidget):
    def __init__(self, cameras, sim_names, joint_limits):
        super().__init__()
        self.setWindowTitle("so-frame calibrate")
        self.setStyleSheet(STYLE)
        self.requests: set[str] = set()          # drained by the caller each tick
        self.sim_names = sim_names

        # -- images ---------------------------------------------------------------------------
        self.panels = {k: qt.Panel("no frame", 140, colour=c, border=2) for k, c in
                       (("real", REAL), ("sim", SIM), ("blend", MUTED))}
        # The other camera is a reference strip, not a working view: capped so it cannot take
        # height from the camera actually being fitted.
        self.others = {k: qt.Panel("no frame", 80, colour=c, border=2, maximum=150) for k, c in
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

        self.joints: dict[str, qt.Value] = {}
        for k, (lo, hi) in joint_limits.items():
            self.joints[k] = _field(lo, hi, (hi - lo) / 200, 4)
        arm = QtWidgets.QGroupBox("arm  (sim units, ranged to the urdf limits)")
        arm.setLayout(_pairs([(k.split(".")[0], f) for k, f in self.joints.items()], 1))

        self.fields: dict[str, qt.Value] = {}
        for key, label, lo, hi, step, dec in MAP_FIELDS:
            self.fields[key] = _field(lo, hi, step, dec)
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
        # setSizes only after the resize, or a splitter still at its default height clamps and
        # rescales them.
        self.resize(1480, 1000)
        split.setSizes([660, 340])

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
