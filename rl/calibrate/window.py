"""The calibrate window: three images and the numbers that move them. Plain Qt.

Real is yellow, sim is blue, on the panel borders and nowhere else, so you can tell the two apart
without reading labels. That is the only styling here.

The caller owns the loop: build a Window, set the images and numbers each tick, call step(). No
Qt event loop, no threads, so it drops into the existing `async for _ in pace(fps)`.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

REAL, SIM, ALARM, MUTED = "#d9a521", "#4a6cf7", "#e5484d", "#7e8797"

# (key, label, min, max, step, decimals)
MAP_FIELDS = [
    ("rot90", "rot90", 0, 3, 1, 0),
    ("angle_deg", "angle (deg)", -20, 20, 0.5, 2),
    ("k1", "k1", -0.6, 0.6, 0.01, 3),
    ("k2", "k2", -0.3, 0.3, 0.01, 3),
    ("focal_px", "focal (px)", 100, 1200, 5, 0),
    ("zoom", "zoom", 0.2, 2.0, 0.01, 3),
    ("offset_x", "offset x (px)", -400, 400, 1, 1),
    ("offset_y", "offset y (px)", -400, 400, 1, 1),
    ("gain_r", "gain R", 0.5, 2.0, 0.01, 3),
    ("gain_g", "gain G", 0.5, 2.0, 0.01, 3),
    ("gain_b", "gain B", 0.5, 2.0, 0.01, 3),
    ("gamma", "gamma", 0.4, 2.5, 0.01, 3),
]

STYLE = """
QWidget { background:#14161a; color:#e4e7ec; font-size:13px; }
QGroupBox { border:1px solid #262b34; border-radius:3px; margin-top:14px; padding-top:8px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; color:#7e8797; }
QDoubleSpinBox, QSpinBox, QComboBox { background:#1f232b; border:1px solid #2d333d;
    border-radius:3px; padding:3px 5px; }
QPushButton { background:#242932; border:1px solid #333a45; border-radius:3px; padding:5px 11px; }
QPushButton:hover { background:#2e3440; }
QSlider::groove:horizontal { height:3px; background:#2d333d; }
QSlider::handle:horizontal { width:11px; margin:-5px 0; border-radius:2px; background:#8a93a3; }
"""


def _pix(rgb, size):
    """HxWx3 uint8 RGB -> a QPixmap scaled to size."""
    if rgb is None:
        return QtGui.QPixmap()
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    img = QtGui.QImage(rgb.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(img).scaled(size, size, QtCore.Qt.KeepAspectRatio,
                                               QtCore.Qt.FastTransformation)


class _Panel(QtWidgets.QLabel):
    def __init__(self, title, colour, size):
        super().__init__()
        self.size_px = size
        self.setFixedSize(size, size)
        self.setStyleSheet(f"border:2px solid {colour}; background:#0f1115;")
        self.setToolTip(title)

    def show_rgb(self, rgb):
        self.setPixmap(_pix(rgb, self.size_px - 4))


class Window(QtWidgets.QWidget):
    def __init__(self, cameras, sim_names, joint_limits, big=430, small=130):
        super().__init__()
        self.setWindowTitle("so-frame calibrate")
        self.setStyleSheet(STYLE)
        self.requests: set[str] = set()          # drained by the caller each tick

        self.panels = {k: _Panel(k, c, big) for k, c in
                       (("real", REAL), ("sim", SIM), ("blend", MUTED))}
        self.others = {k: _Panel(k, c, small) for k, c in
                       (("real", REAL), ("sim", SIM), ("blend", MUTED))}

        row = QtWidgets.QHBoxLayout()
        for k in ("real", "sim", "blend"):
            box = QtWidgets.QVBoxLayout()
            lbl = QtWidgets.QLabel({"real": "REAL", "sim": "SIM", "blend": "OVERLAY"}[k])
            lbl.setStyleSheet(f"color:{ {'real': REAL, 'sim': SIM}.get(k, MUTED) };")
            box.addWidget(lbl)
            box.addWidget(self.panels[k])
            row.addLayout(box)

        self.blend = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blend.setRange(0, 100)
        self.blend.setValue(50)
        blend_row = QtWidgets.QHBoxLayout()
        for text, colour, w in (("REAL", REAL, None), (None, None, self.blend), ("SIM", SIM, None)):
            if w is not None:
                blend_row.addWidget(w)
            else:
                lab = QtWidgets.QLabel(text)
                lab.setStyleSheet(f"color:{colour};")
                blend_row.addWidget(lab)

        small_row = QtWidgets.QHBoxLayout()
        small_row.addWidget(QtWidgets.QLabel("other camera"))
        for k in ("real", "sim", "blend"):
            small_row.addWidget(self.others[k])
        small_row.addStretch()

        left = QtWidgets.QVBoxLayout()
        self.heading = QtWidgets.QLabel("")
        self.heading.setStyleSheet(f"color:{MUTED};")
        left.addWidget(self.heading)
        left.addLayout(row)
        left.addLayout(blend_row)
        left.addLayout(small_row)
        left.addStretch()

        # -- rail -----------------------------------------------------------------------------
        self.camera = QtWidgets.QComboBox()
        self.camera.addItems(cameras)
        self.sim_names = sim_names

        self.joints: dict[str, QtWidgets.QDoubleSpinBox] = {}
        jbox = QtWidgets.QFormLayout()
        for k, (lo, hi) in joint_limits.items():
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(4)
            sb.setSingleStep((hi - lo) / 200)
            self.joints[k] = sb
            jbox.addRow(k.split(".")[0], sb)
        arm = QtWidgets.QGroupBox("arm  (sim units, clamped to the urdf limits)")
        arm.setLayout(jbox)

        self.fields: dict[str, QtWidgets.QDoubleSpinBox] = {}
        mbox = QtWidgets.QFormLayout()
        for key, label, lo, hi, step, dec in MAP_FIELDS:
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(dec)
            sb.setSingleStep(step)
            self.fields[key] = sb
            mbox.addRow(label, sb)
        cam = QtWidgets.QGroupBox("camera mapping")
        cam.setLayout(mbox)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("font-family:Menlo,monospace;")
        self.telemetry = QtWidgets.QLabel("")
        self.telemetry.setStyleSheet("font-family:Menlo,monospace;")

        rail = QtWidgets.QVBoxLayout()
        rail.addWidget(QtWidgets.QLabel("camera"))
        rail.addWidget(self.camera)
        rail.addWidget(arm)
        rail.addWidget(cam)
        for text, tag in (("go to rest", "rest"), ("go to park", "park"),
                          ("hold here", "hold"), ("clear speed peaks", "clear_speed"),
                          ("match colour to sim", "match_colour"),
                          ("reset zoom + offset", "recentre"), ("save mapping", "save")):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, t=tag: self.requests.add(t))
            rail.addWidget(b)
        rail.addWidget(self.status)
        rail.addWidget(self.telemetry)
        rail.addStretch()

        outer = QtWidgets.QHBoxLayout(self)
        outer.addLayout(left)
        wrap = QtWidgets.QWidget()
        wrap.setLayout(rail)
        wrap.setFixedWidth(330)
        outer.addWidget(wrap)

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
        return {k: sb.value() for k, sb in self.joints.items()}

    def set_joints(self, pose: dict) -> None:
        for k, v in pose.items():
            if k in self.joints:
                sb = self.joints[k]
                sb.blockSignals(True)
                sb.setValue(v)
                sb.blockSignals(False)

    def mapping_values(self) -> dict:
        return {k: sb.value() for k, sb in self.fields.items()}

    def set_mapping(self, m: dict) -> None:
        for k, sb in self.fields.items():
            if k in m:
                sb.blockSignals(True)
                sb.setValue(float(m[k]))
                sb.blockSignals(False)

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
