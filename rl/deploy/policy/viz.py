"""The --viz window: what the policy sees, and what it is doing about it. Plain Qt.

Two panels are the rectified camera views fed to the encoder, so a bad mapping is visible while the
policy runs rather than after. Beside them, one bar per joint: the action the last decision produced
and how far the real arm still is from the target it was told to reach.

Resizable, and it scrolls rather than clips when the window gets small.

The caller owns the loop: set the frames and rows each tick, call step(). No Qt event loop and no
threads, so it drops into the existing `async for tick in pace(fps)` without touching inference.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

FG, MUTED, ALARM, POS, NEG = "#e4e7ec", "#7e8797", "#e5484d", "#4a6cf7", "#d9a521"

STYLE = """
QWidget { background:#14161a; color:#e4e7ec; font-size:13px; }
QLabel#status { font-size:17px; padding:2px 0; }
QLabel#info, QLabel#hint { font-family:Menlo,monospace; color:#7e8797; }
QPushButton { background:#242932; border:1px solid #333a45; border-radius:3px; padding:5px 11px; }
QPushButton:hover { background:#2e3440; }
QScrollArea { border:0; }
QScrollBar:vertical, QScrollBar:horizontal { background:#14161a; width:10px; height:10px; }
QScrollBar::handle { background:#333a45; border-radius:5px; min-height:30px; min-width:30px; }
QScrollBar::add-line, QScrollBar::sub-line { height:0; width:0; }
"""

# Keys the window offers, and the run loop's existing single-character handler for each.
KEYS = [("pause / resume", "p"), ("rest", "r"), ("park", "k"), ("zero rail", "0"), ("quit", "q")]


class _Panel(QtWidgets.QLabel):
    """An image that fills whatever the layout gives it, keeping its aspect ratio.

    Both size policies are Ignored so the pixmap's own size never feeds back into the layout;
    without that, setting a frame grows the panel, which grows the window.
    """

    def __init__(self, title, minimum=140):
        super().__init__()
        self._rgb = None
        self.title = title
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("border:1px solid #2d333d; background:#0f1115; color:#7e8797;")
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setMinimumSize(minimum, minimum)
        self.setText(f"{title}\nno frame")

    def show_rgb(self, rgb):
        self._rgb = rgb
        self._render()

    def resizeEvent(self, ev):         # noqa: N802 - Qt naming
        super().resizeEvent(ev)
        self._render()

    def _render(self):
        if self._rgb is None:
            self.setText(f"{self.title}\nno frame")
            return
        rgb = np.ascontiguousarray(self._rgb)
        h, w = rgb.shape[:2]
        img = QtGui.QImage(rgb.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
        side = max(min(self.width(), self.height()) - 4, 16)
        self.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            side, side, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))


class _Actions(QtWidgets.QWidget):
    """One row per joint: the action as a bar out of centre, the tracking error as a tick.

    Drawn in one widget rather than as seven, because the interesting thing is the shape of the
    whole vector: which joints the policy is pushing on at once.
    """

    ROW, PAD, LABEL = 24, 8, 76

    def __init__(self, names):
        super().__init__()
        self.names = list(names)
        self.rows: list = []
        self.setMinimumWidth(260)
        self.setMinimumHeight(self.ROW * len(self.names) + 2 * self.PAD)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_rows(self, rows) -> None:
        self.rows = list(rows or [])
        self.update()

    def paintEvent(self, _ev):          # noqa: N802 - Qt naming
        p = QtGui.QPainter(self)
        p.setFont(QtGui.QFont("Menlo", 10))
        x0 = self.LABEL
        span = max(self.width() - x0 - 58, 40)
        mid = x0 + span / 2
        row = max((self.height() - 2 * self.PAD) / max(len(self.names), 1), self.ROW)
        for i, name in enumerate(self.names):
            y = self.PAD + i * row
            act, resid = (self.rows[i][1], self.rows[i][2]) if i < len(self.rows) else (None, 0.0)
            p.setPen(QtGui.QColor(MUTED))
            p.drawText(4, int(y + 15), name[:11])
            # the track
            p.fillRect(QtCore.QRectF(x0, y + 8, span, 3), QtGui.QColor("#22272f"))
            p.fillRect(QtCore.QRectF(mid - 0.5, y + 4, 1, 11), QtGui.QColor("#39404b"))
            if act is not None:
                a = max(-1.0, min(1.0, float(act)))
                w = a * span / 2
                p.fillRect(QtCore.QRectF(min(mid, mid + w), y + 6, abs(w), 7),
                           QtGui.QColor(POS if a >= 0 else NEG))
                p.setPen(QtGui.QColor(FG))
                p.drawText(int(x0 + span + 6), int(y + 15), f"{a:+.2f}")
            # tracking error, in units of one action step: a full bar means a whole step behind
            r = max(-1.5, min(1.5, float(resid)))
            p.fillRect(QtCore.QRectF(mid + r * span / 3 - 1, y + 2, 2, 15),
                       QtGui.QColor(ALARM if abs(r) > 1.0 else "#5b6472"))
        p.end()


class Window(QtWidgets.QWidget):
    def __init__(self, cameras, joint_names):
        super().__init__()
        self.setWindowTitle("so-frame policy")
        self.setStyleSheet(STYLE)
        self.requests: list[str] = []      # single-char keys, drained by the caller each tick

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        self.info = QtWidgets.QLabel("")
        self.info.setObjectName("info")
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # Views left, actions right, on one row: the two are read together, and side by side keeps
        # the window from growing taller than the images are wide.
        self.panels = {c: _Panel(c) for c in cameras}
        views = QtWidgets.QGridLayout()
        for col, (name, panel) in enumerate(self.panels.items()):
            lab = QtWidgets.QLabel(name)
            lab.setStyleSheet(f"color:{MUTED};")
            views.addWidget(lab, 0, col)
            views.addWidget(panel, 1, col)
            views.setColumnStretch(col, 1)
        views.setRowStretch(1, 1)

        self.actions = _Actions(joint_names)
        hint = QtWidgets.QLabel("action out of centre;\nthe tick is how far the arm lags its target")
        hint.setObjectName("hint")
        acts = QtWidgets.QVBoxLayout()
        acts.addWidget(hint)
        acts.addWidget(self.actions, 1)

        middle = QtWidgets.QHBoxLayout()
        middle.addLayout(views, 3)
        middle.addLayout(acts, 2)

        buttons = QtWidgets.QHBoxLayout()
        for text, ch in KEYS:
            b = QtWidgets.QPushButton(f"{text}  ({ch})")
            b.clicked.connect(lambda _=False, c=ch: self.requests.append(c))
            buttons.addWidget(b)
        buttons.addStretch()

        inner = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(inner)
        box.addWidget(self.status)
        box.addLayout(middle, 1)
        box.addWidget(self.info)
        box.addLayout(buttons)

        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(inner)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(area)
        self.resize(1120, 620)

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

    def take_keys(self) -> list[str]:
        keys, self.requests = list(self.requests), []
        return keys

    def show_stack(self, rgb) -> None:
        """Split a channel-stacked HxWx(3*N) view back into one panel per camera."""
        for i, panel in enumerate(self.panels.values()):
            panel.show_rgb(None if rgb is None else rgb[:, :, 3 * i:3 * i + 3])

    def set_state(self, status: str, alarm: bool, info: str, act_rows) -> None:
        self.status.setText(status)
        self.status.setStyleSheet(f"color:{ALARM if alarm else FG};")
        self.info.setText(info)
        self.actions.set_rows(act_rows)
