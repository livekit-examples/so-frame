"""The --viz window: what the policy sees, and what it is doing about it. Plain Qt.

Two panels are the rectified camera views fed to the encoder, so a bad mapping is visible while the
policy runs rather than after. Beside them, one bar per joint: the action the last decision produced
and how far the real arm still is from the target it was told to reach.

Resizable, and it scrolls rather than clips when the window gets small.

The caller owns the loop: set the frames and rows each tick, call step(). No Qt event loop and no
threads, so it drops into the existing `async for tick in pace(fps)` without touching inference.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from utils import qt

FG, MUTED, ALARM, POS, NEG = "#e4e7ec", "#7e8797", "#e5484d", "#4a6cf7", "#d9a521"

STYLE = qt.STYLE + """
QPushButton { padding:5px 11px; }
QLabel#status { font-size:17px; padding:2px 0; }
QLabel#info, QLabel#hint { font-family:Menlo,monospace; color:#7e8797; }
"""

# Keys the window offers, and the run loop's single-character handler for each.
KEYS = [("pause / resume", "p"), ("rest", "r"), ("park", "k"), ("zero rail", "0"), ("quit", "q")]

# Budget range, in action steps. Below ~0.5 a group can barely glide, since one action step is the
# whole budget; well above that the target is free to run ahead of a slower joint again.
LAG_MIN, LAG_MAX = 0.1, 6.0


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
        self.thresholds: dict = {}
        self.setMinimumWidth(260)
        self.setMinimumHeight(self.ROW * len(self.names) + 2 * self.PAD)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_rows(self, rows, thresholds=None) -> None:
        self.rows = list(rows or [])
        self.thresholds = thresholds or {}
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
            p.fillRect(QtCore.QRectF(x0, y + 8, span, 3), QtGui.QColor("#22272f"))
            p.fillRect(QtCore.QRectF(mid - 0.5, y + 4, 1, 11), QtGui.QColor("#39404b"))
            if act is not None:
                a = max(-1.0, min(1.0, float(act)))
                w = a * span / 2
                p.fillRect(QtCore.QRectF(min(mid, mid + w), y + 6, abs(w), 7),
                           QtGui.QColor(POS if a >= 0 else NEG))
                p.setPen(QtGui.QColor(FG))
                p.drawText(int(x0 + span + 6), int(y + 15), f"{a:+.2f}")
            # Lag in action steps, scaled so this joint's budget sits at half a bar. Red means the
            # budget is spent; a joint with no budget (the gripper) still shows its lag, never red.
            gate = self.thresholds.get(name)
            r = max(-1.0, min(1.0, float(resid) / (2 * max(gate or 1.0, 1e-3))))
            p.fillRect(QtCore.QRectF(mid + r * span / 2 - 1, y + 2, 2, 15),
                       QtGui.QColor(ALARM if gate and abs(resid) > gate else "#5b6472"))
        p.end()


class Window(qt.Driven, QtWidgets.QWidget):
    def __init__(self, cameras, joint_names, max_lag=3.0, rail_step=7.0, gated=None):
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
        self.panels = {c: qt.Panel(f"{c}\nno frame", 140) for c in cameras}
        views = QtWidgets.QGridLayout()
        for col, (name, panel) in enumerate(self.panels.items()):
            lab = QtWidgets.QLabel(name)
            lab.setStyleSheet(f"color:{MUTED};")
            views.addWidget(lab, 0, col)
            views.addWidget(panel, 1, col)
            views.setColumnStretch(col, 1)
        views.setRowStretch(1, 1)

        self.actions = _Actions(joint_names)
        hint = QtWidgets.QLabel("action out of centre; the tick is the joint's lag,\n"
                                "red once it is over its own gate below")
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
        # How far the target may lead the arm, which is both how far it may glide between decisions
        # and when the next one fires. Joints outside `gated` (the gripper) are unbounded, and the
        # bars are the readout: dial it to just above where the ticks stop going red.
        self._gated = {k.split(".")[0] for k in (gated or ())}
        self._lag = qt.Value(LAG_MIN, LAG_MAX, 0.05, 2, max_lag)
        # How far one full-command action moves the carriage. It rescales rail lag too, since lag
        # is counted in these steps.
        self._rail_step = qt.Value(1.0, 20.0, 0.5, 1, rail_step)
        for text, widget in (("max lag", self._lag), ("rail step (mm)", self._rail_step)):
            lab = QtWidgets.QLabel(text)
            lab.setStyleSheet(f"color:{MUTED};")
            buttons.addWidget(lab)
            buttons.addWidget(widget)

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

        # Take focus so key presses land here. A child that wants a key still wins, so digits type
        # into a spin box you have clicked into, including 0.
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus()

    def keyPressEvent(self, ev):       # noqa: N802 - Qt naming
        """Queue a debug key for the caller, same characters the terminal accepts.

        Qt offers a key to the focused child first and only walks up to here if that child ignores
        it, so typing into the sliders still works.
        """
        ch = ev.text().lower()
        if ch in {c for _, c in KEYS}:
            self.requests.append(ch)
            ev.accept()
            return
        super().keyPressEvent(ev)

    def max_lag(self) -> float:
        """Lag budget in action steps, as currently dialled in."""
        return self._lag.value()

    def rail_step(self) -> float:
        """Millimetres of carriage travel per full-command action, as currently dialled in."""
        return self._rail_step.value()

    def take_keys(self) -> list[str]:
        keys, self.requests = list(self.requests), []
        return keys

    def show_stack(self, rgb) -> None:
        """Split a channel-stacked HxWx(3*N) view back into one panel per camera."""
        for i, panel in enumerate(self.panels.values()):
            panel.show_rgb(None if rgb is None else rgb[:, :, 3 * i:3 * i + 3])

    def set_state(self, status: str, alarm: bool, info: str, act_rows, threshold=1.0) -> None:
        self.status.setText(status)
        self.status.setStyleSheet(f"color:{ALARM if alarm else FG};")
        self.info.setText(info)
        self.actions.set_rows(act_rows, {n: threshold for n in self._gated})
