import time

from PyQt5 import QtCore, QtGui, QtWidgets
class ArcTimerWidget(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0
        self._duration_ms = 0          # default to 0 so we don't show 4.0s
        self._tick_ms = 30
        self._elapsed = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self.setMinimumSize(160, 160)

    def set_duration(self, duration_ms: int):
        """Prime the timer display without starting it."""
        self._duration_ms = max(1, int(duration_ms))
        self._elapsed = 0
        self._progress = 0.0
        self.update()

    def start(self, duration_ms=4000):
        # keep start() flexible, but we’ll usually call set_duration() beforehand
        time.sleep(0.2)
        self._duration_ms = max(1, int(duration_ms))
        self._elapsed = 0
        self._progress = 0.0
        self._timer.start(self._tick_ms)
        self.update()

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()
        self._progress, self._elapsed = 0.0, 0
        self.update()

    def is_running(self):
        return self._timer.isActive()

    def _on_tick(self):
        self._elapsed += self._tick_ms * 0.95
        self._progress = min(1.0, self._elapsed / self._duration_ms)
        self.update()
        if self._progress >= 1.0:
            self._timer.stop()
            self.finished.emit()

    def paintEvent(self, event):
        side = min(self.width(), self.height())
        rect = QtCore.QRect(
            (self.width() - side) // 2,
            (self.height() - side) // 2,
            side,
            side,
        )
        start_angle = 90 * 16
        span_angle = -int(self._progress * 360 * 16)

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # background circle
        bg_pen = QtGui.QPen(QtGui.QColor(220, 220, 220), 12)
        p.setPen(bg_pen)
        p.drawEllipse(rect.adjusted(10, 10, -10, -10))

        # foreground arc
        fg_pen = QtGui.QPen(QtGui.QColor(70, 120, 255), 12, cap=QtCore.Qt.RoundCap)
        p.setPen(fg_pen)
        p.drawArc(rect.adjusted(10, 10, -10, -10), start_angle, span_angle)

        # countdown text
        remaining_ms = max(0, self._duration_ms - self._elapsed)
        secs = remaining_ms / 1000.0
        p.setPen(QtGui.QColor(50, 50, 50))
        font = p.font()
        font.setPointSize(int(side * 0.12))
        p.setFont(font)
        p.drawText(rect, QtCore.Qt.AlignCenter, f"{secs:0.1f}s")

