# widgets/live_signal_plot_1ch.py
# pip install pyqtgraph
from typing import Optional
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


class LiveSignalPlot(QtWidgets.QWidget):
    """
    Single-channel live plot that accepts the FULL series each update.

    Use:
        plot = LiveSignalPlot(sampling_rate_hz=2000.0)
        plot.set_series(y)   # y shape: (N,) float/int
    """

    def __init__(
        self,
        sampling_rate_hz: float,
        time_span_sec: float = 5.0,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.fs = float(sampling_rate_hz if sampling_rate_hz > 0 else 2000.0)
        self.time_span_sec = float(max(0.1, time_span_sec))
        self._last_y = None  # keep last series for quick re-draws if needed

        # UI
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        pg.setConfigOptions(antialias=True)
        self.plot.setBackground('w')
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        axis_pen = pg.mkPen(color=(0, 0, 0, 200))
        for ax in ('bottom', 'left'):
            self.plot.getAxis(ax).setPen(axis_pen)
            self.plot.getAxis(ax).setTextPen(axis_pen)
        self.plot.setLabel('bottom', 'Time', units='s')

        self.curve = self.plot.plot([], [], pen=pg.mkPen(color=(0, 0, 0), width=1))
        vb = self.plot.getViewBox()
        vb.enableAutoRange(x=False, y=False)

        layout.addWidget(self.plot)

    # --- public API ---
    def clear(self):
        self._last_y = None
        self.curve.setData([], [])
        self.plot.setXRange(0.0, self.time_span_sec, padding=0)
        self.plot.setYRange(-1.0, 1.0, padding=0)

    def set_series(self, y):
        """
        Replace the entire plotted series with y (shape (N,), (N,1) or (1,N)).
        """
        if y is None:
            return
        a = np.asarray(y)
        if a.ndim == 2 and 1 in a.shape:
            a = a.reshape(-1)
        elif a.ndim != 1:
            return

        if a.size == 0:
            self.clear()
            return

        # Keep a copy/reference
        self._last_y = a.astype(np.float32, copy=False)

        # Build time axis (0..T), clamp to visible time_span_sec
        N = self._last_y.shape[0]
        t = np.arange(N, dtype=np.float32) / (self.fs if self.fs > 0 else 1.0)
        t_max = max(self.time_span_sec, float(N) / (self.fs if self.fs > 0 else 1.0))
        self.plot.setXRange(0.0, t_max, padding=0.0)

        # Symmetric y-range around 0 to show negatives clearly
        ymin = float(np.min(self._last_y))
        ymax = float(np.max(self._last_y))
        amp = max(1e-9, abs(ymin), abs(ymax))
        pad = 0.05 * amp
        self.plot.setYRange(-amp - pad, amp + pad, padding=0.0)

        self.curve.setData(t, self._last_y)
