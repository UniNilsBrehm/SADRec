import pyqtgraph as pg
from scipy.signal import find_peaks


class SpikeDetector:
    def __init__(self, plot_item, time_axis):
        self.enabled = False
        self.threshold = 500
        self.min_distance = 20
        self.plot = plot_item  # the PyQtGraph plot object
        self.time_axis = time_axis
        self.spike_lines = []

    def update_params(self, threshold=None, min_distance=None, enabled=None):
        if threshold is not None:
            self.threshold = threshold
        if min_distance is not None:
            self.min_distance = min_distance
        if enabled is not None:
            self.enabled = enabled

    def set_time_axis(self, time_axis):
        self.time_axis = time_axis

    def detect(self, data):
        if not self.enabled or data is None or len(data) == 0:
            self.clear_spikes()
            return []

        peaks, _ = find_peaks(data, height=self.threshold, distance=self.min_distance)
        return list(zip(peaks, self.time_axis[peaks]))

    def show_spikes(self, spike_list):
        self.clear_spikes()
        for idx, t in spike_list:
            if idx < len(self.time_axis):
                # Use the Y value from the plot's data
                y_val = self.plot.curves[0].yData[idx]
                dot = pg.ScatterPlotItem(
                    x=[self.time_axis[idx]],
                    y=[y_val],
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush('r'),
                    size=6
                )
                self.plot.addItem(dot)
                self.spike_lines.append(dot)

    def clear_spikes(self):
        for line in self.spike_lines:
            self.plot.removeItem(line)
        self.spike_lines.clear()