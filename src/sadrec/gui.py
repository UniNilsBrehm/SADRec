import pyaudio
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, \
    QSpinBox, QFileDialog, QMessageBox, QDialog, QFormLayout, QDoubleSpinBox, QPushButton
from datetime import datetime
from .recorder import LiveAudioRecorder


# Helper dialog ----------------------------------------------------------
class SineDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sine-wave parameters")

        self.freq   = QDoubleSpinBox(); self.freq.setRange(10, 20_000); self.freq.setValue(440)
        self.dur    = QDoubleSpinBox(); self.dur.setRange(0.1, 10);     self.dur.setValue(3); self.dur.setSuffix(" s")
        self.vol    = QDoubleSpinBox(); self.vol.setRange(0.001, 1);    self.vol.setSingleStep(0.01); self.vol.setValue(0.01)

        lay = QFormLayout(self)
        lay.addRow("Frequency (Hz)", self.freq)
        lay.addRow("Duration (s)",   self.dur)
        lay.addRow("Volume (0-1)",   self.vol)

        ok = QPushButton("Start"); ok.clicked.connect(self.accept)
        lay.addRow(ok)

    # convenience accessors
    def parameters(self):
        return self.freq.value(), self.dur.value(), self.vol.value()
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Create a PyAudio Instance for all other stuff
        self.audio = pyaudio.PyAudio()

        self.recorder = LiveAudioRecorder(self.audio)
        self._setup_gui()
        self.create_menu()

        # Mouse Bindings
        self.recorder.plot.scene().sigMouseMoved.connect(self.mouse_moved)

        self.update_filter_text()
        self.show()
        self._add_stimulation_menu()

    def _add_stimulation_menu(self):
        stim_menu = self.menuBar().addMenu("Stimulation")

        sine_act = stim_menu.addAction("Sine-wave …")
        sine_act.triggered.connect(self._open_sine_dialog)

        wav_act = stim_menu.addAction("Play WAV…")
        wav_act.triggered.connect(self._choose_wav)

    def _choose_wav(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select WAV stimulus", filter="WAV files (*.wav)")
        if file:
            self.recorder.play_wav_stimulus(file, volume=1.0)  # or ask user for a scale factor

    def _setup_gui(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Main Layout
        self.layout = QVBoxLayout(self.central_widget)

        # Label for Mouse Position
        self.position_label = QLabel(f"X: {0}, Y: {0}")
        self.layout.addWidget(self.position_label)
        self.layout.addWidget(self.recorder.win)
        self.input_layout = QHBoxLayout()

        # Add input boxes at the bottom
        # High Pass Filter
        self.high_cutoff_input_label = QLabel('High-pass Filter (Hz):')
        self.high_cutoff_input = QSpinBox(self)
        self.high_cutoff_input.setRange(0, 10000)
        self.high_cutoff_input.setValue(0)  # Default to 'off'w
        self.high_cutoff_input.setSingleStep(10)  # Step size for arrows
        self.high_cutoff_input.valueChanged.connect(self.filter_changed)
        # Set a fixed size for the input boxes
        self.high_cutoff_input.setFixedWidth(100)
        vbox = QVBoxLayout()
        vbox.addWidget(self.high_cutoff_input_label)
        vbox.addWidget(self.high_cutoff_input)
        vbox.setSpacing(0)  # Adjust spacing between label and spinbox
        self.input_layout.addLayout(vbox)

        # Low Pass Filter
        self.low_cutoff_input_label = QLabel('Low-pass Filter (Hz):')
        self.low_cutoff_input = QSpinBox(self)
        self.low_cutoff_input.setRange(0, 10000)
        self.low_cutoff_input.setValue(0)  # Default to 'off'
        self.low_cutoff_input.setSingleStep(10)  # Step size for arrows
        self.low_cutoff_input.valueChanged.connect(self.filter_changed)
        self.low_cutoff_input.setFixedWidth(100)
        vbox = QVBoxLayout()
        vbox.addWidget(self.low_cutoff_input_label)
        vbox.addWidget(self.low_cutoff_input)
        vbox.setSpacing(0)  # Adjust spacing between label and spinbox
        self.input_layout.addLayout(vbox)
        self.layout.addLayout(self.input_layout)

        # SPIKE DETECTION
        self._spike_detection_gui()

    def _open_sine_dialog(self):
        dlg = SineDialog(self)
        if dlg.exec():
            f, d, v = dlg.parameters()
            self.recorder.run_stimulation(freq=f, duration=d, volume=v)

    def _spike_detection_gui(self):
        # Spike Threshold
        self.spike_thresh_label = QLabel('Threshold:')
        self.spike_thresh_box = QSpinBox()
        self.spike_thresh_box.setRange(0, 32767)
        self.spike_thresh_box.setValue(10000)
        self.spike_thresh_box.setFixedWidth(100)
        self.spike_thresh_box.valueChanged.connect(self.spike_settings_changed)

        vbox = QVBoxLayout()
        vbox.addWidget(self.spike_thresh_label)
        vbox.addWidget(self.spike_thresh_box)
        vbox.setSpacing(0)
        self.input_layout.addLayout(vbox)

        # Spike Min Distance
        self.spike_dist_label = QLabel('Min Dist:')
        self.spike_dist_box = QSpinBox()
        self.spike_dist_box.setRange(1, 1000)
        self.spike_dist_box.setValue(20)
        self.spike_dist_box.setFixedWidth(100)
        self.spike_dist_box.valueChanged.connect(self.spike_settings_changed)

        vbox = QVBoxLayout()
        vbox.addWidget(self.spike_dist_label)
        vbox.addWidget(self.spike_dist_box)
        vbox.setSpacing(0)
        self.input_layout.addLayout(vbox)

    def create_menu(self):
        menubar = self.menuBar()

        # --- File Menu ---
        file_menu = menubar.addMenu('File')

        # Live View
        start_live = file_menu.addAction('Live View')
        start_live.triggered.connect(
            lambda: (self.recorder.change_viewing_mode('live'), self.disable_spike_menu_toggle()))

        # WAV File Viewer
        open_file = file_menu.addMenu('View Wav File')
        open_file_action = open_file.addAction('Open ...')
        open_file_action.triggered.connect(
            lambda: (self.recorder.change_viewing_mode('wav'), self.disable_spike_menu_toggle()))

        # Output Folder
        save_dir = file_menu.addMenu('Output Directory')
        self.save_dir_label = save_dir.addAction('Set Output Directory ...')
        self.save_dir_label.triggered.connect(lambda: self.open_directory())

        # --- Spike Detection Toggle ---
        spike_menu = menubar.addMenu("Spike Detection")
        self.spike_toggle_action = spike_menu.addAction("Enable Spike Detection")
        self.spike_toggle_action.setCheckable(True)
        self.spike_toggle_action.setChecked(False)
        self.spike_toggle_action.triggered.connect(self.toggle_spike_detection)

        # --- Help Menu ---
        help_menu = menubar.addMenu("Help")
        show_help = help_menu.addAction("Keyboard Shortcuts")
        show_help.triggered.connect(self.show_help_dialog)

    def show_help_dialog(self):
        help_text = """
        <b>Welcome to the EPHYS RECORDER</b><br><br>
        <b>Keyboard Shortcuts:</b><br>
        <ul>
        <li><b>M</b>: Mute / Unmute audio monitor</li>
        <li><b>R</b>: Start / Stop recording</li>
        <li><b>B</b>: Reset view</li>
        <li><b>C</b>: Center view (Y-axis)</li>
        <li><b>T</b>: Zoom out X-axis</li>
        <li><b>Shift+T</b>: Zoom in X-axis</li>
        <li><b>X</b>: Zoom out Y-axis</li>
        <li><b>Shift+X</b>: Zoom in Y-axis</li>
        <li><b>← →</b>: Move along X-axis</li>
        <li><b>↑ ↓</b>: Move along Y-axis</li>
        <li><b>S</b>: Play sine wave stimulation</li>
        </ul>
        """
        QMessageBox.information(self, "Help - Keyboard Shortcuts", help_text)

    def open_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.recorder.save_dir = folder
            self.save_dir_label.setText(self.recorder.save_dir)

    def toggle_spike_detection(self, checked):
        self.recorder.spike_detector.update_params(enabled=checked)

    def spike_settings_changed(self):
        self.recorder.spike_detector.update_params(
            threshold=self.spike_thresh_box.value(),
            min_distance=self.spike_dist_box.value()
        )

    def disable_spike_menu_toggle(self):
        self.spike_toggle_action.setChecked(False)

    def gain_changed(self):
        self.recorder.amp_gain = self.gain.value()
        self.center_axis()

    def filter_changed(self):
        if self.recorder.VIEWING_MODE == 'live':
            self.recorder.set_low_cutoff(self.low_cutoff_input.value())
            self.recorder.set_high_cutoff(self.high_cutoff_input.value())
        else:
            self.recorder.set_low_cutoff(self.low_cutoff_input.value())
            self.recorder.set_high_cutoff(self.high_cutoff_input.value())
            self.recorder.update_wav_plot()
            # print('UPDATE WAV')
        self.update_filter_text()

    def center_axis(self):
        data = self.recorder.plotting_data
        if data is not None and len(data) > 0:
            ymin = float(data.min())
            ymax = float(data.max())
            if ymin == ymax:
                ymin -= 1
                ymax += 1
            padding = 0.05 * abs(ymax - ymin)
            self.recorder.plot.setYRange(ymin - padding, ymax + padding)

    def reset_axis(self):
        self.center_axis()
        self.recorder.plot.setYRange(self.recorder.y_min_range, self.recorder.y_max_range)
        self.recorder.plot.setXRange(0, self.recorder.time_axis[-1])

    def move_axis(self, factor, axis):
        # axis=0: x axis, axis=1: y axis
        # Get current axis range
        xmin, xmax = self.recorder.plot.viewRange()[axis]
        axis_range = abs(xmax - xmin)

        # Calculate new axis range
        new_xmin = xmin + factor*axis_range
        new_xmax = new_xmin + axis_range
        if axis == 0:
            self.recorder.plot.setXRange(new_xmin, new_xmax, padding=0)
        else:
            self.recorder.plot.setYRange(new_xmin, new_xmax, padding=0)

    def zoom_axis(self, factor, axis):
        # Get current axis range
        xmin, xmax = self.recorder.plot.viewRange()[axis]

        # Calculate new axis range
        new_xmin = xmin - (xmax - xmin) * factor
        new_xmax = xmax + (xmax-xmin) * factor

        # Set new axis range
        if axis == 0:
            self.recorder.plot.setXRange(new_xmin, new_xmax, padding=0)
        else:
            self.recorder.plot.setYRange(new_xmin, new_xmax, padding=0)

    def update_filter_text(self):
        self.recorder.plot.setTitle(f"LowPass: {self.recorder.low_cutoff} Hz, HighPass: {self.recorder.high_cutoff} Hz")

    def closeEvent(self, event):
        self.recorder.stop()
        event.accept()

    def mouse_moved(self, event):
        vb = self.recorder.plot.vb
        if self.recorder.plot.sceneBoundingRect().contains(event):
            mouse_point = vb.mapSceneToView(event)
            self.position_label.setText(f'X: {mouse_point.x(): .3f}, Y: {mouse_point.y(): .3f}')

    def keyPressEvent(self, event):
        modifiers = QApplication.keyboardModifiers()

        # R Key - Recording
        if event.key() == Qt.Key.Key_R:
            if self.recorder.is_recording:
                if self.recorder.save_dir is not None:
                    print("Recording stopped.")
                    self.recorder.is_recording = False
                    file_name = f'{self.recorder.save_dir}/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_recording.wav'
                    self.recorder.save_audio(file_name)
                    self.recorder.recorded_frames = []
            else:
                if self.recorder.save_dir is not None:
                    print("Recording started.")
                    self.recorder.is_recording = True
                else:
                    print("ERROR: Please select OUTPUT FOLDER before starting any recordings!")
                    self.recorder.is_recording = False

        # M Key - Mute
        if event.key() == Qt.Key.Key_M:
            self.recorder.audio_monitor_status = np.invert(self.recorder.audio_monitor_status)
            print(f'AUDIO MONITOR: {self.recorder.audio_monitor_status}')

        # S Key - Run Stimulation
        if event.key() == Qt.Key.Key_S:
            print('PLAY SINE WAVE')
            self.recorder.run_stimulation()

        # Zoom in on X axis with 'T'
        if event.key() == Qt.Key.Key_T and modifiers == Qt.KeyboardModifier.ShiftModifier:
            self.zoom_axis(factor=0.1, axis=0)  # Zoom in by 10%

        # Zoom out on X axis with 'Shift+T'
        elif event.key() == Qt.Key.Key_T:
            self.zoom_axis(factor=-0.1, axis=0)  # Zoom out by 10%

        # Zoom in on Y axis with 'X'
        if event.key() == Qt.Key.Key_X and modifiers == Qt.KeyboardModifier.ShiftModifier:
            self.zoom_axis(factor=0.1, axis=1)  # Zoom in by 10%

        # Zoom out on X axis with 'Shift+X'
        elif event.key() == Qt.Key.Key_X:
            self.zoom_axis(factor=-0.1, axis=1)  # Zoom out by 10%

        # B Key - Reset Axis
        if event.key() == Qt.Key.Key_B:
            self.reset_axis()

        # Arrow Keys
        if event.key() == Qt.Key.Key_Left:
            self.move_axis(factor=-0.1, axis=0)

        if event.key() == Qt.Key.Key_Right:
            self.move_axis(factor=0.1, axis=0)

        if event.key() == Qt.Key.Key_Up:
            self.move_axis(factor=0.1, axis=1)

        if event.key() == Qt.Key.Key_Down:
            self.move_axis(factor=-0.1, axis=1)

        # C Key - Center Axis
        if event.key() == Qt.Key.Key_C:
            self.center_axis()
