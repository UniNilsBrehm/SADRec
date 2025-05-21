import pyaudio
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QTimer, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu, QWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, \
    QSpinBox, QFormLayout, QFileDialog, QMessageBox
import sys
import threading
import wave
from scipy.signal import butter, lfilter, filtfilt
from datetime import datetime
from scipy.signal import find_peaks

'''
This is a light weight data recorder app. You can record signals from you soundcard microphone input.
Features:
- Live View 
- Audio Monitor
- Store recording in a wav file
- Low and High Pass Filter
- Navigate the Viewer with short cuts
- Simple Sine Wave Stimulator (via Speaker)
- Open and view wav files (e.g. your recordings)

NOTES:
For the audio stimulation to work you need to connect external speakers, otherwise the internal echo feedback protection
of the soundcard will suppress the microphone signal.
(so far only tested with bluetooth headphones and notebook internal microphone).

TO DO:
- Menu to select input and output channels
- Add running time axis
- Add Stimulation (sine waves, wav files) and display it in the viewer

'''

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

    # def show_spikes(self, spike_list):
    #     self.clear_spikes()
    #     for _, t in spike_list:
    #         line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen('r', width=1))
    #         self.plot.addItem(line)
    #         self.spike_lines.append(line)

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


class LiveAudioRecorder(QObject):
    def __init__(self, audio):
        super().__init__()
        print('')
        print('WELCOME TO THE EPHYS RECORDER')
        print('Press "M" to mute and unmute audio monitor')
        print('Press "R" to start and stop recording')
        print('Press "B" to reset the view')
        print('Press "C" to center the view (y-axis)')
        print('Press "T" and "Shift+T" to zoom the x-axis')
        print('Press "X" and "Shift+X" to zoom the y-axis')
        print('Press "Left" and "Right" to move along the x-axis')
        print('Press "Up" and "Down" to move along the y-axis')
        print('')

        # Parameters
        self.VIEWING_MODE = 'live'
        self.FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
        self.CHANNELS = 1              # Number of channels (mono)
        self.RATE = 44100              # Sample rate (44.1 kHz)
        # self.RATE = 90000          # Sample rate (44.1 kHz)
        self.CHUNK = 1024              # Buffer size
        self.RECORD_SECONDS = 2        # Duration of the recording to display in the plot
        self.BUFFER_CHUNKS = int(self.RATE / self.CHUNK * self.RECORD_SECONDS)  # Number of chunks to display

        self.soundcard_max = 32767
        self.soundcard_min = -32767
        self.y_max_range = self.soundcard_max
        self.y_min_range = self.soundcard_min
        self.amp_gain = 1
        self.wav_fs = 0

        self.save_dir = None

        # Initialize PyAudio
        self.audio = audio

        # Open default stream
        self.stream = self.audio.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)

        self.speaker_stream = self.audio.open(format=self.FORMAT,
                                              channels=self.CHANNELS,
                                              rate=self.RATE,
                                              output=True)

        # Set up live plotting using pyqtgraph
        self.win = pg.GraphicsLayoutWidget(title="Live Audio Data")
        self.plot = self.win.addPlot(title="Audio Waveform")

        self.time_axis = np.arange(0, self.CHUNK * self.BUFFER_CHUNKS, 1) / self.RATE

        # Add Spike Detector
        self.spike_detector = SpikeDetector(self.plot, self.time_axis)

        dummy_data = np.zeros_like(self.time_axis)
        self.curve = self.plot.plot(self.time_axis, dummy_data, pen='b')  # Blue pen for original signal
        self.plot.setLabel('left', 'Signal Voltage (V)')
        self.plot.setLabel('bottom', 'Time (s)')
        self.plot.setYRange(self.y_min_range, self.y_max_range)
        self.plot.setXRange(0, (self.CHUNK * self.BUFFER_CHUNKS)/self.RATE)

        # Text items for filter status
        self.low_filter_text = pg.TextItem(anchor=(0, 1))
        self.high_filter_text = pg.TextItem(anchor=(0, 0))
        self.plot.addItem(self.low_filter_text)
        self.plot.addItem(self.high_filter_text)

        # Filter parameters
        self.low_cutoff = 0  # Initial low cut-off frequency in Hz
        self.high_cutoff = 0  # Initial high cut-off frequency in Hz
        self.low_filter_enabled = False
        self.high_filter_enabled = False
        self.audio_monitor_status = False

        # Buffer to hold the audio data for the plot
        self.audio_buffer = np.zeros(self.CHUNK * self.BUFFER_CHUNKS, dtype=np.int16)
        self.plotting_data = None
        # self.filtered_audio_buffer = np.zeros(self.CHUNK * self.BUFFER_CHUNKS, dtype=np.int16)
        self.data_lock = threading.Lock()

        # Recording variables
        self.is_recording = False
        self.recorded_frames = []

        # Event for stopping the thread
        self.stop_event = threading.Event()

        # Timer for updating the plot
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # Update plot every 50ms

    def change_viewing_mode(self, mode):
        self.VIEWING_MODE = mode
        # Disable spike detection when switching mode
        self.spike_detector.update_params(enabled=False)
        if self.VIEWING_MODE == 'live':
            # LIVE PLOT
            self.time_axis = np.arange(0, self.CHUNK * self.BUFFER_CHUNKS, 1) / self.RATE
            # Timer for updating the plot
            self.timer.start(50)  # Update plot every 50ms
        else:
            # WAV FILE VIEWER
            self.timer.stop()
            self.wav_viewer()

    def update_wav_plot(self):
        if self.plotting_data is None or len(self.plotting_data) == 0:
            print("No data to plot.")
            return

        # Apply filters
        data = self.plotting_data.copy()
        if self.low_filter_enabled:
            b, a = self.butter_lowpass(self.low_cutoff, fs=self.wav_fs)
            data = filtfilt(b, a, data)
        if self.high_filter_enabled:
            b, a = self.butter_highpass(self.high_cutoff, fs=self.wav_fs)
            data = filtfilt(b, a, data)

        self.curve.setData(self.time_axis, data)

        # Better way: use autoRange to fit the view dynamically
        self.plot.enableAutoRange('xy', True)
        self.plot.autoRange(padding=0.02)  # Optional: small padding
        self.spike_detector.set_time_axis(self.time_axis)

    def wav_viewer(self):
        file_dir = QFileDialog.getOpenFileNames()[0][0]
        if file_dir:
            self.VIEWING_MODE = 'wav'
            with wave.open(file_dir) as wf:
                self.wav_fs = wf.getframerate()
                samples = wf.getnframes()
                data = wf.readframes(samples)
                data_as_np_int16 = np.frombuffer(data, dtype=np.int16)
                data_as_np_float32 = data_as_np_int16.astype(np.float32)
                max_int16 = 2**15
                self.time_axis = np.arange(0, len(data_as_np_float32) / self.wav_fs, 1 / self.wav_fs)
                self.plotting_data = data_as_np_float32 / max_int16
                self.spike_detector.set_time_axis(self.time_axis)

            self.update_wav_plot()

    @staticmethod
    def generate_sine_wave(frequency, duration, volume, sample_rate):
        # Calculate the number of frames required for specified duration
        num_frames = int(sample_rate * duration)
        # Generate the time values for the samples
        t = np.linspace(0, duration, num_frames, endpoint=False)
        # Generate the sine wave
        sine_wave = np.sin(2 * np.pi * frequency * t)
        # Normalize to 16-bit range
        sine_wave = (sine_wave * (2 ** 15 - 1) / np.max(np.abs(sine_wave))) * volume
        # Convert to 16-bit PCM format
        sine_wave = sine_wave.astype(np.int16)
        return sine_wave

    @staticmethod
    def butter_lowpass(cutoff, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    @staticmethod
    def butter_highpass(cutoff, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def apply_lowpass_filter(self, data, fs):
        # Live Filtering (adds phase shifts)
        b, a = self.butter_lowpass(self.low_cutoff, fs)
        y = lfilter(b, a, data)
        return y.astype(np.int16)

    def apply_highpass_filter(self, data, fs):
        # Live Filtering (adds phase shifts)
        b, a = self.butter_highpass(self.high_cutoff, fs)
        y = lfilter(b, a, data)
        return y.astype(np.int16)

    def run(self):
        threading.Thread(target=self.audio_thread, daemon=True).start()

    def run_stimulation(self):
        threading.Thread(target=self.stimulation_thread, daemon=True).start()

    def stimulation_thread(self):
        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Generate sine wave
        sine_wave = self.generate_sine_wave(frequency=440, duration=3, volume=0.01, sample_rate=self.RATE)

        # Open stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.RATE,
                        output=True)

        # Play sine wave
        stream.write(sine_wave.tobytes())

        # Stop stream and close PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

    @staticmethod
    def scale_to_new_range(x, old_min, old_max, new_min, new_max, gain):
        return ((new_max - new_min) * (x - old_min) / (old_max - old_min)) + new_min

    def audio_thread(self):
        while not self.stop_event.is_set():
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Apply filters
                # filtered_data = audio_data
                if self.low_filter_enabled:
                    audio_data = self.apply_lowpass_filter(audio_data, fs=self.RATE)

                if self.high_filter_enabled:
                    audio_data = self.apply_highpass_filter(audio_data, fs=self.RATE)

                # Update the buffer with the new data
                with self.data_lock:
                    self.audio_buffer = np.roll(self.audio_buffer, -self.CHUNK)
                    self.audio_buffer[-self.CHUNK:] = audio_data

                # Audio Monitor (Play Sound)
                if self.audio_monitor_status:
                    self.speaker_stream.write(data)

                # Record frames if recording is activated
                if self.is_recording:
                    self.recorded_frames.append(audio_data.tobytes())
            except Exception as e:
                print(f"Error in audio thread: {e}")
                break

    def update_plot(self):
        with self.data_lock:
            self.plotting_data = self.audio_buffer
            # self.curve.setData(self.time_axis, self.audio_buffer)
            self.curve.setData(self.time_axis, self.plotting_data)

            spikes = self.spike_detector.detect(self.plotting_data)
            self.spike_detector.show_spikes(spikes)

            # Change pen color to red when recording
            if self.is_recording:
                self.curve.setPen('r')
            else:
                self.curve.setPen('b')

    def save_audio(self, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.recorded_frames))

    def stop(self):
        self.stop_event.set()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def set_low_cutoff(self, cutoff):
        if cutoff == 'off' or cutoff == 0:
            self.low_filter_enabled = False
            self.low_cutoff = cutoff
        else:
            self.low_filter_enabled = True
            self.low_cutoff = cutoff

    def set_high_cutoff(self, cutoff):
        if cutoff == 'off' or cutoff == 0:
            self.high_filter_enabled = False
            self.high_cutoff = cutoff
        else:
            self.high_filter_enabled = True
            self.high_cutoff = cutoff


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

        # --- Stimulation Menu ---
        stimulation_menu = menubar.addMenu('Stimulation')
        sine_wave_stimulation = stimulation_menu.addAction('Sine Wave')
        sine_wave_stimulation.triggered.connect(self.recorder.run_stimulation)

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


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.recorder.run()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()