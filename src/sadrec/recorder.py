import pyaudio
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QFileDialog
import threading
import wave
import soundfile as sf
from scipy.signal import filtfilt
from .spike_detection import SpikeDetector
from .utils import apply_highpass_filter, apply_lowpass_filter, butter_highpass, butter_lowpass, generate_sine_wave


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
        self.stim_plot = self.win.addPlot(title='Stimulus')
        self.win.nextRow()
        self.plot = self.win.addPlot(title="Audio Waveform")
        # self.stim_plot.setXLink(self.plot)

        # --- tweak row heights ---------------------------------------
        #   larger number  →  more stretch
        lay = self.win.ci.layout  # central item's QGridLayout
        lay.setRowStretchFactor(0, 1)  # row-0 gets 3 parts
        lay.setRowStretchFactor(1, 3)  # row-1 gets 1 part

        self.time_axis = np.arange(0, self.CHUNK * self.BUFFER_CHUNKS, 1) / self.RATE

        # Add Spike Detector
        self.spike_detector = SpikeDetector(self.plot, self.time_axis)

        dummy_data = np.zeros_like(self.time_axis)
        self.curve = self.plot.plot(self.time_axis, dummy_data, pen='b')  # Blue pen for original signal
        self.plot.setLabel('left', 'Signal Voltage (V)')
        self.plot.setLabel('bottom', 'Time (s)')
        self.plot.setYRange(self.y_min_range, self.y_max_range)
        self.plot.setXRange(0, (self.CHUNK * self.BUFFER_CHUNKS)/self.RATE)

        # Curve for stimulus
        self.stim_curve = self.stim_plot.plot(self.time_axis, dummy_data, pen='g')

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
        self.stim_buffer = np.zeros(self.CHUNK * self.BUFFER_CHUNKS, dtype=np.int16)
        self.plotting_data = None
        # self.filtered_audio_buffer = np.zeros(self.CHUNK * self.BUFFER_CHUNKS, dtype=np.int16)
        self.data_lock = threading.Lock()

        # Recording variables
        self.is_recording = False
        self.recorded_frames = []
        self.recorded_frames_stimulus = []

        self.playing_stimulus = False
        self.stimulus_samples = np.array([], dtype=np.int16)
        self.stim_index = 0

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
            b, a = butter_lowpass(self.low_cutoff, fs=self.wav_fs)
            data = filtfilt(b, a, data)
        if self.high_filter_enabled:
            b, a = butter_highpass(self.high_cutoff, fs=self.wav_fs)
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

    def run(self):
        threading.Thread(target=self.audio_thread, daemon=True).start()

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
                    audio_data = apply_lowpass_filter(audio_data, self.low_cutoff, fs=self.RATE)

                if self.high_filter_enabled:
                    audio_data = apply_highpass_filter(audio_data, self.high_cutoff, fs=self.RATE)

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

                    if self.playing_stimulus:
                        stim_chunk = self.stimulus_samples[self.stim_index:self.stim_index + self.CHUNK]
                        self.stim_index += self.CHUNK
                    else:
                        stim_chunk = np.zeros(self.CHUNK, dtype=np.int16)

                    self.recorded_frames_stimulus.append(stim_chunk.tobytes())

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

    # def save_audio(self, filename):
    #     with wave.open(filename, 'wb') as wf:
    #         wf.setnchannels(self.CHANNELS)
    #         wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
    #         wf.setframerate(self.RATE)
    #         wf.writeframes(b''.join(self.recorded_frames))

    def save_audio(self, filename):
        mic_data = np.frombuffer(b''.join(self.recorded_frames), dtype=np.int16)
        stim_data = np.frombuffer(b''.join(self.recorded_frames_stimulus), dtype=np.int16)

        # Match lengths (pad with zeros if necessary)
        length = max(len(mic_data), len(stim_data))
        mic_data = np.pad(mic_data, (0, length - len(mic_data)), mode='constant')
        stim_data = np.pad(stim_data, (0, length - len(stim_data)), mode='constant')

        # Stack as stereo: shape (length, 2)
        stereo = np.column_stack((mic_data, stim_data)).astype(np.int16)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(stereo.tobytes())

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

    # ---------- stimulation helpers ----------
    def run_stimulation(self, *, freq=440.0, duration=3.0, volume=0.01):
        threading.Thread(
            target=self._stimulation_thread,
            args=(freq, duration, volume),
            daemon=True
        ).start()

    def play_wav_stimulus(self, wav_path: str, *, volume: float = 1.0):
        """
        Play an arbitrary mono/stereo WAV file as stimulus,
        plot the left channel, and record it like the sine.
        """
        threading.Thread(
            target=self._wav_stimulation_thread,
            args=(wav_path, volume),
            daemon=True,
        ).start()

    def _wav_stimulation_thread(self, wav_path: str, volume: float):
        data, fs = sf.read(wav_path, dtype='int16')  # data shape: (samples, channels?)

        if fs != self.RATE:
            print(f"Resampling {fs} → {self.RATE} not implemented.")  # keep simple
            return

        # If stereo, take left channel for plotting/recording
        if data.ndim == 2:
            data_plot = data[:, 0]
        else:
            data_plot = data

        # volume scale (0-1)
        data_plot = (data_plot.astype(np.int32) * volume).astype(np.int16)

        # ── plot ──────────────────────────────
        t = np.arange(data_plot.size) / self.RATE
        self._show_stimulus(t, data_plot)

        # store for recording
        self.stimulus_samples = data_plot
        self.playing_stimulus = True
        self.stim_index = 0

        # ── playback ─────────────────────────
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, output=True)
        stream.write(data_plot.tobytes())
        stream.stop_stream();
        stream.close();
        p.terminate()

        # cleanup
        self.playing_stimulus = False
        self._show_stimulus([], [])

    def _stimulation_thread(self, freq, duration, volume):
        p = pyaudio.PyAudio()
        sine = generate_sine_wave(freq, duration, volume, self.RATE)

        # ----- draw it -----
        t = np.linspace(0, duration, sine.size, endpoint=False)
        stim_trace = np.zeros_like(sine)
        # scaled = (sine.astype(np.int16) * volume).astype(np.int16)
        # self._show_stimulus(t, scaled)
        self._show_stimulus(t, sine)
        # self._show_stimulus(t, stim_trace)

        # store entire stimulus (int16) for later saving
        self.stimulus_samples = sine  # 1-D numpy int16

        # ----- play it -----
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, output=True)
        self.playing_stimulus = True
        self.stimulus_samples = sine
        self.stim_index = 0
        stream.write(sine.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.playing_stimulus = False

        # hide after playback
        dummy_data = np.zeros_like(self.time_axis)
        self._show_stimulus([], [])
        self._show_stimulus(self.time_axis, dummy_data)

    def _show_stimulus(self, x, y):
        self.stim_curve.setData(x, y, pen='g')
        # QTimer.singleShot(0, lambda: self.stim_curve.setData(x, y))
