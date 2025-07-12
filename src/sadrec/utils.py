import numpy as np
from scipy.signal import butter, lfilter, filtfilt


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


def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def apply_lowpass_filter(data, low_cutoff, fs):
    # Live Filtering (adds phase shifts)
    b, a = butter_lowpass(low_cutoff, fs)
    y = lfilter(b, a, data)
    return y.astype(np.int16)


def apply_highpass_filter(data, high_cutoff, fs):
    # Live Filtering (adds phase shifts)
    b, a = butter_highpass(high_cutoff, fs)
    y = lfilter(b, a, data)
    return y.astype(np.int16)
