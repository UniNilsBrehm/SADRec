from scipy.signal import butter, lfilter, filtfilt
import numpy as np


def generate_sine_wave(frequency, duration, volume, sample_rate):
    num_frames = int(sample_rate * duration)
    t = np.linspace(0, duration, num_frames, endpoint=False)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    sine_wave = (sine_wave * (2 ** 15 - 1) / np.max(np.abs(sine_wave))) * volume
    return sine_wave.astype(np.int16)


def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)


def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)


def apply_lowpass_filter(data, cutoff, fs):
    b, a = butter_lowpass(cutoff, fs)
    return lfilter(b, a, data).astype(data.dtype)


def apply_highpass_filter(data, cutoff, fs):
    b, a = butter_highpass(cutoff, fs)
    return lfilter(b, a, data).astype(data.dtype)


def apply_lowpass_filter_zero_phase(data, cutoff, fs):
    b, a = butter_lowpass(cutoff, fs)
    return filtfilt(b, a, data).astype(data.dtype)


def apply_highpass_filter_zero_phase(data, cutoff, fs):
    b, a = butter_highpass(cutoff, fs)
    return filtfilt(b, a, data).astype(data.dtype)