from importlib.metadata import version as _v

__all__ = ["LiveAudioRecorder", "SpikeDetector"]
__version__ = _v(__package__ or "sadrec")

from .recorder import LiveAudioRecorder
from .spike_detection import SpikeDetector
