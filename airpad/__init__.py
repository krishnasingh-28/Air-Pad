"""AirPad - draw and write in the air with your index finger.

Webcam -> MediaPipe hand landmarks -> gesture state machine -> smoothed strokes
-> layered canvas composited over the live camera (or a clean board).
"""

import os

# These must be set before OpenCV / MediaPipe initialise their backends.
# Media Foundation with hardware transforms disabled opens a webcam in ~1 s
# (instead of ~12 s) and still delivers full 30 FPS at 720p on Windows.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
# Quieten MediaPipe / TFLite / absl start-up chatter.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__version__ = "1.0.0"
