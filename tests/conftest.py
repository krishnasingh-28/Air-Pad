import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import airpad  # noqa: E402,F401  (sets OpenCV/MediaPipe env vars before cv2 is imported)

FIXTURES = ROOT / "tests" / "fixtures"
MODEL = ROOT / "models" / "hand_landmarker.task"
