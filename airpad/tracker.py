"""MediaPipe HandLandmarker wrapper with square-tile fallback tracking.

Primary path: VIDEO running mode on the full frame. It is synchronous (the
landmarks always belong to the frame being displayed) and tracks the hand
between frames, only re-running the palm detector when tracking is lost.

Why a fallback: MediaPipe's palm detector letterboxes the *whole* frame into
192x192. On a 16:9 webcam a hand at arm's length becomes a ~15 px palm there
and detection becomes position-dependent (measured on a hard sample photo:
~50 % of positions missed). A full-height square crop shows the detector a
1.8x larger hand (measured: 99.7 % detected).

So the frame is covered by a few overlapping square *tiles* (3 for 16:9),
each with its own VIDEO-mode tracker. Tiles never move, so each tile tracker
keeps genuine frame-to-frame tracking (landmark model only, no per-frame palm
detection) while it holds the hand. When the primary loses the hand, the tile
nearest to where it was last seen searches first; otherwise tiles are swept.
As the hand moves, it is handed to a better-centred tile (with hysteresis).
The primary is retried periodically and takes over again when it can.

Fresh detections: the first landmarks after a palm detection are coarse
(measured: fingertip 105 px off and the wrong gesture) and are refined from
the next frame on by tracking. Such observations are flagged ``fresh`` so the
engine can ignore them, and source switches (tile handover, primary takeover)
only happen once the new tracker is tracking, never on its first detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class HandObservation:
    """One detected hand.

    norm  : (21, 3) landmarks; x, y normalised to [0, 1] of the frame, z relative.
    world : (21, 3) metric landmarks in metres, origin at the hand's centre.
    fresh : True if these come from a new palm detection (coarse) rather than
            frame-to-frame tracking.
    """

    norm: np.ndarray
    world: np.ndarray
    handedness: str
    score: float
    fresh: bool = False


def _to_observation(result) -> HandObservation | None:
    if not result.hand_landmarks:
        return None
    best = 0
    if len(result.handedness) > 1:
        scores = [hd[0].score if hd else 0.0 for hd in result.handedness]
        best = int(np.argmax(scores))
    lms = result.hand_landmarks[best]
    wlms = result.hand_world_landmarks[best]
    cat = result.handedness[best][0] if result.handedness and result.handedness[best] else None
    return HandObservation(
        norm=np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float64),
        world=np.array([(p.x, p.y, p.z) for p in wlms], dtype=np.float64),
        handedness=cat.category_name if cat else "",
        score=float(cat.score) if cat else 0.0,
    )


class _VideoLandmarker:
    """A VIDEO-mode landmarker with its own monotonic clock.

    A result counts as *tracked* only if this landmarker also found the hand
    on the immediately preceding frame; otherwise MediaPipe either ran a new
    palm detection or tracked from a stale region, so the result is fresh.
    """

    def __init__(self, factory):
        self._lm = factory()
        self._last_ts = -1
        self._last_frame = -10
        self._had_hand = False

    def detect(self, image, timestamp_ms: int, frame_no: int) -> HandObservation | None:
        ts = max(int(timestamp_ms), self._last_ts + 1)  # VIDEO mode needs strictly increasing ts
        self._last_ts = ts
        obs = _to_observation(self._lm.detect_for_video(image, ts))
        tracked = self._had_hand and self._last_frame == frame_no - 1
        self._had_hand = obs is not None
        self._last_frame = frame_no
        if obs is not None and not tracked:
            obs = replace(obs, fresh=True)
        return obs

    def close(self) -> None:
        self._lm.close()


class HandTracker:
    RECHECK_PRIMARY_EVERY = 6    # while a tile holds the hand, let the primary retry
    HANDOVER_HYSTERESIS = 0.06   # of frame width

    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        infer_width: int = 640,
        tile_search: bool = True,
        frame_size: tuple[int, int] | None = None,
    ):
        """``frame_size`` (w, h), if known, builds the tile trackers now rather
        than on the first frame (creating a landmarker takes ~0.4 s)."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        # Load from bytes: MediaPipe's file loader is unreliable with non-ASCII
        # Windows paths, a buffer always works.
        model_bytes = Path(model_path).read_bytes()
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._factory = lambda: vision.HandLandmarker.create_from_options(options)
        self._primary: _VideoLandmarker | None = _VideoLandmarker(self._factory)
        self.tile_search = tile_search
        self.infer_width = int(infer_width)
        self._tiles: list[tuple[int, _VideoLandmarker]] = []  # (x0, tracker)
        self._tile_geom: tuple[int, int] | None = None
        self._mode: int | None = None  # None = primary holds/searches, else tile index
        self._primary_warm = False      # primary re-found the hand last frame (fresh)
        self._frame_no = 0
        self._misses = 0
        self._last_cx: float | None = None  # pixel x of the last hand seen
        self._sweep = 0
        self.last_source: str | None = None  # "track" | "tile" | None
        if frame_size is not None:
            self._ensure_tiles(int(frame_size[0]), int(frame_size[1]))

    # ----------------------------------------------------------------- helpers
    def _image(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        if 0 < self.infer_width < w:
            # The landmark model sees a 224x224 crop anyway; a smaller frame mostly
            # saves colour-conversion and copy time (fingertip error < 0.3 %).
            bgr = cv2.resize(bgr, (self.infer_width, max(1, round(h * self.infer_width / w))),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

    def _ensure_tiles(self, w: int, h: int) -> None:
        if self._tile_geom == (w, h):
            return
        for _, t in self._tiles:
            t.close()
        self._tiles, self._mode, self._tile_geom = [], None, (w, h)
        if not self.tile_search or w <= h * 1.05:
            return  # (near-)square frames gain nothing from square tiles
        n = math.ceil((w - h) / (0.4 * h)) + 1
        for x0 in np.linspace(0, w - h, n).round().astype(int):
            self._tiles.append((int(x0), _VideoLandmarker(self._factory)))

    def _primary_detect(self, frame: np.ndarray, ts: int) -> HandObservation | None:
        return self._primary.detect(self._image(frame), ts, self._frame_no)

    def _tile_detect(self, k: int, frame: np.ndarray, ts: int) -> HandObservation | None:
        x0, tracker = self._tiles[k]
        h, w = frame.shape[:2]
        obs = tracker.detect(self._image(np.ascontiguousarray(frame[:, x0:x0 + h])), ts, self._frame_no)
        if obs is None:
            return None
        norm = obs.norm.copy()
        norm[:, 0] = (x0 + norm[:, 0] * h) / w  # tile -> full-frame coordinates
        return replace(obs, norm=norm)

    def _nearest_tile(self, cx: float, h: int) -> int:
        return int(np.argmin([abs(cx - (x0 + h / 2)) for x0, _ in self._tiles]))

    def _hit(self, obs: HandObservation, source: str, w: int) -> HandObservation:
        self._misses = 0
        self._last_cx = float(np.clip(obs.norm[:, 0].mean(), 0.0, 1.0)) * w
        self.last_source = source
        return obs

    def _miss(self) -> None:
        self._misses += 1
        self.last_source = None
        if self._misses > 2:
            self._last_cx = None  # stale: sweep the tiles instead
        return None

    def _enter_tile(self, k: int, frame: np.ndarray, ts: int, w: int) -> HandObservation | None:
        obs = self._tile_detect(k, frame, ts)
        if obs is None:
            return self._miss()
        self._mode = k  # this first tile result is fresh; tracking refines it next frame
        return self._hit(obs, "tile", w)

    # --------------------------------------------------------------------- api
    @property
    def mode(self) -> str:
        return "primary" if self._mode is None else f"tile{self._mode}"

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandObservation | None:
        """Detect the most confident hand in ``frame_bgr``; ``None`` if no hand."""
        self._frame_no += 1
        h, w = frame_bgr.shape[:2]
        self._ensure_tiles(w, h)
        ts = int(timestamp_ms)

        if self._mode is None:
            # No hand held. Palm detection is the expensive model, so while
            # sweeping run at most one per frame: odd frames (incl. the very
            # first) use the full-frame primary, even frames try the next tile.
            sweeping = self._last_cx is None
            if sweeping and self._tiles and self._frame_no % 2 == 0:
                k = self._sweep % len(self._tiles)
                self._sweep += 1
                return self._enter_tile(k, frame_bgr, ts, w)
            obs = self._primary_detect(frame_bgr, ts)
            if obs is not None:
                return self._hit(obs, "track", w)
            if not self._tiles or sweeping:
                return self._miss()
            # Just lost the hand: also retry the tile where it was, this frame.
            return self._enter_tile(self._nearest_tile(self._last_cx, h), frame_bgr, ts, w)

        # A tile tracker currently holds the hand.
        k = self._mode
        primary = None
        if self._primary_warm or self._frame_no % self.RECHECK_PRIMARY_EVERY == 0:
            primary = self._primary_detect(frame_bgr, ts)
            self._primary_warm = primary is not None and primary.fresh
            if primary is not None and not primary.fresh:
                self._mode = None  # primary is tracking again: hand back to it
                return self._hit(primary, "track", w)
        obs = self._tile_detect(k, frame_bgr, ts)
        if obs is None:
            self._mode = None
            self._primary_warm = False
            return self._hit(primary, "track", w) if primary is not None else self._miss()
        cx = float(np.clip(obs.norm[:, 0].mean(), 0.0, 1.0)) * w
        best = self._nearest_tile(cx, h)
        if best != k:
            here = abs(cx - (self._tiles[k][0] + h / 2))
            there = abs(cx - (self._tiles[best][0] + h / 2))
            if here - there > self.HANDOVER_HYSTERESIS * w:
                # Warm up the better-centred tile; switch once it is tracking.
                cand = self._tile_detect(best, frame_bgr, ts)
                if cand is not None and not cand.fresh:
                    self._mode = best
                    return self._hit(cand, "tile", w)
        return self._hit(obs, "tile", w)

    def close(self) -> None:
        if self._primary is not None:
            self._primary.close()
            self._primary = None
        for _, t in self._tiles:
            t.close()
        self._tiles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
