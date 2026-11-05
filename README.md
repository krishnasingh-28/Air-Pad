# AirPad — a finger-drawn scratch pad

Write and draw in the air with your index finger. Your webcam tracks your hand
(MediaPipe hand landmarks), gestures lift and lower the pen, and ink is drawn
over the live camera view or on a clean board.

## Quick start (Windows)

```powershell
cd F:\Projects\Scratch-Pad
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m airpad
```

The hand model (`models/hand_landmarker.task`, 7.8 MB) is included; if it is
missing it is downloaded automatically from Google's MediaPipe model storage.

## How to use it

| Gesture | Action |
| --- | --- |
| ☝️ Index finger up | **Draw** (pen down) |
| ✌️ Index + middle finger | **Move** without drawing (pen up), and select |
| ✋ Open palm | **Erase** under your palm |
| Point at a toolbar button and hold ~0.5 s | Press it (a ring fills around your fingertip) |
| Quick tap with the index finger | A dot (for i, j, full stops) |

**Custom ink color:** point at the rainbow **+** button, then point at any color
in the spectrum (or the gray strip) and hold still for ~1 s. The color is picked,
shown in the preview, and saved in the swatch next to the presets so you can
switch back to it. After picking, make the two-finger "move" pose once before
drawing, so the pen does not go down where the picker used to be.

Toolbar: 9 preset colors · custom color · color picker · pen / eraser ·
brush size − / + · undo · redo · clear · save · board mode. Buttons also work with
a mouse click.

| Key | Action |
| --- | --- |
| `1`–`9` / `P` | Preset colors / color picker |
| `E` | Toggle eraser tool |
| `[` `]` (or `-` `=`) | Brush (or eraser) size |
| `Z` / `Y` | Undo / redo |
| `C` | Clear (undoable) |
| `S` | Save PNGs to `outputs/` (on the board color + a transparent version) |
| `B` | Board mode: pure black page, camera shown picture-in-picture |
| `L` / `M` / `F` | Hand skeleton / mirror / fullscreen |
| `H` | Help overlay |
| `Q` or `Esc` | Quit (`Esc` closes the picker/help first) |

Tips for the best tracking: light your hand from the front, keep it roughly
40–80 cm from the camera, and keep your drawing finger clearly separated from
the others.

### Options

```text
python -m airpad --camera 1          # another webcam
python -m airpad --board             # start on the clean board
python -m airpad --dim 1.0           # don't darken the camera behind the ink
python -m airpad --source clip.mp4   # play a video file instead of the webcam
python -m airpad --backend dshow     # if the default Media Foundation backend fails
python -m airpad --help              # everything else
```

## Performance

Measured on this laptop (i7-1165G7) **on battery with the CPU throttled to
1.19 GHz**, i.e. a worst case; on mains power there is roughly twice the headroom.

| | |
| --- | --- |
| Live webcam, end to end | **29.6 FPS** (camera max 30), 44 ms median capture-to-display latency |
| Hand tracking | 16–23 ms per frame while tracking |
| Rendering (strokes, UI, compositing) | ~4 ms per frame, constant however long a stroke gets |
| Pipelined throughput (`scripts/benchmark.py`) | 41 FPS |

What makes it fast and smooth:

- **Pipelining:** hand tracking runs on a worker thread while the main thread
  renders the previous frame; the camera is read on its own thread and only
  the newest frame is ever processed (no queue build-up = low latency).
- **Incremental ink:** spline segments that can no longer change are frozen
  into the canvas; only the last few are redrawn each frame. Undo/redo replays
  the same freeze points, so the result is bit-identical to what you saw.
- **Dirty-rect compositing** with premultiplied ink layers: two saturating
  uint8 OpenCV operations, only inside the ink's bounding box.
- **One Euro filter** on the fingertip (parameters grid-searched on simulated
  handwriting): resting jitter 2.0 → 0.64 px, lag ≈ 0.2 frame when writing fast.
  Slow strokes get extra speed-adaptive smoothing; fast strokes keep sharp corners.
- **Centripetal Catmull-Rom splines** pass exactly through every tracked point
  but fill the gaps between 30 Hz samples with smooth curves.
- **Gesture debouncing** with hysteresis and a pen "hold" while your pose is
  changing, so lifting the pen never leaves a tail.
- **Robust detection:** MediaPipe squeezes the whole 16:9 frame into a 192 px
  detector, so smaller hands can be missed. When that happens, square tiles of
  the frame (each with its own tracker) find and track the hand instead —
  measured 99.7 % vs ~50 % detection on a hard sample. The coarse first
  landmarks after any new detection are ignored, so re-acquiring the hand
  never draws a stray line.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest          # 170 tests, ~40 s
.\.venv\Scripts\python.exe scripts\benchmark.py
```

The suite covers the filter, splines, gesture classification (synthetic hands
at any rotation), the canvas (compositing math, undo/redo, bit-exact replay,
constant per-frame cost), the toolbar / dwell / color picker, full engine
interaction flows driven by scripted hands, **real MediaPipe inference on real
hand photos** (`tests/fixtures`, Google's MediaPipe samples), tile handovers,
and an end-to-end run of the real model over a synthetic video including the
headless app loop.

## Project layout

```text
airpad/
  app.py        window, worker-thread pipeline, keyboard/mouse
  engine.py     per-frame logic: gestures -> strokes -> UI -> rendering (no GUI deps)
  tracker.py    MediaPipe HandLandmarker + square-tile fallback tracking
  gestures.py   finger states (3-D, rotation invariant), gestures, debouncing
  filters.py    One Euro filter
  spline.py     Catmull-Rom + adaptive smoothing
  canvas.py     strokes, layers, incremental rendering, undo/redo, PNG export
  ui.py         toolbar, dwell buttons, color picker, cursor, HUD
  camera.py     low-latency threaded webcam reader, video-file reader
  config.py     settings and command line
  model.py      model download / validation
scripts/benchmark.py
tests/
```

## Credits

- Hand tracking: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
  by Google (Apache 2.0); the model bundle in `models/` is downloaded from Google's model storage.
- Test photos in `tests/fixtures/` are MediaPipe's public sample images.

## Troubleshooting

- **"could not open camera"**: close other apps using the webcam (Teams, Zoom,
  the Camera app), or try `--camera 1` or `--backend dshow`.
- **Hand not detected**: improve lighting, move a little closer, and make sure
  your whole hand (including the wrist) is in view.
- **Low FPS on battery**: Windows' power saver limits the CPU; plug in or pick
  the "Best performance" power mode.
