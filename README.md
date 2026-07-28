# ha-plate-recognizer

Home Assistant add-on that reads an RTSP camera stream, uses **MediaPipe** to detect
whether a vehicle is in a chosen region of the frame, and if so reads its **license
plate** with a local ALPR (`fast-alpr`). Recognized plates are published to MQTT and
auto-exposed in Home Assistant via MQTT discovery.

Runs fully offline on a Raspberry Pi 5 (`aarch64`) — no cloud API or token required.

## Pipeline

1. **RTSP** — a background reader always keeps the newest frame (no latency drift).
2. **ROI crop** — restrict detection to the part of the frame where cars pass.
3. **Vehicle gate** — MediaPipe ObjectDetector (EfficientDet-Lite, COCO classes
   `car/truck/bus/motorcycle`). No vehicle → OCR is skipped, saving CPU.
4. **Plate OCR** — `fast-alpr` detects + reads the plate on the vehicle crop.
5. **Publish** — the plate string + attributes go to MQTT; a HA sensor is auto-created.

## Installation

1. Add this repository to Home Assistant → Settings → Add-ons → Add-on Store → ⋮ → Repositories.
2. Install **ha-plate-recognizer**. First start downloads the ALPR model weights (cached in `/data`).
3. Configure `rtsp_url`, MQTT broker details, and the ROI, then start the add-on.
4. A `sensor.<device_name>_plate` entity appears automatically (MQTT discovery).

## Configuration

| Option | Description | Default |
|---|---|---|
| `rtsp_url` | RTSP stream URL (also accepts a local video file path). | — |
| `mqtt_host` / `mqtt_port` | MQTT broker. | `1883` |
| `mqtt_username` / `mqtt_password` | MQTT credentials. | — |
| `mqtt_topic` | Plain topic the plate is also published to. | `plate_recognizer/plate` |
| `mqtt_enable_topic` | Publish `ON`/`OFF` here to pause/resume analysis. | `plate_recognizer_enable` |
| `mqtt_discovery` | Auto-create the HA sensor via MQTT discovery. | `true` |
| `device_name` | Device/sensor name + MQTT node id. | `ha-plate-recognizer` |
| `min_car_score` | Min vehicle detection score (0..1). | `0.4` |
| `detect_classes` | COCO classes counted as a vehicle. | `car,truck,bus,motorcycle` |
| `min_plate_confidence` | Min OCR confidence to publish (0..1). | `0.5` |
| `cooldown` | Seconds before the same plate is published again. | `30` |
| `analyze_interval` | Seconds between analysed frames. | `0.4` |
| `enhance_contrast` | CLAHE boost for backlit scenes. | `false` |
| `web_ui` / `web_port` | MJPEG debug preview. | `true` / `8099` |
| `roi_top/bottom/left/right` | ROI crop as frame fractions (0..1). | full frame |

## Snapshot on known-plate match

If you keep a set of plates in a Home Assistant `input_select` (a dropdown whose
options are registration strings), the addon can save a snapshot whenever it
reads one of them. When a read plate matches an option of `plates_select_entity`
(default `input_select.plates`), the annotated frame is written to
`snapshot_dir` (default `/share/plate_recognizer`, so `\\<host>\share` or the
HA media/Samba add-on can browse it). Filenames are `YYYYMMDD_HHMMSS_PLATE.jpg`;
saves are de-duplicated per plate by `cooldown`.

Options are read live from HA via the Supervisor API (`homeassistant_api: true`),
so no token is needed as an addon — add/remove plates in the dropdown and the
addon picks them up within ~30 s. For standalone runs, set `HA_URL` + `HA_TOKEN`
(a long-lived token). Matching ignores case, spaces and dashes. Set
`snapshot_on_match: false` to disable.

## Direction filter (ignore cars leaving)

To avoid reading the plate of a car that is *leaving*, the addon tracks the
vehicle bbox centroid across frames and only publishes / snapshots when the car
moves in the "entry" direction. `entry_direction` (down/up/left/right) is which
way an entering car moves in the frame — with the gate at the bottom and the
road at the top, an entering car moves **down**, so `entry_direction: down` and
a leaving car (moving up) is ignored. `motion_min_px` is the minimum centroid
displacement to decide a direction (below it → `unknown`, which is not treated
as entry). The preview shows `dir entry/exit/unknown`. Set `direction_filter:
false` to disable. Note: a single uploaded still has no motion, so it reads as
`unknown` — the filter needs a moving car across several frames.

## Debug preview

Open `http://<host>:8099` to see a live annotated view: the ROI boundary, vehicle
boxes with detection scores, the detected plate box + read text, a crop of the plate
fed to OCR, and status lines (FPS, analysis on/off, last plate, cooldown, diagnostics
`no_vehicle` / `vehicle_no_plate` / `ok`).

## MQTT

- Discovery config (retained): `homeassistant/sensor/<node_id>/plate/config`
- State (plate string): `<node_id>/plate/state`
- Attributes (JSON): `<node_id>/plate/attributes` — `confidence`, `vehicle_class`,
  `detect_score`, `timestamp`.

## Standalone / development

```bash
cp .env.example .env      # fill RTSP_URL, MQTT_*, ROI
pip install -r requirements.txt
python script.py          # --model efficientdet_lite0.tflite
```

## Testing without a camera

You don't need the RTSP stream (or an MQTT broker) to test the recognition
pipeline — feed it a still photo instead.

**A) Upload via the web UI.** Leave `rtsp_url` empty (mock mode) or keep the
stream running, open `http://<host>:8099`, pick an image and click *Analyze
image*. The pipeline runs on that single frame; the annotated result appears in
the preview and the JSON detection (`diag`, `vehicles`, `plates[]` with
per-plate `ocr_confidence`/`accepted`) is shown on the page. Uploads never
publish to MQTT and ignore the cooldown, so you can re-test the same plate.

```bash
# same thing from the shell:
curl -F image=@car.jpg http://localhost:8099/analyze
```

**B) One-shot CLI.** Analyze a local file, print JSON, and exit — no camera, no
MQTT:

```bash
python script.py --image car.jpg
```

The MQTT connection is best-effort: an unreachable broker logs a warning and the
addon keeps running, so image testing works fully offline.

## Notes

- Tune `analyze_interval` if the Pi can't keep up; the MediaPipe gate already avoids
  running OCR on empty frames.
- Model weights (`fast-alpr`) are cached under `/data/cache` and survive restarts.
