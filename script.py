import argparse
import base64
import sys
import time
import json
import os
import re
import threading
import logging

# fast-alpr / open-image-models download their ONNX weights to a cache dir on
# first run. Point that cache at /data so it survives addon restarts (HA maps
# /data:rw). Must be set BEFORE importing fast_alpr.
os.environ.setdefault("XDG_CACHE_HOME", "/data/cache")
os.environ.setdefault("HF_HOME", "/data/cache/huggingface")

import cv2
import numpy as np
import mediapipe as mp
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
load_dotenv()  # load .env into os.environ if present (standalone runs)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from flask import Flask, Response, request, jsonify
except ImportError:
    Flask = None  # web UI simply disabled if flask isn't installed

try:
    from fast_alpr import ALPR
except ImportError:
    ALPR = None  # loaded lazily; run() errors clearly if still missing

json_file_path = '/data/options.json'
if os.path.exists(json_file_path):
    with open(json_file_path, 'r') as file:
        json_data = file.read()
else:
    # /data/options.json only exists when run as an HA addon via Supervisor.
    # Fall back to environment variables for standalone/docker runs.
    json_data = json.dumps({
        "rtsp_url": os.environ.get("RTSP_URL", ""),
        "mqtt_host": os.environ.get("MQTT_HOST", ""),
        "mqtt_port": int(os.environ.get("MQTT_PORT", "1883")),
        "mqtt_username": os.environ.get("MQTT_USERNAME", ""),
        "mqtt_password": os.environ.get("MQTT_PASSWORD", ""),
        "mqtt_topic": os.environ.get("MQTT_TOPIC", "plate_recognizer/plate"),
        "mqtt_enable_topic": os.environ.get("MQTT_ENABLE_TOPIC", "plate_recognizer_enable"),
        "mqtt_discovery": os.environ.get("MQTT_DISCOVERY", "true"),
        "device_name": os.environ.get("DEVICE_NAME", "ha-plate-recognizer"),
        "min_car_score": float(os.environ.get("MIN_CAR_SCORE", "0.4")),
        "detect_classes": os.environ.get("DETECT_CLASSES", "car,truck,bus,motorcycle"),
        "min_plate_confidence": float(os.environ.get("MIN_PLATE_CONFIDENCE", "0.5")),
        "alpr_on_no_vehicle": os.environ.get("ALPR_ON_NO_VEHICLE", "true"),
        "cooldown": float(os.environ.get("COOLDOWN", "30")),
        "analyze_interval": float(os.environ.get("ANALYZE_INTERVAL", "0.4")),
        "enhance_contrast": os.environ.get("ENHANCE_CONTRAST", "false"),
        "web_ui": os.environ.get("WEB_UI", "true"),
        "web_port": int(os.environ.get("WEB_PORT", "8099")),
        "roi_top": float(os.environ.get("ROI_TOP", "0.0")),
        "roi_bottom": float(os.environ.get("ROI_BOTTOM", "1.0")),
        "roi_left": float(os.environ.get("ROI_LEFT", "0.0")),
        "roi_right": float(os.environ.get("ROI_RIGHT", "1.0")),
    })


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Parse the JSON data
data = json.loads(json_data)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RTSP_URL = data.get("rtsp_url", "")
mqtt_broker_address = data.get("mqtt_host")
mqtt_port = int(data.get("mqtt_port", 1883))
mqtt_topic = data.get("mqtt_topic", "plate_recognizer/plate")
mqtt_username = data.get("mqtt_username")
mqtt_password = data.get("mqtt_password")
# Topic HA publishes ON/OFF to, to pause/resume analysis without stopping the addon.
mqtt_enable_topic = data.get("mqtt_enable_topic", "plate_recognizer_enable")
MQTT_DISCOVERY = str(data.get("mqtt_discovery", True)).lower() in ("true", "1", "yes", "on")
DEVICE_NAME = str(data.get("device_name", "ha-plate-recognizer"))

# node id used for MQTT discovery + state topics (slug of the device name).
NODE_ID = re.sub(r"[^a-z0-9_]+", "_", DEVICE_NAME.lower()).strip("_") or "ha_plate_recognizer"
DISCOVERY_TOPIC = "homeassistant/sensor/%s/plate/config" % NODE_ID
STATE_TOPIC = "%s/plate/state" % NODE_ID
ATTR_TOPIC = "%s/plate/attributes" % NODE_ID

# Optional region-of-interest crop, fractions of the frame (0..1). Restricts
# where vehicles are searched -> vehicle is larger after resize -> better
# detection + bigger plate crop for OCR. Default = full frame (no crop).
ROI_TOP = float(data.get("roi_top", 0.0))
ROI_BOTTOM = float(data.get("roi_bottom", 1.0))
ROI_LEFT = float(data.get("roi_left", 0.0))
ROI_RIGHT = float(data.get("roi_right", 1.0))
ROI_ENABLED = (ROI_TOP, ROI_BOTTOM, ROI_LEFT, ROI_RIGHT) != (0.0, 1.0, 0.0, 1.0)

# MediaPipe object-detection gate: minimum score for a vehicle detection, and
# which COCO classes count as a vehicle.
MIN_CAR_SCORE = float(data.get("min_car_score", 0.4))
DETECT_CLASSES = [c.strip() for c in str(data.get("detect_classes", "car,truck,bus,motorcycle")).split(",") if c.strip()]

# ALPR: minimum OCR confidence to accept a plate, and cooldown (seconds) before
# the same plate is published again.
MIN_PLATE_CONFIDENCE = float(data.get("min_plate_confidence", 0.5))
COOLDOWN = float(data.get("cooldown", 30))

# When the vehicle gate finds nothing, still run ALPR on the whole ROI. fast-alpr
# has its own plate detector, so this catches plates the object detector misses
# (e.g. a car filling the frame head-on). Costs extra CPU per frame.
ALPR_ON_NO_VEHICLE = str(data.get("alpr_on_no_vehicle", True)).lower() in ("true", "1", "yes", "on")

# Optional CLAHE contrast boost. Helps backlit / uneven-light scenes.
ENHANCE_CONTRAST = str(data.get("enhance_contrast", False)).lower() in ("true", "1", "yes", "on")

# Web UI: MJPEG debug preview + single-image test upload.
WEB_UI = str(data.get("web_ui", True)).lower() in ("true", "1", "yes", "on")
WEB_PORT = int(data.get("web_port", 8099))

ANALYZE_INTERVAL = float(data.get("analyze_interval", 0.4))


# ---------------------------------------------------------------------------
# Engines + shared analysis state (module globals so both the RTSP loop and the
# web upload handler use the same detector/ALPR and dedup state).
# ---------------------------------------------------------------------------
DETECTOR = None      # MediaPipe ObjectDetector (vehicle gate)
ALPR_ENGINE = None   # fast-alpr ALPR (plate detect + OCR)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if ENHANCE_CONTRAST else None
_plate_last_time = {}  # dedup: plate string -> last publish time
_state = {"last_plate": "-", "last_plate_ts": 0.0, "fps": 0.0, "prev_cycle_t": 0.0}


def build_engines(model):
    """Create the MediaPipe detector and the fast-alpr engine (once)."""
    global DETECTOR, ALPR_ENGINE
    if ALPR is None:
        raise RuntimeError("fast-alpr not installed. Add it to requirements.txt.")
    if DETECTOR is None:
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=MIN_CAR_SCORE,
            category_allowlist=DETECT_CLASSES,
        )
        DETECTOR = vision.ObjectDetector.create_from_options(options)
    if ALPR_ENGINE is None:
        logger.info("Loading fast-alpr models (first run downloads weights)...")
        ALPR_ENGINE = ALPR()  # library defaults: YOLO plate detector + ViT OCR
        logger.info("fast-alpr ready")


# ---------------------------------------------------------------------------
# Web preview (MJPEG) + single-image test upload.
# ---------------------------------------------------------------------------
_web_lock = threading.Lock()
_web_jpeg = None  # latest annotated JPEG bytes for the web UI


def overlay_status(frame_bgr, status_lines):
    """Bake status text into a copy of the frame and return it."""
    disp = frame_bgr.copy()
    y = 22
    for line in status_lines:
        # Black outline + green text so it's readable on any background.
        cv2.putText(disp, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 24
    return disp


def encode_jpeg(frame_bgr):
    """JPEG-encode a frame -> bytes, or None."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


def store_web_frame(frame_bgr):
    """Stash an already-annotated frame for the live MJPEG stream."""
    global _web_jpeg
    if not WEB_UI or Flask is None or frame_bgr is None:
        return
    jpg = encode_jpeg(frame_bgr)
    if jpg is not None:
        with _web_lock:
            _web_jpeg = jpg


_INDEX_HTML = """<!doctype html><title>Plate recognizer</title>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center">
<h3>ha-plate-recognizer</h3>
<form id="f" onsubmit="return up(event)">
  <input type="file" id="img" accept="image/*" required>
  <button type="submit">Analyze image</button>
  <button type="button" onclick="clearRes()">Back to live</button>
</form>
<div id="result" style="display:none">
  <h4>Analyzed image (stays until you upload again)</h4>
  <img id="rimg" style="max-width:100%;height:auto"><br>
  <pre id="out" style="color:#0f0;text-align:left;display:inline-block"></pre>
</div>
<div id="live"><h4>Live stream</h4>
  <img src="stream" style="max-width:100%;height:auto"></div>
<script>
function clearRes(){document.getElementById('result').style.display='none';
 document.getElementById('live').style.display='block';}
async function up(e){e.preventDefault();
 const fd=new FormData(); fd.append('image',document.getElementById('img').files[0]);
 document.getElementById('result').style.display='block';
 document.getElementById('live').style.display='none';
 document.getElementById('out').textContent='analyzing...';
 const r=await fetch('analyze',{method:'POST',body:fd});
 const j=await r.json();
 if(j.image){document.getElementById('rimg').src=j.image; delete j.image;}
 document.getElementById('out').textContent=JSON.stringify(j,null,2);
 return false;}
</script></body>"""


def start_web_server():
    """Start the MJPEG preview + test-upload server in a daemon thread."""
    if not WEB_UI or Flask is None:
        logger.info("Web UI disabled (web_ui=%s, flask=%s)", WEB_UI, Flask is not None)
        return
    app = Flask(__name__)

    @app.route("/")
    def _index():
        return _INDEX_HTML

    @app.route("/stream")
    def _stream():
        def gen():
            while True:
                with _web_lock:
                    frame = _web_jpeg
                if frame is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.1)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/analyze", methods=["POST"])
    def _analyze():
        # Test hook: analyze one uploaded image as if it were a camera frame.
        # Does NOT publish to MQTT and ignores the cooldown, so you can re-test
        # the same plate repeatedly. The annotated frame appears on /stream.
        f = request.files.get("image")
        if f is None:
            return jsonify({"error": "no 'image' file in form-data"}), 400
        buf = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "could not decode image"}), 400
        if DETECTOR is None or ALPR_ENGINE is None:
            return jsonify({"error": "engines not ready yet"}), 503
        result, annotated = analyze(img, publish=False, dedup=False, to_stream=False)
        jpg = encode_jpeg(annotated)
        if jpg is not None:
            result["image"] = "data:image/jpeg;base64," + base64.b64encode(jpg).decode()
        return jsonify(result)

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=WEB_PORT, threaded=True),
        daemon=True,
    ).start()
    logger.info("Web UI on http://<host>:%d", WEB_PORT)


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------
# When False, the loop skips heavy recognition (detect + OCR) but keeps the RTSP
# stream and MQTT connection alive. Defaults True so the addon still works if HA
# never publishes an enable state.
analysis_enabled = True
mqtt_connected = False


def publish_discovery(client):
    """Publish HA MQTT discovery config (retained) so a sensor auto-appears."""
    if not MQTT_DISCOVERY:
        return
    config = {
        "name": "Plate",
        "unique_id": "%s_plate" % NODE_ID,
        "state_topic": STATE_TOPIC,
        "json_attributes_topic": ATTR_TOPIC,
        "icon": "mdi:car",
        "device": {
            "identifiers": [NODE_ID],
            "name": DEVICE_NAME,
            "model": "ha-plate-recognizer",
            "manufacturer": "ha-plate-recognizer",
        },
    }
    client.publish(DISCOVERY_TOPIC, json.dumps(config), retain=True)
    logger.info("Published HA discovery config to %s", DISCOVERY_TOPIC)


def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        logger.info("Connected to MQTT Broker")
        client.subscribe(mqtt_enable_topic)
        logger.info("Subscribed to enable topic: %s", mqtt_enable_topic)
        publish_discovery(client)
    else:
        logger.info("Connection to MQTT Broker failed with code %s", rc)


def on_message(client, userdata, msg):
    global analysis_enabled
    if msg.topic != mqtt_enable_topic:
        return
    payload = msg.payload.decode(errors="ignore").strip().lower()
    analysis_enabled = payload in ("on", "1", "true", "enabled", "yes")
    logger.info("Analysis %s (via %s=%s)",
                "ENABLED" if analysis_enabled else "DISABLED",
                mqtt_enable_topic, payload)


client = mqtt.Client()
client.username_pw_set(username=mqtt_username, password=mqtt_password)
client.on_connect = on_connect
client.on_message = on_message
# Resilient connect: a missing/unreachable broker must not crash the addon
# (e.g. standalone image testing without MQTT). The client keeps retrying.
try:
    client.connect(mqtt_broker_address, mqtt_port, 60)
except Exception as e:
    logger.info("MQTT connect failed (%s); continuing without MQTT", e)
client.loop_start()


def publish_plate(plate, ocr_conf, vehicle_class, detect_score):
    """Publish the plate to the state topic (+ raw topic) with JSON attributes."""
    if not mqtt_connected:
        logger.info("PLATE %s (not published - MQTT not connected)", plate)
        return
    attrs = {
        "confidence": round(float(ocr_conf), 3),
        "vehicle_class": vehicle_class,
        "detect_score": round(float(detect_score), 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    client.publish(STATE_TOPIC, plate)
    client.publish(ATTR_TOPIC, json.dumps(attrs))
    if mqtt_topic and mqtt_topic != STATE_TOPIC:
        client.publish(mqtt_topic, plate)  # plain topic for manual wiring
    logger.info("PLATE %s (ocr %.2f, %s %.2f)", plate, ocr_conf, vehicle_class, detect_score)


# ---------------------------------------------------------------------------
# RTSP reader — unchanged from the original addon.
# ---------------------------------------------------------------------------
class FrameGrabber:
    """Background RTSP reader that always keeps only the NEWEST frame, so the
    analysis loop never falls behind a growing decode buffer. Reconnects if the
    stream drops."""

    def __init__(self, url):
        self.url = url
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.cap = cv2.VideoCapture(url)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        fails = 0
        while self.running:
            ok, f = self.cap.read() if self.cap is not None else (False, None)
            if ok:
                fails = 0
                with self.lock:
                    self.frame = f
            else:
                fails += 1
                logger.info("RTSP read failed (%d) - reconnecting", fails)
                try:
                    self.cap.release()
                except Exception:
                    pass
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(self.url)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ALPR result helpers — fast-alpr's result objects vary slightly by version, so
# pull fields defensively.
# ---------------------------------------------------------------------------
def _alpr_text_conf(res):
    """Return (plate_text, ocr_confidence) from a fast-alpr result, or (None, 0)."""
    ocr = getattr(res, "ocr", None)
    if ocr is None:
        return None, 0.0
    text = getattr(ocr, "text", None)
    conf = getattr(ocr, "confidence", 0.0)
    if not text:
        return None, 0.0
    # fast-alpr gives per-character confidences (a list); reduce to a mean.
    if isinstance(conf, (list, tuple)):
        conf = sum(conf) / len(conf) if conf else 0.0
    return text.strip().upper().replace(" ", ""), float(conf or 0.0)


def _alpr_plate_box(res):
    """Return (x1, y1, x2, y2) of the plate box in crop coords, or None."""
    det = getattr(res, "detection", None)
    box = getattr(det, "bounding_box", None) if det is not None else None
    if box is None:
        return None
    try:
        return int(box.x1), int(box.y1), int(box.x2), int(box.y2)
    except AttributeError:
        return None


# ---------------------------------------------------------------------------
# Core analysis: ROI + gate + OCR + draw. Shared by the RTSP loop and the web
# test-upload handler. Returns a JSON-serialisable result dict.
# ---------------------------------------------------------------------------
def analyze(image, publish=True, dedup=True, to_stream=True):
    """Run the full pipeline on one BGR frame.

    Returns (result_dict, annotated_bgr). The annotated frame is also pushed to
    the live MJPEG stream when to_stream=True (the RTSP loop); web uploads pass
    to_stream=False so a single analysed image can be shown on its own without
    the live stream overwriting it a moment later.
    """
    # ROI crop.
    if ROI_ENABLED:
        h, w = image.shape[:2]
        image = image[int(ROI_TOP * h):int(ROI_BOTTOM * h),
                      int(ROI_LEFT * w):int(ROI_RIGHT * w)]

    # Optional contrast boost on the L channel.
    if _clahe is not None:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _clahe.apply(l)
        image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    roi_h, roi_w = image.shape[:2]
    display = image.copy()
    ts = time.time()
    plates = []
    inset = [None]  # mutable holder for the first plate crop (for the preview)

    def ocr_region(x0, y0, region, vclass, vscore):
        """Run ALPR on a sub-image at (x0,y0); draw + collect any plates."""
        try:
            results = ALPR_ENGINE.predict(region)
        except Exception as e:
            logger.info("ALPR error: %s", e)
            return
        for res in results:
            plate, ocr_conf = _alpr_text_conf(res)
            if not plate:
                continue

            # Draw plate box (region coords -> ROI coords) + label.
            pbox = _alpr_plate_box(res)
            if pbox:
                px0, py0, px1, py1 = pbox
                cv2.rectangle(display, (x0 + px0, y0 + py0), (x0 + px1, y0 + py1),
                              (0, 200, 255), 2)
                if inset[0] is None and (py1 - py0) > 4:
                    inset[0] = region[max(0, py0):py1, max(0, px0):px1].copy()
            cv2.putText(display, "%s (%.2f)" % (plate, ocr_conf),
                        (x0, min(roi_h - 6, y0 + region.shape[0] + 22)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

            accepted = ocr_conf >= MIN_PLATE_CONFIDENCE
            published = False
            if accepted and publish and (not dedup or ts - _plate_last_time.get(plate, 0.0) >= COOLDOWN):
                publish_plate(plate, ocr_conf, vclass, vscore)
                _plate_last_time[plate] = ts
                _state["last_plate"] = plate
                _state["last_plate_ts"] = ts
                published = True

            plates.append({
                "plate": plate,
                "ocr_confidence": round(ocr_conf, 3),
                "vehicle_class": vclass,
                "detect_score": round(float(vscore), 3),
                "accepted": accepted,
                "published": published,
            })

    # --- Gate: detect vehicles ---
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    det_result = DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    vehicles = det_result.detections or []

    for det in vehicles:
        bb = det.bounding_box
        x0 = max(0, bb.origin_x)
        y0 = max(0, bb.origin_y)
        x1 = min(roi_w, bb.origin_x + bb.width)
        y1 = min(roi_h, bb.origin_y + bb.height)
        cat = det.categories[0]
        cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(display, "%s %.2f" % (cat.category_name, cat.score), (x0, max(14, y0 - 6)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        if x1 > x0 and y1 > y0:
            ocr_region(x0, y0, image[y0:y1, x0:x1], cat.category_name, cat.score)

    # Fallback: the gate found nothing but a plate may still be readable (car
    # filling the frame head-on). Run ALPR on the whole ROI.
    fallback = False
    if not vehicles and ALPR_ON_NO_VEHICLE:
        fallback = True
        ocr_region(0, 0, image, "frame", 0.0)

    if plates:
        diag = "ok_fallback" if fallback else "ok"
    elif vehicles:
        diag = "vehicle_no_plate"
    else:
        diag = "no_vehicle"

    # FPS for the overlay.
    _state["fps"] = 1.0 / max(1e-3, ts - _state["prev_cycle_t"])
    _state["prev_cycle_t"] = ts

    # Plate crop inset (top-right) so OCR input is visible.
    if inset[0] is not None and inset[0].size:
        ih = 60
        scale = ih / max(1, inset[0].shape[0])
        iw = min(max(1, int(inset[0].shape[1] * scale)), roi_w - 4)
        try:
            display[4:4 + ih, roi_w - iw - 4:roi_w - 4] = cv2.resize(inset[0], (iw, ih))
            cv2.rectangle(display, (roi_w - iw - 4, 4), (roi_w - 4, 4 + ih), (0, 200, 255), 1)
        except Exception:
            pass

    cd_left = max(0.0, COOLDOWN - (ts - _state["last_plate_ts"])) if _state["last_plate"] != "-" else 0.0
    display = overlay_status(display, [
        "FPS %.1f  analysis %s  diag %s" % (_state["fps"], "ON" if analysis_enabled else "OFF", diag),
        "vehicles %d  last plate %s  cooldown %.0fs" % (len(vehicles), _state["last_plate"], cd_left),
        "ROI x %.2f-%.2f y %.2f-%.2f" % (ROI_LEFT, ROI_RIGHT, ROI_TOP, ROI_BOTTOM),
    ])

    if to_stream:
        store_web_frame(display)

    return {"diag": diag, "vehicles": len(vehicles), "plates": plates}, display


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(model: str) -> None:
    build_engines(model)

    if not RTSP_URL:
        # Mock / test mode: no camera. Serve the web UI and wait for image
        # uploads on /analyze. Keeps the process alive.
        logger.info("No rtsp_url set -> mock mode. Upload an image at http://<host>:%d", WEB_PORT)
        while True:
            time.sleep(1.0)

    grabber = FrameGrabber(RTSP_URL)
    last_analysis = 0.0

    while True:
        if not analysis_enabled:
            time.sleep(0.05)
            continue

        now = time.time()
        if now - last_analysis < ANALYZE_INTERVAL:
            time.sleep(0.01)
            continue
        last_analysis = now

        image = grabber.read()
        if image is None:  # stream not up yet / reconnecting
            time.sleep(0.05)
            continue

        analyze(image, publish=True, dedup=True, to_stream=True)

    grabber.release()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path to the MediaPipe object-detection model (.tflite).',
        required=False,
        default='efficientdet_lite2.tflite')
    parser.add_argument(
        '--image',
        help='Analyze a single local image, print JSON, and exit (offline test).',
        required=False,
        default=None)
    args = parser.parse_args()

    if args.image:
        # Pure offline test: no camera, no MQTT publish.
        build_engines(args.model)
        img = cv2.imread(args.image)
        if img is None:
            logger.error("Could not read image: %s", args.image)
            sys.exit(1)
        result, _ = analyze(img, publish=False, dedup=False, to_stream=False)
        print(json.dumps(result, indent=2))
        return

    start_web_server()
    run(args.model)


if __name__ == '__main__':
    main()
