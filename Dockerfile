# Slim base -> smaller image, faster build + pull on the Pi.
FROM python:3.11-slim

# Runtime libs the wheels need on slim: libgl1/libglib2.0-0 for OpenCV,
# libgomp1 (OpenMP) for MediaPipe/ONNXRuntime/NumPy, ffmpeg for RTSP decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# MediaPipe object-detection model (COCO, includes car/truck/bus/motorcycle).
# Lite2 float32: more accurate than the int8 lite0 (better on head-on cars).
RUN wget -qO efficientdet_lite2.tflite \
    https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite

# Copy the current directory contents into the container at /app
COPY . /app

# Install packages specified in requirements.txt
RUN pip install --no-cache-dir --prefer-binary --timeout 300 -r requirements.txt

# Cache dir for fast-alpr ONNX weights (mapped to /data:rw by the addon).
ENV XDG_CACHE_HOME=/data/cache

# Run script.py when the container launches
CMD ["python", "script.py"]
