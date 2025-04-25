# Object Detection Stream Manager

Object Detection Stream Manager is a Flask-based application that connects to a webcam, processes video frames through an inference server running an object detection model, and provides both a video stream with bounding boxes and a JSON API with detection information.

## Features

- Real-time object detection using HTTP inference server API
- GPU acceleration support when available
- Video streaming with detection overlays
- REST API for current detection information
- Automatic camera detection and configuration
- Support for various object detection models

## Requirements

- Python 3.9+
- Inference server (Triton, TensorFlow Serving, or similar) with REST API
- Camera or video source
- CUDA-compatible GPU (optional, for acceleration)

## Installation

### Building and using podman

Build the podman image:

```bash
podman build -t object-detection-stream-manager:rest -f Containerfile .
```

Run the container:

```bash
sudo podman run -it --rm -p 5000:5000 --device=privileged --net host -e INFERENCE_SERVER_URL=http://localhost:8000/v2/models/hardhat/infer -e CLASS_NAMES=hardhat,no_hardhat object-detection-stream-manager:rest
```

> **Note**: Run it with sudo to simplify the access to the video device

> **Note**: Add `--net host` to easily grant communication with other containers running in the same host

### Using podman and a pre-built image in Quay.io

Run the container:

```bash
sudo podman run -it --rm -p 5000:5000 --privileged --net host -e INFERENCE_SERVER_URL=http://localhost:8000/v2/models/hardhat/infer -e CLASS_NAMES=hardhat,no_hardhat quay.io/luisarizmendi/object-detection-stream-manager:rest
```

> **Note**: Run it with sudo to simplify the access to the video device

> **Note**: Add `--net host` to easily grant communication with other containers running in the same host

### Manual Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python object-detection-stream-manager.py
```

## Configuration

The application can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `INFERENCE_SERVER_URL` | URL of the inference server endpoint | `http://localhost:8000/v2/models/hardhat/infer` |
| `CLASS_NAMES` | Comma-separated list of class names | `class0,class1` |
| `CAMERA_INDEX` | Camera device index (set to -1 for auto-detection) | `-1` |

Example:

```bash
export INFERENCE_SERVER_URL=http://inference-server:8000/v2/models/yolov8/infer
export CLASS_NAMES=person,car,truck,dog,cat
python object-detection-stream-manager.py
```

## API Endpoints

### Video Stream

Access the processed video stream with detection overlays:

```
GET /video_stream
```

View in a browser at `http://localhost:5000/video_stream` or embed in an HTML page:

```html
<img src="http://localhost:5000/video_stream" />
```

### Current Detections

Get JSON information about current detections:

```
GET /current_detections
```

Example response:

```json
{
  "available_classes": ["hardhat", "no_hardhat"],
  "detections": {
    "hardhat": {
      "max_confidence": 0.95,
      "min_confidence": 0.85,
      "count": 2
    },
    "no_hardhat": {
      "max_confidence": 0.78,
      "min_confidence": 0.65,
      "count": 1
    }
  }
}
```

## Implementation Details

The application:

1. Connects to the specified or first available camera
2. Captures frames from the camera in a separate thread
3. Preprocesses each frame (resizing and normalization)
4. Sends the preprocessed frame to the inference server via HTTP
5. Processes the inference results to extract bounding boxes and class information
6. Draws bounding boxes on detected objects
7. Provides the video stream and detection information via API endpoints

GPU acceleration is used automatically when available for:
- Image preprocessing
- Detection postprocessing
- Non-maximum suppression (NMS)

The inference server should follow the standard HTTP inferencing protocol:
- POST request to the endpoint with JSON payload
- Input shape: [1, 3, 640, 640] (batch size, channels, height, width)
- Output shape: [1, X, Y] where X is typically the number of detections and Y contains box coordinates and class scores

## Inference Server Format

The application expects the inference server to return results in this format:

```json
{
  "outputs": [
    {
      "name": "output0",
      "shape": [1, 7, N],
      "datatype": "FP32",
      "data": [...]
    }
  ]
}
```

Where:
- N is the number of potential detections
- Each detection consists of [x_center, y_center, width, height, class1_score, class2_score, ...]

## Troubleshooting

- If the camera doesn't initialize, check the camera index or permissions
- If detections are not appearing, verify that the inference server is running and accessible
- Check the logs for connection errors or inference issues
- Ensure the model input/output format matches what the application expects

