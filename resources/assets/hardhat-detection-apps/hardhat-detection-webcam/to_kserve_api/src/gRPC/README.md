# Object Detection Stream Manager

Object Detection Stream Manager is a Flask-based application that connects to a webcam, processes video frames through a Triton Inference Server running an object detection model, and provides both a video stream with bounding boxes and a JSON API with detection information.

## Features

- Real-time object detection using Triton Inference Server
- GPU acceleration support when available
- Video streaming with detection overlays
- REST API for current detection information
- Automatic camera detection and configuration
- Support for various object detection models

## Requirements

- Python 3.9+
- Triton Inference Server running with your detection model
- Camera or video source
- CUDA-compatible GPU (optional, for acceleration)

## Installation

### Building and using podman

Build the podman image:

```bash
podman build -t object-detection-stream-manager:grpc -f Containerfile .
```

Run the container:

```bash
podman run -it --rm -p 5000:5000 --privileged --net host -e TRITON_SERVER_URL=localhost:8001 -e MODEL_NAME=hardhat -e CLASS_NAMES=hardhat,no_hardhat object-detection-stream-manager:grpc
```

> **Note**: Run it with sudo to simplify the access to the video device

> **Note**: Add `--net host` to easily grant communication with other containers running in the same host

### Using podman and a pre-built image in Quay.io

Run the container:

```bash
podman run -it --rm -p 5000:5000 --privileged --net host -e TRITON_SERVER_URL=localhost:8001 -e MODEL_NAME=hardhat -e CLASS_NAMES=hardhat,no_hardhat quay.io/luisarizmendi/object-detection-stream-manager:grpc
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
| `TRITON_SERVER_URL` | URL of the Triton Inference Server (including port) | `localhost:8001` |
| `MODEL_NAME` | Name of the model in Triton Server | `hardhat` |
| `MODEL_VERSION` | Version of the model (empty for latest) | `` |
| `CLASS_NAMES` | Comma-separated list of class names | `class0,class1` |
| `CAMERA_INDEX` | Camera device index (set to -1 for auto-detection) | `-1` |

Example:

```bash
export TRITON_SERVER_URL=triton-server:8001
export MODEL_NAME=yolov8
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

### Model Information

Get information about the loaded model:

```
GET /model_info
```

Example response:

```json
{
  "model_name": "hardhat",
  "inputs": [
    {
      "name": "images",
      "datatype": "FP32",
      "shape": [-1, 3, 640, 640]
    }
  ],
  "outputs": [
    {
      "name": "output0",
      "datatype": "FP32",
      "shape": [-1, 6, -1]
    }
  ],
  "current_input_name": "images",
  "current_output_names": ["output0"]
}
```

## Implementation Details

The application:

1. Connects to the specified or first available camera
2. Initializes a connection to the Triton Inference Server
3. Captures frames from the camera in a separate thread
4. Processes frames through the detection model
5. Draws bounding boxes on detected objects
6. Provides the video stream and detection information via API endpoints

GPU acceleration is used automatically when available for:
- Image preprocessing
- Model inference (via Triton)
- Detection postprocessing
- Non-maximum suppression (NMS)

## Troubleshooting

- If the camera doesn't initialize, check the camera index or permissions
- If detections are not appearing, verify that the Triton server is running and accessible
- For model issues, check the model metadata using the `/model_info` endpoint
- Enable more verbose logging by setting the logging level to DEBUG in the code

