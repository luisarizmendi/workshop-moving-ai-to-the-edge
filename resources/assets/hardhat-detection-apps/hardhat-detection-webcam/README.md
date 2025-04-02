# Object Detection Camera Stream Manager

## Description

This Flask application provides a streaming interface and batch processing capabilities for object detection, working in conjunction with the YOLO Object Detection Inference Server. It handles camera streaming, frame processing, and visualization of detection results.

## Features

- Live webcam streaming with real-time object detection annotations
- Batch image processing support
- Real-time detection statistics
- Automatic camera detection and configuration
- Multipart streaming with annotations and detection data
- Color-coded visualization for different object classes

## Prerequisites

- Python 3.9+
- Flask
- OpenCV
- NumPy
- Requests

You can run `pip install` using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Requirements

The following dependencies are required:

```
Flask
opencv-python
numpy
requests
```

## Environment Variables

The application supports the following environment variables for configuration:

| Variable               | Description                                     | Default Value                |
|-----------------------|------------------------------------------------|------------------------------|
| `INFERENCE_SERVER_URL`| URL of the YOLO inference server               | `http://localhost:8080`      |
| `CAMERA_INDEX`        | Specific camera index to use                   | `-1` (auto-select)           |
| `MODEL_NAME`          | Model name to use for inference                | `1`                          |

## API Endpoints

### 1. `/video_stream`
- **Purpose**: Stream webcam with object detection annotations
- **Returns**: Multipart stream containing:
  - JPEG frames with annotations
  - JSON with current detections

### 2. `/current_detections` (GET)
- **Purpose**: Get current detection statistics
- **Returns**: JSON with object counts and confidence levels per class

### 3. `/detect_image` (POST)
- **Purpose**: Process uploaded images
- **Accepts**: Multipart form-data with image files
- **Returns**: JSON with:
  - Processed images (base64 encoded)
  - Detection results
  - File information

## Usage Examples

### Running the Application

```bash
# Set environment variables (optional)
export INFERENCE_SERVER_URL=http://localhost:8080
export CAMERA_INDEX=0

# Run the application
python object-detection-stream-manager.py
```

### Container Deployment

The application can be containerized using the provided Containerfile. The base image uses Red Hat's UBI9 Python 3.9 image.

```bash
# Build the container
podman build -t object-detection-stream-manager .

# Run the container
podman run -d -p 5000:5000 --device /dev/video0 \
  -e INFERENCE_SERVER_URL=http://localhost:8080 \
  --privileged object-detection-stream-manager
```

Note: The container requires privileged access to use the camera device.

### Processing Images

You can use Python to send images for processing:

```python
import requests

url = 'http://localhost:5000/detect_image'
files = [
    ('images', open('image1.jpg', 'rb')),
    ('images', open('image2.jpg', 'rb'))
]
response = requests.post(url, files=files)
```

Or use curl:

```bash
curl -X POST -F "images=@example.jpg" http://localhost:5000/detect_image > response.json
```

## System Requirements

The container image requires the following system packages for camera and display support:
- mesa-libGL
- mesa-dri-drivers
- libX11
- libXext
- gstreamer1-plugins-base

## Considerations

- The application requires connection to a running YOLO inference server
- Camera selection is automatic if not specified via `CAMERA_INDEX`
- The stream includes both visual annotations and JSON detection data
- Each object class is assigned a unique color for visualization
- Detection results include confidence scores and object counts
- The application maintains a queue of the most recent frame for efficient streaming
- Detection statistics are thread-safe and updated in real-time
