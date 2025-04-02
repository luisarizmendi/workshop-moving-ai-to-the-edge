# Object Detection Inference Server

## Description

This FastAPI application provides an object detection system using YOLO (You Only Look Once) for computer vision tasks. It offers a RESTful API for making predictions on images using the YOLO model.

## Features

- Object detection using YOLO
- RESTful API endpoints for predictions
- Base64 image processing
- Configurable detection thresholds
- Health check endpoint
- GPU/CPU support
- Model loading and status endpoints

## Prerequisites

- Python 3.9+
- FastAPI
- PyTorch
- Ultralytics YOLO
- OpenCV
- Uvicorn

You can run `pip install` using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Requirements

The following dependencies are required:

```
fastapi
torch
ultralytics
numpy
pydantic
uvicorn[standard]
opencv-python-headless
```

## Environment Variables

The application supports the following environment variables for configuration:

| Variable               | Description                                     | Default Value                |
|-----------------------|------------------------------------------------|------------------------------|
| `YOLO_MODEL_PATH`     | Directory path for YOLO model                  | `./models`                   |
| `YOLO_MODEL_FILE`     | Filename of the YOLO model                     | `model.pt`                   |
| `CONFIDENCE_THRESHOLD`| Global confidence threshold for object detection| `0.25`                       |

## API Endpoints

### 1. `/v1/models/{model_name}/infer` (POST)
- **Purpose**: Make predictions on an image
- **Request Body**: JSON with base64 encoded image and optional confidence threshold
- **Returns**: JSON with detections, inference time, and metadata

### 2. `/v1/models/{model_name}` (GET)
- **Purpose**: Get model status information
- **Returns**: JSON with model name, ready status, load time, and device

### 3. `/v1/models/{model_name}/load` (POST)
- **Purpose**: Load a model
- **Parameters**: model_name and model_path
- **Returns**: Success/failure message

### 4. `/healthz` (GET)
- **Purpose**: Health check endpoint
- **Returns**: System health status, GPU availability, and model status

## Usage Examples

### Running the Application

```bash
# Set environment variables (optional)
export CONFIDENCE_THRESHOLD=0.3

# Run the application
uvicorn object-detection-inference-server:app --host 0.0.0.0 --port 8080
```

### Container Deployment

The application can be containerized using the provided Containerfile. The base image uses Red Hat's UBI9 Python 3.9 image.

```bash
# Build the container
podman build -t object-detection-server .

# Run the container
podman run -d -p 8080:8080 object-detection-server
```

### GPU Support

For GPU support, ensure you have the NVIDIA Container Toolkit installed and properly configured. Run the container with GPU access:

```bash
podman run -d -p 8080:8080 --device nvidia.com/gpu=all --security-opt=label=disable object-detection-server
```

### Making Predictions

You can use Python to send prediction requests:

```python
import requests
import base64

# Encode image to base64
with open('image.jpg', 'rb') as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# Send prediction request
url = 'http://localhost:8080/v1/models/1/infer'
data = {
    'image': base64_image,
    'confidence_threshold': 0.25
}
response = requests.post(url, json=data)
predictions = response.json()
```

## Considerations

- The application automatically detects and uses GPU if available
- Models are loaded from the path specified in environment variables
- All images must be base64 encoded when sending to the API
- The API uses FastAPI's automatic OpenAPI documentation (available at `/docs`)
- The application runs on port 8080 by default
