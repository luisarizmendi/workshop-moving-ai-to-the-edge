# app_server.py
from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import json
from datetime import datetime
import os
import base64
import requests
import threading
import queue
import time
from collections import defaultdict
import random
import platform 

app = Flask(__name__)

# Configuration
INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL', 'http://localhost:8080')
cap = None
selected_camera_index = None
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

current_detections_lock = threading.Lock()
current_detections = defaultdict(lambda: {
    "max_confidence": 0.0,
    "min_confidence": 0.0,
    "last_seen": datetime.min,
    "count": 0
})

class_colors = {}

def get_color_for_class(class_name):
    """
    Generate and return a unique color for the given class
    """
    if class_name not in class_colors:
        class_colors[class_name] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return class_colors[class_name]

def call_inference_server(image):
    try:
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        response = requests.post(
            f"{INFERENCE_SERVER_URL}/v1/models/{model_name}/infer",  
            json={
                "image": image_base64,
                "confidence_threshold": 0.25
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Inference server call failed: {str(e)}")
        return None

def process_frame(frame):
    """
    Process a single frame using inference server
    """
    global current_detections
    detections_this_frame = defaultdict(lambda: {
        "max_confidence": 0.0,
        "min_confidence": float('inf'),
        "last_seen": datetime.min,
        "count": 0
    })

    # Call inference server
    inference_result = call_inference_server(frame)
    
    if inference_result and 'detections' in inference_result:
        for detection in inference_result['detections']:
            class_name = detection['class_name']
            conf = detection['confidence']
            bbox = detection['bbox']
            
            color = get_color_for_class(class_name)
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Update detections
            detections_this_frame[class_name]["max_confidence"] = max(
                detections_this_frame[class_name]["max_confidence"],
                float(conf)
            )
            detections_this_frame[class_name]["min_confidence"] = min(
                detections_this_frame[class_name]["min_confidence"],
                float(conf)
            )
            detections_this_frame[class_name]["last_seen"] = datetime.now()
            detections_this_frame[class_name]["count"] += 1
            
            # Add label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    with current_detections_lock:
        current_detections = {
            class_name: {
                "max_confidence": details["max_confidence"],
                "min_confidence": details["min_confidence"],
                "count": details["count"]
            }
            for class_name, details in detections_this_frame.items()
        }

    return frame, current_detections

def initialize_camera():
    """
    Initialize the camera
    """
    global cap, selected_camera_index
    if selected_camera_index is not None:
        return True

    camera_index = int(os.getenv('CAMERA_INDEX', -1))
    system = platform.system()
    
    if camera_index != -1:
        if system == 'Darwin':
            cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            selected_camera_index = camera_index
            return True

    # Auto-detect camera
    for i in range(10):
        if system == 'Darwin':
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            selected_camera_index = i
            return True

    return False

@app.route('/detect_image', methods=['POST'])
def detect_batch():
    """
    Process a batch of images
    """
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    results = []
    files = request.files.getlist('images')

    for file in files:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        processed_image, detections = process_frame(image)

        _, buffer = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        results.append({
            "filename": file.filename,
            "processed_filename": f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            "detections": detections,
            "image_base64": img_base64
        })

    return jsonify({"results": results})

def generate_frames():
    """
    Generator function for streaming webcam
    """
    while True:
        if frame_queue.empty():
            time.sleep(0.1)
            continue

        frame = frame_queue.get()
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        with current_detections_lock:
            detections_json = json.dumps(current_detections)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
               b'Content-Type: application/json\r\n\r\n' + 
               detections_json.encode() + b'\r\n')


@app.route('/current_detections', methods=['GET'])
def current_detections_endpoint():
    """
    Return the current object counts with confidence scores
    """
    with current_detections_lock:
        formatted_detections = current_detections.copy()

    return jsonify(formatted_detections)


@app.route('/video_stream')
def video_stream():
    """
    Route for webcam streaming
    """
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def continuous_inference_thread():
    """
    Continuously capture frames and process them
    """
    global cap
    while not stop_event.is_set():
        if not cap or not cap.isOpened():
            time.sleep(1)
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        processed_frame, _ = process_frame(frame)

        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass

        frame_queue.put(processed_frame)

if __name__ == '__main__':
    
    import os
    model_name = os.getenv('MODEL_NAME', '1')
    
    # Attempt to load the model
    #response = requests.post(
    #    f"{INFERENCE_SERVER_URL}/v1/models/1/load",
    #    json={"model_path": "/path/to/model.pt"}  
    #)
    #if response.status_code != 200:
    #    print("Failed to load model:", response.json().get("detail"))
    #    exit(1) 

    if initialize_camera():
        threading.Thread(target=continuous_inference_thread, daemon=True).start()
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to initialize camera")
