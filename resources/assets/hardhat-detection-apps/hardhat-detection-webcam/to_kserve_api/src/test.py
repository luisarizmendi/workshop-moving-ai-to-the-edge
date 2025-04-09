from flask import Flask, Response
import cv2
import numpy as np
import json
from datetime import datetime
import os
import threading
import queue
import time
from collections import defaultdict
import random
import platform 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get class names from environment variable (e.g., "hardhat,no_hardhat")
CLASS_NAMES = os.getenv('CLASS_NAMES', 'class0,class1').split(',')
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
    if class_name not in class_colors:
        class_colors[class_name] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return class_colors[class_name]

def preprocess_image(image):
    img_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_NEAREST)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose((2, 0, 1))
    img_batched = np.expand_dims(img_transposed, axis=0)
    return img_batched

def process_frame(frame):
    global current_detections
    detections_this_frame = defaultdict(lambda: {
        "max_confidence": 0.0,
        "min_confidence": float('inf'),
        "last_seen": datetime.min,
        "count": 0
    })

    # Perform preprocessing (no actual inference)
    _ = preprocess_image(frame)
    
    # Skip the actual inference server call
    # Instead, simulate some minimal detection data for visualization
    # This will help determine if preprocessing/postprocessing is the bottleneck
    
    # Simple simulated detections
    class_id = 0
    confidence = 0.95
    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class{class_id}"
    
    # Drawing on frame (simulating postprocessing)
    h, w = frame.shape[:2]
    color = get_color_for_class(class_name)
    
    # Draw a simple rectangle in the middle of the frame
    x1, y1 = int(w * 0.25), int(h * 0.25)
    x2, y2 = int(w * 0.75), int(h * 0.75)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    label = f"{class_name}: {confidence:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Update detection data
    detections_this_frame[class_name]["max_confidence"] = confidence
    detections_this_frame[class_name]["min_confidence"] = confidence
    detections_this_frame[class_name]["last_seen"] = datetime.now()
    detections_this_frame[class_name]["count"] += 1
    
    with current_detections_lock:
        current_detections.clear()
        for class_name, details in detections_this_frame.items():
            current_detections[class_name] = {
                "max_confidence": details["max_confidence"],
                "min_confidence": details["min_confidence"],
                "count": details["count"]
            }

    return frame

def initialize_camera():
    global cap, selected_camera_index
    if selected_camera_index is not None:
        return True

    camera_index = int(os.getenv('CAMERA_INDEX', -1))
    system = platform.system()
    
    def configure_camera(cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    if camera_index != -1:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION if system == 'Darwin' else cv2.CAP_ANY)
        if cap.isOpened():
            configure_camera(cap)
            selected_camera_index = camera_index
            return True

    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION if system == 'Darwin' else cv2.CAP_ANY)
        if cap.isOpened():
            configure_camera(cap)
            selected_camera_index = i
            return True

    return False

def generate_frames():
    while True:
        try:
            frame = frame_queue.get(timeout=0.1)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            continue

@app.route('/video_stream')
def video_stream():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_detections')
def get_current_detections():
    with current_detections_lock:
        # Convert defaultdict to regular dict for JSON serialization
        detections_dict = {}
        for class_name, details in current_detections.items():
            # Use the actual class name from detection (already mapped in process_frame)
            detections_dict[class_name] = {
                "max_confidence": details["max_confidence"],
                "min_confidence": details["min_confidence"],
                "count": details["count"]
            }
        
        # Add class names information to the response
        response_data = {
            "available_classes": CLASS_NAMES,
            "detections": detections_dict
        }
    
    return Response(json.dumps(response_data), 
                   mimetype='application/json')

def continuous_capture_thread():
    global cap
    frame_count = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        if not cap or not cap.isOpened():
            logger.warning("Camera not initialized or opened")
            time.sleep(1)
            continue

        success, frame = cap.read()
        if not success:
            logger.warning("Failed to read frame from camera")
            time.sleep(0.01)
            continue

        # Apply all processing except actual inference
        processed_frame = process_frame(frame)

        # Update frame queue
        try:
            frame_queue.get_nowait()  # Clear existing frame
        except queue.Empty:
            pass
        
        frame_queue.put(processed_frame)
        
        # FPS calculation
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            logger.info(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

if __name__ == '__main__':
    if initialize_camera():
        capture_thread = threading.Thread(target=continuous_capture_thread, daemon=True)
        capture_thread.start()
        
        logger.info(f"Starting Flask server with processing but no inference")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        logger.error("Failed to initialize camera")

    stop_event.set()
    if cap:
        cap.release()