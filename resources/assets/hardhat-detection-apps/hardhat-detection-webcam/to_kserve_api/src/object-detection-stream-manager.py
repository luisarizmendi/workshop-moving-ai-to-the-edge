from flask import Flask, Response
import cv2
import numpy as np
import json
from datetime import datetime
import os
import requests
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

INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL', 'http://localhost:8888/v2/models/hardhat/infer')
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
    return img_batched.tolist()

def call_inference_server(image):
    try:
        payload = {
            "inputs": [
                {
                    "name": "images",
                    "shape": [1, 3, 640, 640],
                    "datatype": "FP32",
                    "data": preprocess_image(image)
                }
            ]
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            INFERENCE_SERVER_URL,
            json=payload,
            headers=headers,
            timeout=15
        )
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
        logger.error(f"Inference server call failed: {str(e)}")
        return None

def postprocess_predictions(output, confidence_threshold=0.1):
    output = output[0]  # Remove batch dimension: (1, 7, 8400) -> (7, 8400)
    detections = []
    
    for detection in output.T:
        class_probs = detection[4:]
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]
        
        if confidence > confidence_threshold:
            x_center, y_center, width, height = detection[:4]
            x = x_center - width / 2
            y = y_center - height / 2
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class{class_id}"
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [x, y, width, height]
            })
    
    return non_max_suppression(detections)

def non_max_suppression(detections, iou_threshold=0.5):
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    final_detections = []
    
    while detections:
        best = detections.pop(0)
        final_detections.append(best)
        detections = [
            det for det in detections
            if calculate_iou(best['bbox'], det['bbox']) < iou_threshold
        ]
    
    return final_detections

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def process_frame(frame):
    global current_detections
    detections_this_frame = defaultdict(lambda: {
        "max_confidence": 0.0,
        "min_confidence": float('inf'),
        "last_seen": datetime.min,
        "count": 0
    })

    inference_result = call_inference_server(frame)
    
    if inference_result and 'outputs' in inference_result:
        try:
            output_data = inference_result['outputs'][0]['data']
            output_shape = inference_result['outputs'][0]['shape']
            output = np.array(output_data).reshape(output_shape)
            
            detections = postprocess_predictions(output, confidence_threshold=0.1)
            
            h, w = frame.shape[:2]
            scale_x, scale_y = 640/w, 640/h

            for det in detections:
                x, y, width, height = det['bbox']
                class_id = det['class_id']
                confidence = det['confidence']
                class_name = det['class_name']
                
                x = x / scale_x
                y = y / scale_y
                width = width / scale_x
                height = height / scale_y
                
                color = get_color_for_class(class_name)
                
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + width), int(y + height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                detections_this_frame[class_name]["max_confidence"] = max(
                    detections_this_frame[class_name]["max_confidence"],
                    float(confidence)
                )
                detections_this_frame[class_name]["min_confidence"] = min(
                    detections_this_frame[class_name]["min_confidence"],
                    float(confidence)
                )
                detections_this_frame[class_name]["last_seen"] = datetime.now()
                detections_this_frame[class_name]["count"] += 1
                
                logger.info(f"Detection: {class_name}, confidence={confidence:.2f}")
        
        except Exception as e:
            logger.error(f"Error processing inference output: {str(e)}")

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

        processed_frame = process_frame(frame)

        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put(processed_frame)

if __name__ == '__main__':
    if initialize_camera():
        capture_thread = threading.Thread(target=continuous_capture_thread, daemon=True)
        capture_thread.start()
        
        def print_fps():
            frame_count = 0
            start_time = time.time()
            while not stop_event.is_set():
                time.sleep(1)
                if frame_count > 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.info(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
                frame_count += 1

        #threading.Thread(target=print_fps, daemon=True).start()
        
        logger.info(f"Starting Flask server with class names: {CLASS_NAMES}")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        logger.error("Failed to initialize camera")

    stop_event.set()
    if cap:
        cap.release()