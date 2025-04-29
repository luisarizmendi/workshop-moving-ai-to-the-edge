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
import torch  
from tritonclient.utils import triton_to_np_dtype
import grpc
import tritonclient.grpc as grpc_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# gRPC settings for Triton server
TRITON_SERVER_URL = os.getenv('TRITON_SERVER_URL', 'localhost:8001')
MODEL_NAME = os.getenv('MODEL_NAME', 'hardhat')
MODEL_VERSION = os.getenv('MODEL_VERSION', '')  # Empty string means use latest version

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

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
logger.info(f"Using device: {device}")

# Global variables for Triton client and model metadata
triton_client = None
model_metadata = None
model_config = None
input_name = None
output_names = []

def initialize_triton_client():
    global triton_client, model_metadata, model_config, input_name, output_names
    try:
        triton_client = grpc_client.InferenceServerClient(
            url=TRITON_SERVER_URL,
            verbose=False,
            ssl=False
        )
        logger.info(f"Connected to Triton server at {TRITON_SERVER_URL}")
        
        # Get model metadata to determine input/output names
        model_metadata = triton_client.get_model_metadata(MODEL_NAME, MODEL_VERSION)
        model_config = triton_client.get_model_config(MODEL_NAME, MODEL_VERSION)
        
        # Get input and output names from metadata
        if model_metadata.inputs:
            input_name = model_metadata.inputs[0].name
            logger.info(f"Using input name: {input_name}")
        else:
            input_name = "images"  # Default fallback
            logger.warning(f"Could not determine input name from metadata, using default '{input_name}'")
        
        if model_metadata.outputs:
            output_names = [output.name for output in model_metadata.outputs]
            logger.info(f"Available output names: {output_names}")
        else:
            # Default fallback - try common output names for detection models
            output_names = ["output0", "detections"]
            logger.warning(f"Could not determine output names from metadata, will try: {output_names}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Triton client or get model metadata: {str(e)}")
        triton_client = None
        return False

def get_color_for_class(class_name):
    if class_name not in class_colors:
        class_colors[class_name] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return class_colors[class_name]

def preprocess_image(image):
    """
    GPU-accelerated image preprocessing if available, otherwise CPU
    """
    if cuda_available:
        try:
            # Convert OpenCV BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to torch tensor and move to GPU
            img_tensor = torch.from_numpy(image_rgb).to(device).float()
            
            # Rearrange dimensions from HWC to CHW
            img_tensor = img_tensor.permute(2, 0, 1)
            
            # Resize using GPU
            img_resized = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),
                size=(640, 640),
                mode='nearest'
            ).squeeze(0)
            
            # Normalize
            img_normalized = img_resized / 255.0
            
            # Add batch dimension
            img_batched = img_normalized.unsqueeze(0)
            
            # Convert to NumPy array for gRPC client
            return img_batched.cpu().numpy()
        except Exception as e:
            logger.error(f"GPU preprocessing failed: {str(e)}. Falling back to CPU")
            return cpu_preprocess_image(image)
    else:
        return cpu_preprocess_image(image)

def cpu_preprocess_image(image):
    """Fallback CPU preprocessing"""
    img_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_NEAREST)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose((2, 0, 1))
    img_batched = np.expand_dims(img_transposed, axis=0)
    return img_batched

def call_inference_server_grpc(image):
    """Use gRPC to call the Triton Inference Server"""
    global triton_client, input_name, output_names
    
    if triton_client is None:
        if not initialize_triton_client():
            return None
    
    try:
        # Preprocess the image
        input_data = preprocess_image(image)
        
        # Create the input tensor
        inputs = []
        inputs.append(grpc_client.InferInput(input_name, input_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(input_data)
        
        # Create the request outputs
        outputs = []
        for output_name in output_names:
            outputs.append(grpc_client.InferRequestedOutput(output_name))
        
        # Make inference request
        response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            model_version=MODEL_VERSION
        )
        
        # Try each output until we find valid data
        for output_name in output_names:
            try:
                output = response.as_numpy(output_name)
                logger.debug(f"Successfully retrieved output from '{output_name}' with shape {output.shape}")
                return output
            except Exception as e:
                logger.debug(f"Failed to get output '{output_name}': {str(e)}")
        
        logger.error(f"None of the expected output names {output_names} were valid")
        return None
        
    except grpc.RpcError as e:
        logger.error(f"gRPC inference server call failed: {e.details()}")
        # Check if connection failed and try to reconnect
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            logger.info("Server unavailable. Attempting to reconnect...")
            triton_client = None
            initialize_triton_client()
        elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
            logger.error("Invalid argument error. Check model input/output names.")
            # Try refreshing metadata
            initialize_triton_client()
        return None
    except Exception as e:
        logger.error(f"General error during inference: {str(e)}")
        return None

def postprocess_predictions(output, confidence_threshold=0.1):
    """
    GPU-accelerated post-processing of predictions if available, otherwise CPU
    """
    if cuda_available:
        try:
            # Convert to tensor and move to GPU
            output_tensor = torch.tensor(output, device=device)
            
            # Remove batch dimension if present
            if len(output_tensor.shape) > 2:
                output_tensor = output_tensor[0]
            
            # Transpose for detection processing
            transposed = output_tensor.transpose(0, 1)
            
            # Extract components
            boxes = transposed[:, :4]  # x_center, y_center, width, height
            confidences = transposed[:, 4:]
            
            # Get class IDs and confidence scores
            class_scores, class_ids = torch.max(confidences, dim=1)
            
            # Filter by confidence threshold
            mask = class_scores > confidence_threshold
            filtered_boxes = boxes[mask]
            filtered_class_ids = class_ids[mask]
            filtered_confidences = class_scores[mask]
            
            # Convert back to CPU for processing
            detections = []
            for i in range(len(filtered_boxes)):
                x_center, y_center, width, height = filtered_boxes[i].cpu().tolist()
                class_id = int(filtered_class_ids[i].item())
                confidence = float(filtered_confidences[i].item())
                x = x_center - width / 2
                y = y_center - height / 2
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class{class_id}"
                
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x, y, width, height]
                })
            
            return torch_non_max_suppression(detections)
        except Exception as e:
            logger.error(f"GPU postprocessing failed: {str(e)}. Falling back to CPU")
            return cpu_postprocess_predictions(output, confidence_threshold)
    else:
        return cpu_postprocess_predictions(output, confidence_threshold)

def cpu_postprocess_predictions(output, confidence_threshold=0.20):
    """Fallback CPU postprocessing"""
    # Handle different output shapes
    if len(output.shape) > 2:
        output = output[0]
    
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

def torch_non_max_suppression(detections, iou_threshold=0.5):
    """PyTorch-based non-maximum suppression"""
    if not detections:
        return []
    
    if cuda_available and len(detections) > 1:
        try:
            # Sort by confidence (descending)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Extract boxes and convert to tensor
            boxes = torch.tensor([[d['bbox'][0], d['bbox'][1], 
                                  d['bbox'][0] + d['bbox'][2], 
                                  d['bbox'][1] + d['bbox'][3]] for d in detections], 
                                device=device)
            
            scores = torch.tensor([d['confidence'] for d in detections], device=device)
            
            # Custom PyTorch NMS implementation
            keep = []
            indices = torch.argsort(scores, descending=True)
            
            while len(indices) > 0:
                # Keep the highest scoring box
                current = indices[0].item()
                keep.append(current)
                
                if len(indices) == 1:
                    break
                
                # Get remaining boxes
                indices = indices[1:]
                
                # Get IoU with current best box
                remaining_boxes = boxes[indices]
                current_box = boxes[current].unsqueeze(0)
                
                # Calculate IoU
                xx1 = torch.max(current_box[:, 0], remaining_boxes[:, 0])
                yy1 = torch.max(current_box[:, 1], remaining_boxes[:, 1])
                xx2 = torch.min(current_box[:, 2], remaining_boxes[:, 2])
                yy2 = torch.min(current_box[:, 3], remaining_boxes[:, 3])
                
                w = torch.clamp(xx2 - xx1, min=0)
                h = torch.clamp(yy2 - yy1, min=0)
                
                intersection = w * h
                box1_area = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
                box2_area = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
                union = box1_area + box2_area - intersection
                
                iou = intersection / union
                
                # Filter indices where IoU is less than threshold
                mask = iou <= iou_threshold
                indices = indices[mask]
            
            # Retrieve kept detections
            kept_detections = [detections[i] for i in keep]
            return kept_detections
            
        except Exception as e:
            logger.error(f"PyTorch NMS failed: {str(e)}. Falling back to CPU")
            return non_max_suppression(detections, iou_threshold)
    else:
        return non_max_suppression(detections, iou_threshold)

def non_max_suppression(detections, iou_threshold=0.5):
    """CPU fallback for non-maximum suppression"""
    if not detections:
        return []
        
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

    # Get inference results using gRPC
    output = call_inference_server_grpc(frame)
    
    if output is not None:
        try:
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
                "min_confidence": details["min_confidence"] if details["min_confidence"] != float('inf') else 0.0,
                "count": details["count"]
            }

    return frame

def initialize_camera():
    global cap, selected_camera_index
    if selected_camera_index is not None:
        return True

    camera_index = int(os.getenv('CAMERA_INDEX', '-1'))
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

@app.route('/model_info')
def model_info():
    """Endpoint to check model information for debugging purposes"""
    global model_metadata, input_name, output_names
    
    if model_metadata is None:
        initialize_triton_client()
    
    if model_metadata is None:
        return Response(json.dumps({"error": "Could not retrieve model metadata"}),
                       mimetype='application/json')
    
    # Extract relevant information for debugging
    inputs = []
    for input_info in model_metadata.inputs:
        inputs.append({
            "name": input_info.name,
            "datatype": input_info.datatype,
            "shape": input_info.shape
        })
    
    outputs = []
    for output_info in model_metadata.outputs:
        outputs.append({
            "name": output_info.name,
            "datatype": output_info.datatype,
            "shape": output_info.shape
        })
    
    info = {
        "model_name": model_metadata.name,
        "inputs": inputs,
        "outputs": outputs,
        "current_input_name": input_name,
        "current_output_names": output_names
    }
    
    return Response(json.dumps(info, indent=2), 
                   mimetype='application/json')

if __name__ == '__main__':
    # Initialize Triton gRPC client
    if not initialize_triton_client():
        logger.warning("Could not connect to Triton server initially. Will retry during inference.")
    
    if initialize_camera():
        capture_thread = threading.Thread(target=continuous_capture_thread, daemon=True)
        capture_thread.start()
        
        logger.info(f"Starting Flask server with class names: {CLASS_NAMES}")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        logger.error("Failed to initialize camera")

    stop_event.set()
    if cap:
        cap.release()