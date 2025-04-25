import gradio as gr
import os
import cv2
import numpy as np
from typing import List, Dict
from math import ceil
import logging
import torch
import tritonclient.grpc as grpc_client
import grpc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get inference URL from environment variable if it exists
TRITON_SERVER_URL = os.getenv('TRITON_SERVER_URL', 'localhost:8001')
MODEL_NAME = os.getenv('MODEL_NAME', 'hardhat')
MODEL_VERSION = os.getenv('MODEL_VERSION', '')  # Empty string means use latest version

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

# Define a fixed set of colors for different classes (BGR format)
DEFAULT_COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 0),      # Dark Blue
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Dark Red
    (128, 128, 0)     # Dark Cyan
]

def initialize_triton_client(model_url, model_name):
    global triton_client, model_metadata, model_config, input_name, output_names
    try:
        triton_client = grpc_client.InferenceServerClient(
            url=model_url,
            verbose=False
        )
        logger.info(f"Connected to Triton server at {model_url}")
        
        # Get model metadata to determine input/output names
        model_metadata = triton_client.get_model_metadata(model_name, MODEL_VERSION)
        model_config = triton_client.get_model_config(model_name, MODEL_VERSION)
        
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

def preprocess_image(image_data):
    """
    GPU-accelerated image preprocessing if available, otherwise CPU
    """
    # Read the image
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    if cuda_available:
        try:
            # Convert OpenCV BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to torch tensor and move to GPU
            img_tensor = torch.from_numpy(img_rgb).to(device).float()
            
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
            processed_img = img_batched.cpu().numpy()
            return processed_img, img
        except Exception as e:
            logger.error(f"GPU preprocessing failed: {str(e)}. Falling back to CPU")
            return cpu_preprocess_image(img), img
    else:
        return cpu_preprocess_image(img), img

def cpu_preprocess_image(img):
    """Fallback CPU preprocessing"""
    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose((2, 0, 1))
    img_batched = np.expand_dims(img_transposed, axis=0)
    return img_batched

def load_model(model_url):
    # In this version, we'll just return the URL as before, but we'll initialize
    # the gRPC client when needed
    logger.info(f"Model URL provided: {model_url}")
    return model_url

def call_inference_server_grpc(model_url, model_name, batch_input):
    """Use gRPC to call the Triton Inference Server with a batch of images"""
    global triton_client, input_name, output_names
    
    if triton_client is None:
        if not initialize_triton_client(model_url, model_name):
            return None
    
    try:
        batch_size = batch_input.shape[0]
        logger.info(f"Sending batch of {batch_size} images to inference server")
        
        # Create the input tensor
        inputs = []
        inputs.append(grpc_client.InferInput(input_name, batch_input.shape, "FP32"))
        inputs[0].set_data_from_numpy(batch_input)
        
        # Create the request outputs
        outputs = []
        for output_name in output_names:
            outputs.append(grpc_client.InferRequestedOutput(output_name))
        
        # Make inference request - USE THE PASSED MODEL_NAME
        response = triton_client.infer(
            model_name=model_name,  # FIX: Use the passed model_name instead of the global MODEL_NAME
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
            initialize_triton_client(model_url, model_name)
        elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
            logger.error("Invalid argument error. Check model input/output names.")
            # Try refreshing metadata
            initialize_triton_client(model_url, model_name)
        return None
    except Exception as e:
        logger.error(f"General error during inference: {str(e)}")
        return None

def detect_objects(model_url, model_name, image_paths, class_names, confidence_threshold, merge_threshold, batch_size):
    if not image_paths:
        return "No images uploaded.", []

    # Convert thresholds to float
    try:
        confidence_threshold = float(confidence_threshold)
        merge_threshold = float(merge_threshold)
        batch_size = int(batch_size)
    except ValueError:
        return "Invalid threshold or batch size value. Please enter valid numbers.", []

    if not 0 <= confidence_threshold <= 1 or not 0 <= merge_threshold <= 1:
        return "Threshold values must be between 0 and 1.", []
    
    if batch_size < 1:
        return "Batch size must be at least 1.", []

    # Parse class names into a list or generate default names
    if class_names.strip():
        class_names_list = [name.strip() for name in class_names.split(",")]
    else:
        # We'll determine the number of classes from the first detection
        class_names_list = []

    results_images = []
    processed_images = []
    original_images = []
    
    # Preprocess all images first
    for image_path in image_paths:
        try:
            # Read the image
            with open(image_path, "rb") as f:
                image_data = f.read()
                
            # Store original image for later annotation and preprocess
            processed_img, original_img = preprocess_image(image_data)
            original_images.append(original_img)
            processed_images.append(processed_img)
            
        except Exception as e:
            return f"Error preprocessing file: {os.path.basename(image_path)}. Exception: {str(e)}", []
    
    # Process in batches
    num_images = len(processed_images)
    num_batches = ceil(num_images / batch_size)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_images)
        
        # Create batch for this iteration
        current_batch = processed_images[start_idx:end_idx]
        current_batch_size = len(current_batch)
        
        # Stack all images into a single array
        if current_batch_size == 1:
            batch_array = current_batch[0]  # Just use the single image
        else:
            batch_array = np.vstack([img for img in current_batch])
            # Reshape to proper batch dimensions
            batch_array = batch_array.reshape(current_batch_size, 3, 640, 640)
        
        # Send batch to inference server
        try:
            outputs = call_inference_server_grpc(model_url, model_name, batch_array)
            
            if outputs is None:
                return f"Error processing batch {batch_idx+1}/{num_batches}. No valid output received.", []
            
            # If class names weren't provided, generate them from the first detection
            if not class_names_list and outputs.shape[0] > 0:
                num_classes = outputs.shape[2] - 4  # Subtract 4 for bbox coordinates
                class_names_list = [f"class{i}" for i in range(num_classes)]
            
            # Process each image in the batch
            for i in range(current_batch_size):
                img_idx = start_idx + i
                original_img = original_images[img_idx]
                
                # Get detections for this image
                detections, annotated_img = postprocess_predictions(
                    outputs[i:i+1], original_img, class_names_list, DEFAULT_COLORS, 
                    confidence_threshold, merge_threshold
                )

                # Convert annotated image to RGB
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                results_images.append(annotated_img_rgb)
                
        except Exception as e:
            return f"Error processing batch {batch_idx+1}/{num_batches}. Exception: {str(e)}", []

    return f"Processing completed. {num_images} images processed in {num_batches} batches.", results_images

def postprocess_predictions(output, original_img, class_names, colors, confidence_threshold, merge_threshold):
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

            detections.append({
                'class_id': int(class_id),  # Ensure class_id is an integer
                'class_name': class_names[class_id] if class_id < len(class_names) else f'class{class_id}',
                'confidence': float(confidence),
                'bbox': [x, y, width, height]
            })

    # Apply single-box-per-object filtering with user-defined merge threshold
    filtered_detections = strict_single_box_nms(detections, iou_threshold=merge_threshold)

    # Scale bounding boxes to original image dimensions
    h, w = original_img.shape[:2]
    scale_x, scale_y = 640/w, 640/h
    
    # Draw bounding boxes on the image
    for det in filtered_detections:
        x, y, width, height = det['bbox']
        x, width = x/scale_x, width/scale_x
        y, height = y/scale_y, height/scale_y
        
        # Get color for this class - use the class_id to index into the fixed color list
        color = colors[det['class_id'] % len(colors)]

        cv2.rectangle(
            original_img,
            (int(x), int(y)),
            (int(x+width), int(y+height)),
            color,
            2
        )

        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.putText(
            original_img,
            label,
            (int(x), int(y-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    return filtered_detections, original_img

def strict_single_box_nms(detections, iou_threshold=0.2):
    """
    Extremely strict NMS that ensures only one box per object regardless of class.
    This function prioritizes the highest confidence detection in any area.
    
    Args:
        detections: List of detection dictionaries with bbox and confidence
        iou_threshold: Threshold for determining when boxes should be merged
        
    Returns:
        List of filtered detections with only one box per object
    """
    if not detections:
        return []
    
    # Sort all detections by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Step 1: Perform a very aggressive global NMS across all classes
    # This ensures only the highest confidence detection remains in any area
    keep_indices = []
    final_detections = []
    
    for i, detection in enumerate(sorted_detections):
        # Skip if this detection was already marked for removal
        if i in keep_indices:
            continue
            
        # Keep this detection (highest confidence not yet processed)
        final_detections.append(detection)
        keep_indices.append(i)
        
        # Mark all other detections with significant overlap for removal
        for j in range(i+1, len(sorted_detections)):
            if j not in keep_indices:  # Only consider detections not yet processed
                if calculate_iou(detection['bbox'], sorted_detections[j]['bbox']) > iou_threshold:
                    keep_indices.append(j)  # Mark for removal (will be skipped in outer loop)
    
    # Step 2: Apply clustering to further combine nearby detections
    # This handles cases where boxes may have IoU just below the threshold
    final_clusters = []
    detection_clusters = [[det] for det in final_detections]
    
    # Merge clusters that have any detection with significant overlap
    merged = True
    while merged:
        merged = False
        for i in range(len(detection_clusters)):
            if not detection_clusters[i]:  # Skip empty clusters
                continue
                
            for j in range(i+1, len(detection_clusters)):
                if not detection_clusters[j]:  # Skip empty clusters
                    continue
                    
                # Check if any detection in cluster i overlaps with any in cluster j
                for det_i in detection_clusters[i]:
                    for det_j in detection_clusters[j]:
                        # Use a fraction of the main iou_threshold for the clustering step
                        # This makes clustering slightly more aggressive than the main NMS
                        cluster_threshold = max(0.1, iou_threshold / 2)
                        if calculate_iou(det_i['bbox'], det_j['bbox']) > cluster_threshold:
                            # Merge clusters
                            detection_clusters[i].extend(detection_clusters[j])
                            detection_clusters[j] = []  # Empty the merged cluster
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break
            if merged:
                break
    
    # Take only the highest confidence detection from each cluster
    for cluster in detection_clusters:
        if cluster:  # Skip empty clusters
            # Sort by confidence and take the highest
            best_detection = sorted(cluster, key=lambda x: x['confidence'], reverse=True)[0]
            final_clusters.append(best_detection)
    
    # Sort final detections by confidence for display purposes
    final_clusters.sort(key=lambda x: x['confidence'], reverse=True)
    
    return final_clusters

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

interface = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Text(label="Inference Endpoint URL", value=os.getenv('TRITON_SERVER_URL', 'localhost:8001')),
        gr.Text(label="Model name", value=os.getenv('MODEL_NAME', 'hardhat')),
        gr.Files(file_types=["image"], label="Select Images"),
        gr.Textbox(label="Class Names (comma-separated)", placeholder="Leave empty to use default class names (class0, class1, etc.)"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Box Merge Threshold"),
        gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch Size")
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Gallery(label="Results")
    ],
    title="Object Detection with Batch Processing (gRPC)",
    description=(
        "Upload images to perform batch object detection using the provided gRPC endpoint. "
        "Displays only one detection box per object. "
        "Adjust the Box Merge Threshold to control how aggressively boxes are merged - "
        "lower values (0.1-0.3) result in more aggressive merging with fewer final boxes, "
        "while higher values (0.4-0.6) preserve more distinct detections. "
        "The Batch Size controls how many images are processed in a single API call."
    )
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8800)