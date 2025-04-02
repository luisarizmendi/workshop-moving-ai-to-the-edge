import gradio as gr
import requests
import os
import cv2
import numpy as np
from typing import List, Dict

# Get inference URL from environment variable if it exists
DEFAULT_INFERENCE_URL = os.getenv('INFERENCE_URL', '')

def generate_distinct_colors(n: int) -> List[tuple]:
    """Generate n visually distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8 + (i % 3) * 0.1  # Varying saturation
        value = 0.9 + (i % 2) * 0.1  # Varying value
        
        # Convert HSV to RGB
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        colors.append((b, g, r))  # BGR format for OpenCV
    
    return colors

def load_model(model_url):
    return model_url

def detect_objects(model_url, image_paths, class_names, confidence_threshold, merge_threshold):
    if not image_paths:
        return "No images uploaded.", []

    # Convert thresholds to float
    try:
        confidence_threshold = float(confidence_threshold)
        merge_threshold = float(merge_threshold)
    except ValueError:
        return "Invalid threshold value. Please enter a number between 0 and 1.", []

    if not 0 <= confidence_threshold <= 1 or not 0 <= merge_threshold <= 1:
        return "Threshold values must be between 0 and 1.", []

    # Parse class names into a list or generate default names
    if class_names.strip():
        class_names_list = [name.strip() for name in class_names.split(",")]
    else:
        # We'll determine the number of classes from the first detection
        class_names_list = []

    results_images = []
    for image_path in image_paths:
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            payload = {
                "inputs": [
                    {
                        "name": "images",
                        "shape": [1, 3, 640, 640],
                        "datatype": "FP32",
                        "data": preprocess_image(image_data)
                    }
                ]
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(model_url, json=payload, headers=headers)

            if response.status_code != 200:
                return f"Error processing file: {os.path.basename(image_path)}. Exception: {response.text}", []

            result = response.json()
            output = np.array(result['outputs'][0]['data']).reshape(result['outputs'][0]['shape'])
            
            # If class names weren't provided, generate them from the first detection
            if not class_names_list:
                num_classes = output.shape[1] - 4  # Subtract 4 for bbox coordinates
                class_names_list = [f"class{i}" for i in range(num_classes)]
            
            # Generate colors for all classes
            colors = generate_distinct_colors(len(class_names_list))
            
            processed_image = cv2.imread(image_path)
            detections, annotated_img = postprocess_predictions(
                output, processed_image, class_names_list, colors, confidence_threshold, merge_threshold
            )

            # Convert annotated image to RGB
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            results_images.append(annotated_img_rgb)

        except Exception as e:
            return f"Error processing file: {os.path.basename(image_path)}. Exception: {str(e)}", []

    return "Processing completed.", results_images

def preprocess_image(image_data):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (640, 640))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose((2, 0, 1))
    img_batched = np.expand_dims(img_transposed, axis=0)
    return img_batched.tolist()

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
                'class_id': class_id,
                'class_name': class_names[class_id] if class_id < len(class_names) else f'class{class_id}',
                'confidence': confidence,
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
        
        # Get color for this class
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
        gr.Text(label="Inference Endpoint URL", value=DEFAULT_INFERENCE_URL),
        gr.Files(file_types=["image"], label="Select Images"),
        gr.Textbox(label="Class Names (comma-separated)", placeholder="Leave empty to use default class names (class0, class1, etc.)"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Box Merge Threshold")
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Gallery(label="Results")
    ],
    title="Object Detection",
    description=(
        "Upload images to perform object detection using the provided API endpoint. "
        "Displays only one detection box per object. "
        "Adjust the Box Merge Threshold to control how aggressively boxes are merged - "
        "lower values (0.1-0.3) result in more aggressive merging with fewer final boxes, "
        "while higher values (0.4-0.6) preserve more distinct detections."
    )
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8800)