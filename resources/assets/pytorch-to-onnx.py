import argparse
from ultralytics import YOLO
import os
import onnx

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Export a YOLO model to ONNX format.")
parser.add_argument("model_path", type=str, help="Path to the YOLO model (.pt file)")
args = parser.parse_args()

# Load and export the model
model = YOLO(args.model_path)
model.export(format='onnx', imgsz=640, dynamic=True)

onnx_model = onnx.load(f"{os.path.dirname(args.model_path)}/best.onnx")
output_tensor = onnx_model.graph.output[0]
inference_outputdims = [
        d.dim_value if (d.dim_value > 0) else -1
        for d in output_tensor.type.tensor_type.shape.dim
    ]
print("Exported model output shape:", inference_outputdims)