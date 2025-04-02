import argparse
from ultralytics import YOLO

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Export a YOLO model to ONNX format.")
parser.add_argument("model_path", type=str, help="Path to the YOLO model (.pt file)")
args = parser.parse_args()

# Load and export the model
model = YOLO(args.model_path)
model.export(format="onnx")
