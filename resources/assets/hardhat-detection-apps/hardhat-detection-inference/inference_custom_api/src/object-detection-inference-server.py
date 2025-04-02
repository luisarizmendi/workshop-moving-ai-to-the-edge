import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from typing import List, Optional, Dict
from datetime import datetime

app = FastAPI(title="YOLO Object Detection Inference Server")

class ModelService:
    def __init__(self):
        self.model = None
        self.name = None
        self.load_time = None
        self.ready = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load(self, model_path: str, model_name: str) -> bool:
        """Load YOLO model from path"""
        try:
            self.model = YOLO(model_path)
            if self.device == 'cuda':
                self.model.to('cuda')
            self.name = model_name
            self.load_time = datetime.now()
            self.ready = True
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict(self, image: np.ndarray, confidence_threshold: float = 0.25) -> Dict:
        """Make prediction using loaded model"""
        if not self.ready:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            start_time = datetime.now()
            results = self.model(image, conf=confidence_threshold)[0]
            inference_time = (datetime.now() - start_time).total_seconds()
            
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2, conf, cls = box.data[0]
                class_name = results.names[int(cls)]
                detections.append({
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })
            
            return {
                "detections": detections,
                "inference_time": inference_time,
                "model_name": self.name,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Initialize model service
model_service = ModelService()

class PredictionRequest(BaseModel):
    image: str  # Base64 encoded image
    confidence_threshold: Optional[float] = None

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

class PredictionResponse(BaseModel):
    detections: List[Detection]
    inference_time: float
    model_name: str
    timestamp: str

def decode_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to image"""
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.post("/v1/models/{model_name}/infer", response_model=PredictionResponse)
async def predict(model_name: str, request: PredictionRequest):
    """Prediction endpoint"""
    if not model_service.ready or model_service.name != model_name:
        raise HTTPException(status_code=404, detail=f"Inferencing failed: Model {model_name} not found")
    
    # Use environment variable or default value for confidence_threshold
    confidence_threshold = request.confidence_threshold if request.confidence_threshold is not None else float(os.getenv('CONFIDENCE_THRESHOLD', 0.25))
    
    img = decode_image(request.image)
    return model_service.predict(img, confidence_threshold)

@app.get("/v1/models/{model_name}")
async def get_model_status(model_name: str):
    """Get model status"""
    if not model_service.ready or model_service.name != model_name:
        raise HTTPException(status_code=404, detail=f"Get model status failed: Model {model_name} not found")
    
    return {
        "name": model_service.name,
        "ready": model_service.ready,
        "load_time": model_service.load_time.isoformat() if model_service.load_time else None,
        "device": model_service.device
    }

@app.post("/v1/models/{model_name}/load")
async def load_model(model_name: str, model_path: str):
    """Load model endpoint"""
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model load error: File not found in {model_path}")
    
    success = model_service.load(model_path, model_name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to load model")
    
    return {"message": f"Model {model_name} loaded successfully"}

@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model_service.ready,
        "model_name": model_service.name if model_service.ready else None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    default_model_path = os.getenv('YOLO_MODEL_PATH', './models')
    default_model_file = os.getenv('YOLO_MODEL_FILE', 'model.pt')
       
    if os.path.exists(f"{default_model_path}/{default_model_file}"):
        model_service.load(f"{default_model_path}/{default_model_file}", "1")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
