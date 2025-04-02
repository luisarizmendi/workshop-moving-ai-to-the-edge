#!/bin/bash
YOLO_MODEL_NAME=${YOLO_MODEL_NAME:-"1"}
YOLO_MODEL_PATH=${YOLO_MODEL_PATH:-"./models"}
YOLO_MODEL_FILE=${YOLO_MODEL_FILE:-"model.pt"}


while ! curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/healthz | grep -q "200"; do sleep 2; done && curl -X POST "http://localhost:8080/v1/models/${YOLO_MODEL_NAME}/load?model_path=${YOLO_MODEL_PATH}/${YOLO_MODEL_FILE}" &

uvicorn main:app --host 0.0.0.0 --port 8080 
