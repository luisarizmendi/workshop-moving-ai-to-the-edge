#!/bin/bash

podman run -it --rm \
  --name object-detection-batch-kserve-api \
  -p 8800:8800 \
  -e INFERENCE_URL="http://localhost:8000/v2/models/hardhat/infer" \
  quay.io/luisarizmendi/object-detection-batch-kserve-api:latest