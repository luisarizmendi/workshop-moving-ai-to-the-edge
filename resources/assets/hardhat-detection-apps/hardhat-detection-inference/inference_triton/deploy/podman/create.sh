#!/bin/bash

podman volume create model-storage

echo "Creating pod with init container..."

podman pod create --name hardhat-inference-pod --publish 8888:8888

# init container 
podman create \
  --name model-downloader \
  --pod hardhat-inference-pod \
  --volume model-storage:/mnt/models:z \
  --init-ctr once \
  --entrypoint /bin/sh \
  quay.io/skopeo/stable \
  -c "mkdir -p /mnt/models && \
    skopeo copy --override-os linux docker://quay.io/luisarizmendi/modelcar-hardhat:v1 oci-archive:/tmp/modelcar.tar && \
    mkdir -p /tmp/image && \
    tar -xf /tmp/modelcar.tar -C /tmp/image && \
    for layer in /tmp/image/blobs/sha256/*; do \
      tar -tf \"\$layer\" | grep '^models/' && tar -xf \"\$layer\" -C /mnt/models --strip-components=1 || true; \
    done && \
    rm -rf /tmp/modelcar.tar /tmp/image"

# main container
podman create \
  --name inference-container \
  --pod hardhat-inference-pod \
  --volume model-storage:/mnt/models:z \
  nvcr.io/nvidia/tritonserver@sha256:eea017611e2231da3a06d1cf47b73efdfe4811a313001cb12f4efe13b1418134 \
  --model_name=hardhat \
  --port=8001 \
  --rest_port=8888 \
  --model_path=/mnt/models \
  --file_system_poll_wait_seconds=0 \
  --grpc_bind_address=0.0.0.0 \
  --rest_bind_address=0.0.0.0 \
  --target_device=AUTO \
  --metrics_enable

echo "Starting the pod..."
podman pod start hardhat-inference-pod

echo "Deployment complete. The service is available at: http://localhost:8888"

echo "To stop the deployment: podman pod stop hardhat-inference-pod && podman pod rm hardhat-inference-pod"