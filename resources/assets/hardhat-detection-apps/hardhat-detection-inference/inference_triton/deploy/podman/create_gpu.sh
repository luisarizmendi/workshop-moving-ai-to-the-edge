#!/bin/bash

# Create volume and pod
podman volume create model-storage-gpu

echo "Creating pod..."
podman pod create --name hardhat-inference-pod-gpu --publish 8000:8000  --security-opt=label=disable --device nvidia.com/gpu=all --share=net

# Init container
podman create \
    --name model-downloader-gpu  \
    --pod hardhat-inference-pod-gpu \
    --volume model-storage-gpu:/mnt/models:z \
    --init-ctr=always \
    --restart no \
    quay.io/luisarizmendi/ocp-job:latest \
    /bin/bash -c "mkdir -p /mnt/models && \
        /usr/bin/skopeo copy --override-os linux docker://quay.io/luisarizmendi/modelcar-hardhat:v1 oci-archive:/tmp/modelcar.tar && \
        mkdir -p /tmp/image && \
        tar -xf /tmp/modelcar.tar -C /tmp/image && \
        for layer in /tmp/image/blobs/sha256/*; do \
            tar -tf \"\$layer\" | grep '^models/' && tar -xf \"\$layer\" -C /mnt/models --strip-components=1 || true; \
        done && \
        rm -rf /tmp/modelcar.tar /tmp/image"

# Main container
podman create \
    --name inference-container-gpu \
    --pod hardhat-inference-pod-gpu \
    --volume model-storage-gpu:/mnt/models:z \
    --mount type=tmpfs,destination=/dev/shm \
    --security-opt=label=disable --device nvidia.com/gpu=all  \
    -e PORT=8000 \
    nvcr.io/nvidia/tritonserver@sha256:eea017611e2231da3a06d1cf47b73efdfe4811a313001cb12f4efe13b1418134 \
    /bin/sh -c 'exec tritonserver "--model-repository=/mnt/models" "--allow-http=true" "--allow-sagemaker=false"'


echo "Starting the pod..."
podman pod start hardhat-inference-pod-gpu 




echo "Deployment complete. The service is available at: http://localhost:8000"

echo "To stop the deployment: podman pod stop hardhat-inference-pod-gpu && podman pod rm hardhat-inference-pod-gpu && podman volume rm model-storage-gpu"


