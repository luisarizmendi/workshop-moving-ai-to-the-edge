#!/bin/bash

# Create volume and pod
podman volume create model-storage

echo "Creating pod..."
podman pod create --name hardhat-inference-pod --publish 8000:8000

# Init container
podman create \
    --name model-downloader  \
    --pod hardhat-inference-pod \
    --volume model-storage:/mnt/models:z \
    --init-ctr=always \
    --restart no \
    quay.io/luisarizmendi/ocp-job:latest \
    /bin/bash -c "mkdir -p /mnt/models && \
        /usr/bin/skopeo copy --override-os linux docker://<INSERT IMAGE NAME AND TAG> oci-archive:/tmp/modelcar.tar && \
        mkdir -p /tmp/image && \
        tar -xf /tmp/modelcar.tar -C /tmp/image && \
        for layer in /tmp/image/blobs/sha256/*; do \
            tar -tf \"\$layer\" | grep '^models/' && tar -xf \"\$layer\" -C /mnt/models --strip-components=1 || true; \
        done && \
        rm -rf /tmp/modelcar.tar /tmp/image"

# Main container
podman create \
    --name inference-container \
    --pod hardhat-inference-pod \
    --volume model-storage:/mnt/models:z \
    --mount type=tmpfs,destination=/dev/shm \
    -e PORT=8000 \
    nvcr.io/nvidia/tritonserver:25.03-py3 \
    /bin/sh -c 'exec tritonserver "--model-repository=/mnt/models" "--allow-http=true" "--allow-sagemaker=false"'


echo "Starting the pod..."
podman pod start hardhat-inference-pod 




echo "Deployment complete. The service is available at: http://localhost:8000"

echo "To clean-up the deployment: podman pod stop hardhat-inference-pod && podman pod rm hardhat-inference-pod && podman volume rm model-storage"




