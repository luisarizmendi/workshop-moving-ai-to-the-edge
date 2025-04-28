#!/bin/bash

# Exit if any command fails
set -e

ACTION="$1"
IMAGE_NAME="$2"

# Constants
POD_NAME="hardhat-inference-pod-gpu"
VOLUME_NAME="model-storage-gpu"

usage() {
    echo "Usage:"
    echo "  $0 create <IMAGE_NAME_AND_TAG>    - Create pod, download model, set up Triton"
    echo "  $0 run                            - Start the existing pod"
    echo "  $0 stop                           - Stop pod"
    echo "  $0 delete                         - Stop and remove pod and volume"
    echo ""
    echo "Example:"
    echo "  $0 create quay.io/luisarizmendi/modelcar-hardhat:v1"
    exit 1
}

has_gpu_support() {
    podman run --rm --device nvidia.com/gpu=all alpine true &>/dev/null
}


case "$ACTION" in
    create)
        if [ -z "$IMAGE_NAME" ]; then
            echo "Error: Image name is required for 'create' action."
            usage
        fi

        podman volume create "$VOLUME_NAME"

        echo "Creating pod..."
        podman pod create --name "$POD_NAME" --publish 8000:8000 --publish 8001:8001 --share=net

        # Init container
        podman create \
            --name model-downloader-gpu  \
            --pod "$POD_NAME" \
            --volume "$VOLUME_NAME":/mnt/models:z \
            --init-ctr=always \
            --restart no \
            quay.io/luisarizmendi/ocp-job:latest \
            /bin/bash -c "mkdir -p /mnt/models && \
                /usr/bin/skopeo copy --override-os linux docker://$IMAGE_NAME oci-archive:/tmp/modelcar.tar && \
                mkdir -p /tmp/image && \
                tar -xf /tmp/modelcar.tar -C /tmp/image && \
                for layer in /tmp/image/blobs/sha256/*; do \
                    tar -tf \"\$layer\" | grep '^models/' && tar -xf \"\$layer\" -C /mnt/models --strip-components=1 || true; \
                done && \
                rm -rf /tmp/modelcar.tar /tmp/image"

        echo "Checking for GPU support..."
        if has_gpu_support; then
            echo "GPU detected. Launching with GPU options."
            GPU_FLAGS="--security-opt=label=disable --device nvidia.com/gpu=all"
        else
            echo "No GPU detected. Launching without GPU options."
            GPU_FLAGS=""
        fi

        # Main container
        podman create \
            --name inference-container-gpu \
            --pod "$POD_NAME" \
            --volume "$VOLUME_NAME":/mnt/models:z \
            --mount type=tmpfs,destination=/dev/shm \
            $GPU_FLAGS \
            -e PORT=8000 \
            -e PORT=8001 \
            nvcr.io/nvidia/tritonserver:25.03-py3 \
            /bin/sh -c 'exec tritonserver "--model-repository=/mnt/models" "--allow-http=true" "--allow-sagemaker=false"'

        podman pod start "$POD_NAME"
        echo "Pod created and started"
        ;;
    run)
        echo "Starting the pod..."
        podman pod start "$POD_NAME"
        echo "Service available at: http://localhost:8000 (REST) and http://localhost:8001 (gRPC)"
        ;;
    stop)
        echo "Stopping the pod..."
        podman pod stop "$POD_NAME"
        echo "Service stopped"
        ;;
    delete)
        echo "Stopping and removing pod and volume..."
        podman pod stop "$POD_NAME" || true
        podman pod rm "$POD_NAME" || true
        podman volume rm "$VOLUME_NAME" || true
        echo "Clean-up complete."
        ;;
    *)
        echo "Error: Unknown action '$ACTION'"
        usage
        ;;
esac
