#!/bin/bash
# Exit if any command fails
set -e

ACTION="$1"
IMAGE_NAME="$2"

# Constants
COMPOSE_FILE="triton-compose.yaml"
COMPOSE_PROJECT="hardhat-inference"

usage() {
    echo "Usage:"
    echo "  $0 create <IMAGE_NAME_AND_TAG> - Create and start services with model download"
    echo "  $0 run - Start the existing services"
    echo "  $0 stop - Stop services"
    echo "  $0 delete - Stop and remove services and volumes"
    echo "  $0 logs - Show service logs"
    echo "  $0 status - Show service status"
    echo ""
    echo "Example:"
    echo "  $0 create quay.io/luisarizmendi/modelcar-hardhat:v1"
    exit 1
}

has_gpu_support() {
    podman run --rm --device nvidia.com/gpu=all alpine true &>/dev/null
}

check_compose_file() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo "Error: $COMPOSE_FILE not found in current directory"
        echo "Make sure you have the triton-compose.yaml file in the current directory"
        exit 1
    fi
}

case "$ACTION" in
    create)
        if [ -z "$IMAGE_NAME" ]; then
            echo "Error: Image name is required for 'create' action."
            usage
        fi
        
        check_compose_file
        
        echo "Checking for GPU support..."
        if has_gpu_support; then
            echo "GPU detected. You may want to uncomment GPU configuration in triton-compose.yaml"
            echo "Edit the compose file and uncomment the devices and security_opt sections"
        else
            echo "No GPU detected. Running without GPU options."
        fi
        
        echo "Creating and starting services with image: $IMAGE_NAME"
        
        # Export the image name for the compose file
        export IMAGE_NAME="$IMAGE_NAME"
        
        # Start the services
        podman-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" up -d
        
        echo "Services created and started"
        echo "Waiting for model download to complete..."
        
        # Wait for model downloader to complete
        podman-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" logs -f model-downloader
        
        echo "Setup complete. Service available at:"
        echo "  REST API: http://localhost:8000"
        echo "  gRPC API: http://localhost:8001"
        ;;
        
    run)
        check_compose_file
        echo "Starting services..."
        podman-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" up -d inference-container
        echo "Service available at:"
        echo "  REST API: http://localhost:8000"
        echo "  gRPC API: http://localhost:8001"
        ;;
        
    stop)
        check_compose_file
        echo "Stopping services..."
        podman-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" stop
        echo "Services stopped"
        ;;
        
    delete)
        check_compose_file
        echo "Stopping and removing services and volumes..."
        podman-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" down -v
        echo "Clean-up complete."
        ;;
        
    logs)
        check_compose_file
        echo "Showing service logs..."
        podman-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" logs -f
        ;;
        
    status)
        check_compose_file
        echo "Service status:"
        podman-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" ps
        ;;
        
    *)
        echo "Error: Unknown action '$ACTION'"
        usage
        ;;
esac