#!/bin/bash

podman manifest create quay.io/luisarizmendi/object-detection-stream-manager:rest

podman build --platform linux/amd64 -t quay.io/luisarizmendi/object-detection-stream-manager:rest-x86 .
podman tag quay.io/luisarizmendi/object-detection-stream-manager:rest-x86 quay.io/luisarizmendi/object-detection-stream-manager:x86

podman build --platform linux/arm64 -t quay.io/luisarizmendi/object-detection-stream-manager:rest-arm .
podman tag quay.io/luisarizmendi/object-detection-stream-manager:rest-arm quay.io/luisarizmendi/object-detection-stream-manager:arm

podman manifest add quay.io/luisarizmendi/object-detection-stream-manager:rest quay.io/luisarizmendi/object-detection-stream-manager:rest-x86
podman manifest add quay.io/luisarizmendi/object-detection-stream-manager:rest quay.io/luisarizmendi/object-detection-stream-manager:rest-arm
podman tag quay.io/luisarizmendi/object-detection-stream-manager:rest quay.io/luisarizmendi/object-detection-stream-manager:prod


podman push quay.io/luisarizmendi/object-detection-stream-manager:rest-x86
podman push quay.io/luisarizmendi/object-detection-stream-manager:rest-arm

podman push quay.io/luisarizmendi/object-detection-stream-manager:x86
podman push quay.io/luisarizmendi/object-detection-stream-manager:arm

podman manifest push quay.io/luisarizmendi/object-detection-stream-manager:rest
podman manifest push quay.io/luisarizmendi/object-detection-stream-manager:prod
