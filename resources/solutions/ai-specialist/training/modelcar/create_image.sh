#!/bin/bash

podman manifest create quay.io/luisarizmendi/modelcar-hardhat:v1
podman build --platform linux/amd64 -t quay.io/luisarizmendi/modelcar-hardhat:v1-x86 .
podman build --platform linux/arm64 -t quay.io/luisarizmendi/modelcar-hardhat:v1-arm .
podman manifest add quay.io/luisarizmendi/modelcar-hardhat:v1 quay.io/luisarizmendi/modelcar-hardhat:v1-x86
podman manifest add quay.io/luisarizmendi/modelcar-hardhat:v1 quay.io/luisarizmendi/modelcar-hardhat:v1-arm
podman manifest push quay.io/luisarizmendi/modelcar-hardhat:v1
podman tag quay.io/luisarizmendi/modelcar-hardhat:v1 quay.io/luisarizmendi/modelcar-hardhat:prod
podman manifest push quay.io/luisarizmendi/modelcar-hardhat:prod

#
# Change model onnx and the associated config (depending on number of classes)
#
#podman manifest create quay.io/luisarizmendi/modelcar-hardhat:v2
#podman build --platform linux/amd64 -t quay.io/luisarizmendi/modelcar-hardhat:v2-x86 .
#podman build --platform linux/arm64 -t quay.io/luisarizmendi/modelcar-hardhat:v2-arm .
#podman manifest add quay.io/luisarizmendi/modelcar-hardhat:v2 quay.io/luisarizmendi/modelcar-hardhat:v2-x86
#podman manifest add quay.io/luisarizmendi/modelcar-hardhat:v2 quay.io/luisarizmendi/modelcar-hardhat:v2-arm
#podman manifest push quay.io/luisarizmendi/modelcar-hardhat:v2
#podman tag quay.io/luisarizmendi/modelcar-hardhat:v2 quay.io/luisarizmendi/modelcar-hardhat:prod
#podman manifest push quay.io/luisarizmendi/modelcar-hardhat:prod




