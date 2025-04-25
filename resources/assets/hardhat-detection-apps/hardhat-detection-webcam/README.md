# Object Detection Camera Stream Manager

## Description

This Flask application provides a streaming interface for object detection, working in conjunction with an inference server. It handles camera streaming, frame processing, and visualization of detection results.

## Versions


### Kserve API

This application is used to send the images to an inference server that supports the Kserve API, such as the NVIDIA Triton server. There are two versions of this application, click on the names to access the description:

* [gRPC](to_kserve_api/src/gRPC/README.md): It uses gRPC to communicate with the inference server. This is better for stream data processing since it includes less overhead.

* [REST](to_kserve_api/src/REST/README.md):: It uses REST API calls when communicating with the inference server.


## Custom API

The [Custom Inference Server](to_custom_api/README.md) implements a custom server that includes the API and the preprocessing.