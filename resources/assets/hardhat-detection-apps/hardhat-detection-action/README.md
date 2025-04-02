# Object detection action service

## Description

This Python script is designed to continuously monitor object detection endpoint provided by the inference server.


## Features

- Generate alerts based on configurable detection thresholds
- Send periodic alive signals to a central monitoring system
- Generate a unique device identifier for tracking

## Prerequisites

- Python 3.8+
- `requests` library
- Access to object detection endpoint


You can run `pip install` using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Environment Variables

The application supports the following environment variables for configuration:

| Variable             | Description                                     | Default Value                |
|----------------------|------------------------------------------------|------------------------------|
| `DETECTIONS_ENDPOINT`| Endpoint for fetching object detections         | `http://localhost:5000/current_detections` |
| `ALERT_ENDPOINT`     | Endpoint for sending alerts                     | `http://localhost:5005/alert` |
| `ALIVE_ENDPOINT`     | Endpoint for sending alive signals              | `http://localhost:5005/alive` |
| `CHECK_INTERVAL`     | Time between detection checks (seconds)         | `1`                          |
| `ALERT_DURATION`     | Time to trigger an alert (seconds)              | `5`                          |
| `RESET_CHECKS`       | Number of checks before resetting alert         | `3`                          |
| `ALIVE_INTERVAL`     | Interval between alive signals (seconds)        | `5`                          |

## Usage

### Running the Script

```bash
# Set environment variables (optional)
export DETECTIONS_ENDPOINT=http://your-detection-server:5000/current_detections
export ALERT_ENDPOINT=http://your-alert-server:5005/alert
export ALIVE_ENDPOINT=http://your-alive-server:5005/alive

# Run the script
python object_detection_action.py
```


You can run it containerized, but it's recommended to make the container use the host network, since the UUID will be based on the MAC address and because the container will need to call the object-detection-server endpoint:

```bash
podman run -d --network=host -e DETECTIONS_ENDPOINT=http://<inference server ip>:<port>/current_detections -e ALERT_ENDPOINT=http://<dashboard backend ip>:<port>/alert -e ALIVE_ENDPOINT=http://<dashboard backend ip>:<port>/alive <image name>
```
> **Note:**
> You can find an image in `quay.io/luisarizmendi/object-detection-action:x86`


### Application Configuration

The script currently monitors specific object classes:
- `no_helmet`: Triggers an alert when more than 3 instances are detected
- `hat`: Triggers an alert when more than 3 instances are detected (this class is commented in the code)
- Additional classes can be easily added in the `monitored_classes` dictionary

## Device Identification

The script generates a unique device identifier using:
1. MAC address of the default network interface
2. Fallback to a randomly generated UUID
3. Persistence across script executions by storing in `./device_uuid`

## Alert Mechanism

- Alerts are sent when the number of detected objects exceeds the threshold
- Supports `ALERT_ON` and `ALERT_OFF` messages
- Configurable alert duration and reset conditions

## Timer Mechanisms
The script uses several timer mechanisms to control its monitoring and alerting behavior:

1. check_interval

* Purpose: Controls the frequency of detection checks
* Default: 1 second
* Behavior: Determines how often the script polls the detection endpoint
* Configurable via: CHECK_INTERVAL environment variable

2. alert_duration

* Purpose: Defines the time threshold for triggering a sustained alert
* Default: 5 seconds
* Behavior:

Tracks how long a monitored condition (e.g., no helmet detected) persists
Only sends an "ALERT_ON" message after the condition continues for this duration


* Configurable via: ALERT_DURATION environment variable

3. reset_checks

* Purpose: Determines the number of checks without the alerting condition before resetting the alert
* Default: 3 checks
* Behavior:

Prevents rapid on/off alert switching
Requires multiple consecutive checks without the alerting condition to turn off the alert


* Configurable via: RESET_CHECKS environment variable

4. alive_interval

* Purpose: Controls the frequency of sending "alive" signals
* Default: 5 seconds
* Behavior:

Periodically sends a signal to indicate the monitoring script is operational
Helps track the health and connectivity of the monitoring system


* Configurable via: ALIVE_INTERVAL environment variable

## Considerations

- Requires network access to detection and alert endpoints
- MAC-based device UUID generation may not work in all network environments
- Alert thresholds and intervals are fully customizable
