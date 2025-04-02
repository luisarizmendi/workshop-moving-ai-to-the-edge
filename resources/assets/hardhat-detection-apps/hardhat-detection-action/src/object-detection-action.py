import os
import time
import requests
import uuid
import hashlib

UUID_FILE = "./device_uuid"

def get_detections(endpoint: str) -> dict:
    """
    Fetch detections from the endpoint and return them as a dictionary.
    """
    try:
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        # Parse the JSON response
        detections = response.json()

        # Ensure the response is a dictionary
        if isinstance(detections, dict):
            return detections
        else:
            print(f"Unexpected response format: {detections}")
            return {}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching detections: {e}")
        return {}
    except ValueError as e:
        print(f"Error parsing JSON from {endpoint}: {e}")
        return {}


def send_alert(alert_endpoint: str, message: str, device_uuid: str) -> None:
    """
    Send an alert message with a device UUID to the specified endpoint.
    """
    try:
        payload = {"message": message, "device_uuid": device_uuid}
        response = requests.post(alert_endpoint, json=payload, timeout=5)
        response.raise_for_status()
        print(f"Sent alert: {message} for device {device_uuid}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending alert: {e}")


def get_device_uuid() -> str:
    """
    Retrieve or generate a persistent UUID for the device.
    The UUID is stored in a file for reuse across script executions.
    """
    # Check if the UUID file exists
    if os.path.exists(UUID_FILE):
        with open(UUID_FILE, "r") as f:
            return f.read().strip()

    # Generate a new UUID based on the MAC address
    # Step 1: Get the IP address of the default gateway
    default_gateway_ip = None
    for line in os.popen("ip route show"):
        if 'default' in line:
            default_gateway_ip = line.split()[2]  # Extract the gateway IP address
            break

    # Step 2: Find the interface associated with that default gateway IP
    interface = None
    if default_gateway_ip:
        for line in os.popen(f"ip route get {default_gateway_ip}"):
            # Extract the interface name from the route details
            if 'dev' in line:
                interface = line.split('dev')[1].split()[0]
                break

    # Step 3: Get the MAC address of that interface
    mac = None
    if interface:
        for line in os.popen(f"ip link show {interface}"):
            if 'link/ether' in line:
                mac_address = line.split()[1]
                # Skip invalid MAC addresses (e.g., 00:00:00:00:00:00)
                if mac_address != "00:00:00:00:00:00":
                    mac = mac_address
                    break

    if mac:
        # Create a UUID from the MAC address to ensure it's unique to the device
        print(f"MAC: {mac}")
        device_uuid = str(mac)
    else:
        # If MAC is not found, generate a random UUID
        device_uuid = str(uuid.uuid4())

    # Save the UUID to the file
    os.makedirs(os.path.dirname(UUID_FILE), exist_ok=True)
    with open(UUID_FILE, "w") as f:
        f.write(device_uuid)

    return device_uuid


def send_alive_signal(alive_endpoint: str) -> None:
    """
    Send an alive signal to the specified endpoint, including a device UUID.
    """
    device_uuid = get_device_uuid()  # Get the unique device UUID
    payload = {"status": "alive", "device_uuid": device_uuid}  # Add UUID to the payload
    try:
        response = requests.post(alive_endpoint, json=payload, timeout=5)
        response.raise_for_status()
        print(f"Sent alive signal with UUID: {device_uuid}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending alive signal: {e}")


def main():
    # Configuration from environment variables
    detections_endpoint = os.getenv("DETECTIONS_ENDPOINT", "http://localhost:5000/current_detections")
    alert_endpoint = os.getenv("ALERT_ENDPOINT", "http://localhost:5005/alert")
    alive_endpoint = os.getenv("ALIVE_ENDPOINT", "http://localhost:5005/alive")
    check_interval = float(os.getenv("CHECK_INTERVAL", 1))  # Seconds
    alert_duration = float(os.getenv("ALERT_DURATION", 5))  # Seconds
    reset_checks = int(os.getenv("RESET_CHECKS", 3))  # Number of checks
    alive_interval = float(os.getenv("ALIVE_INTERVAL", 5))  # Seconds

    last_alert_time = 0
    alert_active = False
    reset_count = 0
    last_alive_signal = time.time()

    device_uuid = get_device_uuid()

    monitored_classes = {
        "no_helmet": {"count_threshold": 0},
        #"hat": {"count_threshold": 0},
    }

    while True:
        detections = get_detections(detections_endpoint)

        class_detected = None
        for cls, config in monitored_classes.items():
            if detections.get(cls, {}).get("count", 0) > config["count_threshold"]:
                class_detected = cls
                break

        if class_detected:
            if not alert_active:
                if last_alert_time == 0:
                    last_alert_time = time.time()
                print(f"Detected: {class_detected}")

            if time.time() - last_alert_time >= alert_duration:
                if not alert_active:
                    send_alert(alert_endpoint, "ALERT_ON", device_uuid)
                    alert_active = True
                reset_count = 0
        else:
            if alert_active:
                reset_count += 1
                if reset_count >= reset_checks:
                    send_alert(alert_endpoint, "ALERT_OFF", device_uuid)
                    alert_active = False
                    reset_count = 0
            else:
                last_alert_time = time.time()

        # alive 
        if time.time() - last_alive_signal >= alive_interval:
            send_alive_signal(alive_endpoint)
            last_alive_signal = time.time()

        time.sleep(check_interval)


if __name__ == "__main__":
    main()
