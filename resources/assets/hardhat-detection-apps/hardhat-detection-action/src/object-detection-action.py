import os
import time
import requests
import uuid
import json

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


def parse_monitored_classes(classes_env_var: str) -> dict:
    """
    Parse the monitored classes from an environment variable.
    Format is expected to be a JSON string or comma-separated list of class names.
    """
    if not classes_env_var:
        # Default to "no_helmet" if no classes are specified
        return {"no_helmet": {"count_threshold": 0}}
    
    try:
        # First try parsing as JSON
        classes = json.loads(classes_env_var)
        if isinstance(classes, dict):
            return classes
        elif isinstance(classes, list):
            return {cls: {"count_threshold": 0} for cls in classes}
    except json.JSONDecodeError:
        # If not valid JSON, try parsing as comma-separated list
        class_names = [cls.strip() for cls in classes_env_var.split(',')]
        return {cls: {"count_threshold": 0} for cls in class_names if cls}
    
    # Fallback to default
    return {"no_helmet": {"count_threshold": 0}}


def main():
    # Configuration from environment variables
    detections_endpoint = os.getenv("DETECTIONS_ENDPOINT", "http://localhost:5000/current_detections")
    alert_endpoint = os.getenv("ALERT_ENDPOINT", "http://localhost:5005/alert")
    alive_endpoint = os.getenv("ALIVE_ENDPOINT", "http://localhost:5005/alive")
    check_interval = float(os.getenv("CHECK_INTERVAL", 1))  # Seconds
    alert_duration = float(os.getenv("ALERT_DURATION", 5))  # Seconds
    reset_checks = int(os.getenv("RESET_CHECKS", 3))  # Number of checks
    alive_interval = float(os.getenv("ALIVE_INTERVAL", 5))  # Seconds
    
    # Parse monitored classes from environment variable
    monitored_classes_env = os.getenv("MONITORED_CLASSES", "no_helmet")
    monitored_classes = parse_monitored_classes(monitored_classes_env)
    
    print(f"Monitoring for classes: {list(monitored_classes.keys())}")

    last_alert_time = 0
    alert_active = False
    reset_count = 0
    last_alive_signal = time.time()

    device_uuid = get_device_uuid()

    while True:
        response_data = get_detections(detections_endpoint)
        
        # Extract the actual detections from the nested structure
        detections = response_data.get("detections", {})
        
        print(f"Current detections: {response_data}")

        class_detected = None
        for cls, config in monitored_classes.items():
            print(f"Checking class: {cls}, threshold: {config['count_threshold']}")
            
            # Check if the class exists in detections
            if cls in detections and "count" in detections[cls]:
                count = detections[cls]["count"]
                print(f"Found {cls} with count {count}")
                
                if count > config["count_threshold"]:
                    class_detected = cls
                    print(f"Threshold exceeded for {cls}")
                    break

        if class_detected:
            print(f"Alert condition detected for class: {class_detected}")
            if not alert_active:
                if last_alert_time == 0:
                    last_alert_time = time.time()
                print(f"Detected: {class_detected}, starting timer")

            if time.time() - last_alert_time >= alert_duration:
                if not alert_active:
                    print(f"Alert duration reached, sending ALERT_ON")
                    send_alert(alert_endpoint, "ALERT_ON", device_uuid)
                    alert_active = True
                reset_count = 0
        else:
            if alert_active:
                reset_count += 1
                print(f"No detection, reset count: {reset_count}/{reset_checks}")
                if reset_count >= reset_checks:
                    print(f"Reset threshold reached, sending ALERT_OFF")
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