# Object detection Dashboard Backend

## Description

This Flask-based enables tracking of device status, handling alive signals, managing device information, and logging alarm events.

## Features

- Device registration and tracking
- Alive signal handling
- Alarm logging
- Device management (list, details, update, delete)
- SQLite database integration
- CORS support

## Prerequisites

- Python 3.8+
- Flask
- SQLAlchemy
- Flask-CORS

You can run `pip install` using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Environment Variables

| Variable         | Description                           | Default Value |
|-----------------|---------------------------------------|---------------|
| `DEVICE_TIMEOUT`| Device inactivity timeout (seconds)   | `30`          |

## Endpoints

### Device Management

| Endpoint                | Method | Description                         |
|------------------------|--------|-------------------------------------|
| `/alive`               | POST   | Receive device alive signal         |
| `/alert`               | POST   | Log device alert                    |
| `/devices`             | GET    | List all devices                    |
| `/devices/<uuid>`      | GET    | Get device details                  |
| `/devices/<uuid>`      | PUT    | Update device information           |
| `/devices/<uuid>`      | DELETE | Remove device from system           |

### Alive Signal

**Payload**:
```json
{
  "device_uuid": "unique-device-identifier"
}
```

### Alert Logging

**Payload**:
```json
{
  "device_uuid": "unique-device-identifier",
  "message": "ALERT_ON/ALERT_OFF"
}
```

## Database

- Uses SQLite for persistent storage
- Automatically creates `device_monitoring.db`
- Supports device and alarm log tracking

## Running the Application

```bash 
python app.py
```

If you want to run it containerized:

```bash
podman run -d -p 5005:5005  <image name>
```
> **Note:**
> You can find an image in `quay.io/luisarizmendi/object-detection-dashboard-backend:v1`

## Access the Application

Go to `http://<ip>:3000`