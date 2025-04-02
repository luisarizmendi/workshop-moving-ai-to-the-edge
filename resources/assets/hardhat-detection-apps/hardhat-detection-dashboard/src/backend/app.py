import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from database import SessionLocal, engine
from models import Device, AlarmLog, Base
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.route('/alive', methods=['POST'])
def receive_alive_signal():
    db = next(get_db())
    data = request.json

    # Find or create device
    device = db.query(Device).filter_by(uuid=data['device_uuid']).first()
    if not device:
        # Device doesn't exist, create it and set the name to uuid if not set
        device = Device(uuid=data['device_uuid'], name=data['device_uuid'])  # Set name to uuid initially
        db.add(device)

    # Update last alive time
    device.last_alive_time = datetime.utcnow()
    device.is_active = True

    db.commit()
    return jsonify({"status": "success"}), 200

@app.route('/alert', methods=['POST'])
def receive_alert():
    db = next(get_db())

    # Attempt to parse incoming JSON data
    try:
        data = request.json
    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

    # Ensure device_uuid is not empty
    device_uuid = data.get('device_uuid')
    if not device_uuid:
        return jsonify({"error": "device_uuid is required"}), 400

    # Ensure message is present
    message = data.get('message')
    if not message:
        return jsonify({"error": "message is required"}), 400

    # Log the alert
    try:
        alert_log = AlarmLog(
            device_uuid=device_uuid,
            message=message,
            is_alarm_on=message == 'ALERT_ON'
        )
        db.add(alert_log)
        db.commit()
    except Exception as e:
        return jsonify({"error": "Failed to log alert"}), 500

    return jsonify({"status": "success"}), 200

@app.route('/devices', methods=['GET'])
def list_devices():
    db = next(get_db())

    # Configure timeout from environment variable
    device_timeout = int(os.getenv('DEVICE_TIMEOUT', 30))  # Default 30 seconds
    timeout_threshold = datetime.utcnow() - timedelta(seconds=device_timeout)

    devices = db.query(Device).all()
    device_list = []

    for device in devices:
        # Check device activity based on last alive time
        is_active = device.last_alive_time and device.last_alive_time > timeout_threshold

        # Get latest alarm status
        latest_alarm = db.query(AlarmLog)\
            .filter_by(device_uuid=device.uuid)\
            .order_by(AlarmLog.timestamp.desc())\
            .first()

        device_list.append({
            'uuid': device.uuid,
            'name': device.name or device.uuid,  # Use uuid as the name if not set
            'last_alive_time': device.last_alive_time.isoformat() if device.last_alive_time else None,
            'is_active': is_active,
            'current_alarm_status': latest_alarm.is_alarm_on if latest_alarm else False
        })

    return jsonify(device_list)

@app.route('/devices/<uuid>', methods=['GET'])
def get_device_details(uuid):
    db = next(get_db())

    device = db.query(Device).filter_by(uuid=uuid).first()
    if not device:
        return jsonify({"error": "Device not found"}), 404

    # Get alarm logs for this device
    alarm_logs = db.query(AlarmLog)\
        .filter_by(device_uuid=uuid)\
        .order_by(AlarmLog.timestamp.desc())\
        .limit(50)\
        .all()

    logs = [{
        'timestamp': log.timestamp.isoformat(),
        'message': log.message,
        'is_alarm_on': log.is_alarm_on
    } for log in alarm_logs]

    return jsonify({
        'uuid': device.uuid,
        'name': device.name or device.uuid,
        'last_alive_time': device.last_alive_time.isoformat() if device.last_alive_time else None,
        'alarm_logs': logs
    })

@app.route('/devices/<uuid>', methods=['PUT'])
def update_device(uuid):
    db = next(get_db())
    device = db.query(Device).filter_by(uuid=uuid).first()

    if not device:
        return jsonify({"error": "Device not found"}), 404

    data = request.json
    # Update the name if provided
    device.name = data.get('name', device.name)

    db.commit()
    return jsonify({
        'uuid': device.uuid,
        'name': device.name
    })

@app.route('/devices/<uuid>', methods=['DELETE'])
def delete_device(uuid):
    db = next(get_db())
    device = db.query(Device).filter_by(uuid=uuid).first()

    if not device:
        return jsonify({"error": "Device not found"}), 404

    db.delete(device)
    db.commit()

    return jsonify({"status": "Device deleted successfully"}), 200

# database
Base.metadata.create_all(bind=engine)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)
