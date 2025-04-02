from database import SessionLocal
from models import Device, AlarmLog
from datetime import datetime

db = SessionLocal()
new_device = Device(uuid='test-uuid', name='Test Device')
db.add(new_device)
db.commit()

# Verify
devices = db.query(Device).all()
for device in devices:
    print(device.uuid, device.name)
