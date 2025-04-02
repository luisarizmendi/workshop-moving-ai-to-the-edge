from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from database import Base

class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String, unique=True, index=True)
    name = Column(String, default=None)
    last_alive_time = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

class AlarmLog(Base):
    __tablename__ = "alarm_logs"

    id = Column(Integer, primary_key=True, index=True)
    device_uuid = Column(String, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    message = Column(String)
    is_alarm_on = Column(Boolean)
