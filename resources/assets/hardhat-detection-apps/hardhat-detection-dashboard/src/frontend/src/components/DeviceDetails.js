import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { DeviceService } from '../services/api';
import {
    AlertCircleIcon,
    CheckCircleIcon
} from 'lucide-react';

const DeviceDetails = () => {
    const { uuid } = useParams();
    const navigate = useNavigate();
    const [device, setDevice] = useState(null);
    const [deviceName, setDeviceName] = useState('');
    const [isEditing, setIsEditing] = useState(false);
    const [loading, setLoading] = useState(true);

    // New state to track if there are new alarms
    const [hasNewAlarms, setHasNewAlarms] = useState(false);

    // Fetch device details function to be reused
    const fetchDeviceDetails = useCallback(async () => {
        try {
            const details = await DeviceService.getDeviceDetails(uuid);
            
            // Check if there are new alarms since last fetch
            if (device && device.alarm_logs.length !== details.alarm_logs.length) {
                setHasNewAlarms(true);
            }

            setDevice(details);
            setDeviceName(details.name);
            setLoading(false);
        } catch (error) {
            console.error('Error fetching device details:', error);
            setLoading(false);
        }
    }, [uuid, device]);

    // Initial fetch
    useEffect(() => {
        fetchDeviceDetails();
    }, [fetchDeviceDetails]);

    // Polling mechanism for live updates
    useEffect(() => {
        // Set up interval to poll for updates every 3 seconds
        const intervalId = setInterval(fetchDeviceDetails, 3000);

        // Cleanup interval on component unmount
        return () => clearInterval(intervalId);
    }, [fetchDeviceDetails]);

    const handleNameUpdate = async () => {
        try {
            await DeviceService.updateDeviceName(uuid, deviceName);
            const updatedDevice = await DeviceService.getDeviceDetails(uuid);
            setDevice(updatedDevice);
            setIsEditing(false);
        } catch (error) {
            console.error('Error updating device name:', error);
        }
    };

    const handleDelete = async () => {
        if (window.confirm('Are you sure you want to delete this device?')) {
            try {
                await DeviceService.deleteDevice(uuid);
                navigate('/');
            } catch (error) {
                console.error('Error deleting device:', error);
            }
        }
    };

    // Reset new alarms flag when user views them
    const handleViewAlarms = () => {
        setHasNewAlarms(false);
    };

    if (loading) {
        return <div>Loading device details...</div>;
    }

    if (!device) {
        return <div>Device not found</div>;
    }

    // Calculate current alarm status and duration
    const currentAlarmStatus = device.alarm_logs.length > 0
        ? device.alarm_logs[0].is_alarm_on
        : false;

    const latestAlarmLog = device.alarm_logs.find(log => log.is_alarm_on);
    const alarmStartTime = latestAlarmLog
        ? new Date(latestAlarmLog.timestamp)
        : null;

    // Format the device creation and last alive times
    const deviceCreatedTime = device.created_at ? new Date(device.created_at) : null;
    const lastAliveTime = device.last_alive_time ? new Date(device.last_alive_time) : null;

    return (
        <div className="p-4">
            {/* Return to Main Screen Button */}
            <div className="mb-4">
                <button
                    onClick={() => navigate('/')}
                    className="bg-gray-300 px-4 py-2 rounded"
                >
                    Return to Main Screen
                </button>
            </div>

            {/* Device Details Header */}
            <div className="flex justify-between items-center mb-4">
                <div className="flex items-center">
                    {isEditing ? (
                        <div className="flex items-center">
                            <input
                                type="text"
                                value={deviceName}
                                onChange={(e) => setDeviceName(e.target.value)}
                                className="border p-1 mr-2"
                            />
                            <button
                                onClick={handleNameUpdate}
                                className="bg-blue-500 text-white px-2 py-1 rounded mr-2"
                            >
                                Save
                            </button>
                            <button
                                onClick={() => {
                                    setDeviceName(device.name);
                                    setIsEditing(false);
                                }}
                                className="bg-gray-300 px-2 py-1 rounded"
                            >
                                Cancel
                            </button>
                        </div>
                    ) : (
                        <>
                            <h1 className="text-2xl font-bold mr-4">
                                {device.name || 'Unnamed Device'}
                            </h1>
                            <button
                                onClick={() => setIsEditing(true)}
                                className="bg-gray-300 px-2 py-1 rounded"
                            >
                                Edit
                            </button>
                        </>
                    )}
                </div>
                <button
                    onClick={handleDelete}
                    className="bg-red-500 text-white px-2 py-1 rounded"
                >
                    Delete
                </button>
            </div>

            {/* Alarm Status */}
            <div>
                <p>
                    {currentAlarmStatus ? (
                        <span className="text-red-500 flex items-center">
                            <AlertCircleIcon className="mr-2" /> Alarm active
                        </span>
                    ) : (
                        <span className="text-green-500 flex items-center">
                            <CheckCircleIcon className="mr-2" /> Alarm inactive
                        </span>
                    )}
                </p>
                <p>
                    -------------------------------------
                </p>
                {alarmStartTime && (
                    <p>
                        Last alarm: {alarmStartTime.toLocaleString()}
                    </p>
                )}
            </div>

            {/* Device Created and Last Alive Times */}
            <div className="mt-4">
                {deviceCreatedTime && (
                    <p>
                        Device Created At: {deviceCreatedTime.toLocaleString()}
                    </p>
                )}
                {lastAliveTime && (
                    <p>
                        Last Alive: {lastAliveTime.toLocaleString()}
                    </p>
                )}
            </div>

            <p>
                -------------------------------------
            </p>
            
            {/* Alarm Log List with New Alarms Indicator */}
            <div className="mt-4">
                <h2 className="text-xl font-bold flex items-center">
                    Alarm History 
                    {hasNewAlarms && (
                        <span className="ml-2 bg-red-500 text-white text-xs px-2 py-1 rounded">
                            New
                        </span>
                    )}
                </h2>
                <ul onClick={handleViewAlarms}>
                    {device.alarm_logs.map((log, index) => (
                        <li key={index} className="border-b py-2">
                            <p>{log.message}</p>
                            <p>
                                Status: {log.is_alarm_on ? 'Active' : 'Inactive'}
                            </p>
                            <p>
                                Time: {new Date(log.timestamp).toLocaleString()}
                            </p>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default DeviceDetails;