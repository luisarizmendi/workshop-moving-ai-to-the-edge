import React, { useState, useEffect } from 'react';
import { DeviceService } from '../services/api';
import { Link } from 'react-router-dom';
import {
    CheckCircleIcon,
    AlertCircleIcon,
    XCircleIcon,
    HelpCircleIcon  // Add the unknown icon
} from 'lucide-react';

const DeviceStatusIcon = ({ isActive }) => {
    if (!isActive) {
        return <XCircleIcon color="red" title="Device Offline" />;  // Device is offline
    }

    return <CheckCircleIcon color="green" title="Device Active" />;  // Device is active
};

const AlarmStatusIcon = ({ isActive, hasAlarm }) => {
    if (!isActive) {
        return <HelpCircleIcon color="gray" title="Unknown (Device Offline)" />;  // Device offline, so alarm status is unknown
    }

    if (hasAlarm) {
        return <AlertCircleIcon color="red" title="Alarm Active" />;  // Alarm active
    }

    return <CheckCircleIcon color="green" title="No Alarm" />;  // No alarm
};

const DeviceList = () => {
    const [devices, setDevices] = useState([]);
    const [loading, setLoading] = useState(true);
    useEffect(() => {
        const fetchDevices = async () => {
            try {
                const deviceList = await DeviceService.getDevices();
                console.log(deviceList);
                setDevices(deviceList);
                setLoading(false);
            } catch (error) {
                console.error('Error fetching devices:', error);
                setLoading(false);
            }
        };

        // Reduce polling interval to detect device status faster
        const intervalId = setInterval(fetchDevices, 2000); // Refresh every 2 seconds

        // Initial fetch
        fetchDevices();

        return () => clearInterval(intervalId);
    }, []);

    if (loading) {
        return <div>Loading devices...</div>;
    }

    return (
        <div className="p-4">
        <h1 className="text-2xl font-bold mb-4">Devices</h1>
        <table className="w-full border-collapse">
        <thead>
        <tr className="bg-gray-200">
        <th className="p-2 text-left">Device</th>
        <th className="p-2 text-left">Alarm</th>
        <th className="p-2 text-left">Device Name</th>
        <th className="p-2 text-left">Last Alive</th>
        </tr>
        </thead>
        <tbody>
        {devices.map(device => (
            <tr key={device.uuid} className="border-b">
            <td className="p-2">
            <DeviceStatusIcon isActive={device.is_active} />
            </td>
            <td className="p-2">
            <AlarmStatusIcon
            isActive={device.is_active}
            hasAlarm={device.current_alarm_status}
            />
            </td>
            <td className="p-2">
            <Link
            to={`/device/${device.uuid}`}
            className="text-blue-600 hover:underline"
            >
            {device.name}
            </Link>
            </td>
            <td className="p-2">
            {device.last_alive_time
                ? new Date(device.last_alive_time).toLocaleString()
                : 'Never'}
                </td>
                </tr>
        ))}
        </tbody>
        </table>
        </div>
    );
};

export default DeviceList;
