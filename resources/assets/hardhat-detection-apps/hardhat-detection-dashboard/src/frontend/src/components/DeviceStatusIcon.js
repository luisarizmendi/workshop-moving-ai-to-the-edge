import React from 'react';
import {
    CheckCircleIcon,
    AlertCircleIcon,
    XCircleIcon
} from 'lucide-react';

const DeviceStatusIcon = ({ isActive, hasAlarm }) => {
    if (!isActive) {
        return <XCircleIcon color="red" title="Device Offline" />;
    }

    if (hasAlarm) {
        return <AlertCircleIcon color="orange" title="Alarm Active" />;
    }

    return <CheckCircleIcon color="green" title="Device Active" />;
};

export default DeviceStatusIcon;
