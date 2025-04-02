import axios from 'axios';

const API_BASE_URL = window.env?.BACKEND_API_BASE_URL || 'http://localhost:5005';

export const DeviceService = {
    // Get all devices
    getDevices: async () => {
        const response = await axios.get(`${API_BASE_URL}/devices`);
        return response.data;
    },

    // Get device details
    getDeviceDetails: async (uuid) => {
        const response = await axios.get(`${API_BASE_URL}/devices/${uuid}`);
        return response.data;
    },

    // Update device name
    updateDeviceName: async (uuid, name) => {
        const response = await axios.put(`${API_BASE_URL}/devices/${uuid}`, { name });
        return response.data;
    },

    // Delete device
    deleteDevice: async (uuid) => {
        const response = await axios.delete(`${API_BASE_URL}/devices/${uuid}`);
        return response.data;
    }
};
