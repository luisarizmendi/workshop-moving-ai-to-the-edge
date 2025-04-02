import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DeviceList from './components/DeviceList';
import DeviceDetails from './components/DeviceDetails';

function App() {
    return (
        <Router>
        <div className="min-h-screen bg-gray-100">
        <nav className="bg-blue-600 p-4">
        <div className="container mx-auto">
        <h1 className="text-white text-2xl font-bold">
        Device Monitoring Dashboard
        </h1>
        </div>
        </nav>

        <div className="container mx-auto mt-4 p-4">
        <Routes>
        <Route path="/" element={<DeviceList />} />
        <Route path="/device/:uuid" element={<DeviceDetails />} />
        </Routes>
        </div>
        </div>
        </Router>
    );
}

export default App;
