# Object detection Dashboard Frontend

## Overview

This is a React-based web application for monitoring devices, their status, and alarm history. The application provides a user-friendly interface to view, manage, and track device information.

## Features

- **Device List View**
  - Device status tracking
  - Quick overview of device connectivity and alarm states
  - Periodic device status updates

- **Device Details**
  - Detailed view of individual device information
  - Ability to edit device name
  - Device deletion functionality
  - Comprehensive alarm history log

## Technologies Used

- React
- React Router
- Axios
- Tailwind CSS
- Lucide React Icons

## Prerequisites

- Node.js (v14 or later)
- npm 

## Running the Application

If you are not running the backend in the same host, you will need first to setup an environment variable:

```bash
export BACKEND_API_BASE_URL=http://<dashboard backend ip>:<port>
```


- Development mode:
  ```bash
  npm start
  ```

- Build for production:
  ```bash
  npm run build
  ```


If you want to run it containerized:

```bash
podman run -d -p 3000:3000 -e BACKEND_API_BASE_URL=http://<dashboard backend ip>:<port> <image name>
```
> **Note:**
> You can find an image in `quay.io/luisarizmendi/object-detection-dashboard-frontend:v1`




## Project Structure

- `src/components/`
  - `DeviceList.js`: Displays a list of all devices
  - `DeviceDetails.js`: Shows detailed information for a specific device
  - `DeviceStatusIcon.js`: Renders device status icons

- `src/services/`
  - `api.js`: Contains API service methods for device operations

## API Endpoints

The application interacts with the following backend endpoints:

- `GET /devices`: Retrieve all devices
- `GET /devices/{uuid}`: Get details of a specific device
- `PUT /devices/{uuid}`: Update device name
- `DELETE /devices/{uuid}`: Delete a device

## Key Components

### DeviceList
- Fetches and displays a list of devices
- Shows device status, alarm state, and last active time
- Polls for updates every 2 seconds

### DeviceDetails
- Displays comprehensive device information
- Allows editing device name
- Shows alarm history
- Provides device deletion option

## Icons

Uses Lucide React icons to represent:
- Device status (active/inactive)
- Alarm status (active/inactive)

## Styling

Utilizes Tailwind CSS for responsive and clean UI design.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/device-monitoring-dashboard](https://github.com/yourusername/device-monitoring-dashboard)
