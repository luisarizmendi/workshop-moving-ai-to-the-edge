#!/bin/bash

# Write the runtime configuration
echo "window.env = { BACKEND_API_BASE_URL: '${BACKEND_API_BASE_URL}' };" > /app/build/runtime-config.js

# Start the server
exec serve -s build -l 3000
