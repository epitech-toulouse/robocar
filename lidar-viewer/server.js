const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const dgram = require('dgram');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.static(path.join(__dirname, 'public')));

// Store active UDP connections per socket
const connections = new Map();

io.on('connection', (socket) => {
    console.log('Web client connected:', socket.id);

    // Handle connection request from web client
    socket.on('connect-lidar', ({ ip, port }) => {
        console.log(`Connecting to LIDAR server at ${ip}:${port}`);

        // Clean up existing connection if any
        if (connections.has(socket.id)) {
            connections.get(socket.id).close();
        }

        const udpClient = dgram.createSocket('udp4');
        connections.set(socket.id, udpClient);

        // Buffer for accumulating data
        let dataBuffer = '';

        udpClient.on('message', (msg) => {
            dataBuffer += msg.toString();

            // Process complete lines
            const lines = dataBuffer.split('\n');
            dataBuffer = lines.pop(); // Keep incomplete line in buffer

            const points = [];
            for (const line of lines) {
                const parts = line.trim().split(',');
                if (parts.length >= 2) {
                    const angle = parseFloat(parts[0]);
                    const distance = parseFloat(parts[1]);
                    const intensity = parts.length > 2 ? parseInt(parts[2]) : 128;

                    if (!isNaN(angle) && !isNaN(distance)) {
                        points.push({ angle, distance, intensity });
                    }
                }
            }

            if (points.length > 0) {
                socket.emit('lidar-data', points);
            }
        });

        udpClient.on('error', (err) => {
            console.error('UDP error:', err.message);
            socket.emit('connection-error', err.message);
        });

        // Send initial message to register with UDP server
        const message = Buffer.from('CONNECT');
        udpClient.send(message, port, ip, (err) => {
            if (err) {
                socket.emit('connection-error', err.message);
            } else {
                socket.emit('connected', { ip, port });
            }
        });
    });

    // Handle disconnect request
    socket.on('disconnect-lidar', () => {
        if (connections.has(socket.id)) {
            connections.get(socket.id).close();
            connections.delete(socket.id);
            socket.emit('disconnected');
        }
    });

    // Cleanup on socket disconnect
    socket.on('disconnect', () => {
        console.log('Web client disconnected:', socket.id);
        if (connections.has(socket.id)) {
            connections.get(socket.id).close();
            connections.delete(socket.id);
        }
    });
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`LIDAR Viewer server running on http://localhost:${PORT}`);
}).on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        const altPort = 3001;
        console.log(`Port ${PORT} in use, trying ${altPort}...`);
        server.listen(altPort);
    } else {
        console.error(err);
    }
});
