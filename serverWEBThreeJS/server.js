const express = require('express');
const path = require('path');
const http = require('http');
const dgram = require('dgram');
const { Server } = require('socket.io');
const { WebSocketServer } = require('ws');
const { Client } = require('ssh2');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

const PORT = process.env.PORT || 3000;
const UDP_PORT = process.env.UDP_PORT || 7070;

// Robot configuration - the robot's UDP server address
const ROBOT_IP = process.env.ROBOT_IP || '192.168.12.1';
const ROBOT_UDP_PORT = process.env.ROBOT_UDP_PORT || 7070;

// UDP socket for communicating with robot
const udpServer = dgram.createSocket('udp4');
let lidarConnected = false;
let lastUdpReceiveTime = 0;
let registrationInterval = null;

// Lidar data processing
let dataBuffer = Buffer.alloc(0);
let scanPoints = [];
let lastAngle = 0;
let lastEmitTime = 0;
const EMIT_INTERVAL = 100; // Emit every 100ms
let isRegistered = false;

/**
 * Register with the robot's UDP server
 * Sends a message to the robot so it adds us to its client list
 * The robot doesn't send a confirmation - it just starts sending data
 */
function registerWithRobot() {
    const message = Buffer.from('REGISTER\n');
    udpServer.send(message, ROBOT_UDP_PORT, ROBOT_IP, (err) => {
        if (err) {
            console.error(`❌ Failed to send registration: ${err.message}`);
            isRegistered = false;
            io.emit('lidar-status', { connected: false, error: err.message });
        } else {
            if (!isRegistered) {
                console.log(`✅ Registered with robot at ${ROBOT_IP}:${ROBOT_UDP_PORT}`);
                isRegistered = true;
                // Consider connected as soon as we register (robot doesn't send confirmation)
                lidarConnected = true;
                io.emit('lidar-status', { connected: true, source: `${ROBOT_IP}:${ROBOT_UDP_PORT}` });
            }
        }
    });
}

/**
 * Parse LDRobot LD19/STL-19P protocol
 * Packet format (47 bytes total):
 * - Header: 0x54 (1 byte)
 * - VerLen: 0x2C (1 byte) - version and length (12 points)
 * - Speed: 2 bytes (little-endian, degrees/sec)
 * - Start Angle: 2 bytes (little-endian, 0.01 degree units)
 * - Data: 36 bytes (12 points × 3 bytes each)
 *   - Distance: 2 bytes (little-endian, mm)
 *   - Intensity: 1 byte
 * - End Angle: 2 bytes (little-endian, 0.01 degree units)
 * - Timestamp: 2 bytes
 * - CRC: 1 byte
 */
function parseLD19Data(buffer) {
    const points = [];
    let offset = 0;
    const PACKET_SIZE = 47;
    const POINTS_PER_PACKET = 12;
    const HEADER = 0x54;
    const VERLEN = 0x2C;

    while (offset + PACKET_SIZE <= buffer.length) {
        // Find header
        if (buffer[offset] !== HEADER) {
            offset++;
            continue;
        }

        // Check VerLen
        if (buffer[offset + 1] !== VERLEN) {
            offset++;
            continue;
        }

        // Parse packet
        const speed = buffer.readUInt16LE(offset + 2);
        const startAngle = buffer.readUInt16LE(offset + 4) / 100.0;
        const endAngle = buffer.readUInt16LE(offset + 42) / 100.0;

        // Calculate angle step
        let angleDiff = endAngle - startAngle;
        if (angleDiff < 0) angleDiff += 360;
        const angleStep = angleDiff / (POINTS_PER_PACKET - 1);

        // Parse 12 data points
        for (let i = 0; i < POINTS_PER_PACKET; i++) {
            const dataOffset = offset + 6 + (i * 3);
            const distance = buffer.readUInt16LE(dataOffset); // mm
            const intensity = buffer[dataOffset + 2];

            // Calculate angle for this point
            let angle = startAngle + (i * angleStep);
            if (angle >= 360) angle -= 360;

            // Filter valid points
            if (distance > 10 && distance < 12000 && intensity > 0) {
                const angleRad = (angle * Math.PI) / 180;
                const distanceM = distance / 1000.0;

                points.push({
                    x: Math.cos(angleRad) * distanceM,
                    y: 0.1,
                    z: Math.sin(angleRad) * distanceM,
                    angle: angle,
                    distance: distanceM,
                    intensity: intensity
                });
            }
        }

        offset += PACKET_SIZE;
    }

    return { points, remaining: buffer.slice(offset) };
}

/**
 * Parse CSV text lidar data format
 * Expected format: angle,distance,intensity\n (one point per line)
 * Example: 354.71,0.108,148
 * - angle in degrees
 * - distance in meters
 * - intensity (0-255)
 */
function parseCSVLidarData(data) {
    try {
        const text = data.toString('utf8');
        const lines = text.split('\n').filter(line => line.trim().length > 0);
        
        const points = [];
        for (const line of lines) {
            const parts = line.split(',');
            if (parts.length >= 2) {
                const angle = parseFloat(parts[0]);
                const distance = parseFloat(parts[1]); // Already in meters
                const intensity = parts[2] ? parseInt(parts[2]) : 100;
                
                if (!isNaN(angle) && !isNaN(distance) && distance > 0.01 && distance < 30) {
                    const angleRad = (angle * Math.PI) / 180;
                    points.push({
                        x: Math.cos(angleRad) * distance,
                        y: 0.1,
                        z: Math.sin(angleRad) * distance,
                        angle: angle,
                        distance: distance,
                        intensity: intensity
                    });
                }
            }
        }
        
        return { points, remaining: Buffer.alloc(0) };
    } catch (e) {
        return null;
    }
}

/**
 * Parse JSON lidar data format
 * Expected format: { "points": [{ "angle": 0, "distance": 1000 }, ...] }
 * or: [{ "angle": 0, "distance": 1000 }, ...]
 */
function parseJSONLidarData(data) {
    try {
        const json = JSON.parse(data.toString());
        let rawPoints = [];

        if (Array.isArray(json)) {
            rawPoints = json;
        } else if (json.points && Array.isArray(json.points)) {
            rawPoints = json.points;
        } else {
            return { points: [], remaining: Buffer.alloc(0) };
        }

        const points = rawPoints.map(p => {
            const angle = p.angle || 0;
            const distance = p.distance || 0; // mm
            const distanceM = distance / 1000.0;
            const angleRad = (angle * Math.PI) / 180;

            return {
                x: Math.cos(angleRad) * distanceM,
                y: 0.1,
                z: Math.sin(angleRad) * distanceM,
                angle: angle,
                distance: distanceM,
                intensity: p.intensity || 100
            };
        }).filter(p => p.distance > 0.01 && p.distance < 12);

        return { points, remaining: Buffer.alloc(0) };
    } catch (e) {
        return null; // Not JSON, try binary parser
    }
}

// Process incoming lidar data
function processLidarData(data) {
    try {
        // First try CSV text format (angle,distance,intensity\n)
        const csvResult = parseCSVLidarData(data);
        if (csvResult && csvResult.points.length > 0) {
            // CSV data - emit directly
            if (messageCount <= 10) {
                console.log(`📍 Parsed CSV: ${csvResult.points.length} points`);
            }
            io.emit('lidar-data', csvResult.points);
            return;
        }
        
        // Then try JSON format
        const jsonResult = parseJSONLidarData(data);
        if (jsonResult && jsonResult.points.length > 0) {
            // JSON data - emit directly
            console.log(`📍 Received JSON scan: ${jsonResult.points.length} points`);
            io.emit('lidar-data', jsonResult.points);
            return;
        }

        // Binary format - accumulate in buffer
        dataBuffer = Buffer.concat([dataBuffer, data]);

        // Parse LD19 binary data
        const result = parseLD19Data(dataBuffer);

        if (result && result.points.length > 0) {
            // Accumulate points for a full scan
            for (const point of result.points) {
                // Detect new scan (angle wraps around)
                if (point.angle < lastAngle - 180) {
                    // New scan started, emit accumulated points
                    if (scanPoints.length > 10) {
                        console.log(`📍 Complete scan: ${scanPoints.length} points`);
                        io.emit('lidar-data', scanPoints);
                    }
                    scanPoints = [];
                }
                scanPoints.push(point);
                lastAngle = point.angle;
            }

            // Also emit periodically for real-time updates
            const now = Date.now();
            if (now - lastEmitTime >= EMIT_INTERVAL && scanPoints.length > 0) {
                io.emit('lidar-data', scanPoints);
                lastEmitTime = now;
            }
        }

        // Keep remaining unparsed data
        if (result) {
            dataBuffer = result.remaining;
        }

        // Prevent buffer from growing too large
        if (dataBuffer.length > 20000) {
            console.log('⚠️ Buffer overflow, trimming...');
            dataBuffer = dataBuffer.slice(-5000);
        }

    } catch (err) {
        console.error('Error processing lidar data:', err.message);
    }
}

// UDP Server for receiving lidar data
udpServer.on('error', (err) => {
    console.error(`UDP server error:\n${err.stack}`);
    udpServer.close();
});

let messageCount = 0;
udpServer.on('message', (msg, rinfo) => {
    // Update last receive time
    lastUdpReceiveTime = Date.now();
    messageCount++;
    
    // Log first few messages in detail for debugging
    if (messageCount <= 5) {
        console.log(`📡 UDP message #${messageCount} from ${rinfo.address}:${rinfo.port}`);
        console.log(`   Size: ${msg.length} bytes`);
        console.log(`   First 50 bytes (hex): ${msg.slice(0, 50).toString('hex')}`);
        console.log(`   As string: ${msg.slice(0, 100).toString('utf8').replace(/[\x00-\x1f]/g, '.')}`);
    }
    
    // Log first data reception
    if (!lidarConnected || rinfo.address !== ROBOT_IP) {
        console.log(`📡 Receiving lidar data from ${rinfo.address}:${rinfo.port}`);
        lidarConnected = true;
        io.emit('lidar-status', { connected: true, source: `${rinfo.address}:${rinfo.port}` });
    }

    // Process the incoming lidar data
    processLidarData(msg);
});

udpServer.on('listening', () => {
    const address = udpServer.address();
    console.log(`📡 UDP socket listening on ${address.address}:${address.port}`);
    console.log(`   Robot address: ${ROBOT_IP}:${ROBOT_UDP_PORT}`);
    
    // Register with robot's UDP server
    registerWithRobot();
    
    // Keep sending registration messages periodically to maintain connection
    registrationInterval = setInterval(() => {
        if (!lidarConnected || Date.now() - lastUdpReceiveTime > 2000) {
            registerWithRobot();
        }
    }, 1000);
});

// Check for lidar connection timeout
setInterval(() => {
    if (lidarConnected && Date.now() - lastUdpReceiveTime > 3000) {
        console.log('📴 Lidar data source disconnected (timeout)');
        lidarConnected = false;
        io.emit('lidar-status', { connected: false });
    }
}, 1000);

// Serve static files from the dist directory (for production)
app.use(express.static(path.join(__dirname, 'dist')));

// API endpoints
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        lidar: lidarConnected,
        uptime: process.uptime()
    });
});

app.get('/api/lidar/status', (req, res) => {
    res.json({
        connected: lidarConnected,
        lastReceive: lastUdpReceiveTime,
        pointsInBuffer: scanPoints.length
    });
});

// Serve the React app for any other route
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// Socket.IO connection handling
io.on('connection', (socket) => {
    console.log(`🔌 Client connected: ${socket.id}`);

    // Send current lidar status
    socket.emit('lidar-status', { connected: lidarConnected });

    // If we have recent scan data, send it immediately
    if (scanPoints.length > 0) {
        socket.emit('lidar-data', scanPoints);
    }

    socket.on('disconnect', () => {
        console.log(`🔌 Client disconnected: ${socket.id}`);
    });
});

// WebSocket server for SSH terminal
const wss = new WebSocketServer({ server, path: '/ssh' });

wss.on('connection', (ws) => {
    console.log('📡 SSH WebSocket connected');
    let sshClient = null;
    let sshStream = null;

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);

            if (data.type === 'connect') {
                sshClient = new Client();

                sshClient.on('ready', () => {
                    console.log(`SSH connected to ${data.host}`);
                    ws.send(JSON.stringify({
                        type: 'connected',
                        message: 'SSH connection established'
                    }));

                    sshClient.shell((err, stream) => {
                        if (err) {
                            ws.send(JSON.stringify({
                                type: 'error',
                                message: err.message
                            }));
                            return;
                        }

                        sshStream = stream;

                        stream.on('data', (data) => {
                            ws.send(JSON.stringify({
                                type: 'data',
                                data: data.toString('utf-8')
                            }));
                        });

                        stream.on('close', () => {
                            console.log('SSH stream closed');
                            ws.close();
                        });

                        stream.stderr.on('data', (data) => {
                            ws.send(JSON.stringify({
                                type: 'data',
                                data: data.toString('utf-8')
                            }));
                        });
                    });
                });

                sshClient.on('error', (err) => {
                    console.error('SSH error:', err.message);
                    ws.send(JSON.stringify({
                        type: 'error',
                        message: err.message
                    }));
                });

                sshClient.on('close', () => {
                    console.log('SSH connection closed');
                });

                sshClient.connect({
                    host: data.host,
                    port: data.port || 22,
                    username: data.username,
                    password: data.password,
                    readyTimeout: 10000
                });

            } else if (data.type === 'input') {
                if (sshStream) {
                    sshStream.write(data.data);
                }
            }
        } catch (error) {
            console.error('WebSocket message error:', error);
        }
    });

    ws.on('close', () => {
        console.log('📡 SSH WebSocket closed');
        if (sshClient) {
            sshClient.end();
        }
    });

    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
    });
});

// Start servers
server.listen(PORT, () => {
    console.log(`🚀 HTTP/WebSocket server running on port ${PORT}`);
    console.log(`📦 Open http://localhost:${PORT} in your browser`);
});

udpServer.bind(UDP_PORT, '0.0.0.0');

console.log('');
console.log('='.repeat(50));
console.log('  Lidar UDP Client/Server');
console.log('='.repeat(50));
console.log(`  Web interface: http://localhost:${PORT}`);
console.log(`  Local UDP port: ${UDP_PORT}`);
console.log(`  Robot address: ${ROBOT_IP}:${ROBOT_UDP_PORT}`);
console.log('');
console.log('  Will register with robot and receive lidar data');
console.log('='.repeat(50));
