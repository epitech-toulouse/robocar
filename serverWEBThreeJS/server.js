const express = require('express');
const path = require('path');
const { WebSocketServer } = require('ws');
const { Client } = require('ssh2');
const http = require('http');
const { Server } = require('socket.io');
const { SerialPort } = require('serialport');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

const PORT = process.env.PORT || 3000;
const LIDAR_BAUD_RATE = parseInt(process.env.LIDAR_BAUD_RATE) || 230400; // LD19 uses 230400
const LIDAR_TYPE = process.env.LIDAR_TYPE || 'ld19'; // 'ld19', 'rplidar', 'ydlidar', 'auto'

// Try to find and connect to lidar on USB0 or USB1
let serialPort = null;
let lidarConnected = false;

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
        const startAngle = buffer.readUInt16LE(offset + 4) / 100.0; // Convert to degrees
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

            // Filter valid points (distance > 0 and < 12m, intensity > 0)
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

async function findLidarPort() {
    const possiblePorts = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1'];

    for (const portPath of possiblePorts) {
        try {
            console.log(`🔍 Trying to connect to lidar on ${portPath}...`);
            const port = new SerialPort({
                path: portPath,
                baudRate: LIDAR_BAUD_RATE,
                autoOpen: false
            });

            const result = await new Promise((resolve) => {
                port.open((err) => {
                    if (err) {
                        console.log(`❌ Could not open ${portPath}: ${err.message}`);
                        resolve(null);
                    } else {
                        console.log(`✅ Connected to lidar on ${portPath}`);
                        resolve(port);
                    }
                });
            });

            if (result) return result;
        } catch (err) {
            console.log(`❌ Error with ${portPath}: ${err.message}`);
        }
    }
    return null;
}

/**
 * Parse RPLidar binary protocol (Express Scan mode)
 * Each measurement packet is 5 bytes:
 * Byte 0: quality (6 bits) + start flag (1 bit) + ~start flag (1 bit)
 * Byte 1-2: angle in q6 format
 * Byte 3-4: distance in mm
 */
function parseRPLidarExpressScan(buffer) {
    const points = [];
    let offset = 0;

    while (offset + 5 <= buffer.length) {
        const byte0 = buffer[offset];
        const byte1 = buffer[offset + 1];
        const byte2 = buffer[offset + 2];
        const byte3 = buffer[offset + 3];
        const byte4 = buffer[offset + 4];

        // Check for valid sync bits (bit 0 and bit 1 should be inverted)
        const S = byte0 & 0x01;
        const notS = (byte0 >> 1) & 0x01;

        if (S === notS) {
            // Invalid sync, skip one byte
            offset++;
            continue;
        }

        // Quality (0-63)
        const quality = byte0 >> 2;

        // Check bit for angle (should be 1)
        const C = byte1 & 0x01;
        if (C !== 1) {
            offset++;
            continue;
        }

        // Angle in degrees (15-bit, value / 64)
        const angleRaw = ((byte2 << 8) | byte1) >> 1;
        const angle = angleRaw / 64.0;

        // Distance in mm (16-bit)
        const distance = (byte4 << 8) | byte3;

        // Only add valid points
        if (quality > 0 && distance > 0 && distance < 12000) {
            const angleRad = (angle * Math.PI) / 180;
            const distanceM = distance / 1000;

            points.push({
                x: Math.cos(angleRad) * distanceM,
                y: 0.1,
                z: Math.sin(angleRad) * distanceM,
                angle: angle,
                distance: distanceM,
                quality: quality
            });
        }

        offset += 5;
    }

    return { points, remaining: buffer.slice(offset) };
}

/**
 * Parse YDLIDAR binary protocol
 * Package header: 0xAA 0x55
 */
function parseYDLidarData(buffer) {
    const points = [];
    let offset = 0;

    while (offset + 10 <= buffer.length) {
        // Look for package header
        if (buffer[offset] !== 0xAA || buffer[offset + 1] !== 0x55) {
            offset++;
            continue;
        }

        const packageType = buffer[offset + 2];
        const sampleCount = buffer[offset + 3];

        // Check if we have enough data
        const packetLength = 10 + sampleCount * 2;
        if (offset + packetLength > buffer.length) {
            break; // Need more data
        }

        const startAngle = ((buffer[offset + 5] << 8) | buffer[offset + 4]) / 128.0;
        const endAngle = ((buffer[offset + 7] << 8) | buffer[offset + 6]) / 128.0;

        // Calculate angle increment
        let angleDiff = endAngle - startAngle;
        if (angleDiff < 0) angleDiff += 360;
        const angleStep = sampleCount > 1 ? angleDiff / (sampleCount - 1) : 0;

        const dataStart = offset + 10;

        for (let i = 0; i < sampleCount; i++) {
            const idx = dataStart + i * 2;
            if (idx + 1 >= buffer.length) break;

            const distanceRaw = (buffer[idx + 1] << 8) | buffer[idx];
            const distance = distanceRaw / 4.0; // mm
            const angle = (startAngle + i * angleStep) % 360;

            if (distance > 10 && distance < 12000) {
                const angleRad = (angle * Math.PI) / 180;
                const distanceM = distance / 1000;

                points.push({
                    x: Math.cos(angleRad) * distanceM,
                    y: 0.1,
                    z: Math.sin(angleRad) * distanceM,
                    angle: angle,
                    distance: distanceM
                });
            }
        }

        offset += packetLength;
    }

    return { points, remaining: buffer.slice(offset) };
}

/**
 * Simple binary parser that looks for any recognizable pattern
 * and extracts angle/distance pairs
 */
function parseGenericLidar(buffer) {
    const points = [];
    let offset = 0;

    // Try to find patterns: look for 2-byte angle followed by 2-byte distance
    while (offset + 4 <= buffer.length) {
        // Interpret as little-endian values
        const val1 = (buffer[offset + 1] << 8) | buffer[offset];
        const val2 = (buffer[offset + 3] << 8) | buffer[offset + 2];

        // Heuristic: if first value looks like angle (0-36000 for 0.01 degree resolution)
        // and second value looks like distance (100-12000 mm)
        if (val1 <= 36000 && val2 >= 100 && val2 <= 12000) {
            const angle = val1 / 100.0; // Assume 0.01 degree resolution
            const distance = val2;

            const angleRad = (angle * Math.PI) / 180;
            const distanceM = distance / 1000;

            points.push({
                x: Math.cos(angleRad) * distanceM,
                y: 0.1,
                z: Math.sin(angleRad) * distanceM,
                angle: angle,
                distance: distanceM
            });

            offset += 4;
        } else {
            offset++;
        }
    }

    return { points, remaining: buffer.slice(offset) };
}

let dataBuffer = Buffer.alloc(0);
let detectedProtocol = null;
let pointsAccumulator = [];
let lastEmitTime = 0;
let scanPoints = []; // Full scan accumulator
let lastAngle = 0;
const EMIT_INTERVAL = 100; // Emit every 100ms

async function initLidar() {
    serialPort = await findLidarPort();

    if (serialPort) {
        lidarConnected = true;
        dataBuffer = Buffer.alloc(0);
        pointsAccumulator = [];
        scanPoints = [];

        console.log('📡 Lidar connected, listening for data...');
        console.log('Lidar connected, listening for data...');

        // Debug: log first few packets to understand protocol
        let debugPacketCount = 0;
        const MAX_DEBUG_PACKETS = 10;

        // Handle binary data directly
        serialPort.on('data', (data) => {
            try {
                // Debug: show raw data format with more details
                if (debugPacketCount < MAX_DEBUG_PACKETS) {
                    const hexStr = data.slice(0, Math.min(64, data.length)).toString('hex').match(/.{1,2}/g).join(' ');
                    console.log(`Raw[${debugPacketCount}] (${data.length} bytes): ${hexStr}`);
                    // Show as ASCII too (for text-based protocols)
                    const asciiStr = data.slice(0, Math.min(32, data.length)).toString('ascii').replace(/[^\x20-\x7E]/g, '.');
                    console.log(`   ASCII: ${asciiStr}`);
                    debugPacketCount++;
                }

                // Append new data to buffer
                dataBuffer = Buffer.concat([dataBuffer, data]);

                // Try different parsers
                let result;

                if (detectedProtocol === 'ld19') {
                    result = parseLD19Data(dataBuffer);
                } else if (detectedProtocol === 'ydlidar') {
                    result = parseYDLidarData(dataBuffer);
                } else if (detectedProtocol === 'rplidar') {
                    result = parseRPLidarExpressScan(dataBuffer);
                } else {
                    // Auto-detect - try LD19 first (STL-19P)
                    // Check for LD19 header (0x54 0x2C)
                    for (let i = 0; i < Math.min(100, dataBuffer.length - 1); i++) {
                        if (dataBuffer[i] === 0x54 && dataBuffer[i + 1] === 0x2C) {
                            console.log('Detected LD19/STL-19P protocol');
                            detectedProtocol = 'ld19';
                            result = parseLD19Data(dataBuffer);
                            break;
                        }
                    }

                    if (!result) {
                        // Check for YDLIDAR header
                        for (let i = 0; i < Math.min(50, dataBuffer.length - 1); i++) {
                            if (dataBuffer[i] === 0xAA && dataBuffer[i + 1] === 0x55) {
                                console.log('🔍 Detected YDLIDAR protocol');
                                detectedProtocol = 'ydlidar';
                                result = parseYDLidarData(dataBuffer);
                                break;
                            }
                        }
                    }

                    if (!result) {
                        // Try RPLidar
                        result = parseRPLidarExpressScan(dataBuffer);
                        if (result.points.length > 5) {
                            console.log('🔍 Detected RPLidar protocol');
                            detectedProtocol = 'rplidar';
                        }
                    }

                    if (!result || result.points.length === 0) {
                        // Try generic parser
                        result = parseGenericLidar(dataBuffer);
                        if (result.points.length > 0) {
                            console.log('🔍 Using generic lidar parser');
                            detectedProtocol = 'generic';
                        }
                    }
                }

                if (result && result.points.length > 0) {
                    // Accumulate points for a full scan
                    for (const point of result.points) {
                        // Detect new scan (angle wraps around)
                        if (point.angle < lastAngle - 180) {
                            // New scan started, emit accumulated points
                            if (scanPoints.length > 10) {
                                console.log(`📍 Complete scan: ${scanPoints.length} points`);
                                // Log sample point for debugging
                                if (scanPoints.length > 0) {
                                    console.log(`   Sample point: x=${scanPoints[0].x.toFixed(2)}, z=${scanPoints[0].z.toFixed(2)}, dist=${scanPoints[0].distance.toFixed(2)}m`);
                                }
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
                console.error('Error parsing lidar data:', err.message);
            }
        });

        serialPort.on('error', (err) => {
            console.error('Serial port error:', err.message);
            lidarConnected = false;
        });

        serialPort.on('close', () => {
            console.log('📴 Lidar connection closed');
            lidarConnected = false;
            setTimeout(initLidar, 5000);
        });

        // Send start scan command for RPLidar (0xA5 0x20)
        console.log('📤 Sending RPLidar start scan command...');
        serialPort.write(Buffer.from([0xA5, 0x20]));

    } else {
        console.log('⚠️ No lidar found. Retrying in 5 seconds...');
        setTimeout(initLidar, 5000);
    }
}

// Serve static files from React build
app.use(express.static('dist'));

// API endpoint for health check
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        lidar: lidarConnected ? 'connected' : 'disconnected'
    });
});

// API endpoint to get lidar status
app.get('/api/lidar/status', (req, res) => {
    res.json({
        connected: lidarConnected,
        baudRate: LIDAR_BAUD_RATE,
        protocol: detectedProtocol || 'detecting',
        pointCount: scanPoints.length
    });
});

// Serve React app for all other routes
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// WebSocket connection handling
io.on('connection', (socket) => {
    console.log('🔌 Client connected:', socket.id);

    // Send current lidar status
    socket.emit('lidar-status', {
        connected: lidarConnected,
        protocol: detectedProtocol
    });

    // Send current scan if available
    if (scanPoints.length > 0) {
        socket.emit('lidar-data', scanPoints);
    }

    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
    });
});

// Start server and initialize lidar
server.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running at http://localhost:${PORT}`);
    console.log(`React + Three.js server ready`);
    console.log(`Lidar baud rate: ${LIDAR_BAUD_RATE}`);
    console.log(`Lidar type: ${LIDAR_TYPE}`);

    // Initialize lidar connection
    initLidar();
});

// WebSocket server for SSH proxy
const wss = new WebSocketServer({ server, path: '/ssh' });

console.log('WebSocket SSH proxy ready on port', PORT);

wss.on('connection', (ws) => {
    console.log('New WebSocket connection');

    let sshClient = null;
    let sshStream = null;

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);

            if (data.type === 'auth') {
                // Create SSH connection
                sshClient = new Client();

                sshClient.on('ready', () => {
                    console.log(`SSH connected to ${data.host}`);

                    ws.send(JSON.stringify({
                        type: 'connected',
                        message: 'SSH connection established'
                    }));

                    // Open shell
                    sshClient.shell((err, stream) => {
                        if (err) {
                            ws.send(JSON.stringify({
                                type: 'error',
                                message: err.message
                            }));
                            return;
                        }

                        sshStream = stream;

                        // Send SSH output to WebSocket
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

                // Connect to SSH
                sshClient.connect({
                    host: data.host,
                    port: data.port || 22,
                    username: data.username,
                    password: data.password,
                    readyTimeout: 10000
                });

            } else if (data.type === 'input') {
                // Forward input to SSH
                if (sshStream) {
                    sshStream.write(data.data);
                }
            }
        } catch (error) {
            console.error('WebSocket message error:', error);
        }
    });

    ws.on('close', () => {
        console.log('📡 WebSocket closed');
        if (sshClient) {
            sshClient.end();
        }
    });

    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
    });
});
