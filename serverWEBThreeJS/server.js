const express = require('express');
const path = require('path');
const { WebSocketServer } = require('ws');
const { Client } = require('ssh2');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files from React build
app.use(express.static('dist'));

// API endpoint for health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Serve React app for all other routes
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Server running at http://localhost:${PORT}`);
    console.log(`📦 React + Three.js server ready`);
});

// WebSocket server for SSH proxy
const wss = new WebSocketServer({ server, path: '/ssh' });

console.log('🔌 WebSocket SSH proxy ready on port', PORT);

wss.on('connection', (ws) => {
    console.log('📡 New WebSocket connection');

    let sshClient = null;
    let sshStream = null;

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);

            if (data.type === 'auth') {
                // Create SSH connection
                sshClient = new Client();

                sshClient.on('ready', () => {
                    console.log(`✓ SSH connected to ${data.host}`);

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
