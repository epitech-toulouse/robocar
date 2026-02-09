/**
 * Main application - handles connection and Socket.IO communication
 */
document.addEventListener('DOMContentLoaded', () => {
    // Initialize LIDAR renderer
    const container = document.getElementById('lidar-container');
    const renderer = new LidarRenderer(container);

    // Socket.IO connection
    const socket = io();

    // DOM elements
    const ipInput = document.getElementById('ip-address');
    const portInput = document.getElementById('port');
    const connectBtn = document.getElementById('connect-btn');
    const disconnectBtn = document.getElementById('disconnect-btn');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const pointCountEl = document.getElementById('point-count');
    const fpsEl = document.getElementById('fps');

    // FPS counter
    let frameCount = 0;
    let lastFpsUpdate = Date.now();

    setInterval(() => {
        const now = Date.now();
        const elapsed = (now - lastFpsUpdate) / 1000;
        const fps = Math.round(frameCount / elapsed);
        fpsEl.textContent = fps;
        frameCount = 0;
        lastFpsUpdate = now;
    }, 1000);

    // Update status
    function setStatus(status, text) {
        statusDot.className = 'status-dot ' + status;
        statusText.textContent = text;
    }

    // Connect button
    connectBtn.addEventListener('click', () => {
        const ip = ipInput.value.trim();
        const port = parseInt(portInput.value);

        if (!ip || !port) {
            alert('Veuillez entrer une adresse IP et un port valides');
            return;
        }

        setStatus('connecting', 'Connexion...');
        connectBtn.disabled = true;

        socket.emit('connect-lidar', { ip, port });
    });

    // Disconnect button
    disconnectBtn.addEventListener('click', () => {
        socket.emit('disconnect-lidar');
    });

    // Socket events
    socket.on('connected', ({ ip, port }) => {
        setStatus('connected', `Connecté à ${ip}:${port}`);
        connectBtn.disabled = true;
        disconnectBtn.disabled = false;
        renderer.clear();
    });

    socket.on('disconnected', () => {
        setStatus('', 'Déconnecté');
        connectBtn.disabled = false;
        disconnectBtn.disabled = true;
    });

    socket.on('connection-error', (error) => {
        setStatus('error', `Erreur: ${error}`);
        connectBtn.disabled = false;
        disconnectBtn.disabled = true;
    });

    socket.on('lidar-data', (points) => {
        renderer.addPoints(points);
        pointCountEl.textContent = renderer.getPointCount();
        frameCount++;
    });

    // Initial status
    setStatus('', 'Déconnecté');
});
