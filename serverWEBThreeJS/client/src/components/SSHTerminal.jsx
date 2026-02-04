import { useEffect, useRef, useState } from 'react';
import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { WebLinksAddon } from 'xterm-addon-web-links';
import 'xterm/css/xterm.css';

/**
 * SSHTerminal - Web-based SSH terminal component
 * Connects to backend WebSocket SSH proxy
 */
export default function SSHTerminal() {
    const terminalRef = useRef(null);
    const terminalInstance = useRef(null);
    const fitAddon = useRef(null);
    const wsRef = useRef(null);

    const [connected, setConnected] = useState(false);
    const [connecting, setConnecting] = useState(false);
    const [credentials, setCredentials] = useState({
        host: 'localhost',
        port: '22',
        username: '',
        password: ''
    });

    // Initialize terminal
    useEffect(() => {
        if (!terminalRef.current) return;

        const term = new Terminal({
            cursorBlink: true,
            fontSize: 14,
            fontFamily: 'Menlo, Monaco, "Courier New", monospace',
            theme: {
                background: '#1a1a1a',
                foreground: '#f0f0f0',
                cursor: '#00ff88',
                selection: 'rgba(255, 255, 255, 0.3)'
            }
        });

        fitAddon.current = new FitAddon();
        term.loadAddon(fitAddon.current);
        term.loadAddon(new WebLinksAddon());

        term.open(terminalRef.current);
        fitAddon.current.fit();

        term.writeln('\x1b[1;36m Robocar SSH Terminal\x1b[0m');
        term.writeln('\x1b[90mEnter connection details and click Connect\x1b[0m');
        term.writeln('');

        terminalInstance.current = term;

        // Handle window resize
        const handleResize = () => {
            if (fitAddon.current) {
                fitAddon.current.fit();
            }
        };
        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => {
            window.removeEventListener('resize', handleResize);
            if (wsRef.current) {
                wsRef.current.close();
            }
            term.dispose();
        };
    }, []);

    // Connect to SSH via WebSocket
    const connect = () => {
        if (!credentials.host || !credentials.username || !credentials.password) {
            terminalInstance.current.writeln('\x1b[1;31m❌ Please fill in all fields\x1b[0m');
            return;
        }

        setConnecting(true);
        const term = terminalInstance.current;

        term.clear();
        term.writeln(`\x1b[1;33m🔌 Connecting to ${credentials.username}@${credentials.host}:${credentials.port}...\x1b[0m`);

        // Determine WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname;
        // In dev (port 5173), connect to server on 3000. In prod, use same port.
        const port = window.location.port === '5173' ? '3000' : window.location.port;
        const wsUrl = `${protocol}//${host}:${port}/ssh`;

        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
            // Send credentials
            ws.send(JSON.stringify({
                type: 'auth',
                ...credentials,
                port: parseInt(credentials.port)
            }));
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'connected') {
                term.writeln('\x1b[1;32m✓ Connected!\x1b[0m\r\n');
                setConnected(true);
                setConnecting(false);

                // Handle terminal input
                term.onData((input) => {
                    ws.send(JSON.stringify({
                        type: 'input',
                        data: input
                    }));
                });
            } else if (data.type === 'data') {
                term.write(data.data);
            } else if (data.type === 'error') {
                term.writeln(`\x1b[1;31m❌ Error: ${data.message}\x1b[0m`);
                setConnecting(false);
            }
        };

        ws.onerror = (error) => {
            term.writeln('\x1b[1;31m❌ WebSocket connection failed\x1b[0m');
            console.error('WebSocket error:', error);
            setConnecting(false);
        };

        ws.onclose = () => {
            if (connected) {
                term.writeln('\r\n\x1b[1;33m🔌 Connection closed\x1b[0m');
            }
            setConnected(false);
            setConnecting(false);
        };
    };

    // Disconnect SSH
    const disconnect = () => {
        if (wsRef.current) {
            wsRef.current.close();
        }
        setConnected(false);
    };

    return (
        <div className="ssh-terminal-container">
            {/* Connection Form */}
            <div className="connection-panel">
                <h3>SSH Connection</h3>

                <div className="form-group">
                    <label>Host IP:</label>
                    <input
                        type="text"
                        placeholder="192.168.1.100"
                        value={credentials.host}
                        onChange={(e) => setCredentials({ ...credentials, host: e.target.value })}
                        disabled={connected}
                    />
                </div>

                <div className="form-row">
                    <div className="form-group">
                        <label>Port:</label>
                        <input
                            type="text"
                            placeholder="22"
                            value={credentials.port}
                            onChange={(e) => setCredentials({ ...credentials, port: e.target.value })}
                            disabled={connected}
                        />
                    </div>
                    <div className="form-group">
                        <label>Username:</label>
                        <input
                            type="text"
                            placeholder="pi"
                            value={credentials.username}
                            onChange={(e) => setCredentials({ ...credentials, username: e.target.value })}
                            disabled={connected}
                        />
                    </div>
                </div>

                <div className="form-group">
                    <label>Password:</label>
                    <input
                        type="password"
                        placeholder="••••••••"
                        value={credentials.password}
                        onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
                        disabled={connected}
                    />
                </div>

                <div className="connection-controls">
                    {!connected ? (
                        <button
                            className="btn-connect"
                            onClick={connect}
                            disabled={connecting}
                        >
                            {connecting ? '⏳ Connecting...' : '🔌 Connect'}
                        </button>
                    ) : (
                        <button className="btn-disconnect" onClick={disconnect}>
                            🔴 Disconnect
                        </button>
                    )}
                </div>
            </div>

            {/* Terminal */}
            <div className="terminal-wrapper">
                <div ref={terminalRef} className="terminal" />
            </div>
        </div>
    );
}
