import { useEffect, useState, useRef, useCallback } from 'react';
import { io } from 'socket.io-client';

/**
 * Custom hook for managing lidar data via WebSocket
 * 
 * @param {Object} options - Configuration options
 * @param {string} options.serverUrl - WebSocket server URL (default: auto-detect)
 * @param {number} options.maxPoints - Maximum number of points to keep (default: 720)
 * @param {boolean} options.accumulate - Whether to accumulate points by angle (default: true)
 * @param {number} options.angleResolution - Angle bucket size in degrees (default: 1)
 * @returns {Object} - { points, isConnected, lidarStatus, clearPoints }
 */
export default function useLidarSocket({
    serverUrl = null,
    maxPoints = 720,
    accumulate = true,
    angleResolution = 1
} = {}) {
    const [points, setPoints] = useState([]);
    const [isConnected, setIsConnected] = useState(false);
    const [lidarStatus, setLidarStatus] = useState({ connected: false });
    const socketRef = useRef(null);
    // Use a Map to store points by angle bucket for efficient updates
    const pointsMapRef = useRef(new Map());

    // Determine the server URL
    const getServerUrl = useCallback(() => {
        if (serverUrl) return serverUrl;
        
        // In production, connect to the same host
        if (import.meta.env.PROD) {
            return window.location.origin;
        }
        
        // In development, connect to the dev server (adjust port if needed)
        return 'http://localhost:3000';
    }, [serverUrl]);

    // Clear accumulated points
    const clearPoints = useCallback(() => {
        pointsMapRef.current.clear();
        setPoints([]);
    }, []);

    useEffect(() => {
        const url = getServerUrl();
        console.log(`🔌 Connecting to WebSocket server at ${url}`);

        const socket = io(url, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionAttempts: Infinity,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000
        });

        socketRef.current = socket;

        socket.on('connect', () => {
            console.log('✅ WebSocket connected');
            setIsConnected(true);
        });

        socket.on('disconnect', () => {
            console.log('❌ WebSocket disconnected');
            setIsConnected(false);
        });

        socket.on('lidar-status', (status) => {
            console.log('📡 Lidar status:', status);
            setLidarStatus(status);
        });

        socket.on('lidar-data', (newPoints) => {
            console.log(`📍 Received ${newPoints?.length || 0} lidar points from server`);
            
            if (!Array.isArray(newPoints) || newPoints.length === 0) {
                return;
            }

            if (accumulate) {
                // Accumulate points by angle bucket
                // This builds up a complete 360° scan over time
                for (const point of newPoints) {
                    if (point.angle !== undefined) {
                        // Round angle to bucket
                        const angleBucket = Math.floor(point.angle / angleResolution) * angleResolution;
                        pointsMapRef.current.set(angleBucket, {
                            x: point.x,
                            z: point.z,
                            angle: point.angle,
                            distance: point.distance,
                            timestamp: Date.now()
                        });
                    }
                }
                
                // Convert map to array and limit size
                const allPoints = Array.from(pointsMapRef.current.values());
                
                // Remove old points if we exceed maxPoints (keep most recent by timestamp)
                if (allPoints.length > maxPoints) {
                    allPoints.sort((a, b) => b.timestamp - a.timestamp);
                    const pointsToKeep = allPoints.slice(0, maxPoints);
                    pointsMapRef.current.clear();
                    for (const p of pointsToKeep) {
                        const bucket = Math.floor(p.angle / angleResolution) * angleResolution;
                        pointsMapRef.current.set(bucket, p);
                    }
                }
                
                setPoints(Array.from(pointsMapRef.current.values()));
            } else {
                // Replace with new scan data (original behavior)
                setPoints([...newPoints]);
            }
        });

        socket.on('connect_error', (error) => {
            console.error('WebSocket connection error:', error.message);
        });

        // Cleanup on unmount
        return () => {
            console.log('🔌 Disconnecting WebSocket');
            socket.disconnect();
        };
    }, [getServerUrl, accumulate, maxPoints]);

    return {
        points,
        isConnected,
        lidarStatus,
        clearPoints
    };
}
