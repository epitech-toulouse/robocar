import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

/**
 * LidarPoints - Component for visualizing lidar point cloud data in 2D
 * 
 * Props:
 * - points: Array of {x, y, z} objects representing lidar points
 * - scene: Three.js scene instance from parent
 * - color: Point color (default: 0x00ff00)
 * - size: Point size (default: 0.1)
 * - visible: Boolean to show/hide points (default: true)
 */
export default function LidarPoints({
    points = [],
    scene,
    color = 0x00ff00,
    size = 0.1,
    visible = true
}) {
    const pointsRef = useRef(null);
    const [initialized, setInitialized] = useState(false);

    useEffect(() => {
        if (!scene) return;

        // Create points geometry
        const geometry = new THREE.BufferGeometry();
        const material = new THREE.PointsMaterial({
            color: color,
            size: size,
            sizeAttenuation: true
        });

        const pointsMesh = new THREE.Points(geometry, material);
        pointsMesh.visible = visible;
        scene.add(pointsMesh);
        pointsRef.current = pointsMesh;
        setInitialized(true);

        // Cleanup
        return () => {
            if (pointsRef.current) {
                scene.remove(pointsRef.current);
                geometry.dispose();
                material.dispose();
            }
        };
    }, [scene, color, size]);

    // Update points when data changes
    useEffect(() => {
        if (!initialized || !pointsRef.current) return;

        const positions = new Float32Array(points.length * 3);

        points.forEach((point, i) => {
            positions[i * 3] = point.x;
            positions[i * 3 + 1] = point.y || 0; // Default to 0 for 2D visualization
            positions[i * 3 + 2] = point.z;
        });

        pointsRef.current.geometry.setAttribute(
            'position',
            new THREE.BufferAttribute(positions, 3)
        );
        pointsRef.current.geometry.attributes.position.needsUpdate = true;
    }, [points, initialized]);

    // Update visibility
    useEffect(() => {
        if (pointsRef.current) {
            pointsRef.current.visible = visible;
        }
    }, [visible]);

    return null; // This component doesn't render DOM elements
}
