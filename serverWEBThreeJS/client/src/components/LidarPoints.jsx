import { useEffect, useRef } from 'react';
import * as THREE from 'three';

/**
 * LidarPoints - Component for visualizing lidar point cloud data in 2D
 */
export default function LidarPoints({
    points = [],
    scene,
    color = 0xff0000,
    size = 8,
    visible = true
}) {
    const meshRef = useRef(null);
    const materialRef = useRef(null);

    // Create the points mesh once when scene is available
    useEffect(() => {
        if (!scene) {
            console.log('❌ No scene provided to LidarPoints');
            return;
        }

        console.log('🎨 Creating LidarPoints mesh');

        // Create material - red color for lidar points
        const material = new THREE.PointsMaterial({
            color: 0xff0000,
            size: 10,
            sizeAttenuation: false,
            transparent: false,
            depthWrite: true,
            depthTest: true
        });
        materialRef.current = material;

        // Create empty geometry (will be filled with lidar data)
        const geometry = new THREE.BufferGeometry();
        const emptyPositions = new Float32Array([0, 0.5, 0]);
        geometry.setAttribute('position', new THREE.BufferAttribute(emptyPositions, 3));

        // Create mesh
        const mesh = new THREE.Points(geometry, material);
        mesh.frustumCulled = false; // Always render
        mesh.renderOrder = 999; // Render on top
        mesh.visible = visible;
        
        scene.add(mesh);
        meshRef.current = mesh;
        
        console.log('✅ LidarPoints mesh added to scene');

        return () => {
            console.log('🗑️ Cleaning up LidarPoints');
            if (meshRef.current && scene) {
                scene.remove(meshRef.current);
                meshRef.current.geometry.dispose();
            }
            if (materialRef.current) {
                materialRef.current.dispose();
            }
        };
    }, [scene]);

    // Update color
    useEffect(() => {
        if (materialRef.current) {
            materialRef.current.color.set(color);
        }
    }, [color]);

    // Update size
    useEffect(() => {
        if (materialRef.current) {
            materialRef.current.size = size;
        }
    }, [size]);

    // Update visibility
    useEffect(() => {
        if (meshRef.current) {
            meshRef.current.visible = visible;
        }
    }, [visible]);

    // Update points positions
    useEffect(() => {
        if (!meshRef.current) {
            console.log('❌ No mesh ref for updating points');
            return;
        }
        
        if (!points || points.length === 0) {
            console.log('⚠️ No points to display');
            return;
        }

        console.log(`🔄 Updating ${points.length} points`);

        // Create new positions array
        const positions = new Float32Array(points.length * 3);

        for (let i = 0; i < points.length; i++) {
            const point = points[i];
            positions[i * 3] = point.x || 0;       // X
            positions[i * 3 + 1] = 0.5;            // Y (above grid)
            positions[i * 3 + 2] = point.z || 0;   // Z
        }

        // Update geometry
        const geometry = meshRef.current.geometry;
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.attributes.position.needsUpdate = true;
        geometry.computeBoundingSphere();

        
    }, [points]);

    return null;
}
