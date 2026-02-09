import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

/**
 * ThreeScene - Reusable Three.js scene component
 * Manages the core Three.js setup: scene, camera, renderer, and animation loop
 */
export default function ThreeScene({ children, onSceneReady }) {
    const containerRef = useRef(null);
    const sceneRef = useRef(null);
    const cameraRef = useRef(null);
    const rendererRef = useRef(null);
    const controlsRef = useRef(null);
    const animationIdRef = useRef(null);

    useEffect(() => {
        if (!containerRef.current) return;

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0a);
        // Disable fog to see all lidar points
        // scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);
        sceneRef.current = scene;

        // Camera setup - 2D top-down view with orthographic camera
        const aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
        const frustumSize = 2; // Smaller to see close lidar points (within ~1m)
        const camera = new THREE.OrthographicCamera(
            frustumSize * aspect / -2,
            frustumSize * aspect / 2,
            frustumSize / 2,
            frustumSize / -2,
            0.1,
            1000
        );
        camera.position.set(0, 50, 0); // Camera higher up
        camera.lookAt(0, 0, 0); // Looking straight down
        cameraRef.current = camera;

        // Renderer setup
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Controls - 2D only (no rotation)
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.enableRotate = false; // Disable rotation for strict 2D view
        controls.mouseButtons = {
            LEFT: THREE.MOUSE.PAN,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN
        };
        controlsRef.current = controls;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        scene.add(directionalLight);

        // Grid helper only (no axes)
        const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
        scene.add(gridHelper);

        // Removed axes helper for cleaner view
        // const axesHelper = new THREE.AxesHelper(5);
        // scene.add(axesHelper);

        // Notify parent that scene is ready
        if (onSceneReady) {
            onSceneReady({ scene, camera, renderer, controls });
        }

        // Animation loop
        function animate() {
            animationIdRef.current = requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // Handle window resize
        function handleResize() {
            if (!containerRef.current) return;

            const aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
            const frustumSize = 2; // Match the initial frustumSize
            camera.left = frustumSize * aspect / -2;
            camera.right = frustumSize * aspect / 2;
            camera.top = frustumSize / 2;
            camera.bottom = frustumSize / -2;
            camera.updateProjectionMatrix();
            renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        }
        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => {
            window.removeEventListener('resize', handleResize);
            if (animationIdRef.current) {
                cancelAnimationFrame(animationIdRef.current);
            }
            if (containerRef.current && renderer.domElement) {
                containerRef.current.removeChild(renderer.domElement);
            }
            renderer.dispose();
            controls.dispose();
        };
    }, [onSceneReady]);

    return (
        <div
            ref={containerRef}
            style={{ width: '100%', height: '100vh' }}
        >
            {children}
        </div>
    );
}
