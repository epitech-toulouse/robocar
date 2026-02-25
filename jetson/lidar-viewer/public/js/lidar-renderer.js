/**
 * LIDAR 2D Renderer using Three.js with top-down orthographic camera
 */
class LidarRenderer {
    constructor(container) {
        this.container = container;
        this.points = [];
        this.maxPoints = 1500;
        this.maxRange = 12; // meters
        this.scale = 30; // pixels per meter

        this.init();
        this.createGrid();
        this.animate();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a15);

        // Get container dimensions
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        // Orthographic camera (top-down view)
        const aspect = width / height;
        const viewSize = this.maxRange * this.scale;
        this.camera = new THREE.OrthographicCamera(
            -viewSize * aspect, viewSize * aspect,
            viewSize, -viewSize,
            1, 1000
        );
        this.camera.position.set(0, 100, 0);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        // Points geometry
        this.pointsGeometry = new THREE.BufferGeometry();
        this.positions = new Float32Array(this.maxPoints * 3);
        this.colors = new Float32Array(this.maxPoints * 3);

        this.pointsGeometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
        this.pointsGeometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));

        const pointsMaterial = new THREE.PointsMaterial({
            size: 4,
            vertexColors: true,
            sizeAttenuation: false
        });

        this.pointsMesh = new THREE.Points(this.pointsGeometry, pointsMaterial);
        this.scene.add(this.pointsMesh);

        // Center marker
        const centerGeom = new THREE.CircleGeometry(5, 16);
        const centerMat = new THREE.MeshBasicMaterial({ color: 0x4ade80 });
        const centerMesh = new THREE.Mesh(centerGeom, centerMat);
        centerMesh.rotation.x = -Math.PI / 2;
        centerMesh.position.y = 0.1;
        this.scene.add(centerMesh);

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }

    createGrid() {
        // Range circles
        for (let r = 2; r <= this.maxRange; r += 2) {
            const circleGeom = new THREE.RingGeometry(
                r * this.scale - 0.5,
                r * this.scale + 0.5,
                64
            );
            const circleMat = new THREE.MeshBasicMaterial({
                color: 0x333355,
                side: THREE.DoubleSide
            });
            const circle = new THREE.Mesh(circleGeom, circleMat);
            circle.rotation.x = -Math.PI / 2;
            this.scene.add(circle);
        }

        // Crosshairs
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0x333355 });
        const size = this.maxRange * this.scale;

        // Horizontal line
        const hPoints = [
            new THREE.Vector3(-size, 0, 0),
            new THREE.Vector3(size, 0, 0)
        ];
        const hGeom = new THREE.BufferGeometry().setFromPoints(hPoints);
        this.scene.add(new THREE.Line(hGeom, lineMaterial));

        // Vertical line
        const vPoints = [
            new THREE.Vector3(0, 0, -size),
            new THREE.Vector3(0, 0, size)
        ];
        const vGeom = new THREE.BufferGeometry().setFromPoints(vPoints);
        this.scene.add(new THREE.Line(vGeom, lineMaterial));
    }

    addPoints(newPoints) {
        // Add new points to buffer
        for (const point of newPoints) {
            this.points.push(point);
            if (this.points.length > this.maxPoints) {
                this.points.shift();
            }
        }

        this.updateGeometry();
    }

    updateGeometry() {
        for (let i = 0; i < this.maxPoints; i++) {
            if (i < this.points.length) {
                const p = this.points[i];
                // Convert polar to cartesian (angle 0 = forward/up, clockwise)
                const angleRad = (90 - p.angle) * Math.PI / 180;
                const x = p.distance * this.scale * Math.cos(angleRad);
                const z = -p.distance * this.scale * Math.sin(angleRad);

                this.positions[i * 3] = x;
                this.positions[i * 3 + 1] = 0;
                this.positions[i * 3 + 2] = z;

                // Color based on intensity (yellow to red)
                const intensity = p.intensity / 255;
                this.colors[i * 3] = 1;
                this.colors[i * 3 + 1] = 1 - intensity * 0.5;
                this.colors[i * 3 + 2] = 0;
            } else {
                // Hide unused points
                this.positions[i * 3] = 0;
                this.positions[i * 3 + 1] = -1000;
                this.positions[i * 3 + 2] = 0;
            }
        }

        this.pointsGeometry.attributes.position.needsUpdate = true;
        this.pointsGeometry.attributes.color.needsUpdate = true;
    }

    getPointCount() {
        return this.points.length;
    }

    clear() {
        this.points = [];
        this.updateGeometry();
    }

    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        const aspect = width / height;
        const viewSize = this.maxRange * this.scale;

        this.camera.left = -viewSize * aspect;
        this.camera.right = viewSize * aspect;
        this.camera.top = viewSize;
        this.camera.bottom = -viewSize;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }
}
