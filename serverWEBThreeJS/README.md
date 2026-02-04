# Robocar Web Three.js Server

React + Three.js web application for visualizing Robocar lidar data with reusable component architecture.

## 🏗️ Architecture

### Component Structure
```
serverWEBThreeJS/
├── client/                 # React application
│   ├── src/
│   │   ├── components/
│   │   │   ├── ThreeScene.jsx      # Reusable Three.js scene manager
│   │   │   └── LidarPoints.jsx     # 2D lidar point cloud component
│   │   ├── App.jsx                 # Main application
│   │   └── App.css                 # Styles
│   ├── vite.config.js              # Build configuration
│   └── package.json                # Client dependencies
├── server.js                       # Express server
├── Dockerfile                      # Multi-stage Docker build
└── package.json                    # Server dependencies
```

### Key Components

**ThreeScene** - Manages Three.js lifecycle
- Handles scene, camera, renderer setup
- Manages animation loop and cleanup
- Responsive canvas with resize handling
- Exposes scene context to child components

**LidarPoints** - 2D point cloud visualization
- Receives point data as props: `[{x, y, z}, ...]`
- Dynamically updates when data changes
- Configurable color, size, and visibility
- Optimized for real-time updates

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Build and start (builds React app + runs server)
docker compose up --build -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Local Development

```bash
# Install dependencies
npm install

# Terminal 1: Start React dev server (with hot reload)
npm run dev:client
# Opens on http://localhost:5173

# Terminal 2: Build React and start Express server
npm run build
npm install  # Install server deps
npm start
# Opens on http://localhost:3000
```

## 📦 Access

- **Production**: http://localhost:3000 (Docker or after build)
- **Development**: http://localhost:5173 (Vite dev server)
- **Health Check**: http://localhost:3000/api/health

## 🎨 Features

- ✅ Reusable Three.js component architecture
- ✅ LidarPoints component for 2D point cloud visualization
- ✅ Sample lidar data generation (circular pattern)
- ✅ Interactive controls (show/hide, regenerate data)
- ✅ OrbitControls for camera manipulation
- ✅ Modern glassmorphism UI
- ✅ Multi-stage Docker build for optimization
- ✅ Health check endpoint

## 🔌 Using the Components

### Example: Adding Lidar Visualization

```jsx
import ThreeScene from './components/ThreeScene';
import LidarPoints from './components/LidarPoints';
import { useState } from 'react';

function App() {
  const [sceneContext, setSceneContext] = useState(null);
  const [points, setPoints] = useState([
    { x: 1, y: 0, z: 2 },
    { x: -1, y: 0, z: 3 },
    // ... more points
  ]);

  return (
    <ThreeScene onSceneReady={setSceneContext}>
      {sceneContext && (
        <LidarPoints 
          points={points}
          scene={sceneContext.scene}
          color={0x00ff88}
          size={0.15}
          visible={true}
        />
      )}
    </ThreeScene>
  );
}
```

## 🔧 Next Steps

Ready for integration:
- Connect to UDP server for real-time lidar data
- Add WebSocket for live data streaming
- Import 3D car models
- Add multiple visualization modes
- Implement data recording/playback

## 📚 Tech Stack

- **React** - Component-based UI
- **Vite** - Fast build tool
- **Three.js** - 3D visualization
- **Express** - Web server
- **Docker** - Containerization
