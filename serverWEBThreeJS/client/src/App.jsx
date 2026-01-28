import { useState, useCallback } from 'react';
import ThreeScene from './components/ThreeScene';
import LidarPoints from './components/LidarPoints';
import useLidarSocket from './hooks/useLidarSocket';
import './App.css';

function App() {
  const [sceneContext, setSceneContext] = useState(null);
  const [showLidar, setShowLidar] = useState(true);
  const [useDemoData, setUseDemoData] = useState(false);
  const [demoPoints, setDemoPoints] = useState([]);

  // Connect to lidar via WebSocket
  // accumulate=true: builds up a complete 360° scan over time
  // angleResolution=1: one point per degree = up to 360 points
  const { points: lidarPoints, isConnected, lidarStatus, clearPoints } = useLidarSocket({
    maxPoints: 360,
    accumulate: true,
    angleResolution: 1
  });

  // Handle scene initialization
  const handleSceneReady = useCallback((context) => {
    setSceneContext(context);
    console.log('🎨 Three.js scene ready');
  }, []);

  // Generate sample lidar points for demonstration
  const generateDemoData = useCallback(() => {
    const points = [];
    const numPoints = 360;
    const radius = 5;

    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * Math.PI * 2;
      const distance = radius + Math.random() * 2;

      points.push({
        x: Math.cos(angle) * distance,
        y: 0.1,
        z: Math.sin(angle) * distance
      });
    }

    setDemoPoints(points);
    setUseDemoData(true);
  }, []);

  // Use real lidar data or demo data based on toggle
  const displayPoints = useDemoData ? demoPoints : lidarPoints;

  return (
    <div className="app">
      {/* Status Panel */}
      <div className="status-panel">
        <div className="status-item">
          <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
          WebSocket: {isConnected ? 'Connecté' : 'Déconnecté'}
        </div>
        <div className="status-item">
          <span className={`status-dot ${lidarStatus.connected ? 'connected' : 'disconnected'}`}></span>
          Lidar: {lidarStatus.connected ? 'Connecté' : 'Déconnecté'}
        </div>
        <div className="status-item">
          Points: {displayPoints.length}
        </div>
        <div className="status-controls">
          <button onClick={() => setShowLidar(!showLidar)}>
            {showLidar ? '🔴 Masquer' : '🟢 Afficher'} Lidar
          </button>
          <button onClick={clearPoints}>
            🗑️ Effacer
          </button>
          <button onClick={generateDemoData}>
            🎲 Démo
          </button>
          {useDemoData && (
            <button onClick={() => setUseDemoData(false)}>
              📡 Données réelles
            </button>
          )}
        </div>
      </div>

      {/* Three.js Scene - 2D Top-Down View */}
      <ThreeScene onSceneReady={handleSceneReady}>
        {sceneContext && (
          <LidarPoints
            points={displayPoints}
            scene={sceneContext.scene}
            color={0xff0000}
            size={8}
            visible={showLidar}
          />
        )}
      </ThreeScene>
    </div>
  );
}

export default App;
