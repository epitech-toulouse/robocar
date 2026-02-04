import { useState, useCallback } from 'react';
import ThreeScene from './components/ThreeScene';
import LidarPoints from './components/LidarPoints';
import SSHTerminal from './components/SSHTerminal';
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
    console.log('Three.js scene ready');
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
    <div className="app-container">
      {/* Left Panel: Visualization */}
      <div className="viz-panel">

        {/* Status Panel Overlay */}
        <div className="status-panel-overlay">
          <div className="status-row">
            <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
            WS: {isConnected ? 'OK' : 'KO'}
          </div>
          <div className="status-row">
            <span className={`status-dot ${lidarStatus.connected ? 'connected' : 'disconnected'}`}></span>
            Lidar: {lidarStatus.connected ? 'OK' : 'KO'}
          </div>
          <div className="status-controls">
            <button className="btn-icon" onClick={() => setShowLidar(!showLidar)} title={showLidar ? "Hide Lidar" : "Show Lidar"}>
              {showLidar ? 'Hide' : 'Show'}
            </button>
            <button className="btn-icon" onClick={clearPoints} title="Clear Points">Clear</button>
            <button className="btn-icon" onClick={generateDemoData} title="Demo Data">Demo</button>
            {useDemoData && (
              <button className="btn-icon" onClick={() => setUseDemoData(false)} title="Real Data">Real</button>
            )}
          </div>
        </div>

        <ThreeScene onSceneReady={handleSceneReady}>
          {sceneContext && (
            <LidarPoints
              points={displayPoints}
              scene={sceneContext.scene}
              color={useDemoData ? 0x00ffff : 0xff0000}
              size={0.2}
              visible={showLidar}
            />
          )}
        </ThreeScene>
      </div>

      {/* Right Panel: SSH Terminal */}
      <div className="terminal-panel">
        <SSHTerminal />
      </div>
    </div>
  );
}

export default App;
