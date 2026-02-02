import { useState, useCallback } from 'react';
import ThreeScene from './components/ThreeScene';
import LidarPoints from './components/LidarPoints';
import SSHTerminal from './components/SSHTerminal';
import './App.css';

function App() {
  const [sceneContext, setSceneContext] = useState(null);
  const [lidarData, setLidarData] = useState([]);
  const [showLidar, setShowLidar] = useState(true);

  // Handle scene initialization
  const handleSceneReady = useCallback((context) => {
    setSceneContext(context);
    console.log('🎨 Three.js scene ready');

    // Generate sample lidar data for demonstration
    generateSampleLidarData();
  }, []);

  // Generate sample lidar points in a circular pattern
  const generateSampleLidarData = () => {
    const points = [];
    const numPoints = 360;
    const radius = 5;

    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * Math.PI * 2;
      const distance = radius + Math.random() * 2; // Add some noise

      points.push({
        x: Math.cos(angle) * distance,
        y: 0.1, // Slightly above ground for visibility
        z: Math.sin(angle) * distance
      });
    }

    setLidarData(points);
  };

  return (
    <div className="app-container">
      {/* Left Panel: Visualization */}
      <div className="viz-panel">
        <ThreeScene onSceneReady={handleSceneReady}>
          {sceneContext && (
            <LidarPoints
              points={lidarData}
              scene={sceneContext.scene}
              color={0x00ff88}
              size={0.2}
              visible={showLidar}
            />
          )}
        </ThreeScene>

        {/* Viz Controls Overlay */}
        <div className="viz-controls">
          <button className="btn-icon" onClick={generateSampleLidarData} title="Regenerate Data">🔄</button>
          <button className="btn-icon" onClick={() => setShowLidar(!showLidar)} title="Toggle Lidar">👁️</button>
        </div>
      </div>

      {/* Right Panel: SSH Terminal */}
      <div className="terminal-panel">
        <SSHTerminal />
      </div>
    </div>
  );
}

export default App;
