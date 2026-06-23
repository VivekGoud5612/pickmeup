import React, { useState, useEffect } from 'react';

// Sub-Component: Metrics tracking for individual entities
function EntityStatCard({ name, stats, type }) {
  const isMonster = type === 'monster';
  // Calculate percentage for CSS width bars
  const hpPercentage = (stats.hp / stats.max) * 100;
  const staminaPercentage = (stats.stamina / stats.max_stamina) * 100;

  return (
    <div style={{
      backgroundColor: '#161B22',
      borderLeft: `4px solid ${isMonster ? '#FF453A' : '#58A6FF'}`,
      padding: '12px',
      borderRadius: '6px',
      marginBottom: '10px'
    }}>
      <h3 style={{ margin: '0 0 8px 0', color: isMonster ? '#FF453A' : '#7EE787' }}>
        {name.toUpperCase()} <span style={{ fontSize: '12px', color: '#8B949E' }}>({stats.x}, {stats.y})</span>
      </h3>
      
      {/* HP Telemetry */}
      <div style={{ fontSize: '12px', marginBottom: '4px' }}>HP: {stats.hp}/{stats.max}</div>
      <div style={{ width: '100%', backgroundColor: '#21262D', height: '6px', borderRadius: '3px', marginBottom: '8px', overflow: 'hidden' }}>
        <div style={{ width: `${hpPercentage}%`, backgroundColor: isMonster ? '#FF453A' : '#30D158', height: '100%', transition: 'width 0.1s ease' }} />
      </div>

      {/* Stamina Telemetry */}
      <div style={{ fontSize: '12px', marginBottom: '4px' }}>Stamina: {stats.stamina}/{stats.max_stamina}</div>
      <div style={{ width: '100%', backgroundColor: '#21262D', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
        <div style={{ width: `${staminaPercentage}%`, backgroundColor: '#0BF', height: '100%', transition: 'width 0.1s ease' }} />
      </div>
    </div>
  );
}

// Master Component
export default function App() {
  // 1. Initial State Definition
  const [gameState, setGameState] = useState({
    step: 0,
    heroes: {},
    monsters: {}
  });

  const GRID_SIZE = 20;

  // 2. The Network Listener (WebSocket)
  useEffect(() => {
    // Dialing your Uvicorn/FastAPI backend exactly on port 6545
    const ws = new WebSocket('http://127.0.0.1:8000/ws/combat');

    ws.onopen = () => {
      console.log('[*] Successfully connected to Python MAPPO Engine!');
    };

    // This triggers automatically every time Python does await websocket.send_text()
    ws.onmessage = (event) => {
      const incomingData = JSON.parse(event.data);
      setGameState(incomingData);
      console.log(event.data) // Triggers React to redraw the screen
    };

    ws.onerror = (error) => {
      console.error('[!] WebSocket Network Error:', error);
    };

    ws.onclose = () => {
      console.log('[*] Disconnected from engine.');
    };

    // Cleanup function when you close the React tab
    return () => {
      ws.close();
    };
  }, []); // Empty array ensures we only connect once

  // 3. Coordinate Projection Helper
  const renderEntityMarker = (name, data, isMonster) => {
    // Converts absolute coordinates (e.g., 5, 10) into CSS percentages (25%, 50%)
    const leftPos = (data.x / GRID_SIZE) * 100;
    const topPos = (data.y / GRID_SIZE) * 100;

    return (
      <div 
        key={name}
        style={{
          position: 'absolute',
          left: `${leftPos}%`,
          top: `${topPos}%`,
          width: isMonster ? '32px' : '26px',
          height: isMonster ? '32px' : '26px',
          backgroundColor: isMonster ? '#FF453A' : '#1F6FEB',
          border: '2px solid #FFFFFF',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#FFFFFF',
          fontWeight: 'bold',
          fontSize: '11px',
          transform: 'translate(-50%, -50%)', // Centers the circle exactly on the coordinate
          transition: 'left 0.1s linear, top 0.1s linear', // Smooth sliding animation
          boxShadow: '0 4px 8px rgba(0,0,0,0.4)'
        }}
        title={name}
      >
        {name[0]}
      </div>
    );
  };

  // 4. The Master Layout Output
  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#0D1117', color: '#C9D1D9', fontFamily: 'sans-serif', padding: '20px' }}>
      
      {/* LEFT COLUMN: MAP VIEWPORT */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <h2 style={{ marginBottom: '15px', color: '#58A6FF' }}>MAPPO Simulation Space (Step: {gameState.step})</h2>
        
        <div style={{
          position: 'relative', // Critical for absolute positioning of children
          width: '550px',
          height: '550px',
          backgroundColor: '#161B22',
          border: '2px solid #30363D',
          borderRadius: '8px'
        }}>
          {/* Loop through dictionary keys and render the dots */}
          {Object.entries(gameState.heroes).map(([name, data]) => renderEntityMarker(name, data, false))}
          {Object.entries(gameState.monsters).map(([name, data]) => renderEntityMarker(name, data, true))}
        </div>
      </div>

      {/* RIGHT COLUMN: METRICS SIDEBAR PANEL */}
      <div style={{ width: '340px', borderLeft: '1px solid #30363D', paddingLeft: '20px', overflowY: 'auto' }}>
        <h2 style={{ color: '#F0F6FC', paddingBottom: '10px', borderBottom: '1px solid #30363D', marginTop: '0' }}>Telemetry Panel</h2>
        
        <h4 style={{ color: '#58A6FF', margin: '15px 0 10px 0' }}>TEAM HEROES</h4>
        {Object.entries(gameState.heroes).map(([name, stats]) => (
          <EntityStatCard key={name} name={name} stats={stats} type="hero" />
        ))}

        <h4 style={{ color: '#FF453A', margin: '20px 0 10px 0' }}>TEAM MONSTERS</h4>
        {Object.entries(gameState.monsters).map(([name, stats]) => (
          <EntityStatCard key={name} name={name} stats={stats} type="monster" />
        ))}
      </div>

    </div>
  );
}