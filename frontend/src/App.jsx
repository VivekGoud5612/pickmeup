import React, { useState, useEffect } from 'react';

// Sub-Component: Metrics tracking for individual entities
function EntityStatCard({ name, stats, type }) {
  const isMonster = type === 'monster';
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
      
      <div style={{ fontSize: '12px', marginBottom: '4px' }}>HP: {stats.hp}/{stats.max}</div>
      <div style={{ width: '100%', backgroundColor: '#21262D', height: '6px', borderRadius: '3px', marginBottom: '8px', overflow: 'hidden' }}>
        <div style={{ width: `${hpPercentage}%`, backgroundColor: isMonster ? '#FF453A' : '#30D158', height: '100%', transition: 'width 0.1s ease' }} />
      </div>

      <div style={{ fontSize: '12px', marginBottom: '4px' }}>Stamina: {stats.stamina}/{stats.max_stamina}</div>
      <div style={{ width: '100%', backgroundColor: '#21262D', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
        <div style={{ width: `${staminaPercentage}%`, backgroundColor: '#0BF', height: '100%', transition: 'width 0.1s ease' }} />
      </div>
    </div>
  );
}

// Master Component
export default function App() {
  const [gameState, setGameState] = useState({
    step: 0,
    heroes: {},
    monsters: {}
  });
  const [isEngineRunning, setIsEngineRunning] = useState(false);

  const GRID_SIZE = 20;

  // 1. The Real-Time Network Listener (WebSocket to Node.js)
  useEffect(() => {
    const ws = new WebSocket('ws://127.0.0.1:3001');

    ws.onopen = () => {
      console.log('[*] Connected to Node.js Real-Time WebSocket Gateway!');
    };

    ws.onmessage = (event) => {
      const incomingData = JSON.parse(event.data);
      setGameState(incomingData);

      // FIX: Seamlessly unlock the UI button when the engine reaches the final step (Step 200)
      if (incomingData.step >= 200) {
        setIsEngineRunning(false);
        console.log('[*] Engine broadcast completed. Resetting UI ignition layout.');
      }
    };

    ws.onerror = (error) => {
      console.error('[!] WebSocket Network Error:', error);
    };

    ws.onclose = () => {
      console.log('[*] Disconnected from Node.js Gateway.');
    };

    return () => {
      ws.close();
    };
  }, []); 

  // 2. The Ignition Handler (HTTP Fetch to Python FastAPI)
  const startSimulation = async () => {
    if (isEngineRunning) return;
    
    setIsEngineRunning(true);
    console.log('[*] Sending HTTP Trigger to Python MAPPO Core...');
    
    try {
      // Direct call to your FastAPI /api/start endpoint on Port 8000
      const response = await fetch('http://127.0.0.1:8000/api/start', { method: 'POST' });
      if (response.ok) {
        console.log('[+] Python MAPPO Engine acknowledges launch request!');
      } else {
        console.error('[!] Python engine rejected launch request.');
        setIsEngineRunning(false);
      }
    } catch (error) {
      console.error('[!] Failed to reach Python server via HTTP:', error);
      setIsEngineRunning(false);
    }
  };

  const renderEntityMarker = (name, data, isMonster) => {
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
          transform: 'translate(-50%, -50%)',
          transition: 'left 0.1s linear, top 0.1s linear',
          boxShadow: '0 4px 8px rgba(0,0,0,0.4)'
        }}
        title={name}
      >
        {name[0]}
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#0D1117', color: '#C9D1D9', fontFamily: 'sans-serif', padding: '20px' }}>
      
      {/* LEFT COLUMN: MAP VIEWPORT */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <h2 style={{ marginBottom: '15px', color: '#58A6FF' }}>MAPPO Simulation Space (Step: {gameState.step})</h2>
        
        <div style={{
          position: 'relative',
          width: '550px',
          height: '550px',
          backgroundColor: '#161B22',
          border: '2px solid #30363D',
          borderRadius: '8px'
        }}>
          {Object.entries(gameState.heroes).map(([name, data]) => renderEntityMarker(name, data, false))}
          {Object.entries(gameState.monsters).map(([name, data]) => renderEntityMarker(name, data, true))}
        </div>
      </div>

      {/* RIGHT COLUMN: METRICS & CONTROLS SIDEBAR PANEL */}
      <div style={{ width: '340px', borderLeft: '1px solid #30363D', paddingLeft: '20px', overflowY: 'auto' }}>
        <h2 style={{ color: '#F0F6FC', paddingBottom: '10px', borderBottom: '1px solid #30363D', marginTop: '0' }}>Telemetry Panel</h2>
        
        {/* PHYSICAL IGNITION BUTTON */}
        <button
          onClick={startSimulation}
          disabled={isEngineRunning}
          style={{
            width: '100%',
            padding: '14px',
            backgroundColor: isEngineRunning ? '#21262D' : '#238636',
            color: isEngineRunning ? '#8B949E' : '#FFFFFF',
            border: '1px solid rgba(240,246,252,0.1)',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: 'bold',
            cursor: isEngineRunning ? 'not-allowed' : 'pointer',
            marginBottom: '20px',
            transition: 'background-color 0.2s',
            boxShadow: '0 4px 12px rgba(0,0,0,0.2)'
          }}
        >
          {isEngineRunning ? '🚀 SIMULATION IN PROGRESS...' : '⚡ START MAPPO ENGINE'}
        </button>

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
