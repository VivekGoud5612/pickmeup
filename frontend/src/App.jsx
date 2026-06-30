import React, { useState, useEffect } from 'react';

function EntityStatCard({ name, stats, type }) {
  const isMonster = type === 'monster';
  const hpPercentage = (stats.hp / (stats.max_hp || 1)) * 100;
  const staminaPercentage = (stats.stamina / (stats.max_stamina || 1)) * 100;

  return (
    <div style={{ backgroundColor: '#161B22', borderLeft: `4px solid ${isMonster ? '#FF453A' : '#58A6FF'}`, padding: '12px', borderRadius: '6px', marginBottom: '10px' }}>
      <h3 style={{ margin: '0 0 8px 0', color: isMonster ? '#FF453A' : '#7EE787', fontSize: '14px' }}>
        {name.toUpperCase()} <span style={{ fontSize: '11px', color: '#8B949E' }}>({stats.x}, {stats.y})</span>
      </h3>
      
      <div style={{ fontSize: '11px', marginBottom: '4px', fontWeight: 'bold' }}>HP: {Math.floor(stats.hp)}/{Math.floor(stats.max_hp)}</div>
      <div style={{ width: '100%', backgroundColor: '#21262D', height: '6px', borderRadius: '3px', marginBottom: '8px', overflow: 'hidden' }}>
        <div style={{ width: `${Math.max(0, Math.min(100, hpPercentage))}%`, backgroundColor: isMonster ? '#FF453A' : '#30D158', height: '100%' }} />
      </div>

      <div style={{ fontSize: '11px', marginBottom: '4px', fontWeight: 'bold' }}>Stamina: {Math.floor(stats.stamina)}/{Math.floor(stats.max_stamina)}</div>
      <div style={{ width: '100%', backgroundColor: '#21262D', height: '6px', borderRadius: '3px', marginBottom: '10px', overflow: 'hidden' }}>
        <div style={{ width: `${Math.max(0, Math.min(100, staminaPercentage))}%`, backgroundColor: '#0BF', height: '100%' }} />
      </div>
    </div>
  );
}

export default function App() {
  const [gameState, setGameState] = useState({ step: 0, agents: {} });
  const [engineState, setEngineState] = useState('idle'); // 'idle', 'running', 'paused'
  
  const GRID_SIZE = 20;
  const BACKEND_URL = "http://127.0.0.1:8000/api";

  useEffect(() => {
    const ws = new WebSocket('ws://127.0.0.1:3001');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setGameState(data);
      if (data.step >= 200) setEngineState('paused');
    };
    return () => ws.close();
  }, []);

  const handleStartResume = async () => {
    setEngineState('running');
    try {
      const res = await fetch(`${BACKEND_URL}/start`, { method: 'POST' });
      const data = await res.json();
      console.log("[Backend]", data.status);
    } catch (err) {
      console.error(err);
      setEngineState('paused');
    }
  };

  const handleStop = async () => {
    setEngineState('paused');
    try {
      const res = await fetch(`${BACKEND_URL}/stop`, { method: 'POST' });
      const data = await res.json();
      console.log("[Backend]", data.status);
    } catch (err) {
      console.error(err);
    }
  };

  const handleReset = async () => {
    setEngineState('idle');
    try {
      const res = await fetch(`${BACKEND_URL}/reset`, { method: 'POST' });
      const data = await res.json();
      console.log("[Backend]", data.status);
    } catch (err) {
      console.error(err);
    }
  };

  const isAgentMonster = (name) => name.toLowerCase().includes('boss') || name.toLowerCase().includes('monster');
  const heroes = Object.entries(gameState.agents || {}).filter(([n]) => !isAgentMonster(n));
  const monsters = Object.entries(gameState.agents || {}).filter(([n]) => isAgentMonster(n));

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#0D1117', color: '#C9D1D9', fontFamily: 'sans-serif', padding: '20px', boxSizing: 'border-box' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <h2 style={{ marginBottom: '15px', color: '#58A6FF' }}>MAPPO Arena (Step: {gameState.step || 0})</h2>
        <div style={{
          position: 'relative', width: '550px', height: '550px', backgroundColor: '#161B22', border: '2px solid #30363D', borderRadius: '8px',
          backgroundImage: 'radial-gradient(#30363D 1px, transparent 1px)', backgroundSize: `${550 / GRID_SIZE}px ${550 / GRID_SIZE}px`
        }}>
          {Object.entries(gameState.agents || {}).map(([name, data]) => {
            const isMonster = isAgentMonster(name);
            return (
              <div key={name} style={{
                position: 'absolute', left: `${(data.x / GRID_SIZE) * 100}%`, top: `${(data.y / GRID_SIZE) * 100}%`,
                width: isMonster ? '32px' : '26px', height: isMonster ? '32px' : '26px',
                backgroundColor: isMonster ? '#FF453A' : '#1F6FEB', border: '2px solid #FFFFFF', borderRadius: '50%',
                display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#FFFFFF', fontWeight: 'bold', fontSize: '11px',
                transform: 'translate(-50%, -50%)', transition: 'left 0.1s linear, top 0.1s linear', zIndex: isMonster ? 10 : 5
              }}>
                {name.replace('AgentID.', '')[0].toUpperCase()}
              </div>
            );
          })}
        </div>
      </div>

      <div style={{ width: '340px', borderLeft: '1px solid #30363D', paddingLeft: '20px', overflowY: 'auto' }}>
        <h2 style={{ color: '#F0F6FC', paddingBottom: '10px', borderBottom: '1px solid #30363D', marginTop: '0' }}>Control Deck</h2>
        <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
          <button onClick={handleStartResume} disabled={engineState === 'running'} style={{ flex: 1, padding: '12px', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', backgroundColor: '#238636', color: '#FFF', border: 'none' }}>START</button>
          <button onClick={handleStop} disabled={engineState !== 'running'} style={{ flex: 1, padding: '12px', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', backgroundColor: '#DA3633', color: '#FFF', border: 'none' }}>STOP</button>
          <button onClick={handleReset} style={{ flex: 1, padding: '12px', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', backgroundColor: '#1F6FEB', color: '#FFF', border: 'none' }}>RESET</button>
        </div>
        {heroes.map(([name, stats]) => <EntityStatCard key={name} name={name.replace('AgentID.', '')} stats={stats} type="hero" />)}
        {monsters.map(([name, stats]) => <EntityStatCard key={name} name={name.replace('AgentID.', '')} stats={stats} type="monster" />)}
      </div>
    </div>
  );
}