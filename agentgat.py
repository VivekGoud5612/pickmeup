import React, { useEffect, useRef, useState } from 'react';

const CombatVisualizer = () => {
  const canvasRef = useRef(null);
  const gameStateRef = useRef(null); 
  const wsRef = useRef(null); 
  const logsEndRef = useRef(null); // Reference to auto-scroll the logs
  
  const [hudStats, setHudStats] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [logs, setLogs] = useState([]); // State to hold our terminal logs

  // Helper function to push logs to the terminal
  const addLog = (message, type = 'info') => {
    setLogs((prev) => {
      const newLogs = [...prev, { time: new Date().toLocaleTimeString(), message, type }];
      return newLogs.slice(-50); // Keep only the last 50 logs to prevent memory leaks
    });
  };

  // Auto-scroll the log container whenever a new log is added
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  useEffect(() => {
    wsRef.current = new WebSocket('ws://127.0.0.1:8000/ws/combat');

    wsRef.current.onopen = () => {
      console.log("WebSocket connected");
      addLog("System Online: Connected to PickMeUp Engine", "success");
    };

    wsRef.current.onclose = () => {
      addLog("System Offline: Connection closed", "error");
    };

    let frameCount = 0;
    wsRef.current.onmessage = (event) => {
      const state = JSON.parse(event.data);
      gameStateRef.current = state;
      
      frameCount++;
      
      // Throttle React State updates to every 5 frames
      if (frameCount % 5 === 0) {
          setHudStats(state.agents);
          // Optional: Log every 10 steps so it doesn't spam too fast
          if (state.step % 10 === 0) {
              addLog(`Processing Step ${state.step}...`, "info");
          }
      }
    };

    const renderLoop = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw Grid Lines
      ctx.strokeStyle = '#1E2532';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 20; i++) {
          ctx.beginPath(); ctx.moveTo(i * 20, 0); ctx.lineTo(i * 20, 400); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(0, i * 20); ctx.lineTo(400, i * 20); ctx.stroke();
      }

      // Draw Agents
      if (gameStateRef.current && gameStateRef.current.agents) {
        const agents = gameStateRef.current.agents;

        Object.entries(agents).forEach(([agentName, data]) => {
          const px = data.x * 20;
          const py = data.y * 20;

          // Agent Square
          ctx.fillStyle = agentName === "BOSS" ? '#F2685B' : '#4FE8C4';
          ctx.fillRect(px, py, 20, 20);

          // Agent Initials
          ctx.fillStyle = '#000000';
          ctx.font = 'bold 10px monospace';
          ctx.fillText(agentName.substring(0, 1), px + 6, py + 14);

          // HP Bar (Red)
          const maxHp = data.max_hp ? data.max_hp : (agentName === 'BOSS' ? 200 : 100); 
          const hpPercent = Math.max(0, data.hp / maxHp);
          ctx.fillStyle = '#ff4444';
          ctx.fillRect(px, py - 6, 20 * hpPercent, 4); 

          // Stamina Bar (Yellow)
          if (data.stamina !== undefined) {
              const maxStam = data.max_stamina ? data.max_stamina : 50;
              const staminaPercent = Math.max(0, data.stamina / maxStam);
              ctx.fillStyle = '#f6e05e';
              ctx.fillRect(px, py - 2, 20 * staminaPercent, 2); 
          } 
        });
      }

      requestAnimationFrame(renderLoop);
    };
    
    requestAnimationFrame(renderLoop);

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // --- BUTTON CONTROLS ---
  const handleStart = () => {
    setIsPlaying(true);
    addLog("Command Sent: START_GAME", "system");
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'start' }));
    }
  };

  const handleStop = () => {
    setIsPlaying(false);
    addLog("Command Sent: STOP_GAME", "system");
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'stop' }));
    }
  };

  // --- UI RENDER ---
  return (
    <div className="min-h-screen bg-[#0A0E17] text-slate-300 font-sans p-8">
      
      {/* Header & Controls */}
      <div className="max-w-6xl mx-auto flex justify-between items-end mb-8 border-b border-slate-800 pb-4">
        <div>
            <h2 className="text-[#4FE8C4] font-mono text-3xl font-bold tracking-widest drop-shadow-[0_0_8px_rgba(79,232,196,0.5)]">
            PICKMEUP
            </h2>
            <p className="text-slate-500 font-mono text-sm mt-1">Engine Inference Visualizer v1.0</p>
        </div>
        
        <div className="flex gap-4">
            <button 
                onClick={handleStart}
                disabled={isPlaying}
                className={`px-6 py-2 rounded-md font-bold tracking-wide transition-all ${
                    isPlaying 
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                    : 'bg-[#4FE8C4] text-slate-900 hover:bg-[#3bc2a1] hover:shadow-[0_0_15px_rgba(79,232,196,0.4)]'
                }`}
            >
                START ENGINE
            </button>
            <button 
                onClick={handleStop}
                disabled={!isPlaying}
                className={`px-6 py-2 rounded-md font-bold tracking-wide transition-all ${
                    !isPlaying 
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                    : 'bg-[#F2685B] text-slate-900 hover:bg-[#d6554a] hover:shadow-[0_0_15px_rgba(242,104,91,0.4)]'
                }`}
            >
                HALT
            </button>
        </div>
      </div>

      {/* Main Dashboard Layout */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Canvas + Logs */}
        <div className="lg:col-span-2 flex flex-col gap-6">
            
            {/* The Canvas Arena */}
            <div className="relative self-center">
                <div className="absolute -inset-1 bg-gradient-to-r from-[#4FE8C4]/20 to-[#F2685B]/20 blur-lg rounded-xl pointer-events-none"></div>
                <canvas 
                    ref={canvasRef} 
                    width={400} 
                    height={400} 
                    className="relative bg-[#12161D] border-2 border-[#2F3A4F] rounded-lg z-10"
                />
            </div>

            {/* Event Logs Terminal */}
            <div className="bg-[#12161D] border border-[#2F3A4F] rounded-lg p-4 h-64 flex flex-col">
                <h3 className="text-white font-bold mb-3 border-b border-slate-700 pb-2 text-sm">SYSTEM LOGS</h3>
                <div className="flex-grow overflow-y-auto font-mono text-xs space-y-2 pr-2">
                    {logs.map((log, idx) => (
                        <div key={idx} className="flex gap-3">
                            <span className="text-slate-500 shrink-0">[{log.time}]</span>
                            <span className={`
                                ${log.type === 'error' ? 'text-red-400' : ''}
                                ${log.type === 'success' ? 'text-[#4FE8C4]' : ''}
                                ${log.type === 'system' ? 'text-blue-400' : ''}
                                ${log.type === 'info' ? 'text-slate-300' : ''}
                            `}>
                                {log.message}
                            </span>
                        </div>
                    ))}
                    <div ref={logsEndRef} /> {/* Invisible anchor for auto-scroll */}
                </div>
            </div>
        </div>

        {/* Right Column: Live Telemetry */}
        <div className="bg-[#12161D] border border-[#2F3A4F] rounded-lg p-6 h-max">
            <h3 className="text-white font-bold mb-4 border-b border-slate-700 pb-2">LIVE TELEMETRY</h3>
            
            {hudStats ? (
                <div className="space-y-4">
                    {Object.entries(hudStats).map(([name, data]) => (
                        <div key={name} className="bg-slate-800/50 border border-slate-700/50 p-4 rounded-lg transition-colors">
                            <div className={`font-black text-lg mb-2 tracking-wider ${name === 'BOSS' ? 'text-[#F2685B]' : 'text-[#4FE8C4]'}`}>
                                {name}
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-sm font-mono">
                                <span className="bg-slate-900/80 px-2 py-1.5 rounded text-white flex justify-between items-center">
                                    <span className="text-slate-500 text-xs">HP</span> 
                                    <span>{Math.floor(data.hp)}</span>
                                </span>
                                <span className="bg-slate-900/80 px-2 py-1.5 rounded text-white flex justify-between items-center">
                                    <span className="text-slate-500 text-xs">POS</span> 
                                    <span>{data.x},{data.y}</span>
                                </span>
                                {name !== 'BOSS' && (
                                    <span className="bg-slate-900/80 px-2 py-1.5 rounded text-white flex justify-between items-center col-span-2">
                                        <span className="text-slate-500 text-xs">STAMINA</span> 
                                        <span>{Math.floor(data.stamina)}</span>
                                    </span>
                                )}
                                <span className="col-span-2 bg-[#4FE8C4]/10 text-[#4FE8C4] px-3 py-2 rounded mt-1 text-center font-bold">
                                  {data.action || 'AWAITING_INPUT'}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="flex items-center justify-center h-48 text-sm text-slate-500 animate-pulse font-mono">
                    Waiting for telemetry...
                </div>
            )}
        </div>
      </div>
    </div>
  );
};

export default CombatVisualizer;