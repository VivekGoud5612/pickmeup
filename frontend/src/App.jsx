import React, { useEffect, useRef, useState } from 'react';

export default function App() {
  const canvasRef = useRef(null);
  const wsRef = useRef(null); 
  const logsEndRef = useRef(null);
  
  // Base State
  const [hudStats, setHudStats] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [logs, setLogs] = useState([]);
  
  // Episode & Evaluation State
  const [episodeNum, setEpisodeNum] = useState(1);
  const [stepNum, setStepNum] = useState(0);
  const [currentReward, setCurrentReward] = useState(0);
  const [speed, setSpeed] = useState(2); // 0.5, 1, 2, 5 FPS
  
  // Eval Mode State
  const [isEvalMode, setIsEvalMode] = useState(false);
  const [gamesRemaining, setGamesRemaining] = useState(0);

  // Running Statistics
  const [stats, setStats] = useState({
    gamesPlayed: 0,
    heroesWins: 0,
    bossWins: 0,
    totalReturn: 0,
    totalSteps: 0
  });

  // Keep a ref of previous states to track action changes for the terminal
  const prevActionsRef = useRef({});

  // --- LOGGING SYSTEM ---
  const addLog = (message, type = 'info') => {
    setLogs((prev) => {
      const newLogs = [...prev, { time: new Date().toLocaleTimeString('en-US', { hour12: false, hour: "numeric", minute: "numeric", second: "numeric" }), message, type }];
      return newLogs.slice(-100); 
    });
  };

  useEffect(() => {
    if (logsEndRef.current) logsEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // --- WEBSOCKET ENGINE ---
  useEffect(() => {
    wsRef.current = new WebSocket('ws://127.0.0.1:8000/ws/combat');

    wsRef.current.onopen = () => {
      addLog("System Online: Connected to Backend Engine", "success");
    };

    wsRef.current.onclose = () => {
      addLog("System Offline: Connection closed", "error");
      setIsPlaying(false);
    };

    wsRef.current.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      
      setHudStats(payload.agents);
      setStepNum(payload.step);
      if (payload.episode_reward !== undefined) setCurrentReward(payload.episode_reward.toFixed(2));

      // Terminal Action Logging
      if (payload.agents) {
        Object.entries(payload.agents).forEach(([name, data]) => {
          if (data.action && data.action !== prevActionsRef.current[name]) {
            addLog(`${name} -> ${data.action}`, "action");
            prevActionsRef.current[name] = data.action;
          }
        });
      }

      // Handle Episode End
      if (payload.done) {
        const winner = payload.winner || "UNKNOWN";
        const finalReturn = payload.final_return || 0;
        
        addLog(`Episode ${episodeNum} Finished`, "system");
        addLog(`Winner: ${winner} | Return: ${finalReturn.toFixed(2)}`, winner === "HEROES" ? "success" : "error");

        // Update Running Stats
        setStats(prev => ({
          gamesPlayed: prev.gamesPlayed + 1,
          heroesWins: prev.heroesWins + (winner === "HEROES" ? 1 : 0),
          bossWins: prev.bossWins + (winner === "BOSS" ? 1 : 0),
          totalReturn: prev.totalReturn + finalReturn,
          totalSteps: prev.totalSteps + payload.step
        }));

        setEpisodeNum(prev => prev + 1);

        // Handle 10-Game Mode Logic
        setGamesRemaining(prev => {
          if (prev > 1) {
            addLog(`Eval Mode: Starting Game ${episodeNum + 1}...`, "info");
            return prev - 1;
          } else if (prev === 1) {
            // Eval Mode Finished
            setIsPlaying(false);
            setIsEvalMode(false);
            wsRef.current.send(JSON.stringify({ command: 'stop' }));
            addLog(`[EVALUATION COMPLETE]`, "system");
            return 0;
          }
          return 0; // Not in eval mode
        });
      }
    };

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [episodeNum]); // episodeNum dependency helps track game counts

  // --- CANVAS RENDERER ---
  useEffect(() => {
    if (!hudStats || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Grid Constants
    const CELL_SIZE = 20;
    const GRID_SIZE = 20;

    // Clear Arena
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Grid Coordinates & Lines
    ctx.strokeStyle = '#1E2532';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#475569';
    ctx.font = '8px monospace';
    
    for (let i = 0; i <= GRID_SIZE; i++) {
        ctx.beginPath(); ctx.moveTo(i * CELL_SIZE, 0); ctx.lineTo(i * CELL_SIZE, 400); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i * CELL_SIZE); ctx.lineTo(400, i * CELL_SIZE); ctx.stroke();
        // Light coordinates on edge
        if (i < GRID_SIZE) {
          ctx.fillText(i, i * CELL_SIZE + 2, 8);
          if (i > 0) ctx.fillText(i, 2, i * CELL_SIZE + 8);
        }
    }

    // 2. Draw Agents
    Object.entries(hudStats).forEach(([agentName, data]) => {
      const isDead = data.hp <= 0;
      ctx.globalAlpha = isDead ? 0.2 : 1.0; // Dead opacity
      
      const px = data.x * CELL_SIZE;
      const py = data.y * CELL_SIZE;
      const isBoss = agentName === "BOSS";

      // Sprite Sizing (Boss is 1.5x larger)
      const spriteSize = isBoss ? CELL_SIZE * 1.5 : CELL_SIZE;
      const offset = isBoss ? -(CELL_SIZE * 0.25) : 0; 
      const drawX = px + offset;
      const drawY = py + offset;

      // Draw Block
      ctx.fillStyle = isBoss ? '#F2685B' : '#4FE8C4';
      ctx.fillRect(drawX, drawY, spriteSize, spriteSize);

      // Draw Initials
      ctx.fillStyle = '#0A0E17';
      ctx.font = `bold ${isBoss ? 14 : 10}px monospace`;
      ctx.fillText(agentName.substring(0, 1), drawX + (isBoss ? 8 : 6), drawY + (isBoss ? 20 : 14));

      if (!isDead) {
        // Draw Action Text above head
        ctx.fillStyle = '#94A3B8';
        ctx.font = '8px monospace';
        ctx.fillText(data.action || '', drawX, drawY - 10);

        // HP Bar
        const maxHp = data.max_hp || (isBoss ? 200 : 100); 
        const hpPercent = Math.max(0, data.hp / maxHp);
        ctx.fillStyle = '#ef4444';
        ctx.fillRect(drawX, drawY - 6, spriteSize * hpPercent, 3); 

        // Stamina Bar
        if (data.stamina !== undefined) {
            const maxStam = data.max_stamina || 50;
            const staminaPercent = Math.max(0, data.stamina / maxStam);
            ctx.fillStyle = '#facc15';
            ctx.fillRect(drawX, drawY - 2, spriteSize * staminaPercent, 2); 
        }
      }
      ctx.globalAlpha = 1.0; // Reset alpha
    });

  }, [hudStats]);

  // --- CONTROLS ---
  const handleStart = () => {
    setIsPlaying(true);
    addLog(`Command: START [${speed} FPS]`, "system");
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'start', speed }));
    }
  };

  const handleStop = () => {
    setIsPlaying(false);
    setIsEvalMode(false);
    setGamesRemaining(0);
    addLog("Command: STOP", "system");
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'stop' }));
    }
  };

  const handleEvalMode = () => {
    setIsPlaying(true);
    setIsEvalMode(true);
    setGamesRemaining(10);
    addLog(`Command: 10-GAME EVALUATION STARTED`, "system");
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'start', speed }));
    }
  };

  const changeSpeed = (newSpeed) => {
    setSpeed(newSpeed);
    if (isPlaying && wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'speed', speed: newSpeed }));
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0E17] text-slate-300 font-sans p-4 flex flex-col h-screen overflow-hidden">
      
      {/* TOP BAR: Controls & Running Stats */}
      <div className="flex justify-between items-start mb-4 border-b border-slate-800 pb-4 shrink-0">
        
        {/* Left: Branding & Core Controls */}
        <div className="flex gap-6 items-center">
            <div>
              <h1 className="text-2xl font-black text-[#4FE8C4] tracking-widest drop-shadow-[0_0_8px_rgba(79,232,196,0.3)]">
                PICKMEUP <span className="text-slate-600 font-normal">| EVAL</span>
              </h1>
              <p className="text-slate-500 font-mono text-[10px] uppercase mt-1">MAPPO Checkpoint: 4.8M Steps</p>
            </div>
            
            <div className="flex bg-[#12161D] rounded-lg border border-slate-800 p-1 gap-1">
              <button onClick={handleStart} disabled={isPlaying} className={`px-4 py-1.5 rounded font-bold font-mono text-xs ${isPlaying ? 'text-slate-600' : 'bg-[#4FE8C4]/10 text-[#4FE8C4] hover:bg-[#4FE8C4]/20'}`}>START</button>
              <button onClick={handleStop} disabled={!isPlaying} className={`px-4 py-1.5 rounded font-bold font-mono text-xs ${!isPlaying ? 'text-slate-600' : 'bg-red-500/10 text-red-500 hover:bg-red-500/20'}`}>STOP</button>
              <div className="w-px bg-slate-800 mx-1"></div>
              <button onClick={handleEvalMode} disabled={isPlaying} className={`px-4 py-1.5 rounded font-bold font-mono text-xs ${isPlaying ? 'text-slate-600' : 'bg-purple-500/10 text-purple-400 hover:bg-purple-500/20'}`}>10 GAMES</button>
            </div>

            {/* Speed Selector */}
            <div className="flex items-center gap-2 bg-[#12161D] rounded-lg border border-slate-800 px-3 py-1.5">
              <span className="text-slate-500 font-mono text-xs">SPEED:</span>
              {[0.5, 1, 2, 5].map(s => (
                <button key={s} onClick={() => changeSpeed(s)} className={`px-2 py-0.5 rounded font-mono text-xs ${speed === s ? 'bg-blue-500/20 text-blue-400' : 'text-slate-600 hover:text-slate-400'}`}>
                  {s}x
                </button>
              ))}
            </div>
        </div>

        {/* Right: Running Statistics */}
        <div className="flex gap-4">
          <StatBox label="GAMES" value={stats.gamesPlayed} />
          <StatBox label="HERO WINS" value={stats.heroesWins} color="text-[#4FE8C4]" />
          <StatBox label="BOSS WINS" value={stats.bossWins} color="text-[#F2685B]" />
          <StatBox label="AVG REWARD" value={stats.gamesPlayed ? (stats.totalReturn / stats.gamesPlayed).toFixed(2) : '0.00'} />
          <StatBox label="AVG LENGTH" value={stats.gamesPlayed ? Math.floor(stats.totalSteps / stats.gamesPlayed) : '0'} />
        </div>
      </div>

      {/* MAIN LAYOUT: 3 Columns */}
      <div className="flex flex-1 gap-6 min-h-0">
        
        {/* LEFT PANEL: Episode Info */}
        <div className="w-64 bg-[#0D111A] border border-[#1E2532] rounded-xl p-5 flex flex-col gap-6 shadow-lg shrink-0">
           <div>
             <h3 className="text-slate-500 font-mono text-xs tracking-widest border-b border-slate-800 pb-2 mb-4">EPISODE STATUS</h3>
             <div className="text-4xl font-black text-white mb-1">#{episodeNum}</div>
             <div className={`text-xs font-mono font-bold px-2 py-1 inline-block rounded ${isPlaying ? 'bg-green-500/10 text-green-400' : 'bg-slate-800 text-slate-500'}`}>
               {isPlaying ? (isEvalMode ? `EVAL RUNNING (${gamesRemaining} LEFT)` : 'RUNNING') : 'HALTED'}
             </div>
           </div>

           <div className="space-y-4">
             <InfoRow label="CURRENT STEP" value={stepNum} />
             <InfoRow label="EPISODE REWARD" value={currentReward} color={currentReward > 0 ? 'text-green-400' : 'text-red-400'} />
           </div>
        </div>

        {/* CENTER PANEL: Arena & Terminal */}
        <div className="flex-1 flex flex-col gap-6 min-w-0">
            {/* The Arena Canvas */}
            <div className="flex-1 flex items-center justify-center bg-[#0D111A] border border-[#1E2532] rounded-xl relative shadow-inner overflow-hidden">
              <canvas 
                  ref={canvasRef} 
                  width={400} 
                  height={400} 
                  className="bg-[#0A0E17] border border-slate-800 shadow-[0_0_30px_rgba(0,0,0,0.5)]"
              />
            </div>

            {/* Bottom Terminal */}
            <div className="h-48 bg-[#0D111A] border border-[#1E2532] rounded-xl p-3 flex flex-col shadow-inner shrink-0">
                <h3 className="text-slate-500 font-mono text-[10px] tracking-widest border-b border-slate-800 pb-1 mb-2">RAW EVENT LOG</h3>
                <div className="flex-1 overflow-y-auto font-mono text-xs space-y-1 custom-scrollbar pr-2">
                    {logs.map((log, idx) => (
                        <div key={idx} className="flex gap-3 items-start leading-relaxed">
                            <span className="text-slate-600 shrink-0">[{log.time}]</span>
                            <span className={`
                                ${log.type === 'error' ? 'text-red-400' : ''}
                                ${log.type === 'success' ? 'text-emerald-400 font-bold' : ''}
                                ${log.type === 'system' ? 'text-blue-400 font-bold' : ''}
                                ${log.type === 'action' ? 'text-slate-300' : ''}
                                ${log.type === 'info' ? 'text-slate-500' : ''}
                            `}>
                                {log.message}
                            </span>
                        </div>
                    ))}
                    <div ref={logsEndRef} />
                </div>
            </div>
        </div>

        {/* RIGHT PANEL: Live Agent State */}
        <div className="w-80 bg-[#0D111A] border border-[#1E2532] rounded-xl p-5 shadow-lg overflow-y-auto custom-scrollbar shrink-0">
            <h3 className="text-slate-500 font-mono text-xs tracking-widest border-b border-slate-800 pb-2 mb-4 flex justify-between">
              LIVE AGENT STATE
              <span className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></span>
            </h3>
            
            {hudStats ? (
                <div className="space-y-4">
                    {Object.entries(hudStats).map(([name, data]) => (
                        <div key={name} className={`bg-[#12161D] border p-3 rounded-lg ${data.hp <= 0 ? 'border-red-900/50 opacity-50' : 'border-slate-800'}`}>
                            <div className="flex justify-between items-center mb-2">
                                <span className={`font-black tracking-wider ${name === 'BOSS' ? 'text-[#F2685B]' : 'text-[#4FE8C4]'}`}>
                                    {name} {data.hp <= 0 && '(DEAD)'}
                                </span>
                                <span className="text-[10px] font-mono text-slate-500 bg-slate-900 px-1.5 py-0.5 rounded">
                                    [{data.x}, {data.y}]
                                </span>
                            </div>

                            <div className="space-y-1.5 text-xs font-mono">
                                <BarRow label="HP" value={data.hp} max={data.max_hp || (name==='BOSS'?200:100)} color="bg-red-500" />
                                {name !== 'BOSS' && (
                                  <BarRow label="STM" value={data.stamina} max={data.max_stamina || 50} color="bg-yellow-400" />
                                )}
                            </div>

                            <div className="mt-2 pt-2 border-t border-slate-800/50 flex justify-between items-center bg-slate-900/50 px-2 py-1 rounded">
                                <span className="text-slate-500 text-[10px] tracking-widest">ACT</span>
                                <span className="text-[#4FE8C4] text-xs font-bold">{data.action || 'IDLE'}</span>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="flex items-center justify-center h-full text-slate-500 font-mono text-sm animate-pulse">Waiting...</div>
            )}
        </div>

      </div>
    </div>
  );
}

// --- HELPER COMPONENTS ---
const StatBox = ({ label, value, color = 'text-white' }) => (
  <div className="bg-[#12161D] border border-slate-800 rounded px-4 py-2 flex flex-col items-center min-w-[80px]">
    <span className="text-slate-500 font-mono text-[9px] tracking-widest mb-1">{label}</span>
    <span className={`font-black text-lg ${color}`}>{value}</span>
  </div>
);

const InfoRow = ({ label, value, color = 'text-white' }) => (
  <div className="flex justify-between items-center font-mono border-b border-slate-800/50 pb-1">
    <span className="text-slate-500 text-xs">{label}</span>
    <span className={`text-sm font-bold ${color}`}>{value}</span>
  </div>
);

const BarRow = ({ label, value, max, color }) => (
  <div className="flex items-center gap-2">
      <span className="text-slate-500 w-6">{label}</span>
      <div className="flex-1 bg-slate-800 h-1.5 rounded-full overflow-hidden">
          <div className={`${color} h-full transition-all duration-300`} style={{ width: `${Math.max(0, (value / max) * 100)}%` }} />
      </div>
      <span className="text-slate-300 w-6 text-right">{Math.floor(value)}</span>
  </div>
);
