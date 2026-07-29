import React, { useState, useEffect, useRef } from 'react';

// ==========================================
// 1. CONFIGURATION & API SERVICES
// ==========================================
const API_BASE_URL = 'http://localhost:8000/api';
const WS_URL = 'ws://localhost:3001';

const apiService = {
  sendCommand: async (endpoint) => {
    try {
      const response = await fetch(`${API_BASE_URL}/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error(`Error sending command to /${endpoint}:`, error);
      throw error;
    }
  },
  start: () => apiService.sendCommand('start'),
  stop: () => apiService.sendCommand('stop'),
  reset: () => apiService.sendCommand('reset'),
};

// ==========================================
// 2. CUSTOM HOOKS
// ==========================================
const useWebSocket = (url) => {
  const [gameState, setGameState] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  useEffect(() => {
    let isMounted = true;

    const connect = () => {
      if (socketRef.current?.readyState === WebSocket.OPEN) return;

      const socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => {
        if (!isMounted) return;
        setIsConnected(true);
      };

      socket.onmessage = (event) => {
        if (!isMounted) return;
        try {
          const payload = JSON.parse(event.data);
          setGameState(payload);
        } catch (err) {
          console.error('Failed to parse WS message:', err);
        }
      };

      socket.onclose = () => {
        if (!isMounted) return;
        setIsConnected(false);
        // Auto-reconnect every 3 seconds if the Node.js server goes down
        reconnectTimeoutRef.current = setTimeout(connect, 3000); 
      };

      socket.onerror = (error) => {
        console.error('WebSocket Error:', error);
        socket.close();
      };
    };

    connect();

    return () => {
      isMounted = false;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (socketRef.current) socketRef.current.close();
    };
  }, [url]);

  return { gameState, isConnected };
};

// ==========================================
// 3. UI COMPONENTS
// ==========================================

const ControlPanel = ({ isConnected }) => {
  const [loading, setLoading] = useState(null);

  const handleAction = async (actionFn, actionName) => {
    setLoading(actionName);
    try {
      await actionFn();
    } catch (err) {
      alert(`Failed to execute ${actionName}. Is FastAPI running on port 8000?`);
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 shadow-lg">
      <div className="flex items-center space-x-3">
        <span className={`h-3 w-3 rounded-full ${isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
        <span className="text-sm font-medium text-slate-300">
          WS Gateway: {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={() => handleAction(apiService.start, 'Start')}
          disabled={loading !== null}
          className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 text-white font-semibold rounded-lg transition active:scale-95"
        >
          {loading === 'Start' ? '...' : 'Start'}
        </button>
        <button
          onClick={() => handleAction(apiService.stop, 'Pause')}
          disabled={loading !== null}
          className="px-5 py-2 bg-amber-600 hover:bg-amber-500 disabled:bg-slate-700 text-white font-semibold rounded-lg transition active:scale-95"
        >
          {loading === 'Pause' ? '...' : 'Pause'}
        </button>
        <button
          onClick={() => handleAction(apiService.reset, 'Reset')}
          disabled={loading !== null}
          className="px-5 py-2 bg-rose-600 hover:bg-rose-500 disabled:bg-slate-700 text-white font-semibold rounded-lg transition active:scale-95"
        >
          {loading === 'Reset' ? '...' : 'Reset'}
        </button>
      </div>
    </div>
  );
};

const GameGrid = ({ agents = {} }) => {
  const GRID_SIZE = 20;
  
  // Adjusted styles to match your agent names exactly as they might come from AgentID enum
  const AGENT_STYLES = {
    Tank: 'bg-blue-500 text-white border-blue-300',
    Dealer: 'bg-emerald-500 text-white border-emerald-300',
    Healer: 'bg-yellow-400 text-slate-900 border-yellow-200',
    Boss: 'bg-red-600 text-white border-red-400',
    // Uppercase fallbacks
    TANK: 'bg-blue-500 text-white border-blue-300',
    DEALER: 'bg-emerald-500 text-white border-emerald-300',
    HEALER: 'bg-yellow-400 text-slate-900 border-yellow-200',
    BOSS: 'bg-red-600 text-white border-red-400',
  };

  const agentPositions = {};
  if (agents && typeof agents === 'object') {
    Object.entries(agents).forEach(([role, data]) => {
      if (data && data.x != null && data.y != null) {
        const key = `${data.x},${data.y}`;
        if (!agentPositions[key]) agentPositions[key] = [];
        agentPositions[key].push({ role, ...data });
      }
    });
  }

  const cells = [];
  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      const occupants = agentPositions[`${x},${y}`] || [];
      cells.push(
        <div key={`${x}-${y}`} className="bg-slate-900/60 border border-slate-800/80 aspect-square flex items-center justify-center rounded-sm">
          {occupants.map((agent) => (
            <div
              key={agent.role}
              className={`w-4/5 h-4/5 rounded flex items-center justify-center font-extrabold text-[10px] shadow-md border ${AGENT_STYLES[agent.role] || 'bg-gray-500'}`}
              title={`${agent.role} HP: ${agent.hp}`}
            >
              {agent.role.charAt(0).toUpperCase()}
            </div>
          ))}
        </div>
      );
    }
  }

  return (
    <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-xl flex flex-col items-center">
      {/* Explicit Grid Style Fix: Prevents 400 vertical rows */}
      <div 
        className="w-full max-w-[500px] aspect-square gap-[1px] bg-slate-950 p-1 rounded-lg"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${GRID_SIZE}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${GRID_SIZE}, minmax(0, 1fr))`
        }}
      >
        {cells}
      </div>
      <div className="flex flex-wrap justify-center gap-4 mt-4 text-xs font-semibold text-slate-300">
        <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-blue-500" /> Tank</div>
        <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-emerald-500" /> Dealer</div>
        <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-yellow-400" /> Healer</div>
        <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-red-600" /> Boss</div>
      </div>
    </div>
  );
};

const HUD = ({ step = 0, agents = {} }) => {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow-xl flex flex-col gap-5">
      <div className="flex justify-between items-center border-b border-slate-700 pb-3">
        <h2 className="text-lg font-bold text-slate-100 uppercase">Simulation State</h2>
        <div className="bg-slate-900 border border-slate-700 px-4 py-1.5 rounded-lg">
          <span className="text-xs text-slate-400 uppercase font-semibold mr-2">Step</span>
          <span className="text-xl font-mono font-bold text-indigo-400">{step}/200</span>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {!agents || Object.keys(agents).length === 0 ? (
          <p className="text-slate-500 text-sm col-span-2 text-center py-6">Awaiting simulation data...</p>
        ) : (
          Object.entries(agents).map(([role, stats]) => {
            const hpPct = stats.max_hp ? Math.max(0, Math.min(100, (stats.hp / stats.max_hp) * 100)) : 0;
            const stamPct = stats.max_stamina ? Math.max(0, Math.min(100, (stats.stamina / stats.max_stamina) * 100)) : 0;

            return (
              <div key={role} className="bg-slate-900/80 border border-slate-700/80 rounded-lg p-4 flex flex-col gap-3">
                <div className="flex justify-between items-center">
                  <span className="px-2.5 py-0.5 rounded text-xs font-black tracking-wider border bg-slate-800 text-slate-200 border-slate-600 uppercase">
                    {role}
                  </span>
                  <span className="text-xs font-mono text-slate-400">Pos: ({stats.x ?? '-'}, {stats.y ?? '-'})</span>
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400 font-medium">HP</span>
                    <span className="font-mono text-slate-200">{Math.round(stats.hp)} / {Math.round(stats.max_hp)}</span>
                  </div>
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                    <div className="bg-rose-500 h-full transition-all duration-300" style={{ width: `${hpPct}%` }} />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400 font-medium">Stamina</span>
                    <span className="font-mono text-slate-200">{Math.round(stats.stamina)} / {Math.round(stats.max_stamina)}</span>
                  </div>
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                    <div className="bg-sky-500 h-full transition-all duration-300" style={{ width: `${stamPct}%` }} />
                  </div>
                </div>

                {stats.cooldowns && (
                  <div className="mt-1 pt-2 border-t border-slate-800 grid grid-cols-3 gap-2 text-center">
                    <div className="bg-slate-800/60 p-1 rounded">
                      <div className="text-[10px] text-slate-400 uppercase">Basic</div>
                      <div className="font-mono font-bold text-slate-200 text-xs">{Math.round(stats.cooldowns.basic)}s</div>
                    </div>
                    <div className="bg-slate-800/60 p-1 rounded">
                      <div className="text-[10px] text-slate-400 uppercase">Utility</div>
                      <div className="font-mono font-bold text-slate-200 text-xs">{Math.round(stats.cooldowns.utility)}s</div>
                    </div>
                    <div className="bg-slate-800/60 p-1 rounded">
                      <div className="text-[10px] text-slate-400 uppercase">Ultimate</div>
                      <div className="font-mono font-bold text-slate-200 text-xs">{Math.round(stats.cooldowns.ultimate)}s</div>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

// ==========================================
// 4. MAIN APPLICATION COMPONENT
// ==========================================
export default function App() {
  const { gameState, isConnected } = useWebSocket(WS_URL);

  // Fallback extraction to handle direct json payloads
  const state = gameState?.data || gameState?.payload || gameState?.state || gameState;
  const step = state?.step ?? 0;
  const agents = state?.agents ?? {};

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto flex flex-col gap-6">
        <header>
          <h1 className="text-2xl font-black tracking-tight text-white">Simulation Dashboard</h1>
          <p className="text-xs text-slate-400">REST (FastAPI) + WebSockets (Node.js Gateway)</p>
        </header>

        <ControlPanel isConnected={isConnected} />

        <main className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <section className="lg:col-span-5">
            <GameGrid agents={agents} />
          </section>
          <section className="lg:col-span-7">
            <HUD step={step} agents={agents} />
          </section>
        </main>
      </div>
    </div>
  );
}