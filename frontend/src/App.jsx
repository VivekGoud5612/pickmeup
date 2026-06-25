import React, {useEffect, useRef, useState} from 'react';  /* Import those from react..*/

const CombatVisualizer = () => { // Our main function
  const canvasRef = useRef(null); // a null pointer . for now.. useRef is used to create a pointer 
  const gameStateRef = useRef(null);   // a way to avoid state updation each frame...

  const [hudStats, setHudStats] = useState(null)  //Although we are using use state here, we throttle the updates to match the FPS of canvas.. and also use state needs two elements to initialize, one is the value and the other is the function to set the value to some data
  
  useEffect(() => {
     const ws = new WebSocket('ws://127.0.0.1:8000/ws/combat');  // a new Websocket cliet to our web socket server

     ws.onopen = () => {
      console.log("web socket connected");
     }

     let framecount = 0;
     ws.onmessage = (event) => {  // even is the game state payload
      gameStateRef.current = JSON.parse(event.data);  // This is a function to take in game state payload and JSON parse that
      console.log("Received Frame : ", event.data);

      framecount++;
      if(framecount % 5 == 0){  //Set this for every 5 frames... still dont understand what ths is//
        setHudStats(gameStateRef.current.agents);
      }
     };

     const renderLoop = () => {  // A function to render the canvas I think...
      const canvas = canvasRef.current;  // We use the current reference of that pointer ig
      if (!canvas) return ;
      console.log("Loop rendered");
      const ctx = canvas.getContext('2d');  // I need to know where did we even import that canvas.. But that is just a pointer .. whta is it pointing

      ctx.clearRect(0, 0, canvas.width, canvas.height); 

      // 2. Draw Grid Lines (optional but looks great)
      ctx.strokeStyle = '#1E2532';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 20; i++) {
          ctx.beginPath(); ctx.moveTo(i * 20, 0); ctx.lineTo(i * 20, 400); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(0, i * 20); ctx.lineTo(400, i * 20); ctx.stroke();
      }

      if(gameStateRef.current && gameStateRef.current.agents){
        const state = gameStateRef.current;
        const agents = gameStateRef.current.agents;

        Object.entries(agents).forEach(([agentName, data]) => { //// Looping over agents inside gamestate payload for each nameand data
          if (agentName == "BOSS"){
            ctx.fillStyle = '#F2685B';
          }
          else{
            ctx.fillStyle = '#4FE8C4';
          }
          const px = data.x * 20;
          const py = data.y * 20;

          ctx.fillRect(px, py, 20, 20)

          // B. Draw Agent Initials inside the square
          ctx.fillStyle = '#000000';
          ctx.font = 'bold 10px monospace';
          ctx.fillText(agentName.substring(0, 1), px + 6, py + 14);

           // C. Draw Floating HP Bar (Red)
          const maxHp = data.max_hp
          const hpPercent = Math.max(0, data.hp / maxHp);
          ctx.fillStyle = '#ff4444';
          ctx.fillRect(px, py - 6, 20 * hpPercent, 4); // 4 pixels tall, floats above head

          // D. Draw Floating Stamina Bar (Yellow) for Heroes
          if (data.stamina !== undefined) {
              const staminaPercent = Math.max(0, data.stamina / 50); // Assumed 50 max stamina
              ctx.fillStyle = '#f6e05e';
              ctx.fillRect(px, py - 2, 20 * staminaPercent, 2); // 2 pixels tall, right below HP
          }
        })
      }
      requestAnimationFrame(renderLoop);
     };
     requestAnimationFrame(renderLoop);
     
     return () => ws.close();  // Clsoe the conneciton .. to that websocket server
  }, []);

  return (
      <div className="min-h-screen bg-[#0A0E17] flex flex-col items-center pt-12 font-sans">
        <h2 className="text-[#4FE8C4] font-mono text-2xl mb-8 tracking-widest drop-shadow-[0_0_8px_rgba(79,232,196,0.5)]">
          PICKMEUP :: ENGINE VISUALIZER
        </h2>

        <div className="flex gap-8 items-start">
          {/* The Canvas Arena */}
          <div className="relative">
              <div className="absolute -inset-1 bg-gradient-to-r from-[#4FE8C4]/20 to-[#F2685B]/20 blur-lg rounded-xl"></div>
              <canvas 
                  ref={canvasRef} 
                  width={400} 
                  height={400} 
                  className="relative bg-[#12161D] border-2 border-[#2F3A4F] rounded-lg z-10"
              />
          </div>

          {/* The Text HUD Side-Panel */}
          <div className="bg-[#12161D] border border-[#2F3A4F] rounded-lg p-6 w-64 text-slate-300">
              <h3 className="text-white font-bold mb-4 border-b border-slate-700 pb-2">LIVE STATS</h3>
              
              {hudStats ? (
                  <div className="space-y-4">
                      {Object.entries(hudStats).map(([name, data]) => (
                          <div key={name} className="bg-slate-800/50 p-3 rounded-md">
                              <div className={`font-bold text-sm ${name === 'BOSS' ? 'text-[#F2685B]' : 'text-[#4FE8C4]'}`}>
                                  {name}
                              </div>
                              <div className="text-xs mt-1 grid grid-cols-2 gap-1">
                                  <span>HP: {Math.floor(data.hp)}</span>
                                  <span>Pos: [{data.x}, {data.y}]</span>
                                  {name !== 'BOSS' && <span>Stam: {Math.floor(data.stamina)}</span>}
                              </div>
                          </div>
                      ))}
                  </div>
              ) : (
                  <div className="text-sm text-slate-500 animate-pulse">Waiting for engine data...</div>
              )}
          </div>
        </div>
      </div>
    );
  };

export default CombatVisualizer;

