import React, {useEffect, useRef} from 'react';  /* Import those from react..*/

const CombatVisualizer = () => {
  const canvasRef = useRef(null); // a null pointer . for now.. useRef is used to create a pointer 
  const gameStateRef = useRef(null);

  useEffect(() => {
     const ws = new Websocket('ws://127.0.0.1:6545/ws/combat');  // a new Websocket cliet to our web socket server
  })
}


