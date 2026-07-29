const WebSocket = require('ws')
const {createClient} = require('redis')

//Configure the redis client
const redisClient = createClient({
    url : 'redis://redis:6379/0'
})

redisClient.on('error', (err) => console.log('[!] Redis Client Error', err))

//Configure the Websocket server
//Port 300 is used for this websocket connection, since our python is occupying port 8000
const wss = new WebSocket.Server({
    port : 3001,
    host : '0.0.0.0'
})

//Main async and await,,when a user clicks and the node.js sends a connection request to redis, instead of waiting for a reply
//It uses await,,so the endpoint can serve another user,till the redis sends the response, and then resolve this connection.
async function bootGateway() {
    //The Connection
    await redisClient.connect()
    console.log("[*] Node.js Gateway connected to Redis")

    //Sunscribe to the game_frames published by our python script
    //(messade)=>{} is a callback function, until python sends the data it will be dormant and uses 0% of CPU
    //When the frame arrives, Redis forces Node.js to wake up and dumps the frame into the message variable
    await redisClient.subscribe('game_frames', (message) => {
        //If the message arrives from pthon,loop for all clients connected to our websocket server
        //If the connection is open,send the message
        wss.clients.forEach((client) => {
            if (client.readyState == WebSocket.OPEN){
                client.send(message)
            }
        })
    })
    console.log('[*] Subscribed to Redis channel : game_frames')
}



//The main trigger
//Connection management (watching for users)
wss.on('connection', (ws) => {
    console.log('[*] New user connected ')

    ws.on('close', () => {
        console.log('[-] User disconnected')
    })
})

//start the main function, on connection only will it send the data so we can safely run this(async)
bootGateway()
console.log('[*] Node.js websocket Gateway running on ws://localhost:3001')
