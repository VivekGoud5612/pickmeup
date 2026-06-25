const WebSocket = require('ws')
const {createClient} = require('redis')

//Configure the redis client
const redisClient = createClient({
    url : 'redis://127.0.0.1:6379/0'
})

redisClient.on('error', (err) => console.log('[!] Redis Client Error', err))

