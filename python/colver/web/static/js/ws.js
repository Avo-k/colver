// WebSocket connection — Set-based multi-handler support

let ws = null;
const messageHandlers = new Map(); // type -> Set<handler>
const openHandlers = new Set();

function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const base = document.querySelector('base')?.getAttribute('href') || '/';
    ws = new WebSocket(`${proto}://${location.host}${base}ws`);

    ws.onopen = () => {
        console.log('Connecte');
        for (const handler of openHandlers) handler();
    };
    ws.onclose = () => {
        console.log('Deconnecte, reconnexion...');
        setTimeout(connect, 1000);
    };
    ws.onmessage = (evt) => {
        const data = JSON.parse(evt.data);
        const handlers = messageHandlers.get(data.type);
        if (handlers) {
            for (const handler of handlers) handler(data);
        } else {
            console.log('Non gere:', data);
        }
    };
}

export function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
    }
}

export function onMessage(type, handler) {
    if (!messageHandlers.has(type)) {
        messageHandlers.set(type, new Set());
    }
    messageHandlers.get(type).add(handler);
}

export function offMessage(type, handler) {
    const handlers = messageHandlers.get(type);
    if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) messageHandlers.delete(type);
    }
}

export function onOpen(handler) {
    openHandlers.add(handler);
}

export function offOpen(handler) {
    openHandlers.delete(handler);
}

export { connect };
