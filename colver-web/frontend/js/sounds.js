// Web Audio API sound effects for Colver
// Synthesized sounds — no external audio files needed

let ctx = null;
let masterGain = null;
let noiseBuffer = null;
let muted = localStorage.getItem('colver_sfx_muted') === '1';
let _yourTurnLast = 0;

function ensureCtx() {
    if (ctx) return ctx;
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    masterGain = ctx.createGain();
    masterGain.gain.value = muted ? 0 : 1;
    masterGain.connect(ctx.destination);
    return ctx;
}

function getNoise() {
    if (noiseBuffer) return noiseBuffer;
    const c = ensureCtx();
    const len = c.sampleRate * 0.2;
    noiseBuffer = c.createBuffer(1, len, c.sampleRate);
    const data = noiseBuffer.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
    return noiseBuffer;
}

function osc(type, freq, startTime, duration, gain) {
    const c = ensureCtx();
    const o = c.createOscillator();
    const g = c.createGain();
    o.type = type;
    o.frequency.value = freq;
    g.gain.setValueAtTime(gain, startTime);
    g.gain.exponentialRampToValueAtTime(0.001, startTime + duration);
    o.connect(g);
    g.connect(masterGain);
    o.start(startTime);
    o.stop(startTime + duration);
}

export function cardPlay() {
    const c = ensureCtx();
    const t = c.currentTime;
    const src = c.createBufferSource();
    src.buffer = getNoise();
    const bp = c.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 800;
    bp.Q.value = 1.5;
    const g = c.createGain();
    g.gain.setValueAtTime(0.35, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
    src.connect(bp);
    bp.connect(g);
    g.connect(masterGain);
    src.start(t);
    src.stop(t + 0.05);
}

export function trickWon() {
    const c = ensureCtx();
    const t = c.currentTime;
    const o2 = c.createOscillator();
    const g2 = c.createGain();
    o2.type = 'sine';
    o2.frequency.setValueAtTime(200, t);
    o2.frequency.exponentialRampToValueAtTime(600, t + 0.15);
    g2.gain.setValueAtTime(0.15, t);
    g2.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
    o2.connect(g2);
    g2.connect(masterGain);
    o2.start(t);
    o2.stop(t + 0.2);
    const src = c.createBufferSource();
    src.buffer = getNoise();
    const hp = c.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = 2000;
    const ng = c.createGain();
    ng.gain.setValueAtTime(0.1, t);
    ng.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
    src.connect(hp);
    hp.connect(ng);
    ng.connect(masterGain);
    src.start(t);
    src.stop(t + 0.2);
}

export function bid() {
    const c = ensureCtx();
    const t = c.currentTime;
    osc('sine', 523.25, t, 0.12, 0.15);
    osc('sine', 659.25, t + 0.08, 0.12, 0.15);
}

export function pass() {
    const c = ensureCtx();
    const t = c.currentTime;
    osc('sine', 1000, t, 0.025, 0.08);
}

export function coinche() {
    const c = ensureCtx();
    const t = c.currentTime;
    osc('triangle', 80, t, 0.15, 0.3);
    osc('square', 400, t, 0.04, 0.2);
    const src = c.createBufferSource();
    src.buffer = getNoise();
    const bp = c.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 500;
    bp.Q.value = 1;
    const g = c.createGain();
    g.gain.setValueAtTime(0.2, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.1);
    src.connect(bp);
    bp.connect(g);
    g.connect(masterGain);
    src.start(t);
    src.stop(t + 0.15);
}

export function surcoinche() {
    const c = ensureCtx();
    const t = c.currentTime;
    osc('triangle', 100, t, 0.15, 0.3);
    osc('square', 500, t, 0.04, 0.2);
    osc('triangle', 120, t + 0.1, 0.15, 0.3);
    osc('square', 600, t + 0.1, 0.04, 0.2);
}

export function belote() {
    const c = ensureCtx();
    const t = c.currentTime;
    osc('sine', 523.25, t, 0.2, 0.15);
    osc('sine', 659.25, t + 0.08, 0.2, 0.15);
    osc('sine', 783.99, t + 0.16, 0.25, 0.15);
    osc('sine', 527, t + 0.16, 0.25, 0.06);
    osc('sine', 788, t + 0.16, 0.25, 0.06);
}

export function victory() {
    const c = ensureCtx();
    const t = c.currentTime;
    osc('sine', 261.63, t, 0.8, 0.12);
    osc('sine', 329.63, t + 0.12, 0.7, 0.12);
    osc('sine', 392.00, t + 0.24, 0.6, 0.12);
    osc('sine', 523.25, t + 0.36, 0.6, 0.15);
    osc('triangle', 261.63, t + 0.36, 0.5, 0.06);
    osc('triangle', 392.00, t + 0.36, 0.5, 0.06);
}

export function defeat() {
    const c = ensureCtx();
    const t = c.currentTime;
    osc('sine', 261.63, t, 0.5, 0.1);
    osc('sine', 233.08, t + 0.2, 0.5, 0.1);
    osc('sine', 207.65, t + 0.4, 0.6, 0.1);
    osc('triangle', 130.81, t + 0.4, 0.5, 0.05);
}

export function yourTurn() {
    const now = Date.now();
    if (now - _yourTurnLast < 2000) return;
    _yourTurnLast = now;
    const c = ensureCtx();
    const t = c.currentTime;
    osc('sine', 783.99, t, 0.06, 0.1);
    osc('sine', 783.99, t + 0.08, 0.06, 0.1);
}

export function playForAction(phase, action) {
    if (phase === 0) {
        if (action === 0) pass();
        else if (action === 41) coinche();
        else if (action === 42) surcoinche();
        else bid();
    } else {
        cardPlay();
    }
}

export function toggleMute() {
    muted = !muted;
    localStorage.setItem('colver_sfx_muted', muted ? '1' : '0');
    if (masterGain) masterGain.gain.value = muted ? 0 : 1;
}

export function isMuted() { return muted; }
