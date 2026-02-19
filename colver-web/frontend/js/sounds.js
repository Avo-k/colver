// Web Audio API sound effects for Colver
// Synthesized sounds — no external audio files needed

const SFX = (function() {
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
        const len = c.sampleRate * 0.2; // 200ms of noise
        noiseBuffer = c.createBuffer(1, len, c.sampleRate);
        const data = noiseBuffer.getChannelData(0);
        for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
        return noiseBuffer;
    }

    // Helper: create oscillator with envelope
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

    // 1. Card slap — short noise burst with bandpass
    function cardPlay() {
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

    // 2. Trick won — sine sweep + noise swoosh
    function trickWon() {
        const c = ensureCtx();
        const t = c.currentTime;
        // Sine sweep 200->600Hz
        const o = c.createOscillator();
        const g = c.createGain();
        o.type = 'sine';
        o.frequency.setValueAtTime(200, t);
        o.frequency.exponentialRampToValueAtTime(600, t + 0.15);
        g.gain.setValueAtTime(0.15, t);
        g.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
        o.connect(g);
        g.connect(masterGain);
        o.start(t);
        o.stop(t + 0.2);
        // Noise swoosh
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

    // 3. Bid — two-note chime C5->E5
    function bid() {
        const c = ensureCtx();
        const t = c.currentTime;
        osc('sine', 523.25, t, 0.12, 0.15);        // C5
        osc('sine', 659.25, t + 0.08, 0.12, 0.15);  // E5
    }

    // 4. Pass — single soft click
    function pass() {
        const c = ensureCtx();
        const t = c.currentTime;
        osc('sine', 1000, t, 0.025, 0.08);
    }

    // 5. Coinche — low triangle + percussive hit
    function coinche() {
        const c = ensureCtx();
        const t = c.currentTime;
        osc('triangle', 80, t, 0.15, 0.3);
        osc('square', 400, t, 0.04, 0.2);
        // Noise burst for impact
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

    // 6. Surcoinche — double coinche, higher pitch
    function surcoinche() {
        const c = ensureCtx();
        const t = c.currentTime;
        osc('triangle', 100, t, 0.15, 0.3);
        osc('square', 500, t, 0.04, 0.2);
        osc('triangle', 120, t + 0.1, 0.15, 0.3);
        osc('square', 600, t + 0.1, 0.04, 0.2);
    }

    // 7. Belote — ascending arpeggio C5-E5-G5 with shimmer
    function belote() {
        const c = ensureCtx();
        const t = c.currentTime;
        osc('sine', 523.25, t, 0.2, 0.15);          // C5
        osc('sine', 659.25, t + 0.08, 0.2, 0.15);    // E5
        osc('sine', 783.99, t + 0.16, 0.25, 0.15);   // G5
        // Shimmer: slightly detuned
        osc('sine', 527, t + 0.16, 0.25, 0.06);
        osc('sine', 788, t + 0.16, 0.25, 0.06);
    }

    // 8. Victory — major arpeggio C4-E4-G4-C5, sustained
    function victory() {
        const c = ensureCtx();
        const t = c.currentTime;
        osc('sine', 261.63, t, 0.8, 0.12);           // C4
        osc('sine', 329.63, t + 0.12, 0.7, 0.12);    // E4
        osc('sine', 392.00, t + 0.24, 0.6, 0.12);    // G4
        osc('sine', 523.25, t + 0.36, 0.6, 0.15);    // C5
        // Warm pad
        osc('triangle', 261.63, t + 0.36, 0.5, 0.06);
        osc('triangle', 392.00, t + 0.36, 0.5, 0.06);
    }

    // 9. Defeat — descending minor C4-Bb3-Ab3
    function defeat() {
        const c = ensureCtx();
        const t = c.currentTime;
        osc('sine', 261.63, t, 0.5, 0.1);            // C4
        osc('sine', 233.08, t + 0.2, 0.5, 0.1);      // Bb3
        osc('sine', 207.65, t + 0.4, 0.6, 0.1);      // Ab3
        // Dark undertone
        osc('triangle', 130.81, t + 0.4, 0.5, 0.05);
    }

    // 10. Your turn — double ping G5 with throttle
    function yourTurn() {
        const now = Date.now();
        if (now - _yourTurnLast < 2000) return;
        _yourTurnLast = now;
        const c = ensureCtx();
        const t = c.currentTime;
        osc('sine', 783.99, t, 0.06, 0.1);           // G5
        osc('sine', 783.99, t + 0.08, 0.06, 0.1);    // G5
    }

    // Map (phase, action) to the right sound
    function playForAction(phase, action) {
        if (phase === 0) {
            // Bidding phase
            if (action === 0) pass();
            else if (action === 41) coinche();
            else if (action === 42) surcoinche();
            else bid();
        } else {
            // Play phase
            cardPlay();
        }
    }

    function toggleMute() {
        muted = !muted;
        localStorage.setItem('colver_sfx_muted', muted ? '1' : '0');
        if (masterGain) masterGain.gain.value = muted ? 0 : 1;
    }

    return {
        cardPlay: cardPlay,
        trickWon: trickWon,
        bid: bid,
        pass: pass,
        coinche: coinche,
        surcoinche: surcoinche,
        belote: belote,
        victory: victory,
        defeat: defeat,
        yourTurn: yourTurn,
        playForAction: playForAction,
        toggleMute: toggleMute,
        get muted() { return muted; }
    };
})();
