// WASM bridge — lazy-loads WASM module and bid model, manages oracle Worker.

let _ready = false;
let _initPromise = null;
let _bidNet = null;
let _oracleWorker = null;
let _oracleReady = false;
let _simIdCounter = 0;
let _currentOracleHandler = null;

// Paths relative to this module (static/js/wasm-bridge.js)
// ../wasm/ resolves to static/wasm/
const WASM_GLUE_URL = new URL('../wasm/colver_wasm.js', import.meta.url).href;
const WASM_BIN_URL = new URL('../wasm/colver_wasm_bg.wasm', import.meta.url).href;
const BID_MODEL_URL = new URL('../wasm/bid_v5_isdd.bin', import.meta.url).href;
const WORKER_URL = new URL('./workers/oracle-worker.js', import.meta.url).href;

/**
 * Ensure WASM module, bid model, and oracle worker are ready.
 * Returns a promise that resolves when everything is initialized.
 * Reuses existing init if already called.
 */
export function ensureReady() {
    if (_ready) return Promise.resolve();
    if (_initPromise) return _initPromise;
    _initPromise = _doInit();
    return _initPromise;
}

export function isReady() {
    return _ready;
}

async function _doInit() {
    // 1. Load WASM module
    const wasmModule = await import(WASM_GLUE_URL);
    await wasmModule.default(WASM_BIN_URL);

    // 2. Fetch bid model and construct WasmBidNet
    const resp = await fetch(BID_MODEL_URL);
    if (!resp.ok) throw new Error(`Failed to fetch bid model: ${resp.status}`);
    const modelBytes = new Uint8Array(await resp.arrayBuffer());
    _bidNet = new wasmModule.WasmBidNet(modelBytes);

    // 3. Spawn oracle Worker
    _oracleWorker = new Worker(WORKER_URL, { type: 'module' });

    const workerReady = new Promise((resolve, reject) => {
        const handler = (e) => {
            if (e.data.type === 'ready') {
                _oracleReady = true;
                _oracleWorker.removeEventListener('message', handler);
                resolve();
            } else if (e.data.type === 'error') {
                _oracleWorker.removeEventListener('message', handler);
                reject(new Error(e.data.message));
            }
        };
        _oracleWorker.addEventListener('message', handler);
    });

    _oracleWorker.postMessage({ type: 'init', wasmUrl: WASM_GLUE_URL });
    await workerReady;

    _ready = true;
    console.log('[wasm-bridge] WASM + BidNet + Oracle Worker ready');
}

/**
 * Evaluate bid NN on a hand with prior actions.
 * @param {number[]} hand - 8 card indices (0-31)
 * @param {number[]} priorActions - prior bid action indices
 * @returns {{ q_values: [number, number][], best_action: number }}
 */
export function evaluateBid(hand, priorActions) {
    if (!_bidNet) throw new Error('BidNet not ready');
    const handArr = new Uint8Array(hand);
    const priorArr = new Uint8Array(priorActions);
    const jsonStr = _bidNet.evaluate(handArr, priorArr);
    return JSON.parse(jsonStr);
}

/**
 * Run oracle simulation in the Worker.
 * @param {number[]} hand - 8 card indices
 * @param {number} numSims - number of simulations
 * @param {function} onUpdate - called with {completed, total, success_counts, elapsed_ms}
 * @param {function} onDone - called with {completed, total, success_counts, sampled_deals, elapsed_ms}
 * @returns {number} simId for cancellation
 */
export function runOracleSim(hand, numSims, onUpdate, onDone) {
    if (!_oracleWorker || !_oracleReady) throw new Error('Oracle Worker not ready');

    // Cancel any previous oracle sim and detach its handler
    cancelOracle();

    const simId = ++_simIdCounter;

    const handler = (e) => {
        const d = e.data;
        if (d.id !== simId) return;

        if (d.type === 'oracle_update') {
            onUpdate({
                completed: d.completed,
                total: d.total,
                success_counts: d.success_counts,
                elapsed_ms: d.elapsed_ms,
            });
        } else if (d.type === 'oracle_done') {
            _oracleWorker.removeEventListener('message', handler);
            _currentOracleHandler = null;
            onDone({
                completed: d.completed,
                total: d.total,
                success_counts: d.success_counts,
                sampled_deals: d.sampled_deals,
                elapsed_ms: d.elapsed_ms,
            });
        } else if (d.type === 'error') {
            _oracleWorker.removeEventListener('message', handler);
            _currentOracleHandler = null;
            onDone({ error: d.message });
        }
    };

    _currentOracleHandler = handler;
    _oracleWorker.addEventListener('message', handler);
    _oracleWorker.postMessage({ type: 'oracle_sim', id: simId, hand, numSims });

    return simId;
}

/**
 * Cancel any running oracle simulation.
 */
export function cancelOracle() {
    if (_oracleWorker) {
        _oracleWorker.postMessage({ type: 'cancel' });
        if (_currentOracleHandler) {
            _oracleWorker.removeEventListener('message', _currentOracleHandler);
            _currentOracleHandler = null;
        }
    }
}
