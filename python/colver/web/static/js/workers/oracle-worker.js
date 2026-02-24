// Oracle Web Worker — runs DD solves in WASM off the main thread.
// Messages:
//   IN:  { type: 'init', wasmUrl: string }
//   IN:  { type: 'oracle_sim', id: number, hand: number[], numSims: number }
//   IN:  { type: 'cancel' }
//   OUT: { type: 'ready' }
//   OUT: { type: 'oracle_update', id, completed, total, success_counts, elapsed_ms }
//   OUT: { type: 'oracle_done', id, completed, total, success_counts, sampled_deals, elapsed_ms }
//   OUT: { type: 'error', message: string }

let oracle = null;
let cancelFlag = false;

const THRESHOLDS = [80, 90, 100, 110, 120, 130, 140, 150, 160, 162];

self.onmessage = async function(e) {
    const { type } = e.data;

    if (type === 'init') {
        try {
            const wasmUrl = e.data.wasmUrl;
            // Dynamic import of the WASM JS glue module
            const wasmModule = await import(wasmUrl);
            await wasmModule.default();
            oracle = new wasmModule.WasmOracle();
            self.postMessage({ type: 'ready' });
        } catch (err) {
            self.postMessage({ type: 'error', message: `WASM init failed: ${err.message || err}` });
        }
        return;
    }

    if (type === 'cancel') {
        cancelFlag = true;
        return;
    }

    if (type === 'oracle_sim') {
        if (!oracle) {
            self.postMessage({ type: 'error', message: 'Oracle not initialized' });
            return;
        }

        cancelFlag = false;
        const { id, hand, numSims } = e.data;
        const handArr = new Uint8Array(hand);
        const startTime = performance.now();

        // success_counts[suit][threshold_idx]
        const success_counts = [
            new Array(10).fill(0),
            new Array(10).fill(0),
            new Array(10).fill(0),
            new Array(10).fill(0),
        ];
        const sampled_deals = [];

        for (let i = 0; i < numSims; i++) {
            if (cancelFlag) break;

            let resultJson;
            try {
                resultJson = oracle.single_sim(handArr);
            } catch (err) {
                self.postMessage({ type: 'error', message: `Sim ${i} failed: ${err.message || err}` });
                return;
            }

            const result = JSON.parse(resultJson);

            // Accumulate success counts
            for (let suit = 0; suit < 4; suit++) {
                const ns = result.suits[suit][0];
                for (let t = 0; t < THRESHOLDS.length; t++) {
                    if (ns >= THRESHOLDS[t]) {
                        success_counts[suit][t]++;
                    }
                }
            }

            // Save deal for sample viewer
            sampled_deals.push(result.hands);

            const completed = i + 1;
            const elapsed_ms = Math.round(performance.now() - startTime);

            // Post update every sim (streaming like the server does)
            self.postMessage({
                type: 'oracle_update',
                id,
                completed,
                total: numSims,
                success_counts,
                elapsed_ms,
            });

            // Yield every 5 sims to let postMessage flush
            if (completed % 5 === 0) {
                await new Promise(r => setTimeout(r, 0));
            }
        }

        const elapsed_ms = Math.round(performance.now() - startTime);
        const completed = cancelFlag ? sampled_deals.length : numSims;

        self.postMessage({
            type: 'oracle_done',
            id,
            completed,
            total: numSims,
            success_counts,
            sampled_deals,
            elapsed_ms,
        });
    }
};
