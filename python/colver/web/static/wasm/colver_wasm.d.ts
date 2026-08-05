/* tslint:disable */
/* eslint-disable */

/**
 * BidNet wrapper for WASM.  Constructed once from weight bytes, reused across calls.
 *
 * Auto-detects architecture: tries hidden sizes 256, 512, 1024 and matches the
 * weight-file layout. Supports obs_dim 108 / 110 / 113 / 117 via build_bid_obs().
 */
export class WasmBidNet {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Evaluate a hand with prior bid actions, at a match score of 0-0.
     *
     * Kept as the zero-score shorthand for `evaluate_at_score`: a lone deal is
     * the site's default, and 0-0 is the truth there rather than a fallback.
     */
    evaluate(hand: Uint8Array, prior_actions: Uint8Array): string;
    /**
     * Evaluate a hand with prior bid actions at a given match score.
     * `hand`: Uint8Array of 8 card indices (0-31)
     * `prior_actions`: Uint8Array of prior bid action indices
     * `score_ns` / `score_ew`: cumulative match score, seat 2 (the evaluated
     * hand) being North-South. Bid v6 reads it (obs 110/113/117): the same
     * hand is bid differently at 900-200 than at 0-0. Ignored by a 108-dim
     * net, which has no room for it.
     * Returns JSON string: {"q_values":[[action,q],...], "best_action":N}
     */
    evaluate_at_score(hand: Uint8Array, prior_actions: Uint8Array, score_ns: number, score_ew: number): string;
    /**
     * Construct from raw weight bytes (little-endian f32).
     */
    constructor(weight_bytes: Uint8Array);
}

/**
 * Oracle DD solver wrapper for WASM.  Owns a reusable TT buffer (2MB).
 */
export class WasmOracle {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
    /**
     * Run one oracle simulation.
     * `hand`: Uint8Array of 8 card indices
     * Returns JSON string: {"suits":[[ns,ew],...], "hands":{"0":[...],"1":[...],"3":[...]}}
     */
    single_sim(hand: Uint8Array): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmbidnet_free: (a: number, b: number) => void;
    readonly __wbg_wasmoracle_free: (a: number, b: number) => void;
    readonly wasmbidnet_evaluate: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmbidnet_evaluate_at_score: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly wasmbidnet_new: (a: number, b: number) => [number, number, number];
    readonly wasmoracle_new: () => number;
    readonly wasmoracle_single_sim: (a: number, b: number, c: number) => [number, number, number, number];
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
