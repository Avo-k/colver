// XGBoost interpretability: feature extraction, tree inference, and
// Saabas path-based feature attribution.
//
// These models are XGBoost distillations of the bid NN — they approximate
// the NN's decisions using interpretable features. The SHAP-like contributions
// shown are from this simplified model, NOT from the neural network itself.

let modelsData = null;
let modelsLoading = null;

// Trump honor evaluation table (index = rank 0-7: 7,8,9,J,Q,K,10,A)
const TRUMP_EVAL = [0, 0, 6, 8, 1, 1, 3, 4];
const TRUMP_POINTS_TABLE = [0, 0, 14, 20, 3, 4, 10, 11];

// ── Feature names for display (French) ──
const FEATURE_LABELS = {
    trump_count:      'Nb atouts',
    has_jack:         'Valet',
    has_nine:         'Neuf',
    has_ace:          'As',
    has_ten:          'Dix',
    has_king:         'Roi',
    has_queen:        'Dame',
    trump_points:     'Points atout',
    trump_score:      'Score atout',
    has_belote:       'Belote (R+D)',
    side_aces:        'As annexes',
    side_tens:        'Dix annexes',
    side_voids:       'Coupes',
    side_singletons:  'Singletons',
    side_doubletons:  'Doubletons',
    total_aces:       'Total as',
    best_side_length: 'Long. max annexe',
    partner_support:  'Soutien partenaire',
    is_partner_suit:  'Couleur partenaire',
    opp_suit_cards:   'Cartes couleur adv.',
    is_opp_suit:      'Couleur adversaire',
    second_trump_score: 'Score 2e couleur',
    second_trump_count: 'Nb 2e couleur',
};

/**
 * Load XGBoost model JSON. Cached after first load.
 * @returns {Promise<Object>} models keyed by scenario_type
 */
export async function loadModels() {
    if (modelsData) return modelsData;
    if (modelsLoading) return modelsLoading;
    modelsLoading = fetch('/static/data/xgb_models.json')
        .then(r => {
            if (!r.ok) throw new Error(`XGB models: ${r.status}`);
            return r.json();
        })
        .then(data => {
            modelsData = data;
            modelsLoading = null;
            return data;
        })
        .catch(err => {
            modelsLoading = null;
            throw err;
        });
    return modelsLoading;
}

/**
 * Extract hand features for a given trump suit.
 * Mirrors the Rust distill_bid.rs feature computation exactly.
 *
 * @param {number[]} hand - 8 card indices (0-31)
 * @param {number} suitIdx - trump suit (0=S, 1=H, 2=D, 3=C)
 * @returns {Object} feature name → value
 */
export function extractFeatures(hand, suitIdx) {
    // Build per-suit bitmasks
    const suitBits = [0, 0, 0, 0];
    for (const card of hand) {
        const s = card >> 3;   // suit
        const r = card & 7;    // rank
        suitBits[s] |= (1 << r);
    }

    const trumpBits = suitBits[suitIdx];
    const trumpCount = popcount(trumpBits);

    // Trump card presence
    const hasJack  = (trumpBits >> 3) & 1;
    const hasNine  = (trumpBits >> 2) & 1;
    const hasAce   = (trumpBits >> 7) & 1;
    const hasTen   = (trumpBits >> 6) & 1;
    const hasKing  = (trumpBits >> 5) & 1;
    const hasQueen = (trumpBits >> 4) & 1;
    const hasBelote = (hasKing && hasQueen) ? 1 : 0;

    // Trump points (raw Belote scoring)
    let trumpPoints = 0;
    let b = trumpBits;
    while (b) {
        const rank = ctz(b);
        trumpPoints += TRUMP_POINTS_TABLE[rank];
        b &= b - 1;
    }

    // Trump score (evaluate_for_trump): honor eval + length bonus + side aces/voids
    let trumpScore = 0;
    b = trumpBits;
    while (b) {
        const rank = ctz(b);
        trumpScore += TRUMP_EVAL[rank];
        b &= b - 1;
    }
    if (trumpCount > 2) trumpScore += (trumpCount - 2) * 2;
    // Side suit contributions to trump_score
    for (let s = 0; s < 4; s++) {
        if (s === suitIdx) continue;
        const sb = suitBits[s];
        const sc = popcount(sb);
        if (sb & (1 << 7)) trumpScore += 3;  // side ace
        if (sc === 0) trumpScore += 3;        // void
        else if (sc === 1) trumpScore += 1;   // singleton
    }

    // Side suit features
    let sideAces = 0, sideTens = 0, sideVoids = 0;
    let sideSingletons = 0, sideDoubletons = 0, bestSideLength = 0;
    for (let s = 0; s < 4; s++) {
        if (s === suitIdx) continue;
        const sb = suitBits[s];
        const sc = popcount(sb);
        if (sb & (1 << 7)) sideAces++;
        if (sb & (1 << 6)) sideTens++;
        if (sc === 0) sideVoids++;
        else if (sc === 1) sideSingletons++;
        else if (sc === 2) sideDoubletons++;
        if (sc > bestSideLength) bestSideLength = sc;
    }

    let totalAces = 0;
    for (let s = 0; s < 4; s++) {
        if (suitBits[s] & (1 << 7)) totalAces++;
    }

    return {
        trump_count: trumpCount,
        has_jack: hasJack,
        has_nine: hasNine,
        has_ace: hasAce,
        has_ten: hasTen,
        has_king: hasKing,
        has_queen: hasQueen,
        trump_points: trumpPoints,
        trump_score: trumpScore,
        has_belote: hasBelote,
        side_aces: sideAces,
        side_tens: sideTens,
        side_voids: sideVoids,
        side_singletons: sideSingletons,
        side_doubletons: sideDoubletons,
        total_aces: totalAces,
        best_side_length: bestSideLength,
    };
}

/**
 * Determine which scenario model to use based on bid history.
 * Returns the model key prefix (e.g., "opening", "pos2_pass", "partner80").
 *
 * @param {number[]} priorActions - list of bid actions before our turn
 * @returns {string} scenario key prefix
 */
export function detectScenario(priorActions) {
    const n = priorActions.length;
    if (n === 0) return 'opening';

    // Check if there's a non-pass bid in history
    const bids = priorActions.filter(a => a >= 1 && a <= 40);
    if (bids.length === 0) {
        // All passes
        if (n === 1) return 'pos2_pass';
        if (n === 2) return 'pos3_pass';
        if (n === 3) return 'pos4_pass';
        return 'opening'; // fallback
    }

    // There's a bid — figure out if partner or opponent
    // Seats: prior_actions[i] is from seat (2 - n + i + 32) % 4
    // Our seat is always 2 (South). Partner = 0 (North).
    for (let i = bids.length - 1; i >= 0; i--) {
        // Find which prior_action index this bid corresponds to
        const actionIdx = priorActions.indexOf(bids[i]);
        const bidderSeat = (2 - n + actionIdx + 32) % 4;
        const isPartner = (bidderSeat % 2) === (2 % 2); // same team
        if (isPartner) return 'partner80';
        return 'opp80';
    }

    return 'opening';
}

/**
 * Add context features for response scenarios (partner/opp bid).
 *
 * @param {Object} features - base features object (modified in place)
 * @param {number[]} hand - 8 card indices
 * @param {number} suitIdx - trump suit being evaluated
 * @param {number[]} priorActions - bid history
 * @param {string} scenario - detected scenario
 */
export function addContextFeatures(features, hand, suitIdx, priorActions, scenario) {
    if (scenario !== 'partner80' && scenario !== 'opp80') return;

    // Find the bid suit from history
    const bids = priorActions.filter(a => a >= 1 && a <= 40);
    if (bids.length === 0) return;
    const lastBid = bids[bids.length - 1];
    let bidSuit;
    if (lastBid >= 37 && lastBid <= 40) {
        bidSuit = lastBid - 37;
    } else {
        bidSuit = (lastBid - 1) % 4;
    }

    // Count cards in bid suit
    const suitBits = [0, 0, 0, 0];
    for (const card of hand) {
        suitBits[card >> 3] |= (1 << (card & 7));
    }
    const bidSuitCards = popcount(suitBits[bidSuit]);

    if (scenario === 'partner80') {
        features.partner_support = bidSuitCards;
        features.is_partner_suit = (suitIdx === bidSuit) ? 1 : 0;
        features.opp_suit_cards = -1;
        features.is_opp_suit = 0;
    } else {
        features.partner_support = -1;
        features.is_partner_suit = 0;
        features.opp_suit_cards = bidSuitCards;
        features.is_opp_suit = (suitIdx === bidSuit) ? 1 : 0;
    }
}

/**
 * Run tree inference and compute Saabas path-based feature contributions.
 *
 * For each tree, walks root→leaf and accumulates:
 *   contribution[feature] += child_value - node_value
 * at each split node. Contributions sum to (prediction - base_score).
 *
 * @param {Object} model - model dict from xgb_models.json
 * @param {Object} featureValues - feature name → value
 * @returns {{prediction: number, probability: number, contributions: Object, base_score: number}}
 */
export function predict(model, featureValues) {
    const features = model.features;
    const x = features.map(f => featureValues[f] ?? 0);

    let totalPred = model.base_score;
    const contribs = {};
    for (const f of features) contribs[f] = 0;

    for (const tree of model.trees) {
        let node = tree;
        while ('f' in node) {
            const featureIdx = node.f;
            const threshold = node.t;
            const nodeValue = node.v;

            // XGBoost: "yes" = left when feature < threshold
            const child = (x[featureIdx] < threshold) ? node.l : node.r;
            const childValue = child.v;

            contribs[features[featureIdx]] += childValue - nodeValue;
            node = child;
        }
        totalPred += node.v;
    }

    // Convert log-odds to probability
    const probability = 1 / (1 + Math.exp(-totalPred));

    return { prediction: totalPred, probability, contributions: contribs, base_score: model.base_score };
}

/**
 * Full analysis: for each suit, compute features and XGBoost attribution.
 * Returns results sorted by NN Q-value (best suit first).
 *
 * @param {number[]} hand - 8 card indices
 * @param {number[]} priorActions - bid history
 * @param {Array<[number, number]>} qValues - NN Q-values [[action, q], ...]
 * @returns {Promise<Array<{suit: number, probability: number, contributions: Object, features: Object}>>}
 */
export async function analyzeAllSuits(hand, priorActions, qValues) {
    const models = await loadModels();
    const scenario = detectScenario(priorActions);

    // Find the per-suit model
    const suitModelKey = `${scenario}_suit`;
    const suitModel = models[suitModelKey];
    if (!suitModel) {
        console.warn(`[xgb] No model for scenario: ${suitModelKey}`);
        return null;
    }

    // Get best Q-value per suit across all thresholds (80-160 + capot)
    const bestQPerSuit = [-Infinity, -Infinity, -Infinity, -Infinity];
    for (const [action, q] of qValues) {
        if (action >= 1 && action <= 36) {
            const suitIdx = (action - 1) % 4;
            if (q > bestQPerSuit[suitIdx]) bestQPerSuit[suitIdx] = q;
        } else if (action >= 37 && action <= 40) {
            const suitIdx = action - 37;
            if (q > bestQPerSuit[suitIdx]) bestQPerSuit[suitIdx] = q;
        }
    }

    const results = [];
    for (let suitIdx = 0; suitIdx < 4; suitIdx++) {
        const features = extractFeatures(hand, suitIdx);
        addContextFeatures(features, hand, suitIdx, priorActions, scenario);

        const result = predict(suitModel, features);
        results.push({
            suit: suitIdx,
            bestQ: bestQPerSuit[suitIdx],
            probability: result.probability,
            contributions: result.contributions,
            features,
            prediction: result.prediction,
            base_score: result.base_score,
        });
    }

    // Sort by best NN Q-value descending (most promising suit first)
    results.sort((a, b) => b.bestQ - a.bestQ);

    return results;
}

/**
 * Get the feature label in French.
 */
export function featureLabel(name) {
    return FEATURE_LABELS[name] || name;
}

// ── bit helpers ──

function popcount(x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

function ctz(x) {
    if (x === 0) return 32;
    let n = 0;
    if ((x & 0xffff) === 0) { n += 16; x >>= 16; }
    if ((x & 0xff) === 0)   { n += 8;  x >>= 8; }
    if ((x & 0xf) === 0)    { n += 4;  x >>= 4; }
    if ((x & 0x3) === 0)    { n += 2;  x >>= 2; }
    if ((x & 0x1) === 0)    { n += 1; }
    return n;
}
