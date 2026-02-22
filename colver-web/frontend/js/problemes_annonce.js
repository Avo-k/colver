// Problèmes d'annonce: single-bid practice problems

let paLegalActions = [];
let paLocked = false;

// Generate a new bid problem
document.getElementById('pa-generate-btn').addEventListener('click', () => {
    paLocked = false;
    document.getElementById('pa-loading').classList.remove('hidden');
    document.getElementById('pa-main').classList.add('hidden');
    send({ type: 'bid_problem_generate' });
});

onMessage('bid_problem_ready', (data) => {
    document.getElementById('pa-loading').classList.add('hidden');
    document.getElementById('pa-main').classList.remove('hidden');
    document.getElementById('pa-correction').classList.add('hidden');
    document.getElementById('pa-bid-panel').style.display = '';
    paLegalActions = data.legal_actions;
    paLocked = false;

    // Render bid history
    const entries = document.getElementById('pa-bid-history-entries');
    entries.innerHTML = '';
    for (const e of data.bid_history) {
        const sp = document.createElement('span');
        sp.className = 'watch-bid-entry ' + (e.player % 2 === 0 ? 'team-ns' : 'team-ew');
        sp.textContent = ['N', 'E', 'S', 'O'][e.player] + ':' + e.name;
        entries.appendChild(sp);
    }
    // Placeholder for South's pending bid
    const you = document.createElement('span');
    you.className = 'watch-bid-entry';
    you.style.cssText = 'color:#d4af37;font-style:italic';
    you.textContent = 'S:?';
    entries.appendChild(you);

    // Render South's hand
    renderHand(document.getElementById('pa-hand-display'), data.south_hand);

    // Configure bid controls based on legal actions
    const legalSet = new Set(data.legal_actions);

    // Disable value options that have no legal suit combination
    const valSel = document.getElementById('pa-bid-value');
    valSel.value = '';
    for (const opt of valSel.options) {
        if (!opt.value) continue;
        const v = parseInt(opt.value);
        let ok = false;
        for (let s = 0; s < 4; s++) {
            if (legalSet.has(encodeBidAction(v, s))) { ok = true; break; }
        }
        opt.disabled = !ok;
        if (ok && !valSel.value) valSel.value = String(opt.value);
    }

    const showBid = [...legalSet].some(a => a >= 1 && a <= 40);
    document.getElementById('pa-bid-selectors').style.display = showBid ? 'flex' : 'none';
    document.getElementById('pa-pass-btn').classList.toggle('hidden', !legalSet.has(0));
    document.getElementById('pa-coinche-btn').classList.toggle('hidden', !legalSet.has(41));
    document.getElementById('pa-surcoinche-btn').classList.toggle('hidden', !legalSet.has(42));
});

// Submit helpers
function paBidSubmit(action) {
    if (paLocked) return;
    if (!new Set(paLegalActions).has(action)) return;
    paLocked = true;
    send({ type: 'bid_problem_submit', action });
}

document.getElementById('pa-bid-submit').onclick = () => {
    const v = parseInt(document.getElementById('pa-bid-value').value);
    const s = parseInt(document.getElementById('pa-bid-suit').value);
    if (isNaN(v)) return;
    const a = encodeBidAction(v, s);
    paBidSubmit(a);
};
document.getElementById('pa-pass-btn').onclick = () => paBidSubmit(0);
document.getElementById('pa-coinche-btn').onclick = () => paBidSubmit(41);
document.getElementById('pa-surcoinche-btn').onclick = () => paBidSubmit(42);

// Handle correction response
onMessage('bid_problem_correction', (data) => {
    document.getElementById('pa-bid-panel').style.display = 'none';
    document.getElementById('pa-correction').classList.remove('hidden');

    // Badge: green if matches NN recommendation, else red
    const correct = data.nn_action !== null && data.player_action === data.nn_action;
    const badge = document.getElementById('pa-player-badge');
    badge.className = 'prob-badge ' + (correct ? 'prob-badge-correct' : 'prob-badge-wrong');
    badge.textContent = 'Votre annonce : ' + data.player_action_name;

    // NN Q-value bars
    const nnBestEl = document.getElementById('pa-nn-best');
    const nnBarsEl = document.getElementById('pa-nn-bars');
    nnBarsEl.innerHTML = '';
    if (data.nn_q_values && data.nn_q_values.length) {
        nnBestEl.textContent = 'NN : ' + (data.nn_action_name || '—');
        const sorted = [...data.nn_q_values].sort((a, b) => b[1] - a[1]).slice(0, 8);
        const mn = Math.min(...sorted.map(x => x[1]));
        const mx = Math.max(...sorted.map(x => x[1]));
        const rng = mx - mn || 1;
        const div = document.createElement('div');
        div.className = 'visit-bars';
        for (const [a, q] of sorted) {
            const isBest = a === data.nn_action;
            const isPlayer = a === data.player_action;
            const row = document.createElement('div');
            row.className = 'visit-row' + (isBest ? ' best' : '') + (isPlayer ? ' prob-player-pick' : '');
            const pct = ((q - mn) / rng * 100).toFixed(0);
            row.innerHTML = `<span class="visit-name">${bidActionName(a)}</span>` +
                `<div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${pct}%"></div></div>` +
                `<span class="visit-count">${q.toFixed(3)}</span>`;
            div.appendChild(row);
        }
        nnBarsEl.appendChild(div);
    } else {
        nnBestEl.textContent = 'NN : non disponible';
    }

    // Heuristic action
    document.getElementById('pa-heuristic-best').textContent = 'Amélioré : ' + data.heuristic_action_name;

    // DD table
    const keys = ['s', 'h', 'd', 'c'];
    if (data.dd_suits) {
        for (let i = 0; i < 4; i++) {
            document.getElementById(`pa-dd-${keys[i]}-ns`).textContent = data.dd_suits[i][0];
            document.getElementById(`pa-dd-${keys[i]}-ew`).textContent = data.dd_suits[i][1];
        }
    }
    document.getElementById('pa-dd-elapsed').textContent = `DD : ${data.dd_elapsed_ms}ms`;
});

document.getElementById('pa-next-btn').onclick = () => document.getElementById('pa-generate-btn').click();
