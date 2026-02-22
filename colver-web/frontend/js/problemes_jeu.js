// Problèmes de jeu: single-card practice problems

let pjLegalActions = [];
let pjLocked = false;
let pjHasDmc = false;

// Generate a new play problem
document.getElementById('pj-generate-btn').addEventListener('click', () => {
    pjLocked = false;
    document.getElementById('pj-loading').classList.remove('hidden');
    document.getElementById('pj-main').classList.add('hidden');
    send({ type: 'play_problem_generate' });
});

onMessage('play_problem_ready', (data) => {
    document.getElementById('pj-loading').classList.add('hidden');
    document.getElementById('pj-main').classList.remove('hidden');
    document.getElementById('pj-correction').classList.add('hidden');
    pjLegalActions = data.legal_actions;
    pjHasDmc = data.has_dmc_model || false;
    pjLocked = false;

    // Info bar
    document.getElementById('pj-contract-display').textContent = contractStr(data.contract);
    document.getElementById('pj-score-ns').textContent = `NS : ${data.points[0]} (${data.tricks_won[0]}P)`;
    document.getElementById('pj-score-ew').textContent = `EO : ${data.points[1]} (${data.tricks_won[1]}P)`;
    const trickNum = data.tricks_won[0] + data.tricks_won[1] + 1;
    document.getElementById('pj-tricks-info').textContent = `Pli ${trickNum}/8`;

    // Bid history
    const bh = document.getElementById('pj-bid-history-entries');
    bh.innerHTML = '';
    for (const e of data.bid_history) {
        const sp = document.createElement('span');
        sp.className = 'watch-bid-entry ' + (e.player % 2 === 0 ? 'team-ns' : 'team-ew');
        sp.textContent = ['N', 'E', 'S', 'O'][e.player] + ':' + e.name;
        bh.appendChild(sp);
    }

    // Current trick
    renderTrick('pj-trick', data.current_trick);

    // South's hand — clickable, legal cards highlighted
    const legalSet = new Set(pjLegalActions);
    const trump = data.contract ? data.contract.trump : -1;
    renderHand(document.getElementById('pj-hand-display'), data.south_hand, true, pjPlayCard, legalSet, trump);
});

function pjPlayCard(cardIdx) {
    if (pjLocked) return;
    if (!new Set(pjLegalActions).has(cardIdx)) return;
    pjLocked = true;
    SFX.cardPlay();

    // Optimistic: place card in south trick slot
    const el = document.getElementById('pj-trick-s');
    el.innerHTML = '';
    el.appendChild(cardToHtml(cardIdx));

    // Visually remove from hand
    const handEl = document.getElementById('pj-hand-display');
    const cardEl = handEl.querySelector(`[data-card="${cardIdx}"]`);
    if (cardEl) cardEl.remove();

    send({ type: 'play_problem_submit', action: cardIdx });
}

onMessage('play_problem_correction', (data) => {
    document.getElementById('pj-correction').classList.remove('hidden');

    // Badge
    const correct = data.player_action === data.oracle_action;
    const badge = document.getElementById('pj-player-badge');
    badge.className = 'prob-badge ' + (correct ? 'prob-badge-correct' : 'prob-badge-wrong');
    badge.textContent = 'Vous : ' + data.player_action_name;

    // Oracle DD
    document.getElementById('pj-oracle-best').textContent = 'Oracle DD : ' + data.oracle_action_name;
    document.getElementById('pj-oracle-elapsed').textContent = `${data.oracle_elapsed_ms}ms`;

    // IS-DD bars
    document.getElementById('pj-isdd-best').textContent = 'IS-DD : ' + data.isdd_action_name;
    const isddBarsEl = document.getElementById('pj-isdd-bars');
    isddBarsEl.innerHTML = '';

    // Render IS-DD bars with player's pick highlighted
    if (data.isdd_card_scores && data.isdd_card_scores.length) {
        const sorted = [...data.isdd_card_scores].sort((a, b) => b[1] - a[1]);
        const vals = sorted.map(x => x[1]);
        const maxS = Math.max(...vals);
        const minS = Math.min(...vals);
        const rng = maxS - minS || 1;
        const div = document.createElement('div');
        div.className = 'visit-bars';
        for (const [action, score] of sorted) {
            const isBest = action === data.isdd_action;
            const isPlayer = action === data.player_action;
            const row = document.createElement('div');
            row.className = 'visit-row' + (isBest ? ' best' : '') + (isPlayer ? ' prob-player-pick' : '');
            const pct = ((score - minS) / rng * 100).toFixed(0);
            row.innerHTML = `<span class="visit-name">${actionName(action, 1)}</span>` +
                `<div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${pct}%"></div></div>` +
                `<span class="visit-count">${score.toFixed(1)}</span>`;
            div.appendChild(row);
        }
        isddBarsEl.appendChild(div);
    }
    document.getElementById('pj-isdd-meta').textContent =
        `${data.isdd_determinizations} det. / ${data.isdd_elapsed_ms}ms`;

    // DMC Q-value bars (if available)
    const dmcSec = document.getElementById('pj-dmc-section');
    if (data.dmc_q_values && data.dmc_q_values.length) {
        dmcSec.classList.remove('hidden');
        document.getElementById('pj-dmc-best').textContent = 'DouDou : ' + data.dmc_action_name;
        const dmcBarsEl = document.getElementById('pj-dmc-bars');
        dmcBarsEl.innerHTML = '';

        const sorted = [...data.dmc_q_values].sort((a, b) => b[1] - a[1]);
        const vals = sorted.map(x => x[1]);
        const maxQ = Math.max(...vals);
        const minQ = Math.min(...vals);
        const rng = maxQ - minQ || 1;
        const div = document.createElement('div');
        div.className = 'visit-bars';
        for (const [action, q] of sorted) {
            const isBest = action === data.dmc_action;
            const isPlayer = action === data.player_action;
            const row = document.createElement('div');
            row.className = 'visit-row' + (isBest ? ' best' : '') + (isPlayer ? ' prob-player-pick' : '');
            const pct = ((q - minQ) / rng * 100).toFixed(0);
            row.innerHTML = `<span class="visit-name">${actionName(action, 1)}</span>` +
                `<div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${pct}%"></div></div>` +
                `<span class="visit-count">${q.toFixed(3)}</span>`;
            div.appendChild(row);
        }
        dmcBarsEl.appendChild(div);
    } else {
        dmcSec.classList.add('hidden');
    }

    // Reveal all 4 hands in a 2×2 grid (N/W top, S/E bottom)
    const container = document.getElementById('pj-all-hands');
    container.innerHTML = '';
    const seatNames = ['Nord', 'Est', 'Sud', 'Ouest'];
    const trump = data.oracle_action !== undefined ? -1 : -1; // trump info not directly available here
    // Layout order: N, W, S, E for a 2×2 grid
    for (const seat of [0, 3, 2, 1]) {
        const div = document.createElement('div');
        div.className = 'pj-reveal-seat';
        const lbl = document.createElement('div');
        lbl.className = 'section-title';
        lbl.style.marginBottom = '4px';
        lbl.textContent = seatNames[seat];
        div.appendChild(lbl);
        const handEl = document.createElement('div');
        handEl.className = 'hand';
        handEl.style.setProperty('--card-w', 'clamp(28px,3vw,48px)');
        renderHand(handEl, data.all_hands[seat]);
        div.appendChild(handEl);
        container.appendChild(div);
    }
});

document.getElementById('pj-next-btn').onclick = () => document.getElementById('pj-generate-btn').click();
