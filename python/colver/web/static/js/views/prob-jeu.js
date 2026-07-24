// Problèmes de jeu view — single-card practice problems

import { send, onMessage, offMessage } from '../ws.js';
import * as SFX from '../sounds.js';
import { RANKS, SUITS, SEAT_NAMES_FR, cardRank, cardSuit, cardToHtml, renderHand, renderTrick, contractStr, actionName, bidActionHtml } from '../shared/cards.js';

const TEMPLATE = `
<div id="pj-config">
    <span class="annonces-title">Probl\u00e8mes de jeu</span>
    <button id="pj-generate-btn">Nouveau probl\u00e8me</button>
    <span id="pj-loading" class="hidden" style="color:#888;font-size:.85rem">G\u00e9n\u00e9ration\u2026</span>
</div>
<div id="pj-main" class="hidden">
    <div id="pj-left">
        <div class="prob-box" id="pj-info-bar">
            <span id="pj-contract-display" style="color:#d4af37;font-weight:600"></span>
            <span id="pj-score-ns">NS : 0</span>
            <span id="pj-score-ew">EO : 0</span>
            <span id="pj-tricks-info" style="color:#888"></span>
        </div>
        <div class="prob-box">
            <div class="section-title">Ench\u00e8res</div>
            <div id="pj-bid-history-entries" style="display:flex;gap:4px;flex-wrap:wrap;min-height:20px"></div>
        </div>
        <div class="prob-box">
            <div class="section-title">Pli en cours</div>
            <div id="pj-trick-area">
                <div class="trick-card" id="pj-trick-n"></div>
                <div class="trick-card" id="pj-trick-w"></div>
                <div class="trick-card" id="pj-trick-e"></div>
                <div class="trick-card" id="pj-trick-s"></div>
            </div>
        </div>
        <div class="prob-box">
            <div class="section-title">Votre main (Sud) \u2014 cliquez une carte</div>
            <div class="hand" id="pj-hand-display"></div>
        </div>
    </div>
    <div id="pj-correction-overlay" class="prob-overlay hidden">
        <div id="pj-correction" class="prob-correction-modal">
            <div class="prob-correction-header">
                <span class="section-title" style="margin:0">Correction</span>
                <button id="pj-close-correction" class="prob-close-btn">\u00d7</button>
            </div>
            <div class="prob-correction-body">
                <div id="pj-player-badge" class="prob-badge"></div>
                <div class="prob-correction-grid">
                    <div class="prob-section">
                        <div class="prob-label">Oracle DD (information parfaite)</div>
                        <div id="pj-oracle-best" class="prob-best"></div>
                        <div id="pj-oracle-elapsed" class="visit-total"></div>
                    </div>
                    <div class="prob-section">
                        <div class="prob-label">IS-DD (D\u00e9d\u00e9)</div>
                        <div id="pj-isdd-best" class="prob-best"></div>
                        <div id="pj-isdd-bars"></div>
                        <div id="pj-isdd-meta" class="visit-total"></div>
                    </div>
                    <div id="pj-dmc-section" class="prob-section hidden">
                        <div class="prob-label">DouDou50 (DMC)</div>
                        <div id="pj-dmc-best" class="prob-best"></div>
                        <div id="pj-dmc-bars"></div>
                    </div>
                    <div class="prob-section">
                        <div class="section-title" style="margin-bottom:6px">Distribution compl\u00e8te</div>
                        <div id="pj-all-hands"></div>
                    </div>
                </div>
            </div>
            <button id="pj-next-btn" class="prob-next-btn">Probl\u00e8me suivant</button>
        </div>
    </div>
</div>
`;

let pjLegalActions = [];
let pjLocked = false;

function pjPlayCard(cardIdx) {
    if (pjLocked) return;
    if (!new Set(pjLegalActions).has(cardIdx)) return;
    pjLocked = true;
    SFX.cardPlay();

    const el = document.getElementById('pj-trick-s');
    el.innerHTML = '';
    el.appendChild(cardToHtml(cardIdx));

    const handEl = document.getElementById('pj-hand-display');
    const cardEl = handEl.querySelector(`[data-card="${cardIdx}"]`);
    if (cardEl) cardEl.remove();

    send({ type: 'play_problem_submit', action: cardIdx });
}

function handleProblemReady(data) {
    document.getElementById('pj-loading').classList.add('hidden');
    document.getElementById('pj-main').classList.remove('hidden');
    document.getElementById('pj-correction-overlay').classList.add('hidden');
    pjLegalActions = data.legal_actions;
    pjLocked = false;

    document.getElementById('pj-contract-display').innerHTML = contractStr(data.contract);
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
        sp.innerHTML = SEAT_NAMES_FR[e.player] + ' ' + bidActionHtml(e.action);
        bh.appendChild(sp);
    }

    renderTrick('pj-trick', data.current_trick);

    const legalSet = new Set(pjLegalActions);
    const trump = data.contract ? data.contract.trump : -1;
    renderHand(document.getElementById('pj-hand-display'), data.south_hand, true, pjPlayCard, legalSet, trump);
}

function handleCorrection(data) {
    document.getElementById('pj-correction-overlay').classList.remove('hidden');

    const correct = data.player_action === data.oracle_action;
    const badge = document.getElementById('pj-player-badge');
    badge.className = 'prob-badge ' + (correct ? 'prob-badge-correct' : 'prob-badge-wrong');
    badge.textContent = 'Vous : ' + data.player_action_name;

    document.getElementById('pj-oracle-best').textContent = 'Oracle DD : ' + data.oracle_action_name;
    document.getElementById('pj-oracle-elapsed').textContent = `${data.oracle_elapsed_ms}ms`;

    document.getElementById('pj-isdd-best').textContent = 'IS-DD : ' + data.isdd_action_name;
    const isddBarsEl = document.getElementById('pj-isdd-bars');
    isddBarsEl.innerHTML = '';

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

    const dmcSec = document.getElementById('pj-dmc-section');
    if (data.dmc_q_values && data.dmc_q_values.length) {
        dmcSec.classList.remove('hidden');
        document.getElementById('pj-dmc-best').textContent = 'DouDou50 : ' + data.dmc_action_name;
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

    const handContainer = document.getElementById('pj-all-hands');
    handContainer.innerHTML = '';
    const seatNames = ['Nord', 'Est', 'Sud', 'Ouest'];
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
        handContainer.appendChild(div);
    }
}

export function mount(container) {
    container.innerHTML = TEMPLATE;
    pjLegalActions = [];
    pjLocked = false;

    document.getElementById('pj-generate-btn').addEventListener('click', () => {
        pjLocked = false;
        document.getElementById('pj-loading').classList.remove('hidden');
        document.getElementById('pj-main').classList.add('hidden');
        send({ type: 'play_problem_generate' });
    });

    document.getElementById('pj-next-btn').onclick = () => document.getElementById('pj-generate-btn').click();
    document.getElementById('pj-close-correction').onclick = () =>
        document.getElementById('pj-correction-overlay').classList.add('hidden');

    onMessage('play_problem_ready', handleProblemReady);
    onMessage('play_problem_correction', handleCorrection);

    // Auto-load first problem
    document.getElementById('pj-generate-btn').click();
}

export function unmount() {
    offMessage('play_problem_ready', handleProblemReady);
    offMessage('play_problem_correction', handleCorrection);
    pjLegalActions = [];
    pjLocked = false;
}
