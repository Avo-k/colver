// Score keeper for real-life Belote Contrée matches.
// All state lives in localStorage — no server.

const STORAGE_CURRENT = 'colver:score:current';
const STORAGE_HISTORY = 'colver:score:history';

const BID_VALUES = [80, 90, 100, 110, 120, 130, 140, 150, 160];

// ===== Scoring (FFB rules, 2026-04-16 update — surcoinche ×3, base 160+contrat×mult) =====

/**
 * Compute round score from form inputs. Exact values, no rounding.
 * @returns {{scores: [number, number], reussi: boolean}}
 */
function computeRoundScore(r) {
    const taker = r.taker;
    const defense = 1 - taker;
    const beloteBonus = [0, 0];
    if (r.belote === 0 || r.belote === 1) beloteBonus[r.belote] = 20;
    const totalBelote = beloteBonus[0] + beloteBonus[1];
    const contractValue = r.isCapot ? 250 : r.value * 10;
    const scores = [0, 0];

    if (r.isCapot) {
        const reussi = !!r.capotRealise;
        if (reussi) {
            if (r.coinche === 0) {
                scores[taker] = 252 + contractValue + beloteBonus[taker];
                scores[defense] = beloteBonus[defense];
            } else if (r.coinche === 1) {
                scores[taker] = 250 + contractValue * 2 + totalBelote;
            } else {
                scores[taker] = 250 + contractValue * 3 + totalBelote;
            }
        } else {
            const mult = r.coinche === 0 ? 1 : r.coinche === 1 ? 2 : 3;
            scores[defense] = 160 + contractValue * mult + totalBelote;
        }
        return { scores, reussi };
    }

    const takerPts = r.takerPts | 0;
    const takerTotal = takerPts + beloteBonus[taker];
    const reussi = takerTotal >= contractValue;
    const total = r.capotRealise ? 252 : 162;
    const defensePtsActual = Math.max(0, total - takerPts);

    if (reussi) {
        const contreBase = r.capotRealise ? 250 : 160;
        if (r.coinche === 0) {
            scores[taker] = takerPts + contractValue + beloteBonus[taker];
            scores[defense] = defensePtsActual + beloteBonus[defense];
        } else if (r.coinche === 1) {
            scores[taker] = contreBase + contractValue * 2 + totalBelote;
        } else {
            scores[taker] = contreBase + contractValue * 3 + totalBelote;
        }
    } else {
        const mult = r.coinche === 0 ? 1 : r.coinche === 1 ? 2 : 3;
        scores[defense] = 160 + contractValue * mult + totalBelote;
    }
    return { scores, reussi };
}

// ===== Win probability (calibrated on 10k matches en 2000 pts) =====
// σ(1.7 × Δ / (R^0.8 + 340)) where R = (2000 - s_me) + (2000 - s_opp), Δ = s_me - s_opp.
// Cf. colver-core/src/bid/bid_train_env.rs::win_probability.
// Pour un score max ≠ 2000, on rescale linéairement vers une partie virtuelle en 2000 pts —
// approximation, pas exacte (la calibration ignore que les matchs courts/longs varient).

function winProbability(sMe, sOpp, maxScore) {
    const k = 2000 / maxScore;
    const me = sMe * k;
    const opp = sOpp * k;
    if (me >= 2000 && opp >= 2000) return me >= opp ? 1 : 0;
    if (me >= 2000) return 1;
    if (opp >= 2000) return 0;
    const rSum = (2000 - me) + (2000 - opp);
    const denom = Math.pow(Math.max(1, rSum), 0.8) + 340;
    const x = 1.7 * (me - opp) / denom;
    return 1 / (1 + Math.exp(-x));
}

// ===== Storage =====

function loadCurrent() {
    try {
        const raw = localStorage.getItem(STORAGE_CURRENT);
        return raw ? JSON.parse(raw) : null;
    } catch { return null; }
}

function saveCurrent(g) {
    if (g) localStorage.setItem(STORAGE_CURRENT, JSON.stringify(g));
    else localStorage.removeItem(STORAGE_CURRENT);
}

function loadHistory() {
    try {
        const raw = localStorage.getItem(STORAGE_HISTORY);
        return raw ? JSON.parse(raw) : [];
    } catch { return []; }
}

function saveHistory(h) {
    localStorage.setItem(STORAGE_HISTORY, JSON.stringify(h));
}

function newGame(teams, maxScore) {
    return {
        id: 'g' + Date.now() + Math.random().toString(36).slice(2, 6),
        teams: [teams[0] || 'Nous', teams[1] || 'Eux'],
        maxScore,
        rounds: [],
        createdAt: Date.now(),
        finishedAt: null,
    };
}

// ===== Game logic helpers =====

function totalsOf(game) {
    const t = [0, 0];
    for (const r of game.rounds) {
        t[0] += r.scores[0];
        t[1] += r.scores[1];
    }
    return t;
}

function isFinished(game) {
    const [a, b] = totalsOf(game);
    return a >= game.maxScore || b >= game.maxScore;
}

function winnerOf(game) {
    const [a, b] = totalsOf(game);
    if (a === b) return -1;
    return a > b ? 0 : 1;
}

// ===== UI =====

const app = {
    container: null,
    game: null,
    history: [],
    formState: null, // { editingIndex: null|number, draft: {...} }
};

function $(sel, root) { return (root || document).querySelector(sel); }

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function formatDate(ts) {
    const d = new Date(ts);
    const dd = String(d.getDate()).padStart(2, '0');
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    return `${dd}/${mm} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

// ===== Render =====

function render() {
    if (!app.game) renderSetup();
    else renderGame();
}

function renderSetup() {
    const last = app.history[0]; // last finished game for default names
    const defaultA = last?.teams[0] || 'Nous';
    const defaultB = last?.teams[1] || 'Eux';
    const maxOpts = [1000, 1500, 2000, 3000];
    const defaultMax = 2000;

    app.container.innerHTML = `
        <div id="score-page">
            <h2>Compteur de points</h2>
            <p class="score-subtitle">Tenez le score d'une partie de Belote Contrée &mdash; calcul automatique aux règles FFB.</p>

            <div class="score-setup">
                <div class="setup-row">
                    <span class="setup-label">Équipes</span>
                    <div class="team-inputs">
                        <input id="setup-team-a" type="text" maxlength="24" value="${escapeHtml(defaultA)}" placeholder="Équipe A">
                        <span class="vs-sep">VS</span>
                        <input id="setup-team-b" type="text" maxlength="24" value="${escapeHtml(defaultB)}" placeholder="Équipe B">
                    </div>
                </div>

                <div class="setup-row">
                    <span class="setup-label">Score maximum</span>
                    <div class="max-row" id="setup-max-row">
                        ${maxOpts.map(v => `
                            <button class="max-btn ${v === defaultMax ? 'active' : ''}" data-max="${v}">${v}</button>
                        `).join('')}
                    </div>
                </div>

                <button class="start-btn" id="setup-start">Commencer la partie</button>
            </div>

            ${renderHistory()}
        </div>
    `;

    let chosenMax = defaultMax;
    $('#setup-max-row').addEventListener('click', e => {
        const btn = e.target.closest('.max-btn');
        if (!btn) return;
        chosenMax = parseInt(btn.dataset.max, 10);
        $('#setup-max-row').querySelectorAll('.max-btn').forEach(b => b.classList.toggle('active', b === btn));
    });

    $('#setup-start').addEventListener('click', () => {
        const a = $('#setup-team-a').value.trim() || 'Nous';
        const b = $('#setup-team-b').value.trim() || 'Eux';
        app.game = newGame([a, b], chosenMax);
        saveCurrent(app.game);
        render();
    });

    bindHistory();
}

function renderHistory() {
    if (app.history.length === 0) {
        return `
            <div class="score-history">
                <h3>Parties précédentes</h3>
                <div class="score-history-empty">Aucune partie terminée pour le moment.</div>
            </div>
        `;
    }
    return `
        <div class="score-history">
            <h3>Parties précédentes</h3>
            <div class="score-history-list">
                ${app.history.map((g, i) => {
                    const t = totalsOf(g);
                    const w = winnerOf(g);
                    return `
                        <div class="score-history-item" data-h-idx="${i}">
                            <div class="h-teams">
                                <span class="h-team ${w === 0 ? 'winner' : ''}">${escapeHtml(g.teams[0])}</span>
                                <span class="h-vs">vs</span>
                                <span class="h-team ${w === 1 ? 'winner' : ''}">${escapeHtml(g.teams[1])}</span>
                            </div>
                            <span class="h-scores">${t[0]} &ndash; ${t[1]}</span>
                            <span class="h-date">${formatDate(g.finishedAt || g.createdAt)}</span>
                            <button class="h-delete" data-h-delete="${i}" title="Supprimer">✕</button>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
    `;
}

function bindHistory() {
    const list = $('.score-history-list');
    if (!list) return;
    list.addEventListener('click', e => {
        const del = e.target.closest('.h-delete');
        if (del) {
            e.stopPropagation();
            const idx = parseInt(del.dataset.hDelete, 10);
            if (confirm(`Supprimer cette partie de l'historique ?`)) {
                app.history.splice(idx, 1);
                saveHistory(app.history);
                render();
            }
        }
    });
}

function renderGame() {
    const totals = totalsOf(app.game);
    const finished = isFinished(app.game);
    const winner = finished ? winnerOf(app.game) : -1;
    const winProbs = [
        winProbability(totals[0], totals[1], app.game.maxScore),
        winProbability(totals[1], totals[0], app.game.maxScore),
    ];

    app.container.innerHTML = `
        <div id="score-page">
            <div class="score-game">
                <div class="score-header">
                    ${[0, 1].map(t => {
                        const pct = Math.min(100, (totals[t] / app.game.maxScore) * 100);
                        const lead = totals[t] > totals[1 - t];
                        const pPct = (winProbs[t] * 100).toFixed(winProbs[t] > 0.99 || winProbs[t] < 0.01 ? 2 : 1);
                        return `
                            <div class="score-team-block ${lead ? 'lead' : ''}">
                                <div class="score-team-name" data-team="${t}" contenteditable="true" spellcheck="false">${escapeHtml(app.game.teams[t])}</div>
                                <div class="score-team-total">${totals[t]}</div>
                                <div class="score-progress">
                                    <div class="score-progress-bar ${pct >= 100 ? 'over' : ''}" style="width: ${pct}%"></div>
                                </div>
                                <div class="score-max-label">
                                    <span class="score-max-pts">/ ${app.game.maxScore}</span>
                                    <span class="score-winprob" title="Probabilité de gagner la partie">P = ${pPct}%</span>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>

                ${finished ? `
                    <div class="score-finished">
                        <div class="winner-text">
                            ${winner >= 0
                                ? `<span class="winner-name">${escapeHtml(app.game.teams[winner])}</span> remporte la partie !`
                                : 'Match nul (égalité au plafond).'}
                        </div>
                        <button class="btn-pill" id="archive-game">Archiver et nouvelle partie</button>
                    </div>
                ` : `
                    <div class="score-add-row">
                        <button class="score-add-btn" id="add-round">+ Ajouter une manche</button>
                    </div>
                `}

                ${renderRounds()}

                ${renderWinprobPanel(totals, winProbs)}

                <div class="score-actions">
                    <button class="secondary-btn" id="rename-teams">Renommer les équipes</button>
                    <button class="secondary-btn danger-btn" id="reset-game">Abandonner la partie</button>
                </div>
            </div>

            ${renderHistory()}
        </div>
    `;

    bindGame();
}

function renderWinprobPanel(totals, winProbs) {
    const max = app.game.maxScore;
    const k = 2000 / max;
    const me = totals[0] * k;
    const opp = totals[1] * k;
    const rSum = Math.max(0, (2000 - me) + (2000 - opp));
    const denom = Math.pow(Math.max(1, rSum), 0.8) + 340;
    const delta = me - opp;
    const x = 1.7 * delta / denom;
    const scaledNote = max !== 2000
        ? `<div class="winprob-rescale">Score max ${max} ≠ 2000 → scores rééchelonnés ×${k.toFixed(2)} avant calcul. Approximation : la calibration sur partie en 2000 reste raisonnable mais pas exacte pour d'autres formats.</div>`
        : '';

    return `
        <details class="score-winprob-panel">
            <summary>
                <span class="winprob-summary-label">Probabilités de gagner</span>
                <span class="winprob-summary-vals">
                    <span>${escapeHtml(app.game.teams[0])} ${(winProbs[0] * 100).toFixed(1)}%</span>
                    <span class="sep">·</span>
                    <span>${escapeHtml(app.game.teams[1])} ${(winProbs[1] * 100).toFixed(1)}%</span>
                </span>
                <span class="winprob-toggle-hint">détails ▾</span>
            </summary>
            <div class="winprob-body">
                <div class="winprob-formula">
                    <span class="formula-line">P = σ(1.7 × Δ / (R<sup>0.8</sup> + 340))</span>
                    <span class="formula-where">avec Δ = s<sub>moi</sub> − s<sub>adv</sub>, R = (2000 − s<sub>moi</sub>) + (2000 − s<sub>adv</sub>), σ = sigmoïde</span>
                </div>
                <div class="winprob-numerics">
                    <div><span class="lbl">Δ</span><span class="val">${delta >= 0 ? '+' : ''}${delta.toFixed(0)}</span></div>
                    <div><span class="lbl">R</span><span class="val">${rSum.toFixed(0)}</span></div>
                    <div><span class="lbl">R<sup>0.8</sup>+340</span><span class="val">${denom.toFixed(0)}</span></div>
                    <div><span class="lbl">x</span><span class="val">${x.toFixed(3)}</span></div>
                    <div><span class="lbl">σ(x)</span><span class="val">${winProbs[0].toFixed(3)}</span></div>
                </div>
                <div class="winprob-credits">
                    Calibrée par régression logistique sur 10 000 matchs en 2 000 pts (DouDou50 + Bid v6).
                    Cf. <code>colver-core/src/bid/bid_train_env.rs::win_probability</code>.
                </div>
                ${scaledNote}
            </div>
        </details>
    `;
}

function renderRounds() {
    if (app.game.rounds.length === 0) {
        return `
            <div class="score-rounds">
                <div class="score-rounds-empty">Aucune manche enregistrée. Cliquez sur &laquo; Ajouter une manche &raquo; pour commencer.</div>
            </div>
        `;
    }

    let cum = [0, 0];
    const cumulatives = app.game.rounds.map(r => {
        cum = [cum[0] + r.scores[0], cum[1] + r.scores[1]];
        return [...cum];
    });

    // Most recent on top
    const rows = app.game.rounds.map((r, i) => {
        const cumA = cumulatives[i][0];
        const cumB = cumulatives[i][1];
        const beloteTag = r.belote >= 0 ? `<span class="belote-tag">B&middot;${escapeHtml(app.game.teams[r.belote])}</span>` : '';
        const multTag = r.coinche === 1 ? '<span class="mult-tag">CONTRÉ</span>' : r.coinche === 2 ? '<span class="mult-tag">SURCONTRÉ</span>' : '';
        const valLabel = r.isCapot ? 'Capot' : `${r.value * 10}`;
        const ptsLabel = r.isCapot ? (r.capotRealise ? 'réussi' : 'chuté') : `${r.takerPts}`;
        const reussi = r.isCapot ? !!r.capotRealise : (r.takerPts + (r.belote === r.taker ? 20 : 0)) >= (r.value * 10);
        const chuteTag = !reussi ? '<span class="chute-tag">CHUTE</span>' : '';
        return `
            <tr class="round-row" data-round-idx="${i}">
                <td class="col-num">${i + 1}</td>
                <td class="col-contract">
                    <strong>${escapeHtml(app.game.teams[r.taker])}</strong>
                    &middot; ${valLabel}
                    ${multTag} ${chuteTag} ${beloteTag}
                </td>
                <td>${r.isCapot ? '—' : `<span style="color:#aaa">${ptsLabel} pts</span>`}</td>
                <td class="col-score">
                    <span class="pts">${r.scores[0]}</span><span class="cum">(${cumA})</span>
                    <span class="row-actions">
                        <button class="delete" data-delete-idx="${i}" title="Supprimer">✕</button>
                    </span>
                </td>
                <td class="col-score">
                    <span class="pts">${r.scores[1]}</span><span class="cum">(${cumB})</span>
                </td>
            </tr>
        `;
    }).reverse().join('');

    return `
        <div class="score-rounds">
            <table>
                <thead>
                    <tr>
                        <th class="col-num">#</th>
                        <th>Contrat</th>
                        <th>Pts pris</th>
                        <th class="col-score">${escapeHtml(app.game.teams[0])}</th>
                        <th class="col-score">${escapeHtml(app.game.teams[1])}</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

function bindGame() {
    // Editable team names
    app.container.querySelectorAll('.score-team-name').forEach(el => {
        el.addEventListener('blur', () => {
            const t = parseInt(el.dataset.team, 10);
            const newName = el.textContent.trim() || (t === 0 ? 'Nous' : 'Eux');
            if (newName !== app.game.teams[t]) {
                app.game.teams[t] = newName;
                saveCurrent(app.game);
                render();
            } else {
                el.textContent = app.game.teams[t];
            }
        });
        el.addEventListener('keydown', e => {
            if (e.key === 'Enter') { e.preventDefault(); el.blur(); }
            if (e.key === 'Escape') { el.textContent = app.game.teams[parseInt(el.dataset.team, 10)]; el.blur(); }
        });
    });

    const addBtn = $('#add-round');
    if (addBtn) addBtn.addEventListener('click', () => openForm(null));

    const archive = $('#archive-game');
    if (archive) archive.addEventListener('click', archiveGame);

    $('#rename-teams').addEventListener('click', () => {
        const a = prompt('Nom de l\'équipe 1 :', app.game.teams[0]);
        if (a == null) return;
        const b = prompt('Nom de l\'équipe 2 :', app.game.teams[1]);
        if (b == null) return;
        app.game.teams = [a.trim() || 'Nous', b.trim() || 'Eux'];
        saveCurrent(app.game);
        render();
    });

    $('#reset-game').addEventListener('click', () => {
        if (confirm('Abandonner la partie en cours ? Elle ne sera pas archivée.')) {
            app.game = null;
            saveCurrent(null);
            render();
        }
    });

    // Round table actions
    const tbody = app.container.querySelector('.score-rounds tbody');
    if (tbody) {
        tbody.addEventListener('click', e => {
            const del = e.target.closest('[data-delete-idx]');
            if (del) {
                e.stopPropagation();
                const idx = parseInt(del.dataset.deleteIdx, 10);
                if (confirm(`Supprimer la manche #${idx + 1} ?`)) {
                    app.game.rounds.splice(idx, 1);
                    saveCurrent(app.game);
                    render();
                }
                return;
            }
            const row = e.target.closest('tr.round-row');
            if (row) openForm(parseInt(row.dataset.roundIdx, 10));
        });
    }

    bindHistory();
}

function archiveGame() {
    app.game.finishedAt = Date.now();
    app.history.unshift(app.game);
    if (app.history.length > 30) app.history.length = 30;
    saveHistory(app.history);
    app.game = null;
    saveCurrent(null);
    render();
}

// ===== Round form =====

function defaultDraft() {
    return {
        taker: 0,
        value: 8, // 8 → 80 pts (×10)
        isCapot: false,
        coinche: 0,
        takerPts: 80,
        belote: -1,
        capotRealise: false,
    };
}

function openForm(editingIndex) {
    let draft;
    if (editingIndex == null) {
        draft = defaultDraft();
    } else {
        const r = app.game.rounds[editingIndex];
        draft = { ...r };
    }
    app.formState = { editingIndex, draft };
    renderForm();
}

function closeForm() {
    app.formState = null;
    const m = $('#score-modal-root');
    if (m) m.remove();
}

function renderForm() {
    const old = $('#score-modal-root');
    if (old) old.remove();

    const root = document.createElement('div');
    root.id = 'score-modal-root';
    document.body.appendChild(root);

    const d = app.formState.draft;
    const editing = app.formState.editingIndex != null;

    root.innerHTML = `
        <div class="score-modal" id="score-modal-bg">
            <div class="score-modal-content">
                <h3>${editing ? 'Modifier' : 'Nouvelle'} manche${editing ? ` #${app.formState.editingIndex + 1}` : ''}</h3>
                <div class="score-form" id="score-form-body"></div>
            </div>
        </div>
    `;

    renderFormBody();

    $('#score-modal-bg').addEventListener('click', e => {
        if (e.target.id === 'score-modal-bg') closeForm();
    });
    document.addEventListener('keydown', escClose);
}

function escClose(e) {
    if (e.key === 'Escape' && app.formState) {
        closeForm();
        document.removeEventListener('keydown', escClose);
    }
}

function renderFormBody() {
    const d = app.formState.draft;
    const editing = app.formState.editingIndex != null;
    const body = $('#score-form-body');

    const valueButtons = BID_VALUES.map(v => `
        <button type="button" class="btn-pill ${!d.isCapot && d.value === v / 10 ? 'active' : ''}" data-bid-val="${v / 10}">${v}</button>
    `).join('') + `
        <button type="button" class="btn-pill ${d.isCapot ? 'active' : ''}" data-bid-capot="1">Capot</button>
    `;

    const coincheButtons = [
        ['0', 'Aucune'],
        ['1', 'Coinché ×2'],
        ['2', 'Surcoinché ×3'],
    ].map(([v, lbl]) => `
        <button type="button" class="btn-pill ${d.coinche === parseInt(v, 10) ? 'active' : ''}" data-coinche="${v}">${lbl}</button>
    `).join('');

    const teamButtons = [0, 1].map(t => `
        <button type="button" class="btn-pill ${d.taker === t ? 'active' : ''}" data-taker="${t}">${escapeHtml(app.game.teams[t])}</button>
    `).join('');

    const beloteButtons = [
        [-1, 'Aucune'],
        [0, app.game.teams[0]],
        [1, app.game.teams[1]],
    ].map(([v, lbl]) => `
        <button type="button" class="btn-pill ${d.belote === v ? 'active' : ''}" data-belote="${v}">${escapeHtml(lbl)}</button>
    `).join('');

    // Points input section: hidden if capot announced
    const ptsSection = d.isCapot ? `
        <div class="field">
            <span class="field-label">Capot réussi ?</span>
            <div class="btn-row">
                <button type="button" class="btn-pill ${d.capotRealise ? 'active' : ''}" data-capot-yes="1">Oui (8 plis pris)</button>
                <button type="button" class="btn-pill ${!d.capotRealise ? 'active' : ''}" data-capot-yes="0">Non (chute)</button>
            </div>
        </div>
    ` : `
        <div class="field">
            <span class="field-label">Points faits par les preneurs (sur 162)</span>
            <div class="pts-input-row">
                <button type="button" class="pts-step" data-pts-step="-10">−10</button>
                <button type="button" class="pts-step" data-pts-step="-1">−1</button>
                <input type="number" id="pts-input" min="0" max="162" inputmode="numeric" value="${d.takerPts}">
                <button type="button" class="pts-step" data-pts-step="1">+1</button>
                <button type="button" class="pts-step" data-pts-step="10">+10</button>
            </div>
        </div>

        ${d.coinche > 0 ? `
            <div class="field">
                <label class="toggle-row">
                    <input type="checkbox" id="capot-realise" ${d.capotRealise ? 'checked' : ''}>
                    <span class="toggle-label">Tous les plis pris (capot réalisé)</span>
                    <span class="toggle-hint">base 250</span>
                </label>
            </div>
        ` : ''}
    `;

    const { scores, reussi } = computeRoundScore(d);

    body.innerHTML = `
        <div class="field">
            <span class="field-label">Preneurs</span>
            <div class="btn-row team-row" id="team-row">${teamButtons}</div>
        </div>

        <div class="field">
            <span class="field-label">Contrat</span>
            <div class="btn-row" id="bid-row">${valueButtons}</div>
        </div>

        <div class="field">
            <span class="field-label">Multiplicateur</span>
            <div class="btn-row" id="coinche-row">${coincheButtons}</div>
        </div>

        ${ptsSection}

        <div class="field">
            <span class="field-label">Belote / Rebelote</span>
            <div class="btn-row" id="belote-row">${beloteButtons}</div>
        </div>

        <div class="preview">
            <span class="preview-status ${reussi ? 'reussi' : 'chute'}">${reussi ? 'Réussi' : 'Chute'}</span>
            <div class="preview-scores">
                <span>${escapeHtml(app.game.teams[0])} : <strong>+${scores[0]}</strong></span>
                <span>${escapeHtml(app.game.teams[1])} : <strong>+${scores[1]}</strong></span>
            </div>
        </div>

        <div class="form-actions">
            <button type="button" class="cancel-btn" id="form-cancel">Annuler</button>
            <button type="button" class="submit-btn" id="form-submit">${editing ? 'Enregistrer' : 'Ajouter'}</button>
        </div>
    `;

    bindForm();
}

function bindForm() {
    const d = app.formState.draft;

    $('#team-row').addEventListener('click', e => {
        const b = e.target.closest('[data-taker]');
        if (b) { d.taker = parseInt(b.dataset.taker, 10); renderFormBody(); }
    });

    $('#bid-row').addEventListener('click', e => {
        const v = e.target.closest('[data-bid-val]');
        if (v) {
            d.value = parseInt(v.dataset.bidVal, 10);
            d.isCapot = false;
            d.takerPts = d.value * 10;
            renderFormBody();
            return;
        }
        const c = e.target.closest('[data-bid-capot]');
        if (c) { d.isCapot = true; d.capotRealise = true; renderFormBody(); }
    });

    $('#coinche-row').addEventListener('click', e => {
        const b = e.target.closest('[data-coinche]');
        if (b) { d.coinche = parseInt(b.dataset.coinche, 10); renderFormBody(); }
    });

    $('#belote-row').addEventListener('click', e => {
        const b = e.target.closest('[data-belote]');
        if (b) { d.belote = parseInt(b.dataset.belote, 10); renderFormBody(); }
    });

    if (d.isCapot) {
        document.querySelectorAll('[data-capot-yes]').forEach(btn => {
            btn.addEventListener('click', () => { d.capotRealise = btn.dataset.capotYes === '1'; renderFormBody(); });
        });
    } else {
        const input = $('#pts-input');
        if (input) {
            input.addEventListener('input', () => {
                let v = parseInt(input.value, 10);
                if (Number.isNaN(v)) v = 0;
                v = Math.max(0, Math.min(162, v));
                d.takerPts = v;
                updatePreviewOnly();
            });
            input.addEventListener('blur', () => { renderFormBody(); });
        }
        document.querySelectorAll('[data-pts-step]').forEach(b => {
            b.addEventListener('click', () => {
                const step = parseInt(b.dataset.ptsStep, 10);
                d.takerPts = Math.max(0, Math.min(162, d.takerPts + step));
                renderFormBody();
            });
        });
        const cr = $('#capot-realise');
        if (cr) cr.addEventListener('change', () => { d.capotRealise = cr.checked; renderFormBody(); });
    }

    $('#form-cancel').addEventListener('click', closeForm);
    $('#form-submit').addEventListener('click', submitRound);
}

function updatePreviewOnly() {
    const d = app.formState.draft;
    const { scores, reussi } = computeRoundScore(d);
    const status = document.querySelector('.preview-status');
    if (status) {
        status.textContent = reussi ? 'Réussi' : 'Chute';
        status.classList.toggle('reussi', reussi);
        status.classList.toggle('chute', !reussi);
    }
    const scoreEls = document.querySelectorAll('.preview-scores strong');
    if (scoreEls.length === 2) {
        scoreEls[0].textContent = `+${scores[0]}`;
        scoreEls[1].textContent = `+${scores[1]}`;
    }
}

function submitRound() {
    const d = app.formState.draft;
    const { scores } = computeRoundScore(d);
    const round = { ...d, scores };

    if (app.formState.editingIndex != null) {
        app.game.rounds[app.formState.editingIndex] = round;
    } else {
        app.game.rounds.push(round);
    }
    saveCurrent(app.game);
    closeForm();
    document.removeEventListener('keydown', escClose);
    render();
}

// ===== Lifecycle =====

export function mount(container) {
    app.container = container;
    app.game = loadCurrent();
    app.history = loadHistory();
    render();
}

export function unmount() {
    closeForm();
    document.removeEventListener('keydown', escClose);
    app.container = null;
    app.game = null;
    app.history = [];
    app.formState = null;
}
