// Play mode logic

const HUMAN_SEAT = 2; // South

let bidHistory = [];
let playLocked = false; // Prevent double-clicks while waiting for server
let _pendingPlayState = null; // Queued state during trick animation

function getMoveDelay() {
    return parseInt(document.getElementById('move-delay').value);
}

document.getElementById('move-delay').addEventListener('input', (e) => {
    document.getElementById('move-delay-val').textContent = `${e.target.value}s`;
});

document.getElementById('start-game').addEventListener('click', () => {
    const opponentAi = document.getElementById('opponent-ai').value;
    const partnerAi = document.getElementById('partner-ai').value;
    bidHistory = [];
    playLocked = false;
    _prevTrick['trick'] = [];
    _animatingTrick = null;
    _pendingPlayState = null;
    send({ type: 'start_game', opponent_ai: opponentAi, partner_ai: partnerAi, human_seat: HUMAN_SEAT, move_delay: getMoveDelay() });
    document.getElementById('play-table').classList.remove('hidden');
    document.getElementById('game-result').classList.add('hidden');
    document.getElementById('game-result').innerHTML = '';
    document.getElementById('confetti-container').innerHTML = '';
    document.getElementById('play-status').textContent = 'Lancement de la partie...';
});

function renderPlayState(state) {
    // Queue state if trick animation is running — render after it completes
    if (_animatingTrick === 'trick') {
        _pendingPlayState = state;
        return;
    }

    // Score
    document.getElementById('score-ns').textContent = `NS : ${state.points[0]} (${state.tricks_won[0]}P)`;
    document.getElementById('score-ew').textContent = `EO : ${state.points[1]} (${state.tricks_won[1]}P)`;
    document.getElementById('contract-display').textContent = contractStr(state.contract);
    updateCfnBox('play-cfn', state.cfn);

    // Belote badges
    if (state.belote) {
        renderBeloteBadge('score-ns', state.belote[0]);
        renderBeloteBadge('score-ew', state.belote[1]);
    }

    // Hands
    const handEls = {
        0: document.getElementById('hand-north'),
        1: document.getElementById('hand-east'),
        2: document.getElementById('hand-south'),
        3: document.getElementById('hand-west'),
    };

    const isHumanTurn = state.current_player === HUMAN_SEAT && !state.is_terminal;
    const isPlayPhase = state.phase === 1;
    const isBidPhase = state.phase === 0;

    const legalSet = (isHumanTurn && isPlayPhase) ? new Set(state.legal_actions) : null;

    const trumpSuit = (state.contract && state.contract.trump !== undefined) ? state.contract.trump : -1;

    for (let seat = 0; seat < 4; seat++) {
        const cards = state.hands[seat];
        if (seat === HUMAN_SEAT) {
            const clickable = isHumanTurn && isPlayPhase;
            renderHand(handEls[seat], cards, clickable, clickable ? playCard : null, legalSet, trumpSuit);
        } else {
            const tricksPlayed = state.tricks_won[0] + state.tricks_won[1];
            const hasPlayedThisTrick = state.current_trick[seat] >= 0 && state.current_trick[seat] < 32;
            const count = cards.length || Math.max(0, 8 - tricksPlayed - (hasPlayedThisTrick ? 1 : 0));
            renderFaceDownHand(handEls[seat], count);
        }
    }

    // Trick (with flush animation)
    const completedCards = detectTrickCompletion('trick', state.current_trick);
    if (completedCards && _animatingTrick !== 'trick') {
        // Trick just completed: animate flush, then update last-trick
        animateTrickFlush('trick', () => {
            const lastTrickEl = document.getElementById('last-trick');
            if (lastTrickEl && isPlayPhase && state.last_trick) {
                renderLastTrick(lastTrickEl, state.last_trick, state.last_trick_winner, state.last_trick_points, HUMAN_SEAT);
            }
            // Flush any state that arrived during animation
            if (_pendingPlayState) {
                const pending = _pendingPlayState;
                _pendingPlayState = null;
                renderPlayState(pending);
            }
        });
        // Render the new (empty/partial) trick underneath the overlay
        renderTrick('trick', state.current_trick);
    } else {
        renderTrick('trick', state.current_trick);
        // Last completed trick — skip update when 4 cards are showing
        // (server sends last_trick early; wait for the flush animation instead)
        const trickFull = state.current_trick && state.current_trick.filter(c => c >= 0 && c < 32).length === 4;
        if (!trickFull) {
            const lastTrickEl = document.getElementById('last-trick');
            if (lastTrickEl) {
                if (isPlayPhase && state.last_trick) {
                    renderLastTrick(lastTrickEl, state.last_trick, state.last_trick_winner, state.last_trick_points, HUMAN_SEAT);
                } else {
                    lastTrickEl.classList.add('hidden');
                    lastTrickEl.innerHTML = '';
                }
            }
        }
    }

    // Bidding panel
    const biddingPanel = document.getElementById('bidding-panel');
    if (isBidPhase) {
        biddingPanel.classList.remove('hidden');
        renderBidHistory();
        if (isHumanTurn) {
            showBidControls(state.legal_actions, state);
        } else {
            hideBidControls();
        }
    } else {
        biddingPanel.classList.add('hidden');
    }

    // Status
    if (state.is_terminal) {
        showGameResult(state);
        document.getElementById('play-status').textContent = '';
    } else if (isHumanTurn) {
        document.getElementById('play-status').textContent = isBidPhase ? 'A vous d\'annoncer' : 'A vous de jouer';
    } else {
        document.getElementById('play-status').textContent = `${SEAT_NAMES_FR[state.current_player]} reflechit...`;
    }
}

function renderBidHistory() {
    const el = document.getElementById('bid-history');
    el.innerHTML = '';
    for (const entry of bidHistory) {
        const span = document.createElement('span');
        // Team: 0=NS (players 0,2), 1=EO (players 1,3)
        const isPartnerTeam = (entry.player % 2) === (HUMAN_SEAT % 2);
        span.className = 'bid-entry' + (isPartnerTeam ? ' team-partner' : ' team-opponent');
        span.textContent = `${SEAT_NAMES_FR[entry.player]} : ${entry.name}`;
        el.appendChild(span);
    }
}

// Encode a bid from value+suit dropdowns to action index
function encodeBidAction(value, suitIdx) {
    const values = [80,90,100,110,120,130,140,150,160];
    if (value === 250) {
        // Capot
        return 37 + suitIdx;
    }
    const valIdx = values.indexOf(value);
    if (valIdx < 0) return -1;
    return valIdx * 4 + suitIdx + 1;
}

function showBidControls(legalActions, state) {
    const legalSet = new Set(legalActions);

    // Show/hide bid selectors + submit
    const bidSelectors = document.getElementById('bid-selectors');
    const bidSubmit = document.getElementById('bid-submit');
    const bidValue = document.getElementById('bid-value');
    const bidSuit = document.getElementById('bid-suit');

    // Check if any bid (non-pass, non-coinche) is legal
    const hasBids = legalActions.some(a => a >= 1 && a <= 40);
    bidSelectors.style.display = hasBids ? 'flex' : 'none';

    if (hasBids) {
        // Enable only legal value options
        const values = [80,90,100,110,120,130,140,150,160,250];
        for (const opt of bidValue.options) {
            if (opt.value === '') continue; // placeholder
            const v = parseInt(opt.value);
            // A value is available if any suit with that value is legal
            let available = false;
            for (let s = 0; s < 4; s++) {
                if (legalSet.has(encodeBidAction(v, s))) { available = true; break; }
            }
            opt.disabled = !available;
        }
        // Default to lowest legal value
        let firstLegal = '';
        for (const opt of bidValue.options) {
            if (opt.value !== '' && !opt.disabled) { firstLegal = opt.value; break; }
        }
        bidValue.value = firstLegal;

        // Default suit to the one with highest trump potential
        if (state && state.best_trump_suit !== undefined) {
            bidSuit.value = String(state.best_trump_suit);
        }

        bidSubmit.onclick = () => {
            if (playLocked) return;
            const val = parseInt(bidValue.value);
            const suit = parseInt(bidSuit.value);
            if (isNaN(val)) return;
            const action = encodeBidAction(val, suit);
            if (action < 0 || !legalSet.has(action)) return;
            playLocked = true;
            const name = actionName(action, 0);
            bidHistory.push({ player: HUMAN_SEAT, action, name });
            send({ type: 'play', action, human_seat: HUMAN_SEAT, move_delay: getMoveDelay() });
        };
    }

    // Pass button
    const passBtn = document.getElementById('bid-pass');
    if (legalSet.has(0)) {
        passBtn.classList.remove('hidden');
        passBtn.disabled = false;
        passBtn.onclick = () => {
            if (playLocked) return;
            playLocked = true;
            bidHistory.push({ player: HUMAN_SEAT, action: 0, name: 'Passe' });
            send({ type: 'play', action: 0, human_seat: HUMAN_SEAT, move_delay: getMoveDelay() });
        };
    } else {
        passBtn.classList.add('hidden');
    }

    // Coinche button
    const coincheBtn = document.getElementById('bid-coinche');
    if (legalSet.has(41)) {
        coincheBtn.classList.remove('hidden');
        coincheBtn.onclick = () => {
            if (playLocked) return;
            playLocked = true;
            bidHistory.push({ player: HUMAN_SEAT, action: 41, name: 'Coinche' });
            send({ type: 'play', action: 41, human_seat: HUMAN_SEAT, move_delay: getMoveDelay() });
        };
    } else {
        coincheBtn.classList.add('hidden');
    }

    // Surcoinche button
    const surcoincheBtn = document.getElementById('bid-surcoinche');
    if (legalSet.has(42)) {
        surcoincheBtn.classList.remove('hidden');
        surcoincheBtn.onclick = () => {
            if (playLocked) return;
            playLocked = true;
            bidHistory.push({ player: HUMAN_SEAT, action: 42, name: 'Surcoinche' });
            send({ type: 'play', action: 42, human_seat: HUMAN_SEAT, move_delay: getMoveDelay() });
        };
    } else {
        surcoincheBtn.classList.add('hidden');
    }
}

function hideBidControls() {
    document.getElementById('bid-selectors').style.display = 'none';
    document.getElementById('bid-pass').classList.add('hidden');
    document.getElementById('bid-coinche').classList.add('hidden');
    document.getElementById('bid-surcoinche').classList.add('hidden');
}

function playCard(cardIdx) {
    if (playLocked) return;
    playLocked = true;
    // Optimistic update: show card in trick area and remove from hand immediately
    const trickEl = document.getElementById('trick-s');
    trickEl.innerHTML = '';
    trickEl.appendChild(cardToHtml(cardIdx));
    const handEl = document.getElementById('hand-south');
    const cardEl = handEl.querySelector(`[data-card="${cardIdx}"]`);
    if (cardEl) cardEl.remove();
    send({ type: 'play', action: cardIdx, human_seat: HUMAN_SEAT, move_delay: getMoveDelay() });
}

// Message handlers
onMessage('game_state', (data) => {
    renderPlayState(data.state);
    // Unlock input when it's the human's turn or game is over
    if (data.state.is_terminal || data.state.current_player === HUMAN_SEAT) {
        playLocked = false;
    }
    if (data.game_id) setPlayGameId(data.game_id);
    if (data.belote_event) {
        const text = data.belote_event === 'belote' ? 'Belote !' : 'Rebelote !';
        showBeloteAnnouncement('trick-area', text);
    }
    // Disable DouDou options if not available on server, fall back to Smart
    if (data.doudou_available === false) {
        for (const selId of ['opponent-ai', 'partner-ai']) {
            const opt = document.querySelector(`#${selId} option[value="doudou"]`);
            if (opt) {
                opt.disabled = true;
                opt.textContent = 'DouDou35 (non dispo)';
            }
            const sel = document.getElementById(selId);
            if (sel.value === 'doudou') sel.value = 'smart';
        }
    }
});

onMessage('ai_move', (data) => {
    if (data.player !== undefined) {
        // Use JS actionName for proper suit symbols instead of server-provided name
        const name = actionName(data.action, 0);
        bidHistory.push({ player: data.player, action: data.action, name });
    }
    if (data.belote_event) {
        const text = data.belote_event === 'belote' ? 'Belote !' : 'Rebelote !';
        showBeloteAnnouncement('trick-area', text);
    }
});

onMessage('error', (data) => {
    console.error('Erreur serveur:', data.msg);
    document.getElementById('play-status').textContent = `Erreur : ${data.msg}`;
});

function showGameResult(state) {
    const resultEl = document.getElementById('game-result');
    resultEl.classList.remove('hidden');

    const ns = state.points[0], ew = state.points[1];
    // Human is South (NS team)
    const isVictory = ns > ew;
    const isDraw = ns === ew;
    const titleText = isVictory ? 'Victoire' : isDraw ? 'Egalite' : 'Defaite';
    const titleClass = isVictory ? 'victory' : isDraw ? 'draw' : 'defeat';

    const contract = contractStr(state.contract);

    // Belote info (state.belote[team] === 2 means belote+rebelote declared)
    const beloteNS = (state.belote && state.belote[0] === 2) ? ' <span class="belote-note">(dont 20 belote)</span>' : '';
    const beloteEW = (state.belote && state.belote[1] === 2) ? ' <span class="belote-note">(dont 20 belote)</span>' : '';

    resultEl.innerHTML =
        `<div class="result-title ${titleClass}">${titleText}</div>` +
        (contract ? `<div class="result-contract">${contract}</div>` : '') +
        `<div class="result-scores">` +
            `<div class="team-ns">NS : ${ns}${beloteNS}</div>` +
            `<div class="team-ew">EO : ${ew}${beloteEW}</div>` +
        `</div>` +
        `<button class="result-restart" onclick="document.getElementById('start-game').click()">Nouvelle partie</button>`;

    if (isVictory) {
        launchConfetti();
    }
}

function launchConfetti() {
    const container = document.getElementById('confetti-container');
    container.innerHTML = '';
    const colors = ['#d4af37', '#4caf50', '#42a5f5', '#ef5350', '#ab47bc', '#ff9800', '#e0e0e0'];
    const count = 35;
    const rect = container.getBoundingClientRect();
    const fallDist = rect.height || 500;

    for (let i = 0; i < count; i++) {
        const el = document.createElement('div');
        el.className = 'confetti-piece';
        const color = colors[Math.floor(Math.random() * colors.length)];
        const left = Math.random() * 100;
        const delay = Math.random() * 0.8;
        const duration = 1.8 + Math.random() * 1.4;
        const spin = (Math.random() * 720 - 360) + 'deg';
        const size = 5 + Math.random() * 6;
        const shape = Math.random() > 0.5 ? '50%' : '2px';

        el.style.cssText =
            `left:${left}%;` +
            `width:${size}px;height:${size}px;` +
            `background:${color};` +
            `border-radius:${shape};` +
            `animation-duration:${duration}s;` +
            `animation-delay:${delay}s;` +
            `--fall-dist:${fallDist}px;` +
            `--spin:${spin};`;

        container.appendChild(el);
        el.addEventListener('animationend', () => el.remove());
    }
}
