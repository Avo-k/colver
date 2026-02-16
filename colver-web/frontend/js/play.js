// Play mode logic

const HUMAN_SEAT = 2; // South

let bidHistory = [];

document.getElementById('start-game').addEventListener('click', () => {
    const ai = document.getElementById('ai-type').value;
    const timeMs = parseInt(document.getElementById('ai-time').value);
    bidHistory = [];
    send({ type: 'start_game', ai, time_ms: timeMs, human_seat: HUMAN_SEAT });
    document.getElementById('play-table').classList.remove('hidden');
    document.getElementById('game-result').classList.add('hidden');
    document.getElementById('play-status').textContent = 'Lancement de la partie...';
});

function renderPlayState(state) {
    // Score
    document.getElementById('score-ns').textContent = `NS : ${state.points[0]} (${state.tricks_won[0]}P)`;
    document.getElementById('score-ew').textContent = `EO : ${state.points[1]} (${state.tricks_won[1]}P)`;
    document.getElementById('contract-display').textContent = contractStr(state.contract);

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
            const count = cards.length || Math.max(0, 8 - (state.tricks_won[0] + state.tricks_won[1]));
            renderFaceDownHand(handEls[seat], count);
        }
    }

    // Trick
    renderTrick('trick', state.current_trick);

    // Last completed trick (from server)
    const lastTrickEl = document.getElementById('last-trick');
    if (lastTrickEl) {
        if (isPlayPhase && state.last_trick) {
            renderLastTrick(lastTrickEl, state.last_trick, state.last_trick_winner, state.last_trick_points, HUMAN_SEAT);
        } else {
            lastTrickEl.classList.add('hidden');
            lastTrickEl.innerHTML = '';
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
        const resultEl = document.getElementById('game-result');
        resultEl.classList.remove('hidden');
        const ns = state.points[0], ew = state.points[1];
        const winner = ns > ew ? 'NS gagne !' : ew > ns ? 'EO gagne !' : 'Egalite !';
        resultEl.textContent = `Partie terminee ! NS : ${ns} - EO : ${ew}. ${winner}`;
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
            const val = parseInt(bidValue.value);
            const suit = parseInt(bidSuit.value);
            if (isNaN(val)) return;
            const action = encodeBidAction(val, suit);
            if (action < 0 || !legalSet.has(action)) return;
            const name = actionName(action, 0);
            bidHistory.push({ player: HUMAN_SEAT, action, name });
            send({ type: 'play', action, human_seat: HUMAN_SEAT });
        };
    }

    // Pass button
    const passBtn = document.getElementById('bid-pass');
    if (legalSet.has(0)) {
        passBtn.classList.remove('hidden');
        passBtn.disabled = false;
        passBtn.onclick = () => {
            bidHistory.push({ player: HUMAN_SEAT, action: 0, name: 'Passe' });
            send({ type: 'play', action: 0, human_seat: HUMAN_SEAT });
        };
    } else {
        passBtn.classList.add('hidden');
    }

    // Coinche button
    const coincheBtn = document.getElementById('bid-coinche');
    if (legalSet.has(41)) {
        coincheBtn.classList.remove('hidden');
        coincheBtn.onclick = () => {
            bidHistory.push({ player: HUMAN_SEAT, action: 41, name: 'Coinche' });
            send({ type: 'play', action: 41, human_seat: HUMAN_SEAT });
        };
    } else {
        coincheBtn.classList.add('hidden');
    }

    // Surcoinche button
    const surcoincheBtn = document.getElementById('bid-surcoinche');
    if (legalSet.has(42)) {
        surcoincheBtn.classList.remove('hidden');
        surcoincheBtn.onclick = () => {
            bidHistory.push({ player: HUMAN_SEAT, action: 42, name: 'Surcoinche' });
            send({ type: 'play', action: 42, human_seat: HUMAN_SEAT });
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
    send({ type: 'play', action: cardIdx, human_seat: HUMAN_SEAT });
}

// Message handlers
onMessage('game_state', (data) => {
    renderPlayState(data.state);
    if (data.belote_event) {
        const text = data.belote_event === 'belote' ? 'Belote !' : 'Rebelote !';
        showBeloteAnnouncement('trick-area', text);
    }
    // Disable DouDou option if not available on server, fall back to Smart
    if (data.doudou_available === false) {
        const opt = document.querySelector('#ai-type option[value="doudou"]');
        if (opt) {
            opt.disabled = true;
            opt.textContent = 'DouDou (non dispo)';
        }
        const sel = document.getElementById('ai-type');
        if (sel.value === 'doudou') sel.value = 'smart';
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
