// Play mode logic

const HUMAN_SEAT = 2; // South

let bidHistory = [];
let playLocked = false; // Prevent double-clicks while waiting for server
let _pendingPlayState = null; // Queued state during trick animation
let _initialHands = null; // Stored at game start for end-of-game review
let _playGameId = null; // Current game ID for analyse button
let _serverBidHistory = null; // Bid history from server (terminal message)
let _serverCompletedTricks = null; // Completed tricks from server (terminal message)

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
    _initialHands = null;
    _playGameId = null;
    _serverBidHistory = null;
    _serverCompletedTricks = null;
    send({ type: 'start_game', opponent_ai: opponentAi, partner_ai: partnerAi, human_seat: HUMAN_SEAT, move_delay: getMoveDelay() });
    document.getElementById('play-table').classList.remove('hidden');
    document.getElementById('game-result').classList.add('hidden');
    document.getElementById('game-result').innerHTML = '';
    document.getElementById('confetti-container').innerHTML = '';
    document.getElementById('play-review').classList.add('hidden');
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
        }, state.last_trick_winner);
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
        showEndOfGameReview(state);
        document.getElementById('play-status').textContent = '';
    } else if (isHumanTurn) {
        document.getElementById('play-status').textContent = isBidPhase ? 'A vous d\'annoncer' : 'A vous de jouer';
        SFX.yourTurn();
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
            SFX.bid();
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
            SFX.pass();
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
            SFX.coinche();
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
            SFX.surcoinche();
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
    SFX.cardPlay();
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
    // Store initial hands when first received (game start or terminal)
    if (data.initial_hands) _initialHands = data.initial_hands;
    // Store bid_history and completed_tricks from terminal messages
    if (data.bid_history) _serverBidHistory = data.bid_history;
    if (data.completed_tricks) _serverCompletedTricks = data.completed_tricks;
    if (data.game_id) {
        _playGameId = data.game_id;
        setPlayGameId(data.game_id);
    }
    renderPlayState(data.state);
    // Unlock input when it's the human's turn or game is over
    if (data.state.is_terminal || data.state.current_player === HUMAN_SEAT) {
        playLocked = false;
    }
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
        SFX.playForAction(0, data.action);
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

    // Use rewards (contract-aware scoring) to determine victory/defeat
    const rewards = state.rewards;
    const isVictory = rewards ? rewards[0] > rewards[1] : state.points[0] > state.points[1];
    const isDraw = rewards ? rewards[0] === rewards[1] : state.points[0] === state.points[1];
    const titleText = isVictory ? 'Victoire' : isDraw ? 'Egalite' : 'Defaite';
    const titleClass = isVictory ? 'victory' : isDraw ? 'draw' : 'defeat';

    const contract = contractStr(state.contract);
    const sd = state.score_detail;

    let scoresHtml = '';
    if (sd) {
        const teamNames = ['NS', 'EO'];
        const contractTeamName = teamNames[sd.contract_team];
        const contractResult = sd.contract_made ? 'Reussi' : 'Chute';
        const contractClass = sd.contract_made ? 'contract-made' : 'contract-failed';
        const suitSymbols = ['\u2660', '\u2665', '\u2666', '\u2663'];
        scoresHtml += `<div class="result-contract-detail ${contractClass}">${sd.contract_value}${suitSymbols[state.contract.trump]} par ${contractTeamName} — ${contractResult}</div>`;
        scoresHtml += `<div class="result-score-line">Plis : NS ${sd.trick_points[0]} — EO ${sd.trick_points[1]}</div>`;
        if (sd.belote[0] > 0 || sd.belote[1] > 0) {
            const parts = [];
            if (sd.belote[0] > 0) parts.push(`+${sd.belote[0]} belote NS`);
            if (sd.belote[1] > 0) parts.push(`+${sd.belote[1]} belote EO`);
            scoresHtml += `<div class="result-score-line">${parts.join(' / ')}</div>`;
        }
        scoresHtml += `<div class="result-final-scores">Score : NS ${sd.final_scores[0]} — EO ${sd.final_scores[1]}</div>`;
    } else {
        let ns = state.points[0], ew = state.points[1];
        const hasBeloteNS = state.belote && state.belote[0] === 2;
        const hasBeloteEW = state.belote && state.belote[1] === 2;
        if (hasBeloteNS) ns += 20;
        if (hasBeloteEW) ew += 20;
        const beloteNS = hasBeloteNS ? ' <span class="belote-note">(dont belote)</span>' : '';
        const beloteEW = hasBeloteEW ? ' <span class="belote-note">(dont belote)</span>' : '';
        scoresHtml = `<div class="team-ns">NS : ${ns}${beloteNS}</div><div class="team-ew">EO : ${ew}${beloteEW}</div>`;
    }

    resultEl.innerHTML =
        `<div class="result-title ${titleClass}">${titleText}</div>` +
        (contract ? `<div class="result-contract">${contract}</div>` : '') +
        `<div class="result-scores">${scoresHtml}</div>` +
        `<div class="result-buttons">` +
            `<button class="result-restart" onclick="document.getElementById('start-game').click()">Nouvelle partie</button>` +
            `<button class="result-analyse" id="result-analyse-btn">Analyser</button>` +
        `</div>`;

    document.getElementById('result-analyse-btn').addEventListener('click', () => {
        if (!_playGameId) return;
        // Switch to Watch tab and load the replay
        document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        const watchTab = document.querySelector('.tab[data-tab="watch"]');
        watchTab.classList.add('active');
        document.getElementById('watch-panel').classList.add('active');
        // Trigger history load and then load this game's replay
        if (!watchTabVisited) {
            watchTabVisited = true;
            loadGameHistory(false);
        } else {
            loadGameHistory(false);
        }
        loadReplay(_playGameId);
    });

    if (isVictory) {
        SFX.victory();
        launchConfetti();
    } else if (!isDraw) {
        SFX.defeat();
    }
}

function showEndOfGameReview(state) {
    // Clear trick area (no cards behind the result overlay)
    renderTrick('trick', [-1, -1, -1, -1]);

    // Hide last trick box
    const lastTrickEl = document.getElementById('last-trick');
    if (lastTrickEl) {
        lastTrickEl.classList.add('hidden');
        lastTrickEl.innerHTML = '';
    }

    // Show initial hands (all 4 face-up)
    if (_initialHands) {
        const handEls = {
            0: document.getElementById('hand-north'),
            1: document.getElementById('hand-east'),
            2: document.getElementById('hand-south'),
            3: document.getElementById('hand-west'),
        };
        const trumpSuit = (state.contract && state.contract.trump !== undefined) ? state.contract.trump : -1;
        for (let seat = 0; seat < 4; seat++) {
            renderHand(handEls[seat], _initialHands[seat], false, null, null, trumpSuit);
        }
    }

    // Show review panel with bid history and tricks
    const reviewEl = document.getElementById('play-review');
    reviewEl.classList.remove('hidden');

    // Render bid history
    const bidContainer = document.getElementById('play-bid-entries');
    bidContainer.innerHTML = '';
    const bids = _serverBidHistory || bidHistory.map(b => ({ player: b.player, action: b.action }));
    for (const bid of bids) {
        const el = document.createElement('span');
        const team = bid.player % 2 === 0 ? 'team-ns' : 'team-ew';
        el.className = `watch-bid-entry ${team}`;
        const seatLetter = ['N', 'E', 'S', 'O'][bid.player];
        const name = actionName(bid.action, 0);
        el.textContent = `${seatLetter}:${name}`;
        bidContainer.appendChild(el);
    }

    // Render tricks history
    const tricksContainer = document.getElementById('play-tricks-list');
    tricksContainer.innerHTML = '';
    const tricks = _serverCompletedTricks || [];
    for (let i = 0; i < tricks.length; i++) {
        const t = tricks[i];
        const row = document.createElement('div');
        const winnerTeam = t.winner % 2 === 0 ? 'team-ns' : 'team-ew';
        row.className = `trick-history-row ${winnerTeam}`;

        const SEAT_L = ['N', 'E', 'S', 'O'];
        const leadSeat = t.lead !== undefined ? t.lead : -1;

        let orderedCards = '';
        if (leadSeat >= 0) {
            for (let j = 0; j < 4; j++) {
                const seat = (leadSeat + j) % 4;
                const c = t.cards[seat];
                if (c >= 0 && c < 32) {
                    orderedCards += `${RANKS[cardRank(c)]}${SUITS[cardSuit(c)]} `;
                }
            }
            orderedCards = orderedCards.trim();
        } else {
            orderedCards = t.cards.map(c => {
                if (c >= 0 && c < 32) return `${RANKS[cardRank(c)]}${SUITS[cardSuit(c)]}`;
                return '?';
            }).join(' ');
        }

        const leadLabel = leadSeat >= 0 ? SEAT_L[leadSeat] : '?';
        const winnerName = SEAT_NAMES_FR[t.winner];
        row.innerHTML = `<span class="trick-num">#${i + 1}</span>` +
            `<span class="trick-lead-label">${leadLabel}</span>` +
            `<span class="trick-cards">${orderedCards}</span>` +
            `<span class="trick-winner">${winnerName} +${t.points}</span>`;
        tricksContainer.appendChild(row);
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
