// Shared game table — rendering + interaction extracted from the solo play
// view so multiplayer rooms reuse the exact same UI. The viewer is ALWAYS
// display-seat 2 (South): the solo server filters for seat 2, the multiplayer
// server rotates every state into the viewer's frame before sending.

import * as SFX from '../sounds.js';
import {
    RANKS, SUITS, SEAT_NAMES_FR,
    cardSuit, cardRank, cardToHtml,
    renderHand, renderFaceDownHand, renderTrick, renderLastTrick,
    contractStr, actionName, encodeBidAction, bidActionHtml,
    showBeloteAnnouncement, renderBeloteBadge,
    _prevTrick, _animatingTrick, setAnimatingTrick,
    detectTrickCompletion, animateTrickFlush
} from './cards.js';
import { setGameId as setBugReportGameId, openBugReport } from './bug-report.js';

export const MY_SEAT = 2; // South, always (server-side rotation guarantees it)

export const TABLE_TEMPLATE = `
<div id="play-table" class="table hidden">
    <div id="score-bar">
        <span id="score-ns">NS : 0</span>
        <span id="play-game-id" class="game-id-tag hidden"></span>
        <span id="contract-display"></span>
        <span id="score-ew">EO : 0</span>
        <button id="play-report-btn" class="report-btn hidden" title="Signaler un bug">Bug</button>
        <button id="play-config-toggle" class="config-toggle-btn" title="Options">⚙</button>
    </div>
    <div class="seats">
        <div class="seat north" id="seat-north">
            <div class="seat-label" id="seat-label-north">Nord (Partenaire)</div>
            <div class="hand" id="hand-north"></div>
        </div>
        <div class="seat west" id="seat-west">
            <div class="seat-label" id="seat-label-west">Ouest</div>
            <div class="hand" id="hand-west"></div>
        </div>
        <div id="trick-area">
            <div class="trick-card" id="trick-n"></div>
            <div class="trick-card" id="trick-w"></div>
            <div class="trick-card" id="trick-e"></div>
            <div class="trick-card" id="trick-s"></div>
        </div>
        <div class="seat east" id="seat-east">
            <div class="seat-label" id="seat-label-east">Est</div>
            <div class="hand" id="hand-east"></div>
        </div>
        <div class="seat south" id="seat-south">
            <div class="seat-label" id="seat-label-south">Sud (Vous)</div>
            <div class="hand" id="hand-south"></div>
        </div>
        <div id="game-result" class="hidden"></div>
        <div id="confetti-container"></div>
        <div id="play-status"></div>
        <div id="bidding-panel" class="hidden">
            <div id="bid-history"></div>
            <div id="bid-controls">
                <div id="bid-selectors">
                    <select id="bid-value">
                        <option value="">Valeur...</option>
                        <option value="80">80</option>
                        <option value="90">90</option>
                        <option value="100">100</option>
                        <option value="110">110</option>
                        <option value="120">120</option>
                        <option value="130">130</option>
                        <option value="140">140</option>
                        <option value="150">150</option>
                        <option value="160">160</option>
                        <option value="250">Capot</option>
                    </select>
                    <select id="bid-suit">
                        <option value="0">♠ Pique</option>
                        <option value="1">♥ Coeur</option>
                        <option value="2">♦ Carreau</option>
                        <option value="3">♣ Trefle</option>
                    </select>
                    <button id="bid-submit" class="bid-btn bid-action">Annoncer</button>
                </div>
                <div id="bid-special">
                    <button id="bid-pass" class="bid-btn pass">Passe</button>
                    <button id="bid-coinche" class="bid-btn coinche hidden">Coinche</button>
                    <button id="bid-surcoinche" class="bid-btn coinche hidden">Surcoinche</button>
                </div>
            </div>
        </div>
    </div>
    <div id="last-trick" class="hidden"></div>
    <div id="bid-history-panel" class="hidden"></div>
    <div id="play-review" class="hidden">
        <div id="play-bid-history">
            <div class="section-title">Encheres</div>
            <div id="play-bid-entries"></div>
        </div>
        <div id="play-tricks-history">
            <div class="section-title">Plis</div>
            <div id="play-tricks-list"></div>
        </div>
    </div>
</div>
`;

const SEAT_LABEL_ELS = ['seat-label-north', 'seat-label-east', 'seat-label-south', 'seat-label-west'];

export class GameTable {
    /**
     * opts:
     *   sendMove(action)   — transport for the local player's action (required)
     *   localEchoBids      — push own bids into bidHistory on click (solo=true;
     *                        multi=false, the server echoes every move)
     *   resultButtons      — [{label, className, onClick(gameId)}] on the result overlay
     */
    constructor(opts) {
        this.sendMove = opts.sendMove;
        this.localEchoBids = opts.localEchoBids !== false;
        this.resultButtons = opts.resultButtons || [];
        this.seatNames = null;    // display-ordered [N,E,S,W] override
        this.reset();
    }

    reset() {
        this.bidHistory = [];
        this.playLocked = false;
        this._pendingPlayState = null;
        this._initialHands = null;
        this.gameId = null;
        this._serverBidHistory = null;
        this._serverCompletedTricks = null;
        _prevTrick['trick'] = [];
        if (_animatingTrick === 'trick') setAnimatingTrick(null);
    }

    /** Call after the TABLE_TEMPLATE is in the DOM. */
    bind() {
        document.getElementById('play-report-btn').addEventListener('click', openBugReport);
    }

    show() {
        document.getElementById('play-table').classList.remove('hidden');
        document.getElementById('game-result').classList.add('hidden');
        document.getElementById('game-result').innerHTML = '';
        document.getElementById('confetti-container').innerHTML = '';
        document.getElementById('play-review').classList.add('hidden');
    }

    unbind() {
        if (_animatingTrick === 'trick') setAnimatingTrick(null);
        this._pendingPlayState = null;
    }

    playerName(seat) {
        if (this.seatNames && this.seatNames[seat]) return this.seatNames[seat];
        return SEAT_NAMES_FR[seat];
    }

    setSeatLabels(names) {
        // names: display-ordered [N, E, S, W]
        this.seatNames = names;
        const suffixes = ['(Partenaire)', '', '(Vous)', ''];
        for (let d = 0; d < 4; d++) {
            const el = document.getElementById(SEAT_LABEL_ELS[d]);
            if (el && names[d]) el.textContent = `${names[d]} ${suffixes[d]}`.trim();
        }
    }

    setGameId(id) {
        this.gameId = id;
        setBugReportGameId(id);
        const el = document.getElementById('play-game-id');
        if (id) {
            el.textContent = id;
            el.classList.remove('hidden');
            document.getElementById('play-report-btn').classList.remove('hidden');
        } else {
            el.classList.add('hidden');
            document.getElementById('play-report-btn').classList.add('hidden');
        }
    }

    // ===== WS-ish message ingestion =====

    handleGameState(data) {
        if (data.initial_hands) this._initialHands = data.initial_hands;
        if (data.bid_history) {
            this._serverBidHistory = data.bid_history;
            // Live sync (multi rejoin): rebuild the bidding panel history
            if (data.state && !data.state.is_terminal && data.state.phase === 0) {
                this.bidHistory = data.bid_history.map(
                    b => ({ player: b.player, action: b.action }));
            }
        }
        if (data.completed_tricks) this._serverCompletedTricks = data.completed_tricks;
        if (data.game_id) this.setGameId(data.game_id);
        this.renderState(data.state);
        if (data.state.is_terminal || data.state.current_player === MY_SEAT) {
            this.playLocked = false;
        }
        if (data.belote_event) {
            const text = data.belote_event === 'belote' ? 'Belote !' : 'Rebelote !';
            showBeloteAnnouncement('trick-area', text);
        }
    }

    handleMove(data) {
        if (data.player !== undefined) {
            // In multi the server echoes our own move too — we appended locally
            // only when localEchoBids is on, so skip the duplicate then.
            const isOwnEcho = this.localEchoBids && data.player === MY_SEAT;
            if (!isOwnEcho) {
                const name = actionName(data.action, 0);
                this.bidHistory.push({ player: data.player, action: data.action, name });
                SFX.playForAction(data.phase || 0, data.action);
            }
        }
        if (data.belote_event) {
            const text = data.belote_event === 'belote' ? 'Belote !' : 'Rebelote !';
            showBeloteAnnouncement('trick-area', text);
        }
    }

    // ===== Core rendering =====

    renderState(state) {
        if (_animatingTrick === 'trick') {
            this._pendingPlayState = state;
            return;
        }

        document.getElementById('score-ns').textContent = `NS : ${state.points[0]} (${state.tricks_won[0]}P)`;
        document.getElementById('score-ew').textContent = `EO : ${state.points[1]} (${state.tricks_won[1]}P)`;
        document.getElementById('contract-display').textContent = contractStr(state.contract);

        if (state.belote) {
            renderBeloteBadge('score-ns', state.belote[0]);
            renderBeloteBadge('score-ew', state.belote[1]);
        }

        const handEls = {
            0: document.getElementById('hand-north'),
            1: document.getElementById('hand-east'),
            2: document.getElementById('hand-south'),
            3: document.getElementById('hand-west'),
        };

        const isMyTurn = state.current_player === MY_SEAT && !state.is_terminal;
        const isPlayPhase = state.phase === 1;
        const isBidPhase = state.phase === 0;

        const legalSet = (isMyTurn && isPlayPhase) ? new Set(state.legal_actions) : null;
        const trumpSuit = (state.contract && state.contract.trump !== undefined) ? state.contract.trump : -1;

        for (let seat = 0; seat < 4; seat++) {
            const cards = state.hands[seat];
            if (seat === MY_SEAT) {
                const clickable = isMyTurn && isPlayPhase;
                renderHand(handEls[seat], cards, clickable,
                    clickable ? (c) => this.playCard(c) : null, legalSet, trumpSuit);
            } else {
                const tricksPlayed = state.tricks_won[0] + state.tricks_won[1];
                const hasPlayedThisTrick = state.current_trick[seat] >= 0 && state.current_trick[seat] < 32;
                const count = cards.length || Math.max(0, 8 - tricksPlayed - (hasPlayedThisTrick ? 1 : 0));
                renderFaceDownHand(handEls[seat], count);
            }
        }

        // Mobile: compute card overlap so south hand spans full width
        if (window.innerWidth <= 600) {
            const handEl = handEls[MY_SEAT];
            const n = handEl.children.length;
            if (n > 1) {
                const cardW = handEl.children[0].offsetWidth;
                const availW = handEl.offsetWidth;
                const overlap = (availW - n * cardW) / (n - 1);
                handEl.style.setProperty('--card-overlap', Math.min(0, overlap) + 'px');
            } else {
                handEl.style.removeProperty('--card-overlap');
            }
        }

        // Trick (with flush animation)
        const completedCards = detectTrickCompletion('trick', state.current_trick);
        if (completedCards && _animatingTrick !== 'trick') {
            animateTrickFlush('trick', () => {
                const lastTrickEl = document.getElementById('last-trick');
                if (lastTrickEl && isPlayPhase && state.last_trick) {
                    renderLastTrick(lastTrickEl, state.last_trick, state.last_trick_winner, state.last_trick_points, MY_SEAT);
                }
                if (this._pendingPlayState) {
                    const pending = this._pendingPlayState;
                    this._pendingPlayState = null;
                    this.renderState(pending);
                }
            }, state.last_trick_winner);
            renderTrick('trick', state.current_trick);
        } else {
            renderTrick('trick', state.current_trick);
            const trickFull = state.current_trick && state.current_trick.filter(c => c >= 0 && c < 32).length === 4;
            if (!trickFull) {
                const lastTrickEl = document.getElementById('last-trick');
                if (lastTrickEl) {
                    if (isPlayPhase && state.last_trick) {
                        renderLastTrick(lastTrickEl, state.last_trick, state.last_trick_winner, state.last_trick_points, MY_SEAT);
                    } else {
                        lastTrickEl.classList.add('hidden');
                        lastTrickEl.innerHTML = '';
                    }
                }
            }
        }

        // Bidding panel
        const biddingPanel = document.getElementById('bidding-panel');
        const bidHistoryPanel = document.getElementById('bid-history-panel');
        if (isBidPhase) {
            biddingPanel.classList.remove('hidden');
            bidHistoryPanel.classList.add('hidden');
            this.renderBidHistory();
            const bidControls = document.getElementById('bid-controls');
            if (isMyTurn) {
                bidControls.classList.remove('hidden');
                this.showBidControls(state.legal_actions, state);
            } else {
                bidControls.classList.add('hidden');
                this.hideBidControls();
            }
        } else {
            biddingPanel.classList.add('hidden');
            bidHistoryPanel.classList.add('hidden');
        }

        // Status
        if (state.is_terminal) {
            this.showGameResult(state);
            this.showEndOfGameReview(state);
            document.getElementById('play-status').textContent = '';
        } else if (isMyTurn) {
            document.getElementById('play-status').textContent = isBidPhase ? '' : 'A vous de jouer';
            SFX.yourTurn();
        } else {
            document.getElementById('play-status').textContent = `${this.playerName(state.current_player)} reflechit...`;
        }
    }

    renderBidHistory() {
        const el = document.getElementById('bid-history');
        el.innerHTML = '';
        for (const entry of this.bidHistory) {
            const span = document.createElement('span');
            const isPartnerTeam = (entry.player % 2) === (MY_SEAT % 2);
            span.className = 'bid-entry' + (isPartnerTeam ? ' team-partner' : ' team-opponent');
            span.innerHTML = `${this.playerName(entry.player)} ${bidActionHtml(entry.action)}`;
            el.appendChild(span);
        }
    }

    _sendBid(action, sfx) {
        if (this.playLocked) return;
        this.playLocked = true;
        sfx();
        if (this.localEchoBids) {
            this.bidHistory.push({ player: MY_SEAT, action, name: actionName(action, 0) });
        }
        this.sendMove(action);
    }

    showBidControls(legalActions, state) {
        const legalSet = new Set(legalActions);

        const bidSelectors = document.getElementById('bid-selectors');
        const bidSubmit = document.getElementById('bid-submit');
        const bidValue = document.getElementById('bid-value');
        const bidSuit = document.getElementById('bid-suit');

        const hasBids = legalActions.some(a => a >= 1 && a <= 40);
        bidSelectors.style.display = hasBids ? 'flex' : 'none';

        if (hasBids) {
            for (const opt of bidValue.options) {
                if (opt.value === '') continue;
                const v = parseInt(opt.value);
                let available = false;
                for (let s = 0; s < 4; s++) {
                    if (legalSet.has(encodeBidAction(v, s))) { available = true; break; }
                }
                opt.disabled = !available;
            }
            let firstLegal = '';
            for (const opt of bidValue.options) {
                if (opt.value !== '' && !opt.disabled) { firstLegal = opt.value; break; }
            }
            bidValue.value = firstLegal;

            if (state && state.best_trump_suit !== undefined) {
                bidSuit.value = String(state.best_trump_suit);
            }

            bidSubmit.onclick = () => {
                const val = parseInt(bidValue.value);
                const suit = parseInt(bidSuit.value);
                if (isNaN(val)) return;
                const action = encodeBidAction(val, suit);
                if (action < 0 || !legalSet.has(action)) return;
                this._sendBid(action, SFX.bid);
            };
        }

        const passBtn = document.getElementById('bid-pass');
        if (legalSet.has(0)) {
            passBtn.classList.remove('hidden');
            passBtn.disabled = false;
            passBtn.onclick = () => this._sendBid(0, SFX.pass);
        } else {
            passBtn.classList.add('hidden');
        }

        const coincheBtn = document.getElementById('bid-coinche');
        if (legalSet.has(41)) {
            coincheBtn.classList.remove('hidden');
            coincheBtn.onclick = () => this._sendBid(41, SFX.coinche);
        } else {
            coincheBtn.classList.add('hidden');
        }

        const surcoincheBtn = document.getElementById('bid-surcoinche');
        if (legalSet.has(42)) {
            surcoincheBtn.classList.remove('hidden');
            surcoincheBtn.onclick = () => this._sendBid(42, SFX.surcoinche);
        } else {
            surcoincheBtn.classList.add('hidden');
        }
    }

    hideBidControls() {
        document.getElementById('bid-selectors').style.display = 'none';
        document.getElementById('bid-pass').classList.add('hidden');
        document.getElementById('bid-coinche').classList.add('hidden');
        document.getElementById('bid-surcoinche').classList.add('hidden');
    }

    playCard(cardIdx) {
        if (this.playLocked) return;
        this.playLocked = true;
        SFX.cardPlay();
        // Optimistic update: show card in trick area and remove from hand
        const trickEl = document.getElementById('trick-s');
        trickEl.innerHTML = '';
        trickEl.appendChild(cardToHtml(cardIdx));
        const handEl = document.getElementById('hand-south');
        const cardEl = handEl.querySelector(`[data-card="${cardIdx}"]`);
        if (cardEl) cardEl.remove();
        this.sendMove(cardIdx);
    }

    // ===== Game result =====

    showGameResult(state) {
        const resultEl = document.getElementById('game-result');
        resultEl.classList.remove('hidden');

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
            const suitSymbols = ['♠', '♥', '♦', '♣'];
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

        const buttonsHtml = this.resultButtons.map((b, i) =>
            `<button class="${b.className}" data-result-btn="${i}">${b.label}</button>`
        ).join('');

        resultEl.innerHTML =
            `<div class="result-title ${titleClass}">${titleText}</div>` +
            (contract ? `<div class="result-contract">${contract}</div>` : '') +
            `<div class="result-scores">${scoresHtml}</div>` +
            (buttonsHtml ? `<div class="result-buttons">${buttonsHtml}</div>` : '');

        this.resultButtons.forEach((b, i) => {
            resultEl.querySelector(`[data-result-btn="${i}"]`)
                .addEventListener('click', () => b.onClick(this.gameId));
        });

        if (isVictory) {
            SFX.victory();
            this.launchConfetti();
        } else if (!isDraw) {
            SFX.defeat();
        }
    }

    showEndOfGameReview(state) {
        renderTrick('trick', [-1, -1, -1, -1]);

        const lastTrickEl = document.getElementById('last-trick');
        if (lastTrickEl) {
            lastTrickEl.classList.add('hidden');
            lastTrickEl.innerHTML = '';
        }

        if (this._initialHands) {
            const handEls = {
                0: document.getElementById('hand-north'),
                1: document.getElementById('hand-east'),
                2: document.getElementById('hand-south'),
                3: document.getElementById('hand-west'),
            };
            const trumpSuit = (state.contract && state.contract.trump !== undefined) ? state.contract.trump : -1;
            for (let seat = 0; seat < 4; seat++) {
                renderHand(handEls[seat], this._initialHands[seat], false, null, null, trumpSuit);
            }
        }

        const reviewEl = document.getElementById('play-review');
        reviewEl.classList.remove('hidden');

        const bidContainer = document.getElementById('play-bid-entries');
        bidContainer.innerHTML = '';
        const bids = this._serverBidHistory || this.bidHistory.map(b => ({ player: b.player, action: b.action }));
        for (const bid of bids) {
            const el = document.createElement('span');
            const team = bid.player % 2 === 0 ? 'team-ns' : 'team-ew';
            el.className = `watch-bid-entry ${team}`;
            el.innerHTML = `${this.playerName(bid.player)} ${bidActionHtml(bid.action)}`;
            bidContainer.appendChild(el);
        }

        const tricksContainer = document.getElementById('play-tricks-list');
        tricksContainer.innerHTML = '';
        const tricks = this._serverCompletedTricks || [];
        for (let i = 0; i < tricks.length; i++) {
            const t = tricks[i];
            const row = document.createElement('div');
            const winnerTeam = t.winner % 2 === 0 ? 'team-ns' : 'team-ew';
            row.className = `trick-history-row ${winnerTeam}`;

            const SEAT_L = ['N', 'E', 'S', 'O'];
            const leadSeat = t.lead !== undefined && t.lead !== null ? t.lead : -1;

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
            const winnerName = this.playerName(t.winner);
            row.innerHTML = `<span class="trick-num">#${i + 1}</span>` +
                `<span class="trick-lead-label">${leadLabel}</span>` +
                `<span class="trick-cards">${orderedCards}</span>` +
                `<span class="trick-winner">${winnerName} +${t.points}</span>`;
            tricksContainer.appendChild(row);
        }
    }

    launchConfetti() {
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
}
