// Shared game table — rendering + interaction extracted from the solo play
// view so multiplayer rooms reuse the exact same UI. The viewer is ALWAYS
// display-seat 2 (South): the solo server filters for seat 2, the multiplayer
// server rotates every state into the viewer's frame before sending.

import * as SFX from '../sounds.js';
import {
    SEAT_NAMES_FR, teamName, teamNameMid, cardToHtml, suitHtml,
    renderHand, renderFaceDownHand, renderTrick, renderLastTrick,
    actionName, encodeBidAction, bidChipHtml,
    showBeloteAnnouncement, renderBeloteBadge,
    _prevTrick, _animatingTrick, setAnimatingTrick,
    detectTrickCompletion, animateTrickFlush
} from './cards.js';
import { setGameId as setBugReportGameId, openBugReport } from './bug-report.js';
import { createSuitPicker } from './suits.js';
import { renderBidEntries, renderTrickHistory } from './panels.js';

export const MY_SEAT = 2; // South, always (server-side rotation guarantees it)

export const TABLE_TEMPLATE = `
<div id="play-table" class="table hidden">
    <div id="score-bar">
        <div class="score-team team-ns" id="score-ns">
            <span class="score-team-label">Nous</span>
            <span class="score-team-pts">0</span>
        </div>
        <div id="contract-display"></div>
        <div class="score-side">
            <div class="score-team team-ew" id="score-ew">
                <span class="score-team-label">Eux</span>
                <span class="score-team-pts">0</span>
            </div>
            <button id="play-report-btn" class="report-btn hidden" title="Signaler un bug">Bug</button>
            <button id="play-config-toggle" class="config-toggle-btn" title="Options">⚙</button>
        </div>
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
                    <span id="bid-suit-mount"></span>
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

        // Choix de l'atout : segmented control plutôt qu'un <select>, dont les
        // <option> ne peuvent pas porter de couleur de façon portable. Le
        // picker expose `.value` comme un select, le reste du code est inchangé.
        const mount = document.getElementById('bid-suit-mount');
        if (mount) {
            const picker = createSuitPicker({ value: 0, name: 'atout' });
            picker.id = 'bid-suit';
            mount.replaceWith(picker);
        }
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

    // Le code de partie n'est plus affiché pendant le jeu : c'était du texte
    // technique de plus dans un bandeau déjà chargé, et le bouton « Bug »
    // transporte l'identifiant tout seul.
    setGameId(id) {
        this.gameId = id;
        setBugReportGameId(id);
        document.getElementById('play-report-btn')
            .classList.toggle('hidden', !id);
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
            // In multi the server echoes our own move too, and in solo it
            // echoes a pass it played for us (`auto`) — neither was echoed
            // locally, so only a bid we actually clicked is a duplicate.
            const isOwnEcho = this.localEchoBids && data.player === MY_SEAT && !data.auto;
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

    /**
     * Bandeau de score : deux colonnes « camp au-dessus, points en dessous »
     * encadrant le contrat. Le nombre de plis ramassés n'y figure plus — écrit
     * « (8P) » il se lisait comme du code, et l'information est déjà donnée par
     * les cartes restantes puis par le récapitulatif de fin de partie.
     */
    renderScoreBar(state) {
        document.querySelector('#score-ns .score-team-pts').textContent = state.points[0];
        document.querySelector('#score-ew .score-team-pts').textContent = state.points[1];

        const c = state.contract;
        const el = document.getElementById('contract-display');
        // `c.value > 0` : avant la première annonce le serveur envoie un contrat
        // à zéro, qui s'affichait « 0♠ par nous » au milieu du bandeau.
        if (c && Object.keys(c).length > 0 && c.value > 0) {
            const team = teamNameMid(c.team, true);
            const coinche = c.coinche === 1 ? ' x' : c.coinche === 2 ? ' xx' : '';
            el.innerHTML =
                `<span class="contract-val">${c.value}${suitHtml(c.trump)}${coinche}</span>` +
                `<span class="contract-by">par ${team}</span>`;
        } else {
            el.innerHTML = '';
        }
    }

    renderState(state) {
        if (_animatingTrick === 'trick') {
            this._pendingPlayState = state;
            return;
        }

        this.renderScoreBar(state);

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
        // Pass as our sole legal bid: nothing to decide, the server plays it.
        const forcedPass = isMyTurn && isBidPhase
            && state.legal_actions && state.legal_actions.length === 1
            && state.legal_actions[0] === 0;

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

        // Le chevauchement des cartes est calculé en CSS (cf. --card-step dans
        // cards.css) : plus de mesure offsetWidth suivie d'une écriture de
        // style, qui forçait un reflow à chaque rendu et faisait sauter la main.

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
            // Forced pass (our side coinched then partner declined the
            // surcoinche, or partner's capot): the server passes for us, so
            // don't put up a panel whose only button is "Passer".
            if (isMyTurn && !forcedPass) {
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
        } else if (forcedPass) {
            document.getElementById('play-status').textContent =
                'Vous ne pouvez que passer — passe automatique';
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
            span.innerHTML = `${this.playerName(entry.player)} ${bidChipHtml(entry.action)}`;
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

    /**
     * Fin de donne. Le titre dit ce qui s'est passé au jeu — « Contrat réussi »
     * ou « Contrat chuté », pas « Victoire / Défaite » — et c'est sa couleur qui
     * dit ce que ça vaut pour le joueur (vert = notre camp encaisse ; défendre
     * une chute est une victoire). En dessous les scores en gros, puis les
     * points de plis en plus petit.
     */
    showGameResult(state) {
        const resultEl = document.getElementById('game-result');
        resultEl.classList.remove('hidden');

        const rewards = state.rewards;
        const isVictory = rewards ? rewards[0] > rewards[1] : state.points[0] > state.points[1];
        const isDraw = rewards ? rewards[0] === rewards[1] : state.points[0] === state.points[1];
        const titleClass = isVictory ? 'victory' : isDraw ? 'draw' : 'defeat';

        const sd = state.score_detail;
        const c = state.contract;
        const hasContract = !!c && Object.keys(c).length > 0 && c.value > 0;

        let titleText;
        if (!hasContract) titleText = 'Personne n\'a pris';
        else if (sd) titleText = sd.contract_made ? 'Contrat réussi' : 'Contrat chuté';
        else titleText = isVictory ? 'Donne gagnée' : isDraw ? 'Égalité' : 'Donne perdue';

        // Rappel du contrat : c'est lui que le titre juge.
        const contractLine = hasContract
            ? `${sd ? sd.contract_value : c.value}${suitHtml(c.trump)}`
              + (c.coinche === 1 ? ' contré' : c.coinche === 2 ? ' surcontré' : '')
              + ` demandé par ${teamNameMid(sd ? sd.contract_team : c.team, true)}`
            : '';

        // Scores finaux de la donne, en gros.
        let finals;
        if (sd) {
            finals = sd.final_scores;
        } else {
            const bel = [
                state.belote && state.belote[0] === 2 ? 20 : 0,
                state.belote && state.belote[1] === 2 ? 20 : 0,
            ];
            finals = [state.points[0] + bel[0], state.points[1] + bel[1]];
        }

        // Points de plis, et la belote s'il y en a une, en plus petit.
        const tricks = sd ? sd.trick_points : state.points;
        let detailLine = `Plis : ${teamNameMid(0, true)} ${tricks[0]} — ${teamNameMid(1, true)} ${tricks[1]}`;
        const belote = sd ? sd.belote : [
            state.belote && state.belote[0] === 2 ? 20 : 0,
            state.belote && state.belote[1] === 2 ? 20 : 0,
        ];
        const beloteParts = [];
        for (const t of [0, 1]) {
            if (belote[t] > 0) beloteParts.push(`+${belote[t]} pour ${teamNameMid(t, true)}`);
        }
        if (beloteParts.length) {
            detailLine += ` <span class="belote-note">· belote ${beloteParts.join(' et ')}</span>`;
        }

        const buttonsHtml = this.resultButtons.map((b, i) =>
            `<button class="${b.className}" data-result-btn="${i}">${b.label}</button>`
        ).join('');

        resultEl.innerHTML =
            `<div class="result-title ${titleClass}">${titleText}</div>` +
            (contractLine ? `<div class="result-contract">${contractLine}</div>` : '') +
            `<div class="result-final">` +
                `<div class="result-final-team team-ns">` +
                    `<span class="result-final-pts">${finals[0]}</span>` +
                    `<span class="result-final-name">${teamName(0, true)}</span>` +
                `</div>` +
                `<div class="result-final-team team-ew">` +
                    `<span class="result-final-pts">${finals[1]}</span>` +
                    `<span class="result-final-name">${teamName(1, true)}</span>` +
                `</div>` +
            `</div>` +
            `<div class="result-tricks">${detailLine}</div>` +
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

        const bids = this._serverBidHistory || this.bidHistory.map(b => ({ player: b.player, action: b.action }));
        renderBidEntries(document.getElementById('play-bid-entries'), bids, (s) => this.playerName(s));

        renderTrickHistory(
            document.getElementById('play-tricks-list'),
            this._serverCompletedTricks || [],
            (s) => this.playerName(s),
        );
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
