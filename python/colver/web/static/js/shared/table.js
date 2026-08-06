// Shared game table — rendering + interaction extracted from the solo play
// view so multiplayer rooms reuse the exact same UI. The viewer is ALWAYS
// display-seat 2 (South): the solo server filters for seat 2, the multiplayer
// server rotates every state into the viewer's frame before sending.

import * as SFX from '../sounds.js';
import {
    SEAT_NAMES_FR, teamName, teamNameMid, cardToHtml, suitHtml,
    renderHand, renderFaceDownHand, renderTrick, renderLastTrick,
    actionName, encodeBidAction, bidChipHtml, contractChipHtml,
    showBeloteAnnouncement, renderBeloteBadge,
    _prevTrick, _animatingTrick, setAnimatingTrick,
    detectTrickCompletion, animateTrickFlush
} from './cards.js';
import { setGameId as setBugReportGameId, openBugReport } from './bug-report.js';
import { botLabel } from './agents.js';
import { createSuitPicker } from './suits.js';
import { renderAuctionTable, renderBidEntries, renderTrickHistory } from './panels.js';

export const MY_SEAT = 2; // South, always (server-side rotation guarantees it)

export const TABLE_TEMPLATE = `
<div id="play-table" class="table hidden">
    <div id="score-bar">
        <div class="score-team team-ns" id="score-ns">
            <span class="score-team-label">Nous</span>
            <span class="score-team-pts">0</span>
            <span class="score-team-match hidden"></span>
        </div>
        <div class="score-center">
            <div id="contract-display"></div>
            <div id="match-info" class="match-info hidden"></div>
        </div>
        <div class="score-side">
            <div class="score-team team-ew" id="score-ew">
                <span class="score-team-label">Eux</span>
                <span class="score-team-pts">0</span>
                <span class="score-team-match hidden"></span>
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
        <div id="auction-drop" class="hidden">
            <div id="auction-drop-card">
                <div class="section-title">Enchères</div>
                <div id="auction-grid"></div>
                <div class="auction-hint">Appuyez n'importe où pour refermer</div>
            </div>
        </div>
    </div>
    <div id="last-trick" class="hidden"></div>
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

// Les étiquettes telles que `TABLE_TEMPLATE` les pose : celles d'une table où
// les trois autres sièges sont tenus par l'IA. Elles doivent exister ailleurs
// que dans le gabarit parce qu'une partie de salon les réécrit avec des pseudos
// et que **le solo n'envoie pas de `seat_names`** — sans remise à zéro, la
// partie suivante gardait le nom de l'humain qui occupait le siège à la table
// précédente.
const DEFAULT_SEAT_LABELS = ['Nord (Partenaire)', 'Est', 'Sud (Vous)', 'Ouest'];

export class GameTable {
    /**
     * opts:
     *   sendMove(action)   — transport for the local player's action (required)
     *   localEchoBids      — push own bids into bidHistory on click (solo=true;
     *                        multi=false, the server echoes every move)
     *   resultButtons      — [{label, className, disabled, onClick(gameId)}] on
     *                        the result overlay, or a function ({gameId, match})
     *                        => [...] when the buttons depend on the match
     *                        (« Donne suivante » plutôt que « Nouvelle partie »)
     */
    constructor(opts) {
        this.sendMove = opts.sendMove;
        this.localEchoBids = opts.localEchoBids !== false;
        this.resultButtons = opts.resultButtons || [];
        this.seatOccupants = null;   // display-ordered [N,E,S,O] : {name, bot}
        this._onKeyDown = (e) => {
            if (e.key === 'Escape' && this._auctionOpen) this.closeAuction();
        };
        this.reset();
    }

    reset() {
        // La partie en cours, telle que le serveur la décrit : {target, totals,
        // deal_no, finished, winner}. Renvoyée à chaque nouvelle donne, donc
        // remise à zéro ici sans risque de la perdre.
        this.match = null;
        this.bidHistory = [];
        this.playLocked = false;
        this._pendingPlayState = null;
        this._initialHands = null;
        this.gameId = null;
        this._serverBidHistory = null;
        this._serverCompletedTricks = null;
        // Qui tient les sièges est une propriété de la *table*, pas de la donne,
        // mais c'est bien ici qu'il faut l'oublier : une nouvelle donne est le
        // seul moment où la table peut avoir changé, et un état de salon qui
        // survit à une partie solo nomme les bots avec le pseudo d'un humain.
        // Le salon repose ses noms tout de suite après (`setSeatLabels`).
        this.clearSeatLabels();
        this.closeAuction();
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

        // Le rappel des enchères se referme d'un appui n'importe où : il est
        // là pour être consulté puis chassé, et sur téléphone viser une croix
        // de 20px après avoir lu coûte plus cher que de retoucher l'écran.
        // Un vrai glissement (défilement d'une longue enchère) n'émet pas de
        // `click`, donc il ne le ferme pas.
        const drop = document.getElementById('auction-drop');
        if (drop) drop.addEventListener('click', () => this.closeAuction());
    }

    show() {
        document.getElementById('play-table').classList.remove('hidden');
        document.getElementById('game-result').classList.add('hidden');
        document.getElementById('game-result').innerHTML = '';
        document.getElementById('confetti-container').innerHTML = '';
        document.getElementById('play-review').classList.add('hidden');
        document.addEventListener('keydown', this._onKeyDown);
    }

    unbind() {
        if (_animatingTrick === 'trick') setAnimatingTrick(null);
        this._pendingPlayState = null;
        this.closeAuction();
        document.removeEventListener('keydown', this._onKeyDown);
    }

    /**
     * Comment nommer un siège en toutes lettres — puces d'enchères, colonnes du
     * rappel, « … réfléchit », gagnant d'un pli.
     *
     * Un bot est nommé par sa **position**, jamais par son nom : en salon les
     * quatre sièges vides sont tenus par le même bot, et « Dédé 90♠ » puis
     * « Dédé 110♥ » ne disaient pas s'il s'agissait du même joueur. La position
     * est la seule chose qui les distingue, et c'est déjà ce que le solo
     * affiche. Le nom du bot, lui, ne vit plus que sur l'étiquette du siège.
     */
    playerName(seat) {
        const who = this.seatOccupants && this.seatOccupants[seat];
        if (who && !who.bot && who.name) return who.name;
        return SEAT_NAMES_FR[seat];
    }

    /**
     * `seats` : ordre d'affichage [N, E, S, O], entrées `{name, bot}` — même
     * forme que `db.game_seat_names`. Un humain est nommé par son pseudo, qui
     * n'indique pas où il est assis : d'où le « (Vous) » / « (Partenaire) »
     * qui le raccroche à la table. Un bot est déjà nommé par sa position, donc
     * le suffixe n'ajouterait rien — c'est le nom du bot qui vient en
     * qualificatif, comme dans Rejouer (« NORD (DÉDÉ) »).
     */
    setSeatLabels(seats) {
        this.seatOccupants = seats;
        const suffixes = ['(Partenaire)', '', '(Vous)', ''];
        for (let d = 0; d < 4; d++) {
            const el = document.getElementById(SEAT_LABEL_ELS[d]);
            const who = seats && seats[d];
            if (!el || !who || !who.name) continue;
            el.textContent = who.bot
                ? `${SEAT_NAMES_FR[d]} (${botLabel(who.name)})`
                : `${who.name} ${suffixes[d]}`.trim();
        }
    }

    /**
     * Rend aux quatre sièges leur étiquette par défaut et oublie qui les tenait
     * — ce que `playerName` lit pour les puces d'enchères, les colonnes du
     * rappel, « … réfléchit » et le gagnant d'un pli.
     *
     * Écrire `textContent` retire aussi les signes que la page a pu poser sur
     * l'étiquette (pendule, ⏱, ⚡, 🤖) : ils décrivent la partie qu'on quitte.
     */
    clearSeatLabels() {
        this.seatOccupants = null;
        for (let d = 0; d < 4; d++) {
            // Le constructeur appelle `reset()`, éventuellement avant que le
            // gabarit soit dans le document.
            const el = document.getElementById(SEAT_LABEL_ELS[d]);
            if (el) el.textContent = DEFAULT_SEAT_LABELS[d];
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
        // Le score de partie ne voyage qu'avec la première et la dernière
        // position d'une donne : entre les deux il ne bouge pas.
        if (data.match) this.match = data.match;
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
                const phase = data.phase || 0;
                // `bidHistory` ne prend que les annonces. Chaque coup y était
                // versé, cartes comprises : invisible tant que le panneau ne
                // s'ouvrait que pendant les enchères, mais le rappel en cours
                // de jeu montrait alors les cartes jouées en fin d'enchère
                // (une carte lue comme une annonce donne « 130♠ »).
                if (phase === 0) {
                    const name = actionName(data.action, 0);
                    this.bidHistory.push({ player: data.player, action: data.action, name });
                }
                SFX.playForAction(phase, data.action);
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
            // Le contrat *est* le résumé de l'enchère : c'est donc lui qui la
            // rouvre, plutôt qu'un bouton de plus dans un bandeau qui doit déjà
            // tenir sur 360px. Pendant les annonces le panneau d'enchère porte
            // déjà l'historique, et à la fin de la donne le récapitulatif : le
            // chevron n'apparaît qu'entre les deux, quand l'enchère a disparu
            // de l'écran.
            const reviewable = state.phase === 1 && !state.is_terminal
                && this._auctionBids().length > 0;
            const body =
                contractChipHtml(c, 'contract-val') +
                `<span class="contract-by">par ${team}` +
                (reviewable ? '<span class="contract-caret" aria-hidden="true">▾</span>' : '') +
                '</span>';
            if (reviewable) {
                el.innerHTML =
                    `<button type="button" class="contract-toggle" aria-controls="auction-drop"` +
                    ` aria-expanded="${this._auctionOpen ? 'true' : 'false'}"` +
                    ` title="Revoir les enchères">${body}</button>`;
                el.querySelector('.contract-toggle')
                    .addEventListener('click', () => this.toggleAuction());
            } else {
                el.innerHTML = body;
                this.closeAuction();
            }
        } else {
            el.innerHTML = '';
            this.closeAuction();
        }

        this.renderMatchBar();
    }

    // ===== Rappel des enchères pendant le jeu =====

    /** L'enchère de la donne : celle du serveur si on l'a (rejoint en cours,
     *  reprise de partie), sinon celle qu'on a accumulée coup par coup. */
    _auctionBids() {
        return this._serverBidHistory
            || this.bidHistory.map(b => ({ player: b.player, action: b.action }));
    }

    toggleAuction() {
        if (this._auctionOpen) this.closeAuction();
        else this.openAuction();
    }

    openAuction() {
        const drop = document.getElementById('auction-drop');
        if (!drop) return;
        renderAuctionTable(
            document.getElementById('auction-grid'),
            this._auctionBids(),
            (s) => this.playerName(s),
            MY_SEAT,
        );
        drop.classList.remove('hidden');
        this._auctionOpen = true;
        this._setCaretState();
    }

    closeAuction() {
        const drop = document.getElementById('auction-drop');
        if (drop) drop.classList.add('hidden');
        this._auctionOpen = false;
        this._setCaretState();
    }

    _setCaretState() {
        const btn = document.querySelector('#contract-display .contract-toggle');
        if (btn) btn.setAttribute('aria-expanded', this._auctionOpen ? 'true' : 'false');
    }

    /**
     * Score de la partie dans le bandeau, sous les points de la donne. Les deux
     * chiffres cohabitent parce qu'ils répondent à deux questions : les points
     * de plis disent où en est le contrat, le cumul dit où en est la partie.
     * Rien ne s'affiche sur une donne isolée (`target = 0`).
     */
    renderMatchBar() {
        const m = this.match;
        const inMatch = !!m && m.target > 0;

        const infoEl = document.getElementById('match-info');
        if (infoEl) {
            infoEl.classList.toggle('hidden', !inMatch);
            if (inMatch) infoEl.textContent = `Partie en ${m.target} · donne ${m.deal_no}`;
        }
        for (const [team, id] of [[0, 'score-ns'], [1, 'score-ew']]) {
            const el = document.querySelector(`#${id} .score-team-match`);
            if (!el) continue;
            el.classList.toggle('hidden', !inMatch);
            if (inMatch) el.textContent = `partie ${m.totals[team]}`;
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
            renderTrick('trick', state.current_trick, state.trick_lead);
        } else {
            renderTrick('trick', state.current_trick, state.trick_lead);
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
        if (isBidPhase) {
            biddingPanel.classList.remove('hidden');
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
        }

        // Status
        if (state.is_terminal) {
            // `deal_end_hold` : image de la dernière levée, envoyée avant
            // l'état terminal réel. Le panneau de fin recouvre la table et
            // remplace les mains par la donne initiale — l'afficher ici, c'est
            // effacer le pli dans l'image même où il apparaît. On le garde donc
            // pour l'état suivant, que le serveur envoie après la pause
            // (`pacing.DEAL_END_HOLD`).
            if (!state.deal_end_hold) {
                this.showGameResult(state);
                this.showEndOfGameReview(state);
            }
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

        const buttons = typeof this.resultButtons === 'function'
            ? this.resultButtons({ gameId: this.gameId, match: this.match })
            : this.resultButtons;
        const buttonsHtml = buttons.map((b, i) =>
            `<button class="${b.className}" data-result-btn="${i}"` +
            `${b.disabled ? ' disabled' : ''}>${b.label}</button>`
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
            this.matchResultHtml() +
            (buttonsHtml ? `<div class="result-buttons">${buttonsHtml}</div>` : '');

        buttons.forEach((b, i) => {
            if (b.disabled) return;
            resultEl.querySelector(`[data-result-btn="${i}"]`)
                .addEventListener('click', () => b.onClick(this.gameId));
        });

        // Une partie gagnée passe avant la donne : perdre la dernière donne en
        // remportant la partie mérite les confettis, pas le son de défaite.
        const m = this.match;
        if (m && m.target > 0 && m.finished) {
            if (m.winner === 0) { SFX.victory(); this.launchConfetti(); }
            else SFX.defeat();
        } else if (isVictory) {
            SFX.victory();
            this.launchConfetti();
        } else if (!isDraw) {
            SFX.defeat();
        }
    }

    /** Bloc « où en est la partie » sous le résultat de la donne. */
    matchResultHtml() {
        const m = this.match;
        if (!m || m.target <= 0) return '';
        const head = m.finished
            ? `<div class="result-match-title ${m.winner === 0 ? 'victory' : 'defeat'}">`
              + `${m.winner === 0 ? 'Partie gagnée' : 'Partie perdue'}</div>`
            : `<div class="result-match-head">Partie en ${m.target}`
              + ` · donne ${m.deal_no}</div>`;
        return `<div class="result-match">${head}` +
            `<div class="result-match-scores">` +
                `<span class="team-ns">${teamName(0, true)} ${m.totals[0]}</span>` +
                `<span class="result-match-sep">—</span>` +
                `<span class="team-ew">${teamName(1, true)} ${m.totals[1]}</span>` +
            `</div></div>`;
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
