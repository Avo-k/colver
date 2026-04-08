// BoardRenderer: shared board rendering for Watch and Replay views (ES module)

import * as SFX from '../sounds.js';
import {
    RANKS, SUITS, SEAT_NAMES_FR, cardRank, cardSuit,
    renderHand, renderTrick, renderLastTrick, contractStr, actionName, bidActionHtml,
    showBeloteAnnouncement, renderBeloteBadge,
    _prevTrick, _animatingTrick, setAnimatingTrick,
    detectTrickCompletion, animateTrickFlush
} from './cards.js';
import { updateCfnBox } from './cfn-box.js';

const SUIT_LABELS = ['\u2660', '\u2665', '\u2666', '\u2663'];

export class BoardRenderer {
    constructor(opts) {
        this.prefix = opts.prefix;
        this.trickPrefix = `${opts.prefix}-trick`;
        this.isReplay = opts.isReplay || false;
        this.renderMoveStats = opts.renderMoveStats || (() => {});
        this.renderCardAnnotations = opts.renderCardAnnotations || (() => {});
        this.onRequestStep = opts.onRequestStep || (() => {});

        this.moveHistory = [];
        this.historyIndex = -1;
        this._prevHistoryIndex = -1;
        this.initialState = null;
        this.autoPlayMode = null;
        this.autoPlayTimer = null;
        this.waitingForStep = false;
        this.active = false;
        this.finished = false;

        this._keyHandler = null;
    }

    el(id) { return document.getElementById(`${this.prefix}-${id}`); }

    bindTransport() {
        this.el('prev-btn')?.addEventListener('click', () => {
            if (!this.active) return;
            this.stopAutoPlay();
            this.goToPreviousMove();
        });
        this.el('step-btn')?.addEventListener('click', () => {
            if (!this.active || this.waitingForStep) return;
            this.stopAutoPlay();
            this.goToNextMove();
        });
        this.el('start-btn')?.addEventListener('click', () => {
            if (!this.active) return;
            this.stopAutoPlay();
            this.goToStart();
        });
        this.el('auto-btn')?.addEventListener('click', () => {
            if (this.autoPlayMode) {
                this.stopAutoPlay();
            } else if (this.active && !this.isAtEnd()) {
                this.startAutoPlay('game');
            }
        });
        this.el('end-btn')?.addEventListener('click', () => {
            if (!this.active || this.waitingForStep) return;
            this.stopAutoPlay();
            if (this.isAtEnd()) return;
            this.startAutoPlay('end');
        });
    }

    bindKeyboard() {
        this._keyHandler = (e) => {
            const tag = document.activeElement?.tagName;
            if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
            if (!this.active) return;

            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                this.stopAutoPlay();
                this.goToPreviousMove();
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                this.stopAutoPlay();
                this.goToNextMove();
            }
        };
        document.addEventListener('keydown', this._keyHandler);
    }

    unbindKeyboard() {
        if (this._keyHandler) {
            document.removeEventListener('keydown', this._keyHandler);
            this._keyHandler = null;
        }
    }

    reset(initialState) {
        this.moveHistory = [];
        this.historyIndex = -1;
        this._prevHistoryIndex = -1;
        this.initialState = initialState;
        this.autoPlayMode = null;
        if (this.autoPlayTimer) clearTimeout(this.autoPlayTimer);
        this.autoPlayTimer = null;
        this.waitingForStep = false;
        this.active = true;
        this.finished = false;
        _prevTrick[this.trickPrefix] = [];
        if (_animatingTrick === this.trickPrefix) setAnimatingTrick(null);
    }

    pushMove(data) {
        this.moveHistory.push(data);
        this.historyIndex = this.moveHistory.length - 1;
    }

    isAtEnd() {
        if (this.historyIndex >= 0 && this.historyIndex === this.moveHistory.length - 1) {
            const last = this.moveHistory[this.historyIndex];
            if (last && last.finished) return true;
        }
        return this.finished && this.historyIndex === this.moveHistory.length - 1;
    }

    startAutoPlay(mode) {
        this.autoPlayMode = mode;
        const btn = this.el('auto-btn');
        if (btn) btn.textContent = '\u23F8';
        this.goToNextMove();
    }

    stopAutoPlay() {
        this.autoPlayMode = null;
        if (this.autoPlayTimer) {
            clearTimeout(this.autoPlayTimer);
            this.autoPlayTimer = null;
        }
        const btn = this.el('auto-btn');
        if (btn) btn.textContent = '\u25B6';
    }

    goToPreviousMove() {
        if (this.historyIndex <= -1) return;
        if (_animatingTrick === this.trickPrefix) return;
        this.historyIndex--;
        this.renderHistoryEntry(this.historyIndex);
    }

    goToNextMove() {
        if (_animatingTrick === this.trickPrefix) return;
        if (this.historyIndex < this.moveHistory.length - 1) {
            this.historyIndex++;
            this.renderHistoryEntry(this.historyIndex);
            if (this.autoPlayMode) this._continueAutoPlayFromBuffer();
        } else if (!this.finished && !this.isAtEnd()) {
            this.onRequestStep();
            this.waitingForStep = true;
        } else if (this.autoPlayMode) {
            this.stopAutoPlay();
        }
    }

    goToStart() {
        this.historyIndex = -1;
        this.renderHistoryEntry(-1);
    }

    renderHistoryEntry(index) {
        const isForward = index > this._prevHistoryIndex;
        const skipAnimation = this.autoPlayMode === 'end';
        this._prevHistoryIndex = index;

        if (index < 0) {
            _prevTrick[this.trickPrefix] = [];
            if (this.initialState) {
                this.renderState(this.initialState);
                this.renderBidHistory([], this.initialState.phase);
                this.renderTricksHistory([]);
                this._renderInitialStats();
                this.renderCardAnnotations(null, this.initialState);
            }
        } else if (index < this.moveHistory.length) {
            const data = this.moveHistory[index];

            const completedCards = detectTrickCompletion(this.trickPrefix, data.state.current_trick);

            if (completedCards && isForward && !skipAnimation && _animatingTrick !== this.trickPrefix) {
                animateTrickFlush(this.trickPrefix, () => {
                    const lastTrickEl = this.el('last-trick');
                    if (lastTrickEl && data.state.phase === 1 && data.state.last_trick) {
                        renderLastTrick(lastTrickEl, data.state.last_trick, data.state.last_trick_winner, data.state.last_trick_points, 0);
                    }
                }, data.state.last_trick_winner);
            }

            if (isForward && !skipAnimation && data.move) {
                SFX.playForAction(data.move.phase, data.move.action);
            }

            this.renderState(data.state);

            if (data.finished) {
                this._renderFinishedStats(data.state);
            } else if (data.move) {
                this.renderMoveStats(data.move, data.state);
            } else {
                this._clearStats();
            }

            this.renderBidHistory(data.bid_history, data.state.phase);
            this.renderTricksHistory(data.completed_tricks);
            this.renderCardAnnotations(data, data.state);
        }
        this.updateTransportButtons();
    }

    renderState(state) {
        this.el('score-ns').textContent = `NS : ${state.points[0]} (${state.tricks_won[0]}P)`;
        this.el('score-ew').textContent = `EO : ${state.points[1]} (${state.tricks_won[1]}P)`;
        this.el('contract-display').textContent = contractStr(state.contract);
        updateCfnBox(`${this.prefix}-cfn`, state.cfn);

        if (state.belote) {
            renderBeloteBadge(`${this.prefix}-score-ns`, state.belote[0]);
            renderBeloteBadge(`${this.prefix}-score-ew`, state.belote[1]);
        }

        const handEls = {
            0: this.el('hand-north'),
            1: this.el('hand-east'),
            2: this.el('hand-south'),
            3: this.el('hand-west'),
        };
        const trumpSuit = (state.contract && state.contract.trump !== undefined) ? state.contract.trump : -1;
        for (let seat = 0; seat < 4; seat++) {
            renderHand(handEls[seat], state.hands[seat], false, null, null, trumpSuit);
        }

        renderTrick(this.trickPrefix, state.current_trick);

        const lastTrickEl = this.el('last-trick');
        if (lastTrickEl && _animatingTrick !== this.trickPrefix) {
            if (state.phase === 1 && state.last_trick) {
                renderLastTrick(lastTrickEl, state.last_trick, state.last_trick_winner, state.last_trick_points, 0);
            } else {
                lastTrickEl.classList.add('hidden');
                lastTrickEl.innerHTML = '';
            }
        }

        const labelMap = {
            0: `${this.prefix}-label-n`,
            1: `${this.prefix}-label-e`,
            2: `${this.prefix}-label-s`,
            3: `${this.prefix}-label-w`,
        };
        for (let s = 0; s < 4; s++) {
            const el = document.getElementById(labelMap[s]);
            if (el) el.classList.toggle('active-player', s === state.current_player && !state.is_terminal);
        }
    }

    renderBidHistory(bidHistory, phase) {
        const container = this.el('bid-entries');
        if (!container) return;
        container.innerHTML = '';
        if (!bidHistory || bidHistory.length === 0) {
            this._updateBidOverlay([], phase);
            return;
        }

        for (const bid of bidHistory) {
            const el = document.createElement('span');
            const team = bid.player % 2 === 0 ? 'team-ns' : 'team-ew';
            el.className = `watch-bid-entry ${team}`;
            el.innerHTML = `${SEAT_NAMES_FR[bid.player]} : ${bidActionHtml(bid.action)}`;
            container.appendChild(el);
        }

        this._updateBidOverlay(bidHistory, phase);
    }

    _updateBidOverlay(bidHistory, phase) {
        const overlay = document.getElementById(`${this.prefix}-bid-overlay`);
        if (!overlay) return;
        const entries = document.getElementById(`${this.prefix}-bid-overlay-entries`);
        if (!entries) return;

        if (phase !== 0 || !bidHistory || bidHistory.length === 0) {
            overlay.classList.add('hidden');
            entries.innerHTML = '';
            return;
        }

        overlay.classList.remove('hidden');
        entries.innerHTML = '';

        for (const bid of bidHistory) {
            const el = document.createElement('span');
            const team = bid.player % 2 === 0 ? 'team-ns' : 'team-ew';
            el.className = `watch-bid-entry ${team}`;
            el.innerHTML = `${SEAT_NAMES_FR[bid.player]} : ${bidActionHtml(bid.action)}`;
            entries.appendChild(el);
        }
    }

    renderTricksHistory(tricks) {
        const container = this.el('tricks-list');
        if (!container) return;
        container.innerHTML = '';
        if (!tricks || tricks.length === 0) return;

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
            container.appendChild(row);
        }

        container.scrollTop = container.scrollHeight;
    }

    _renderInitialStats() {
        const header = this.el('stats-header');
        const body = this.el('stats-body');
        if (header) header.innerHTML = '';
        if (body) body.innerHTML = '<div class="stats-placeholder">Cliquez sur un bouton pour avancer</div>';
    }

    _clearStats() {
        const header = this.el('stats-header');
        const body = this.el('stats-body');
        if (header) header.innerHTML = '';
        if (body) body.innerHTML = '';
    }

    _renderFinishedStats(state) {
        const header = this.el('stats-header');
        const body = this.el('stats-body');

        if (this.isReplay && !state.is_terminal) {
            header.innerHTML = '<span class="stats-replay-tag">REPLAY</span>';
            body.innerHTML = '<div class="stats-incomplete">Partie incomplete</div>';
            return;
        }

        const rewards = state.rewards;
        const nsWon = rewards ? rewards[0] > rewards[1] : state.points[0] > state.points[1];
        const isDraw = rewards ? rewards[0] === rewards[1] : false;
        const resultText = isDraw ? 'Egalite' : (nsWon ? 'NS gagne' : 'EO gagne');

        if (this.isReplay) {
            header.innerHTML = `<span class="stats-replay-tag">REPLAY</span> <span class="stats-result">${resultText}</span>`;
        } else {
            header.innerHTML = `<span class="stats-result">${resultText}</span>`;
        }

        let bodyHtml = '';
        const sd = state.score_detail;
        if (sd) {
            const teamNames = ['NS', 'EO'];
            const contractTeamName = teamNames[sd.contract_team];
            const contractResult = sd.contract_made ? 'Reussi' : 'Chute';
            const contractClass = sd.contract_made ? 'contract-made' : 'contract-failed';
            bodyHtml += `<div class="stats-contract-result ${contractClass}">${sd.contract_value}${SUIT_LABELS[state.contract.trump]} par ${contractTeamName} — ${contractResult}</div>`;
            bodyHtml += `<div class="stats-score-line">Plis : NS ${sd.trick_points[0]} — EO ${sd.trick_points[1]}</div>`;
            if (sd.belote[0] > 0 || sd.belote[1] > 0) {
                const parts = [];
                if (sd.belote[0] > 0) parts.push(`+${sd.belote[0]} belote NS`);
                if (sd.belote[1] > 0) parts.push(`+${sd.belote[1]} belote EO`);
                bodyHtml += `<div class="stats-score-line">${parts.join(' / ')}</div>`;
            }
            bodyHtml += `<div class="stats-final-scores">Score : NS ${sd.final_scores[0]} — EO ${sd.final_scores[1]}</div>`;
        } else {
            bodyHtml = `<div class="stats-final">NS: ${state.points[0]}pts (${state.tricks_won[0]}P) / EO: ${state.points[1]}pts (${state.tricks_won[1]}P)</div>`;
        }
        body.innerHTML = bodyHtml;
    }

    updateTransportButtons() {
        const prevBtn = this.el('prev-btn');
        const startBtn = this.el('start-btn');
        const stepBtn = this.el('step-btn');
        const endBtn = this.el('end-btn');

        const canGoBack = this.historyIndex >= 0;
        const canGoForward = !this.isAtEnd() || this.historyIndex < this.moveHistory.length - 1;

        if (prevBtn) prevBtn.disabled = !canGoBack;
        if (startBtn) startBtn.disabled = !canGoBack;
        if (stepBtn) stepBtn.disabled = !canGoForward || this.waitingForStep;
        if (endBtn) endBtn.disabled = !canGoForward || this.waitingForStep;
    }

    _continueAutoPlayFromBuffer() {
        if (!this.autoPlayMode) return;
        const data = this.moveHistory[this.historyIndex];
        if (!data) { this.stopAutoPlay(); return; }

        if (data.finished) {
            this.stopAutoPlay();
            return;
        }

        const delay = this.autoPlayMode === 'end' ? 0 : 1000;
        this.autoPlayTimer = setTimeout(() => {
            if (!this.autoPlayMode) return;
            if (_animatingTrick === this.trickPrefix) {
                this._continueAutoPlayFromBuffer();
                return;
            }
            this.goToNextMove();
        }, delay);
    }

    continueAutoPlay(data) {
        if (!this.autoPlayMode || data.finished) {
            this.stopAutoPlay();
            return;
        }

        const delay = this.autoPlayMode === 'end' ? 0 : 1000;
        this.autoPlayTimer = setTimeout(() => {
            if (!this.autoPlayMode) return;
            if (_animatingTrick === this.trickPrefix) {
                this.continueAutoPlay(data);
                return;
            }
            this.goToNextMove();
        }, delay);
    }

    handleBeloteEvent(data) {
        if (data.belote_event) {
            const text = data.belote_event === 'belote' ? 'Belote !' : 'Rebelote !';
            showBeloteAnnouncement(`${this.prefix}-trick-area`, text);
        }
    }
}
