// Replay view — browse and replay saved games with oracle analysis.
// The whole game is preloaded at replay_load, so navigation (buttons,
// arrows, clicking any move in the "Coups" list) is instant and local.

import { send, onMessage, offMessage, onOpen, offOpen } from '../ws.js';
import {
    SEAT_NAMES_FR, suitHtml, cardCode, cardChipHtml,
    bidChipHtml, actionName, SUIT_DISPLAY_ORDER, _animatingTrick
} from '../shared/cards.js';
import { botLabel } from '../shared/agents.js';
import { BoardRenderer } from '../shared/board.js';
import { initCfnBox, updateCfnBox } from '../shared/cfn-box.js';
import { setGameId, setActionIdx, openBugReport } from '../shared/bug-report.js';

const SEAT_INITIALS = ['N', 'E', 'S', 'O'];

const TEMPLATE = `
<a id="replay-resume" class="analyse-back analyse-back-go hidden" href="/jouer/humain"></a>
<div id="replay-main">
    <div id="replay-history">
        <div class="section-title" id="replay-history-title">Historique</div>
        <div id="replay-search">
            <input type="text" id="replay-search-input" placeholder="ID..." maxlength="4">
            <button id="replay-search-btn">Charger</button>
        </div>
        <div id="replay-list"></div>
    </div>

    <div id="replay-left">
        <div id="replay-score-bar">
            <span id="replay-score-ns">Nord-Sud : 0</span>
            <span id="replay-game-id" class="game-id-tag hidden"></span>
            <span id="replay-contract-display"></span>
            <span id="replay-score-ew">Est-Ouest : 0</span>
            <button id="replay-report-btn" class="report-btn hidden" title="Signaler un bug">Bug</button>
        </div>
        <div id="replay-match-bar" class="match-bar hidden"></div>
        <div id="replay-cfn" class="cfn-box hidden" title="Cliquer pour copier"></div>

        <div class="seats">
            <div class="seat north">
                <div class="seat-label" id="replay-label-n">Nord</div>
                <div class="hand" id="replay-hand-north"></div>
            </div>
            <div class="seat west">
                <div class="seat-label" id="replay-label-w">Ouest</div>
                <div class="hand" id="replay-hand-west"></div>
            </div>
            <div id="replay-trick-area">
                <div class="trick-card" id="replay-trick-n"></div>
                <div class="trick-card" id="replay-trick-w"></div>
                <div class="trick-card" id="replay-trick-e"></div>
                <div class="trick-card" id="replay-trick-s"></div>
            </div>
            <div class="seat east">
                <div class="seat-label" id="replay-label-e">Est</div>
                <div class="hand" id="replay-hand-east"></div>
            </div>
            <div class="seat south">
                <div class="seat-label" id="replay-label-s">Sud</div>
                <div class="hand" id="replay-hand-south"></div>
            </div>
        </div>

        <div id="replay-curve" class="hidden"></div>
        <div id="replay-last-trick" class="hidden"></div>
    </div>

    <div id="replay-right">
        <div id="replay-transport">
            <div class="transport-row">
                <button id="replay-prev-btn" title="Coup precedent">|◀</button>
                <button id="replay-step-btn" title="Prochain coup">▶|</button>
            </div>
            <div class="transport-row">
                <button id="replay-start-btn" title="Retour au debut">|◀◀</button>
                <button id="replay-auto-btn" title="Auto-play">▶</button>
                <button id="replay-end-btn" title="Fin de partie">▶▶|</button>
            </div>
        </div>

        <div id="replay-stats-panel">
            <div id="replay-stats-header"></div>
            <div id="replay-stats-body"></div>
        </div>

        <div id="replay-errors" class="hidden">
            <div class="section-title">Moments de la donne</div>
            <div id="replay-errors-body"></div>
        </div>

        <div id="replay-analysis">
            <div class="section-title">Analyse Oracle</div>
            <div id="replay-analysis-body" class="analysis-body"></div>
        </div>

        <div id="replay-moves">
            <div class="section-title">Coups</div>
            <div id="replay-moves-list"></div>
        </div>
    </div>
</div>
`;

let replayBoard = null;
let replayTotalActions = 0;
let _pendingLoadId = null;
let _analysisByIdx = null;   // action_idx -> move analysis
let _analysisSummary = null;
let _bidsByIdx = null;       // action_idx -> bid analysis (model annonce)
let _oracleBids = null;      // deal-level DD contracts {suits, best}
let _curve = null;           // séries de la courbe (points, projection, seuil)
let _initialHands = null;    // all 4 hands at deal start (replay = full info)
let _gameCfn = null;         // full-game CFN (auction + play) — porte le lien vers /analyse/jeu
let _agentsByIdx = null;     // action_idx -> {doudou, oracle, isdd} card choices
let _agentsPending = false;  // the review is still being computed server-side
let _agentsDone = 0;         // cards reviewed so far (streamed, in play order)
let _agentsTotal = 0;        // cards the review will cover
let _pendingJumpIdx = null;  // coup demandé par l'URL, appliqué au chargement
let _currentGameId = null;   // guards late answers from a previously-open game
// Où en était la partie quand cette donne s'est jouée (`replay_loaded.match`),
// ou null pour une donne isolée — le cas par défaut du site. `before` est le
// cumul d'avant la donne : c'est celui-là qui a compté au moment d'annoncer,
// pas le total final.
let _match = null;

// The three reference bots, in the order they are shown under a played card.
const REVIEW_BOTS = [
    { key: 'doudou', label: 'DouDou50', title: 'Réseau Q direct, sans recherche' },
    { key: 'oracle', label: 'Oracle',   title: 'Solveur double-dummy — voit les 4 mains' },
    { key: 'isdd',   label: 'Dédé',     title: 'IS-DD : mondes playgen résolus en double-dummy' },
];

// Cinq états, deux familles qui ne se mélangent pas.
//
// `parfait` / `imprecision` / `decisive` sont des **jugements** : le coup a
// coûté quelque chose, en score de donne (contrat compris), et `decisive` dit
// que le contrat a basculé — perdre des points dans un contrat acquis et faire
// chuter le contrat ne sont pas le même événement.
//
// `malchance` / `aubaine` sont des **explications**, pas des reproches : elles
// naissent du désaccord entre l'Oracle, qui voit les quatre mains, et Dédé, qui
// ne voit que ce que le siège pouvait savoir. Elles ont donc leur propre palette
// (froide), délibérément hors de celle du blâme — un joueur ne doit pas lire
// « malchance » comme un cran de faute.
const CATEGORY_UI = {
    parfait:   { tag: '✓',  cls: 'an-best',   label: 'Meilleur coup' },
    imprecision: { tag: '?!', cls: 'an-inacc', label: 'Imprécision',
                   hint: 'des points perdus, le contrat tient quand même' },
    decisive:  { tag: '??', cls: 'an-blund',  label: 'Faute décisive',
                 hint: 'ce coup fait basculer le contrat' },
    malchance: { tag: '≈',  cls: 'an-unlucky', label: 'Malchance',
                 hint: 'bon choix vu de ce siège — la donne était contre vous' },
    aubaine:   { tag: '!',  cls: 'an-lucky',  label: 'Coup heureux',
                 hint: 'ça paraissait mauvais, ça ne l’était pas' },
    // Catégories des analyses d'avant la v7 (échelle en points cartes). Une
    // ligne périmée est recalculée, mais elle peut être servie le temps que
    // `get_or_compute` rende la main.
    bon:       { tag: '✓',  cls: 'an-good',   label: 'Bon coup' },
    erreur:    { tag: '?',  cls: 'an-error',  label: 'Erreur' },
    faute:     { tag: '??', cls: 'an-blund',  label: 'Faute' },
};

// La catégorie affichée pour un coup : celle de l'Oracle, corrigée par l'avis de
// Dédé dès que la revue est arrivée.
//
// L'Oracle voit les quatre mains ; le désigner seul comme juge accuse un joueur
// d'erreurs qu'il ne pouvait pas voir. Dédé, lui, ne voit que l'information du
// siège. D'où la grille : les deux d'accord = vraie erreur ; l'Oracle seul =
// malchance ; Dédé seul = coup heureux.
//
// `isdd_cost` est en écart de score de donne, la même échelle que `cost_score`.
// Le seuil n'est pas zéro : IS-DD est une moyenne sur des mondes échantillonnés,
// donc il rend rarement exactement 0 même quand il approuve.
const ISDD_NOISE = 1.0;

function moveCategory(an, idx) {
    if (!an || an.forced) return null;
    const base = an.category || 'parfait';
    const review = _agentsByIdx && _agentsByIdx[idx];
    const isdd = review && review.isdd_cost;
    // Pas encore de revue, ou une revue v2 sans le coût : on s'en tient à
    // l'Oracle plutôt que d'inventer un verdict.
    if (isdd === null || isdd === undefined) return base;

    const oracleBlames = (an.cost_score || 0) > 0;
    const isddBlames = isdd > ISDD_NOISE;
    if (oracleBlames && !isddBlames) return 'malchance';
    if (!oracleBlames && isddBlames) return 'aubaine';
    return base;
}

function pct(p) {
    return `${(p * 100).toFixed(p >= 0.1 ? 0 : 1)} %`;
}

// Score de partie d'avant la donne, **dans le repère de la page qui le reçoit**
// — le paramètre `s` vaut toujours « Nord-Sud − Est-Ouest » chez son
// destinataire. La page annonces assied le siège analysé en Sud, donc un siège
// Est-Ouest voit ses deux nombres échangés ici ; /analyse/jeu dessine les
// sièges physiques et les reçoit tels quels. La rotation se fait une seule
// fois, à la fabrication du lien — même principe que `rooms.rotate_state` à la
// diffusion.
//
// Rendu vide hors partie : un `s` absent vaut 0-0, et 0-0 est alors la vérité.
function matchScoreParam(seat) {
    if (!_match) return '';
    const [ns, ew] = _match.before;
    if (!ns && !ew) return '';
    const rot = (seat !== null && seat % 2 === 1) ? [ew, ns] : [ns, ew];
    return `&s=${rot[0]}-${rot[1]}`;
}

// URL de la page annonces pré-remplie avec la main du siège qui parle et
// l'enchère jusqu'à ce coup (exclu). `from`/`i` portent le chemin du retour.
// Retourne null si le coup n'est pas une annonce analysable.
function bidAnalysisUrl(idx) {
    if (!replayBoard || !_initialHands) return null;
    const data = replayBoard.moveHistory[idx];
    if (!data || !data.move || data.move.phase !== 0) return null;
    const hand = _initialHands[data.move.player];
    if (!hand || hand.length !== 8) return null;
    const history = [];
    for (let i = 0; i < idx; i++) {
        const m = replayBoard.moveHistory[i].move;
        if (m && m.phase === 0) history.push(m.action);
    }
    let url = `/analyse/annonces?hand=${hand.map(cardCode).join(',')}`;
    if (history.length) url += `&history=${history.join(',')}`;
    if (_currentGameId) url += `&from=${encodeURIComponent(_currentGameId)}&i=${idx}`;
    url += matchScoreParam(data.move.player);
    return url;
}

// URL de la page jeu de la carte pour ce coup, ou null si on n'a pas le CFN.
function cardAnalysisUrl(idx) {
    if (!_gameCfn) return null;
    let url = `/analyse/jeu?cfn=${encodeURIComponent(_gameCfn)}&i=${idx}`;
    if (_currentGameId) url += `&from=${encodeURIComponent(_currentGameId)}`;
    // Sièges physiques sur cette page-là : aucune rotation.
    url += matchScoreParam(null);
    return url;
}

// ===== "Qui aurait joué quoi" (the three reference bots) =====

// One row of chips: what each bot would have played at this exact position,
// whoever actually played it. A chip is green when the bot agrees with the
// card that was played, grey otherwise.
function botsHtml(idx, played) {
    const entry = _agentsByIdx && _agentsByIdx[idx];
    if (!entry) {
        // Cards are reviewed in play order, so an absent entry on a pending
        // review just means the search has not reached this one yet.
        if (!_agentsPending) return '';
        const progress = _agentsTotal ? ` ${_agentsDone}/${_agentsTotal}` : '';
        return `<div class="an-bots an-bots-loading">Analyse des bots…${progress}</div>`;
    }
    if (entry.forced) return '';

    const chips = REVIEW_BOTS.map(bot => {
        const card = entry[bot.key];
        if (card === null || card === undefined) return '';
        const same = card === played;
        return `<span class="an-bot ${same ? 'an-bot-same' : 'an-bot-diff'}" ` +
            `title="${bot.title}">` +
            `<span class="an-bot-name">${bot.label}</span>` +
            `<span class="an-bot-card">${cardChipHtml(card)}</span></span>`;
    }).filter(Boolean).join('');

    return chips ? `<div class="an-bots">${chips}</div>` : '';
}

// Ce que l'Oracle aurait joué. **La classe, pas son représentant** : en score de
// donne plusieurs cartes valent exactement pareil (cinq, dans les donnes qu'on a
// examinées), et n'en désigner qu'une promet une précision qui n'existe pas —
// le départage de `solve_best_card` est un ordre interne, pas un conseil.
function oracleAltHtml(an) {
    const klass = (an.best_class && an.best_class.length) ? an.best_class : [an.best];
    const shown = klass.slice(0, 3).map(cardChipHtml).join('');
    const more = klass.length > 3 ? ` +${klass.length - 3}` : '';
    const lead = klass.length > 1 ? 'Oracle, au choix :' : 'Oracle :';
    return `<span class="an-alt">${lead} ${shown}${more}</span>`;
}

// ===== « Analyse rapide » d'une annonce =====
//
// Deux chiffres sur l'annonce qu'on regarde, sans quitter la page : combien de
// fois le contrat passe, et l'espérance de points. C'est le chiffre-phare de la
// page annonces ramené à une ligne.
//
// **Ce que ça juge, et pourquoi ça compte** : Rejouer note les annonces au Q de
// bid v6, donc appelle « erreur » tout désaccord avec lui. Un taux de réussite
// simulé ne dépend d'aucun modèle de référence — il note l'humain et v6 au même
// barème, et v6 se trompe aussi. D'où la seconde ligne quand son avis diffère.
//
// Le résultat est gardé par index de coup **et** mis en cache côté serveur
// (`sim_cache`, clé `(main, enchère précédente, annonce forcée)`), donc revenir
// sur une annonce déjà chiffrée est instantané, y compris d'une donne à l'autre.
const _quickBid = new Map();   // idx -> {state, lines:[], progress}

function quickBidHtml(idx, played) {
    const q = _quickBid.get(idx);
    if (!q) {
        // Un passe sans alternative de V6 n'a rien à simuler : pas de contrat
        // dont on puisse mesurer la réussite.
        const bid = _bidsByIdx && _bidsByIdx[idx];
        const v6 = bid && bid.model_best;
        if (played === 0 && !(v6 !== undefined && v6 !== null && v6 !== 0)) return '';
        return `<button class="qb-btn" data-qb="${idx}">Analyse rapide</button>`;
    }
    if (q.state === 'error') {
        return `<div class="qb-box qb-err">${q.error}</div>` +
            `<button class="qb-btn" data-qb="${idx}">Réessayer</button>`;
    }
    let html = '<div class="qb-box">';
    for (const l of q.lines) {
        // `made_pct` est conditionnel — « quand ce camp tient le contrat » —
        // parce qu'une annonce n'empêche pas les adversaires de surenchérir.
        // `taker_pct` dit à quelle fréquence c'est arrivé, donc à quel point le
        // premier chiffre porte sur un sous-ensemble.
        const made = l.made_pct === null || l.made_pct === undefined
            ? '—' : `${l.made_pct} %`;
        const exp = l.expected === null || l.expected === undefined
            ? '—' : (l.expected > 0 ? `+${l.expected}` : `${l.expected}`);
        html += `<div class="qb-line">` +
            `<span class="qb-label">${l.label}</span>` +
            `<span class="qb-bid">${bidChipHtml(l.action)}</span>` +
            `<span class="qb-made" title="Part des simulations où ce camp a tenu le contrat et l'a réussi — ` +
            `il garde l'enchère ${l.taker_pct ?? '?'} % du temps">passe ${made}</span>` +
            `<span class="qb-exp" title="Espérance d'écart de points sur toutes les simulations, ` +
            `surenchères et donnes passées comprises">${exp} pts</span></div>`;
    }
    if (q.state === 'running') {
        html += `<div class="qb-progress">${q.progress || 'simulation…'}</div>`;
    }
    html += '</div>';
    return html;
}

function bindQuickBid(idx) {
    const btn = document.querySelector(`[data-qb="${idx}"]`);
    if (!btn) return;
    btn.addEventListener('click', () => {
        _quickBid.set(idx, { state: 'running', lines: [], progress: 'simulation…' });
        refreshMoveStats();
        send({ type: 'replay_bid_quick', game_id: _currentGameId, idx, req_id: idx });
    });
}

// `req_id` est l'index du coup : sans lui, le résultat d'une analyse lancée puis
// quittée atterrirait sur l'annonce affichée au moment de la réponse. Même
// raison que les onglets de la page annonces.
function handleQuickStart(data) {
    const q = _quickBid.get(data.req_id);
    if (!q) return;
    // Combien de lignes attendre : sans ça, la première réponse ferait passer
    // l'analyse en « terminée » et la progression disparaîtrait pendant que la
    // seconde annonce tourne encore.
    q.expected = data.lines || 1;
    q.progress = `0/${data.sims}`;
    refreshMoveStats();
}

function handleQuickUpdate(data) {
    const q = _quickBid.get(data.req_id);
    if (!q) return;
    q.progress = `${data.label} — ${data.completed}/${data.total}`;
    refreshMoveStats();
}

function handleQuickDone(data) {
    const q = _quickBid.get(data.req_id);
    if (!q) return;
    q.lines.push(data);
    if (q.lines.length >= (q.expected || 1)) {
        q.state = 'done';
        q.progress = null;
    } else {
        q.progress = 'simulation…';
    }
    refreshMoveStats();
}

function handleQuickError(data) {
    const q = _quickBid.get(data.req_id);
    if (!q) return;
    q.state = 'error';
    q.error = data.error || 'Simulation impossible';
    refreshMoveStats();
}

// ===== Move stats (current move annotation) =====

function replayRenderMoveStats(move, state) {
    const header = replayBoard.el('stats-header');
    const body = replayBoard.el('stats-body');

    if (!move) {
        header.innerHTML = '';
        body.innerHTML = '';
        return;
    }

    const seatName = SEAT_NAMES_FR[move.player];
    const teamClass = move.player % 2 === 0 ? 'team-ns' : 'team-ew';

    header.innerHTML = `<span class="stats-replay-tag">REPLAY</span> ` +
        `<span class="stats-player ${teamClass}">${seatName}</span>` +
        // `move.name` rend la carte en ASCII (« 8S ») là où toute la page passe
        // par `cardChipHtml` (« 8♠ », rouge ou noir selon la couleur).
        `<span class="stats-action">${move.phase === 0 ? bidChipHtml(move.action) : cardChipHtml(move.action)}</span>`;
    body.innerHTML = '';

    // Bid move: model annonce + oracle annonce + link to the annonces page
    if (move.phase === 0) {
        const idx = replayBoard.historyIndex;
        let html = '';
        const bid = _bidsByIdx && _bidsByIdx[idx];
        if (bid && bid.model_best !== undefined) {
            const agree = bid.model_best === move.action;
            html += `<div class="an-move ${agree ? 'an-best' : 'an-inacc'}">` +
                `<span class="an-tag">${agree ? '✓' : '≠'}</span>` +
                `Bid V6 : ${bidChipHtml(bid.model_best)}` +
                `<span class="an-bid-q">Q ${bid.q_best.toFixed(2)}` +
                (!agree && bid.q_played !== null && bid.q_played !== undefined
                    ? ` · joué ${bid.q_played.toFixed(2)}` : '') +
                `</span></div>`;
        }
        // Playgen a son mot à dire sur l'enchère : sa tête d'annonce (v2), lue
        // depuis la vue du siège qui parle. C'est un modèle du monde, pas un
        // bidder entraîné — d'où la probabilité plutôt qu'un Q.
        if (bid && bid.playgen_best !== undefined) {
            const agree = bid.playgen_best === move.action;
            html += `<div class="an-move ${agree ? 'an-best' : 'an-inacc'}">` +
                `<span class="an-tag">${agree ? '✓' : '≠'}</span>` +
                `Playgen : ${bidChipHtml(bid.playgen_best)}` +
                `<span class="an-bid-q">p ${pct(bid.playgen_p)}` +
                (!agree && bid.playgen_p_played !== null && bid.playgen_p_played !== undefined
                    ? ` · joué ${pct(bid.playgen_p_played)}` : '') +
                `</span></div>`;
        }
        // Le chiffre, en place. Le lien juste en dessous mène au tableau
        // complet : le bouton répond « est-ce que ça passe ? », la page répond
        // « pourquoi ».
        html += quickBidHtml(idx, move.action);
        // Une vraie ancre, pas un bouton : le routeur laisse passer Ctrl+clic et
        // clic-milieu, donc l'utilisateur choisit onglet courant ou nouvel
        // onglet clic par clic, sans qu'on ait à trancher pour lui.
        const bidUrl = bidAnalysisUrl(idx);
        if (bidUrl) {
            html += `<a class="an-bid-analyse-btn" href="${bidUrl}">Analyser cette annonce →</a>`;
        }
        body.innerHTML = html;
        bindQuickBid(idx);
        return;
    }

    if (move.phase !== 1) return;

    const idx = replayBoard.historyIndex;
    // Oracle annotation for this move (history entry i == action index i)
    const an = _analysisByIdx && _analysisByIdx[idx];
    let html = '';
    if (an) {
        if (an.forced) {
            html = `<div class="an-move an-forced">Carte forcée</div>`;
        } else {
            const cat = moveCategory(an, idx);
            const ui = CATEGORY_UI[cat] || CATEGORY_UI.parfait;
            html = `<div class="an-move ${ui.cls}"` +
                (ui.hint ? ` title="${ui.hint}"` : '') + '>' +
                `<span class="an-tag">${ui.tag}</span> ${ui.label}`;
            // Le chiffre est en **score de donne** : ce que ce coup a coûté à
            // la marque, contrat compris. Les points cartes restent en
            // infobulle — ils répondent à « quelle carte prend le plus de
            // plis », pas à « qu'est-ce que ça m'a coûté ».
            if (an.cost_score > 0) {
                html += ` <span class="an-cost" title="${an.cost} points cartes">` +
                    `−${an.cost_score}</span>`;
            }
            if (an.cost_score > 0 || an.cost > 0) html += oracleAltHtml(an);
            html += '</div>';
        }
    }
    // Le pendant du lien des annonces, pour une carte : la page /analyse/jeu
    // repart du CFN complet et de l'index, donc rien à recalculer ici. Inutile
    // sur une carte forcée — il n'y a pas de décision à peser.
    const cardUrl = (an && an.forced) ? null : cardAnalysisUrl(idx);
    if (cardUrl) {
        html += `<a class="an-bid-analyse-btn" href="${cardUrl}">Analyser cette carte →</a>`;
    }
    body.innerHTML = html + botsHtml(idx, move.action);
}

// ===== Navigable moves list =====

function jumpTo(i) {
    if (!replayBoard || !replayBoard.active) return;
    if (_animatingTrick === 'replay-trick') return;
    replayBoard.stopAutoPlay();
    replayBoard._pendingFlushData = null;
    replayBoard.historyIndex = i;
    replayBoard.renderHistoryEntry(i);
}

// Garder le coup courant en vue en ne bougeant QUE l'ascenseur de la liste.
// scrollIntoView() remonte tous les conteneurs défilants : sur mobile, où la
// liste s'affiche en entier dans la page, il faisait sauter la page sur la
// liste à chaque coup — le tapis disparaissait de l'écran.
function scrollIntoList(list, el) {
    const lr = list.getBoundingClientRect();
    const er = el.getBoundingClientRect();
    if (er.top < lr.top) list.scrollTop -= lr.top - er.top;
    else if (er.bottom > lr.bottom) list.scrollTop += er.bottom - lr.bottom;
}

// L'URL dit toujours quelle partie et quel coup sont à l'écran, pour qu'un
// Retour depuis une page d'analyse retombe exactement ici — et pour qu'un coup
// précis se partage. `replaceState` et non `pushState` : sinon chaque coup
// parcouru créerait une entrée d'historique et le Retour du navigateur
// remonterait la partie coup par coup au lieu de quitter la page.
function syncReplayUrl() {
    if (!_currentGameId) return;
    const idx = replayBoard ? replayBoard.historyIndex : -1;
    const q = new URLSearchParams({ game: _currentGameId });
    if (idx >= 0) q.set('i', String(idx));
    history.replaceState(null, '', `${window.location.pathname}?${q}`);
}

function updateMovesHighlight() {
    const list = document.getElementById('replay-moves-list');
    if (!list || !replayBoard) return;
    const idx = replayBoard.historyIndex;
    setActionIdx(idx + 1);
    syncReplayUrl();
    let current = null;
    list.querySelectorAll('[data-idx]').forEach(el => {
        const isCurrent = parseInt(el.dataset.idx) === idx;
        el.classList.toggle('mv-current', isCurrent);
        if (isCurrent) current = el;
    });
    if (current) scrollIntoList(list, current);
    renderCurve();
}

function buildMovesList() {
    const list = document.getElementById('replay-moves-list');
    if (!list || !replayBoard) return;
    list.innerHTML = '';

    const bids = document.createElement('div');
    bids.className = 'mv-bids';
    let trickRow = null;
    let cardsInRow = 0;
    let trickNum = 0;

    replayBoard.moveHistory.forEach((data, i) => {
        const m = data.move;
        if (!m) return;
        if (m.phase === 0) {
            const chip = document.createElement('span');
            chip.className = 'mv-bid ' + (m.player % 2 === 0 ? 'team-ns' : 'team-ew');
            chip.dataset.idx = i;
            chip.innerHTML = `${SEAT_INITIALS[m.player]}&nbsp;${bidChipHtml(m.action)}`;
            chip.title = SEAT_NAMES_FR[m.player];
            const bid = _bidsByIdx && _bidsByIdx[i];
            // Le liseré ne signale que le désaccord de Bid V6 : c'est lui le
            // bidder de référence. Playgen ne parle que dans l'infobulle.
            if (bid && bid.model_best !== undefined && bid.model_best !== m.action) {
                chip.classList.add('mv-bid-diff');
                chip.title += ` — Bid V6 : ${actionName(bid.model_best, 0)}`;
            }
            if (bid && bid.playgen_best !== undefined && bid.playgen_best !== m.action) {
                chip.title += ` — Playgen : ${actionName(bid.playgen_best, 0)}`;
            }
            chip.addEventListener('click', () => jumpTo(i));
            bids.appendChild(chip);
        } else {
            if (cardsInRow === 0) {
                trickNum++;
                trickRow = document.createElement('div');
                trickRow.className = 'mv-trick';
                const num = document.createElement('span');
                num.className = 'mv-num';
                num.textContent = trickNum;
                trickRow.appendChild(num);
                list.appendChild(trickRow);
            }
            const an = _analysisByIdx && _analysisByIdx[i];
            const cardEl = document.createElement('span');
            let cls = 'mv-card';
            let tip = SEAT_NAMES_FR[m.player];
            if (an) {
                if (an.forced) {
                    cls += ' mv-forced';
                    tip += ' — carte forcée';
                } else {
                    const cat = moveCategory(an, i);
                    cls += ' mv-' + cat;
                    const ui = CATEGORY_UI[cat];
                    tip += ` — ${ui ? ui.label : cat}`;
                    if (an.cost_score > 0) tip += ` (−${an.cost_score})`;
                }
            }
            cardEl.className = cls;
            cardEl.dataset.idx = i;
            cardEl.innerHTML = cardChipHtml(m.action);
            cardEl.title = tip;
            cardEl.addEventListener('click', () => jumpTo(i));
            trickRow.appendChild(cardEl);
            cardsInRow++;

            if (cardsInRow === 4) {
                cardsInRow = 0;
                const w = data.state.last_trick_winner;
                if (w !== null && w !== undefined) {
                    const win = document.createElement('span');
                    win.className = 'mv-winner ' + (w % 2 === 0 ? 'team-ns' : 'team-ew');
                    win.textContent = `${SEAT_INITIALS[w]} +${data.state.last_trick_points}`;
                    trickRow.appendChild(win);
                }
            }
        }
    });

    if (bids.children.length) list.prepend(bids);
    updateMovesHighlight();
}

// ===== Oracle analysis =====

function renderAnalysisSummary() {
    const el = document.getElementById('replay-analysis-body');
    if (!el) return;
    if (!_analysisSummary) {
        el.innerHTML = '<div class="an-loading">Analyse en cours…</div>';
        return;
    }
    let html = '';
    if (_oracleBids && _oracleBids.suits) {
        // Deal-level DD scores per trump suit — static info, no bid "decision"
        html += '<table class="an-dd-table" title="Points réalisables en jeu parfait (double-dummy) pour chaque atout">' +
            '<tr><th></th><th class="team-ns">Nord-Sud</th><th class="team-ew">Est-Ouest</th></tr>';
        for (const suit of SUIT_DISPLAY_ORDER) {
            const [ns, ew] = _oracleBids.suits[suit];
            html += `<tr><td>${suitHtml(suit)}</td>` +
                `<td>${ns}</td><td>${ew}</td></tr>`;
        }
        html += '</table>';
    }
    // « Regrets » et non « Perdu » : chaque coût est mesuré **indépendamment à
    // sa position**, en supposant un jeu parfait ensuite. Deux coups peuvent
    // donc « perdre » la même donne et leurs coûts s'additionner au-delà de ce
    // que la donne vaut — on a vu 904 sur une donne à ~460. C'est correct comme
    // somme de regrets, et ce serait faux comme total de points perdus.
    // « Cartes » garde l'ancienne échelle : elle reste la bonne réponse à
    // « quelle carte prend le plus de plis », elle n'est simplement plus ce qui
    // définit une erreur.
    html += '<table class="an-table"><tr><th></th>' +
        '<th title="Somme des regrets, chacun mesuré indépendamment à sa position — ' +
        'deux coups peuvent perdre la même donne, donc ce total peut dépasser ce ' +
        'que la donne vaut">Regrets</th>' +
        '<th title="Coût total en points cartes — une autre échelle, à ne pas soustraire">Cartes</th>' +
        '<th title="Imprécision : des points perdus, le contrat tient">?!</th>' +
        '<th title="Faute décisive : le contrat bascule">??</th></tr>';
    // Les deux colonnes de comptes sont **recalculées ici**, pas lues dans
    // `p.counts`. Le serveur ne connaît que l'avis de l'Oracle ; sans ce
    // recalcul le tableau annonçait « Sud : 2 fautes décisives » pendant que le
    // panneau des moments classait les deux mêmes coups en malchance et que le
    // compteur affichait 0 — trois chiffres contradictoires sur un écran.
    //
    // La colonne « Regrets », elle, reste l'avis brut de l'Oracle, et c'est
    // voulu : un siège peut avoir laissé 900 de regret sans une seule faute
    // qu'il pouvait voir. C'est même le message que la page doit faire passer.
    const counts = [0, 1, 2, 3].map(() => ({ imprecision: 0, decisive: 0 }));
    for (const idx of Object.keys(_analysisByIdx || {})) {
        const an = _analysisByIdx[idx];
        if (!an || an.forced) continue;
        const cat = moveCategory(an, Number(idx));
        if (cat in counts[an.player]) counts[an.player][cat]++;
    }
    for (const p of _analysisSummary.players) {
        if (p.moves === 0) continue;
        const team = p.player % 2 === 0 ? 'team-ns' : 'team-ew';
        const c = counts[p.player];
        html += `<tr><td class="${team}">${SEAT_NAMES_FR[p.player]}</td>` +
            `<td>${p.total_cost_score ?? 0}</td><td>${p.total_cost}</td>` +
            `<td class="an-inacc">${c.imprecision}</td>` +
            `<td class="an-blund">${c.decisive}</td></tr>`;
    }
    html += '</table>';
    el.innerHTML = html;
}

// ===== La courbe de la donne =====
//
// Un seul axe, en points cartes du **preneur**, et trois tracés qui s'y lisent
// ensemble :
//
//   - l'**aire** : ce qu'il a déjà ramassé, huit marches ;
//   - la **ligne** : ce qu'il fera en jeu parfait depuis chaque position —
//     plate tant que personne ne se trompe, elle décroche d'un coup à une
//     erreur ;
//   - l'**horizontale** : le seuil à atteindre, qui monte par paliers pendant
//     l'enchère puis se fige.
//
// Les deux premiers convergent forcément au 8e pli : quand il ne reste rien à
// jouer, la projection *est* le réalisé. Et le passage de la ligne sous le seuil
// est exactement une « faute décisive » — la courbe et le panneau des moments
// racontent donc la même histoire, par construction.
//
// **Un seul camp tracé.** Les points cartes sont à somme constante (162), donc
// la courbe de l'autre camp en est le miroir exact : elle doublerait l'encre
// sans ajouter un bit. C'est aussi pour ça que tout est orienté preneur et
// jamais Nord-Sud.
//
// La ligne suppose un **jeu parfait des quatre joueurs**, défense comprise :
// elle peut donc plonger sous le seuil sur une donne finalement gagnée, si
// l'adversaire a rendu l'erreur au coup suivant. C'est correct, et c'est dit
// dans la légende — sans quoi la courbe a l'air de contredire le score affiché.
const CURVE_W = 320;
const CURVE_H = 116;
const CURVE_PAD = { l: 4, r: 4, t: 8, b: 12 };

function curveSvg(curve, total) {
    const innerW = CURVE_W - CURVE_PAD.l - CURVE_PAD.r;
    const innerH = CURVE_H - CURVE_PAD.t - CURVE_PAD.b;
    const nAct = Math.max(1, total - 1);
    const x = (i) => CURVE_PAD.l + (i / nAct) * innerW;
    const yMax = curve.capot || curve.points.some(p => p[1] > 162) ? 252 : 162;
    const y = (v) => CURVE_PAD.t + innerH * (1 - Math.min(v, yMax) / yMax);

    const parts = [];

    // Le seuil : paliers pendant l'enchère, horizontale ensuite. Tracé en
    // premier pour passer sous les autres.
    const steps = [];
    let prev = 0;
    for (const [idx, value] of curve.bids) {
        steps.push(`${steps.length ? 'L' : 'M'}${x(idx)} ${y(prev)}`,
                   `L${x(idx)} ${y(value)}`);
        prev = value;
    }
    // Le seuil du jeu tient compte de la belote — elle abaisse la barre au lieu
    // d'ajouter 20 points au bout.
    const firstPlay = curve.points.length ? curve.points[0][0] : nAct;
    steps.push(`${steps.length ? 'L' : 'M'}${x(firstPlay)} ${y(curve.threshold)}`,
               `L${x(nAct)} ${y(curve.threshold)}`);
    parts.push(`<path class="cv-threshold" d="${steps.join(' ')}"
        fill="none" vector-effect="non-scaling-stroke"/>`);

    // L'aire des points ramassés : un escalier, une marche par pli.
    if (curve.points.length) {
        const d = [`M${x(curve.points[0][0])} ${y(0)}`];
        let last = 0;
        for (const [idx, pts] of curve.points) {
            d.push(`L${x(idx)} ${y(last)}`, `L${x(idx)} ${y(pts)}`);
            last = pts;
        }
        d.push(`L${x(nAct)} ${y(last)}`, `L${x(nAct)} ${y(0)}`, 'Z');
        parts.push(`<path class="cv-taken" d="${d.join(' ')}"/>`);
    }

    // La projection, segment par segment : verte au-dessus du seuil, rouge en
    // dessous. Ce changement de couleur **est** le message de la courbe.
    const dd = curve.dd;
    for (let k = 0; k < dd.length; k++) {
        const [idx, v] = dd[k];
        const nextX = k + 1 < dd.length ? x(dd[k + 1][0]) : x(nAct);
        const cls = v >= curve.threshold ? 'cv-made' : 'cv-down';
        parts.push(`<path class="cv-dd ${cls}" d="M${x(idx)} ${y(v)} L${nextX} ${y(v)}"
            fill="none" vector-effect="non-scaling-stroke"/>`);
        if (k + 1 < dd.length && dd[k + 1][1] !== v) {
            const nv = dd[k + 1][1];
            const jcls = nv >= curve.threshold ? 'cv-made' : 'cv-down';
            parts.push(`<path class="cv-dd ${jcls}" d="M${nextX} ${y(v)} L${nextX} ${y(nv)}"
                fill="none" vector-effect="non-scaling-stroke"/>`);
        }
    }

    // Où on en est dans la lecture.
    const cur = replayBoard ? replayBoard.historyIndex : -1;
    if (cur >= 0) {
        parts.push(`<line class="cv-cursor" x1="${x(cur)}" y1="${CURVE_PAD.t}"
            x2="${x(cur)}" y2="${CURVE_H - CURVE_PAD.b}"
            vector-effect="non-scaling-stroke"/>`);
    }

    return `<svg viewBox="0 0 ${CURVE_W} ${CURVE_H}" preserveAspectRatio="none"
        role="img" aria-label="Progression de la donne">${parts.join('')}</svg>`;
}

function renderCurve() {
    const el = document.getElementById('replay-curve');
    if (!el || !replayBoard) return;
    const curve = _curve;
    if (!curve || !curve.dd || !curve.dd.length) {
        el.classList.add('hidden');
        el.innerHTML = '';
        return;
    }
    const total = replayBoard.moveHistory.length;
    const taker = SEAT_NAMES_FR[0] && (curve.taker === 0 ? 'Nord-Sud' : 'Est-Ouest');
    const mult = curve.coinche === 2 ? ' ×3' : curve.coinche === 1 ? ' ×2' : '';
    const bel = curve.belote[curve.taker]
        ? ` − ${curve.belote[curve.taker]} de belote` : '';
    el.innerHTML = curveSvg(curve, total) +
        `<div class="cv-legend">` +
        `<span class="cv-key cv-k-taken">ramassé</span>` +
        `<span class="cv-key cv-k-dd" title="Ce que le preneur ferait si les quatre joueurs jouaient parfaitement à partir de là — un plafond, pas une prédiction">jeu parfait</span>` +
        `<span class="cv-key cv-k-thr">seuil ${curve.threshold}${bel}</span>` +
        `<span class="cv-taker">${curve.value}${suitHtml(curve.trump)}${mult} · ${taker}</span>` +
        `</div>`;
    el.classList.remove('hidden');

    // Cliquer dans la courbe amène au coup correspondant.
    const svg = el.querySelector('svg');
    svg.addEventListener('click', (e) => {
        const r = svg.getBoundingClientRect();
        const frac = (e.clientX - r.left) / r.width;
        const inner = (CURVE_W - CURVE_PAD.l - CURVE_PAD.r) / CURVE_W;
        const i = Math.round(((frac - CURVE_PAD.l / CURVE_W) / inner) * (total - 1));
        jumpTo(Math.max(0, Math.min(total - 1, i)));
    });
}

// ===== Moments de la donne =====
//
// Le panneau qui répond à « où faut-il regarder ». Il est court par
// construction : mesuré à ~2 décisions coûteuses par donne (médiane 2, max 7,
// et un tiers des donnes n'en ont aucune), donc tout tient déplié et il n'y a
// jamais de pagination à prévoir.
//
// Il se remplit en **deux temps**, et c'est assumé : l'Oracle est là avec
// l'analyse (~1 s), l'avis de Dédé arrive avec la revue (~9 s) et peut alors
// reclasser une erreur en malchance. Annoncer un compteur définitif puis le voir
// baisser tout seul serait pire que de dire qu'il est provisoire.

// Les coups qui ont coûté quelque chose, du plus cher au moins cher.
//
// Le critère d'entrée est **le coût pour l'Oracle**, pas la catégorie affichée :
// une malchance a sa place ici (elle explique un écart réel au score), mais un
// coup heureux n'en a pas. Mesuré sur 485 décisions
// (`scripts/analysis/replay_error_grid.py`) : 40 erreurs, 12 malchances, et
// **76 coups heureux** — les lister noierait les moments qui comptent sous
// deux fois leur nombre de coups qui n'ont rien coûté. Ils restent visibles là
// où ils ont du sens : sur le coup lui-même, et dans la couleur de la liste.
function keyMoments() {
    if (!_analysisByIdx) return [];
    const out = [];
    for (const idx of Object.keys(_analysisByIdx)) {
        const an = _analysisByIdx[idx];
        if (!an || an.forced || !(an.cost_score > 0)) continue;
        out.push({ idx: Number(idx), an, cat: moveCategory(an, Number(idx)) });
    }
    out.sort((a, b) => b.an.cost_score - a.an.cost_score || a.idx - b.idx);
    return out;
}

function renderErrors() {
    const wrap = document.getElementById('replay-errors');
    const body = document.getElementById('replay-errors-body');
    if (!wrap || !body) return;
    if (!_analysisByIdx) {
        wrap.classList.add('hidden');
        return;
    }
    wrap.classList.remove('hidden');
    const moments = keyMoments();

    // Le compteur par siège, groupé par camp. Ne comptent que les fautes —
    // malchance et coup heureux sont des explications, pas des reproches, donc
    // les additionner à un « nombre d'erreurs » serait un contresens.
    const blamed = moments.filter(m => m.cat === 'imprecision' || m.cat === 'decisive');
    const perSeat = [0, 0, 0, 0];
    for (const m of blamed) perSeat[m.an.player]++;

    // « N-S 0 » / « E-O 3 · Est 1, Ouest 2 ». Le détail par siège n'apparaît que
    // s'il y a quelque chose à détailler, et **les sièges sont nommés en
    // toutes lettres** : l'initiale d'Ouest est un « O » que la police rend
    // indistinguable d'un zéro dès qu'un chiffre la suit.
    let html = '<div class="err-counts">';
    for (const team of [0, 1]) {
        const seats = team === 0 ? [0, 2] : [1, 3];
        const total = seats.reduce((s, p) => s + perSeat[p], 0);
        const detail = total
            ? ` <span class="err-seat">${seats.map(p =>
                `${SEAT_NAMES_FR[p]} ${perSeat[p]}`).join(', ')}</span>`
            : '';
        html += `<span class="err-team ${team === 0 ? 'team-ns' : 'team-ew'}">` +
            `${team === 0 ? 'N-S' : 'E-O'} <b>${total}</b>${detail}</span>`;
    }
    html += '</div>';

    if (!moments.length) {
        html += _agentsPending
            ? '<div class="err-empty">Aucune faute pour l’instant — Dédé finit de relire.</div>'
            : '<div class="err-empty">Donne jouée sans faute.</div>';
    } else {
        html += '<div class="err-list">';
        for (const { idx, an, cat } of moments) {
            const ui = CATEGORY_UI[cat] || CATEGORY_UI.parfait;
            const cost = an.cost_score > 0 ? `−${an.cost_score}` : '';
            html += `<button class="err-row ${ui.cls}" data-idx="${idx}"` +
                (ui.hint ? ` title="${ui.hint}"` : '') + '>' +
                `<span class="err-tag">${ui.tag}</span>` +
                `<span class="err-seat-name">${SEAT_NAMES_FR[an.player]}</span>` +
                `<span class="err-card">${cardChipHtml(an.action)}</span>` +
                `<span class="err-label">${ui.label}</span>` +
                `<span class="err-cost">${cost}</span></button>`;
        }
        html += '</div>';
    }
    if (_agentsPending) {
        html += '<div class="err-pending">Classement provisoire — ' +
            'l’avis de Dédé peut encore reclasser une faute en malchance.</div>';
    }
    body.innerHTML = html;

    // Le clic amène **sur** le coup, pas avant.
    //
    // Une première version visait `idx - 1`, pour montrer la position avec la
    // carte encore en main. Ça cassait la correspondance : le panneau de stats
    // affiche l'annotation de `historyIndex`, donc on cliquait sur « Faute
    // décisive −462 » et on atterrissait sur le coup d'avant, dont l'annotation
    // n'a rien à voir — la faute restait invisible jusqu'à ce qu'on avance d'un
    // cran. Et l'information « qu'aurait-il fallu jouer » ne se perd pas en
    // arrivant ici : l'annotation porte déjà la classe de cartes de l'Oracle.
    body.querySelectorAll('.err-row').forEach(el => {
        el.addEventListener('click', () => jumpTo(parseInt(el.dataset.idx)));
    });
}

async function loadAnalysis(gameId) {
    _analysisByIdx = null;
    _analysisSummary = null;
    _bidsByIdx = null;
    _oracleBids = null;
    _curve = null;
    renderAnalysisSummary();
    renderCurve();
    try {
        const base = document.querySelector('base')?.getAttribute('href') || '/';
        const resp = await fetch(`${base}api/games/${gameId}/analysis`);
        if (!resp.ok) {
            const el = document.getElementById('replay-analysis-body');
            if (el) el.innerHTML = '<div class="an-loading">Analyse indisponible</div>';
            return;
        }
        const data = await resp.json();
        _analysisByIdx = {};
        for (const m of data.moves) _analysisByIdx[m.idx] = m;
        _bidsByIdx = {};
        for (const b of data.bids || []) _bidsByIdx[b.idx] = b;
        _oracleBids = data.oracle_bids || null;
        _curve = data.curve || null;
        _analysisSummary = data.summary;
        renderAnalysisSummary();
        renderErrors();
        renderCurve();
        // Recolor the moves list and refresh the current annotation
        if (replayBoard) {
            buildMovesList();
            replayBoard.renderHistoryEntry(replayBoard.historyIndex);
        }
    } catch {
        const el = document.getElementById('replay-analysis-body');
        if (el) el.innerHTML = '<div class="an-loading">Analyse indisponible</div>';
    }
}

// ===== Bot review (streamed over the WebSocket) =====

// Re-render only the annotation panel. Going through renderHistoryEntry would
// redraw the board, and it carries navigation state (flush pending, forward vs
// backward) that a mid-playback refresh would trample.
function refreshMoveStats() {
    if (!replayBoard || !replayBoard.active) return;
    const data = replayBoard.moveHistory[replayBoard.historyIndex];
    if (!data || !data.move || data.finished) return;
    replayRenderMoveStats(data.move, data.state);
}

// The review is ~9s of IS-DD search on first load and cached server-side
// afterwards. It streams card by card in play order, so the opening lead is
// annotated while the endgame is still being searched.
function requestAgentReview(gameId) {
    _agentsByIdx = null;
    _agentsPending = true;
    _agentsDone = 0;
    _agentsTotal = 0;
    send({ type: 'replay_agents', game_id: gameId });
}

function handleAgentReviewStart(data) {
    if (data.game_id !== _currentGameId) return;
    _agentsByIdx = {};
    _agentsPending = true;
    _agentsDone = 0;
    _agentsTotal = data.total || 0;
    refreshMoveStats();
}

function handleAgentReviewMove(data) {
    if (data.game_id !== _currentGameId || !data.move) return;
    if (_agentsByIdx === null) _agentsByIdx = {};
    _agentsByIdx[data.move.idx] = data.move;
    _agentsDone++;
    // Repaint when this is the card on screen, or while the panel is still
    // showing the progress counter for a card that has not landed yet.
    const shown = _agentsByIdx[replayBoard ? replayBoard.historyIndex : -1];
    if (!shown || shown === data.move) refreshMoveStats();
    // Chaque carte relue peut reclasser une faute en malchance : le panneau des
    // moments et la coloration de la liste suivent au fil de l'eau.
    if (data.move.isdd_cost !== undefined) {
        renderErrors();
        renderAnalysisSummary();
        buildMovesList();
    }
}

function handleAgentReviewDone(data) {
    if (data.game_id !== _currentGameId) return;
    const byIdx = {};
    for (const m of data.moves || []) byIdx[m.idx] = m;
    _agentsByIdx = byIdx;
    _agentsPending = false;
    refreshMoveStats();
    renderErrors();
    renderAnalysisSummary();
    buildMovesList();
}

function handleAgentReviewError(data) {
    if (data.game_id !== _currentGameId) return;
    _agentsPending = false;
    refreshMoveStats();
    renderErrors();
}

// ===== Load / history =====

function handleReplayLoaded(data) {
    replayTotalActions = data.total_actions || 0;
    // Un index de coup ne veut rien dire d'une donne à l'autre : sans ce vidage,
    // l'annonce n°3 d'une nouvelle donne afficherait le chiffre de la précédente.
    _quickBid.clear();
    _currentGameId = data.game_id;
    setActionIdx(0);
    setReplayGameId(data.game_id);
    setSeatLabels(data.seat_names);
    // Replay states expose all 4 hands; deal-start hands feed the annonces link
    _initialHands = (data.state && data.state.hands) || null;

    replayBoard.reset(data.state);
    if (data.moves) {
        for (const m of data.moves) replayBoard.moveHistory.push(m);
        replayBoard.finished = true;
    }
    replayBoard.historyIndex = -1;
    replayBoard._prevHistoryIndex = -1;

    document.getElementById('replay-main').classList.remove('hidden');
    replayBoard.renderHistoryEntry(-1);

    // Full-game CFN (auction + play): click-to-copy, paste into the belief page,
    // et l'état que le lien « Analyser cette carte » passe à /analyse/jeu.
    _gameCfn = data.game_cfn || null;
    updateCfnBox('replay-cfn', data.game_cfn);

    const header = replayBoard.el('stats-header');
    header.innerHTML = `<span class="stats-replay-tag">REPLAY</span> <span class="stats-agent">${data.game_id}</span>`;

    buildMovesList();

    // Coup demandé par l'URL. `buildMovesList` doit être passé avant, sinon
    // le surlignage de la liste n'a rien à surligner. Borné à la partie : un
    // index hérité d'une autre partie ne doit pas laisser la vue vide.
    if (_pendingJumpIdx !== null) {
        const target = Math.min(_pendingJumpIdx, replayBoard.moveHistory.length - 1);
        _pendingJumpIdx = null;
        if (target >= 0) jumpTo(target);
    }
    syncReplayUrl();

    _match = data.match || null;
    renderMatchBar();
    renderResumeLink(data.resume);

    loadAnalysis(data.game_id);
    requestAgentReview(data.game_id);
}

// « Donne 7 · N-S 940 – E-O 620 avant la donne · objectif 2000 ».
//
// Une donne ne se lit pas pareil selon l'endroit de la partie où elle tombe :
// à 1900-200 on n'annonce pas comme à 0-0, et bid v6 le sait — c'est le même
// score qui part dans les liens d'analyse. Le cumul montré est celui d'**avant**
// la donne, seul état qui existait au moment d'annoncer ; ce qu'elle a marqué
// vient après, en incrément.
//
// Le repère reste Nord-Sud / Est-Ouest : un lien partagé n'a pas de « nous ».
function renderMatchBar() {
    const el = document.getElementById('replay-match-bar');
    if (!el) return;
    if (!_match) {
        el.classList.add('hidden');
        el.innerHTML = '';
        return;
    }
    const [ns, ew] = _match.before;
    const parts = [];
    parts.push(`<a class="match-bar-link" href="/analyse/partie?id=${encodeURIComponent(_match.id)}"` +
        ` title="Voir la feuille de marque">Donne ${_match.deal_no ?? '?'}/${_match.deals}</a>`);
    parts.push(`<span class="match-bar-score">` +
        `<span class="team-ns">N-S ${ns}</span> – <span class="team-ew">E-O ${ew}</span>` +
        `<span class="match-bar-when"> avant la donne</span></span>`);
    if (_match.score) {
        const sign = (v) => (v > 0 ? `+${v}` : `${v}`);
        parts.push(`<span class="match-bar-delta">` +
            `donne : ${sign(_match.score[0])} / ${sign(_match.score[1])}</span>`);
    }
    parts.push(`<span class="match-bar-target">objectif ${_match.target}</span>`);
    // Un cumul amputé se dit : `integrity.backfill_scores` n'a pas encore
    // rattrapé ces donnes-là, donc les nombres ci-dessus sont trop bas.
    if (_match.unscored_before) {
        parts.push(`<span class="match-bar-warn" title="Ces donnes n'ont pas encore de score marqué en base">` +
            `${_match.unscored_before} donne${_match.unscored_before > 1 ? 's' : ''} non comptée${_match.unscored_before > 1 ? 's' : ''}</span>`);
    }
    el.innerHTML = parts.join('');
    el.classList.remove('hidden');
}

// Chemin de retour vers la partie en cours, quand la donne analysée en est une
// et qu'elle appartient au joueur connecté. C'est l'inverse du bouton
// « Analyser » de la table : sans lui, analyser une donne au milieu d'une
// partie en 2000 points était un aller sans retour.
function renderResumeLink(match) {
    const el = document.getElementById('replay-resume');
    if (!el) return;
    if (!match) {
        el.classList.add('hidden');
        return;
    }
    const us = (match.human_seat ?? 2) % 2 === 0 ? match.points_ns : match.points_ew;
    const them = (match.human_seat ?? 2) % 2 === 0 ? match.points_ew : match.points_ns;
    el.setAttribute('href', `/jouer/humain?resume=${encodeURIComponent(match.id)}`);
    el.textContent = `↩ Reprendre la partie — ${us} – ${them}, objectif ${match.target}`;
    el.classList.remove('hidden');
}

// « EST (DOUDOU) », « SUD (AVOK) » — qui tenait le siège dans cette partie-là.
// Le nom sert de repère de lecture : sans lui, quatre sièges anonymes ne disent
// pas de qui est le coup qu'on est en train d'analyser. `seat_names` est résolu
// côté serveur (`db.game_seat_names`), la clé de bot est traduite ici.
// Les libellés sont en majuscules par CSS (`.seat-label`).
function setSeatLabels(seats) {
    const ids = ['replay-label-n', 'replay-label-e', 'replay-label-s', 'replay-label-w'];
    for (let s = 0; s < 4; s++) {
        const el = document.getElementById(ids[s]);
        if (!el) continue;
        const who = seats && seats[s];
        const name = who ? (who.bot ? botLabel(who.name) : who.name) : null;
        el.textContent = name ? `${SEAT_NAMES_FR[s]} (${name})` : SEAT_NAMES_FR[s];
    }
}

function setReplayGameId(id) {
    setGameId(id);
    const el = document.getElementById('replay-game-id');
    if (id) {
        el.textContent = id;
        el.classList.remove('hidden');
        document.getElementById('replay-report-btn').classList.remove('hidden');
    } else {
        el.classList.add('hidden');
        document.getElementById('replay-report-btn').classList.add('hidden');
    }
}

// `send()` jette silencieusement quand le socket n'est pas encore OPEN. Au
// chargement à froid — lien partagé, signet, F5 — la vue se monte et demande la
// partie avant l'ouverture, et la page restait vide sans rien dire. La demande
// est donc mémorisée et rejouée à l'ouverture. C'était une course, gagnée en
// navigation SPA (socket déjà ouverte) et perdue à froid, d'où l'intermittence.
let _wantedGameId = null;

function loadReplay(gameId) {
    _wantedGameId = gameId;
    send({ type: 'replay_load', game_id: gameId });
}

function flushPendingLoad() {
    if (_wantedGameId && !_currentGameId) loadReplay(_wantedGameId);
}

async function loadGameHistory(autoLoadFirst = false) {
    try {
        const base = document.querySelector('base')?.getAttribute('href') || '/';
        let mine = false;
        try {
            const meResp = await fetch(`${base}api/me`);
            mine = meResp.ok && !!(await meResp.json()).user;
        } catch { /* anonymous */ }
        const title = document.getElementById('replay-history-title');
        if (title) title.textContent = mine ? 'Mes donnes' : 'Historique';
        const url = mine ? `${base}api/me/games?limit=50` : `${base}api/games?limit=50`;
        const resp = await fetch(url);
        if (!resp.ok) return;
        const games = await resp.json();
        renderGameHistory(games);
        if (autoLoadFirst && games.length > 0) {
            loadReplay(games[0].id);
        }
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

// Le glyphe passe par `suitHtml` : ♠ et ♣ héritaient sinon de l'or de la
// ligne, et seuls ♥ ♦ étaient colorés. Rouge / noir, comme partout ailleurs.
function contractLabel(g) {
    const c = g.contract;
    if (!c || !c.value) return '<span class="history-nocontract">passée</span>';
    const mult = c.coinche === 2 ? ' ×3' : c.coinche === 1 ? ' ×2' : '';
    const val = c.value === 250 ? 'Capot' : c.value;
    return `${val}${suitHtml(c.trump)}${mult}`;
}

function renderGameHistory(games) {
    const list = document.getElementById('replay-list');
    list.innerHTML = '';
    if (games.length === 0) {
        list.innerHTML = '<div class="history-empty">Aucune partie</div>';
        return;
    }
    for (const g of games) {
        const row = document.createElement('div');
        row.className = 'history-row';
        row.addEventListener('click', () => loadReplay(g.id));

        const contract = document.createElement('span');
        contract.className = 'history-contract';
        contract.innerHTML = contractLabel(g);

        // Score from the viewer's team perspective when known
        const seat = g.user_seat ?? g.human_seat;
        const mine = seat !== null && seat !== undefined;
        const isNS = mine ? seat % 2 === 0 : true;
        const a = isNS ? g.points_ns : g.points_ew;
        const b = isNS ? g.points_ew : g.points_ns;
        const info = document.createElement('span');
        info.className = 'history-info';
        info.textContent = `${a}-${b}`;
        info.classList.add(a > b ? 'ns-won' : 'ew-won');

        const date = document.createElement('span');
        date.className = 'history-date';
        const d = new Date(g.created_at);
        date.textContent = d.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' });

        row.title = `${g.id} — ${d.toLocaleString()}`;
        row.appendChild(contract);
        row.appendChild(info);
        row.appendChild(date);
        list.appendChild(row);
    }
}

// Public API for cross-view navigation (play -> replay)
export function loadReplayById(gameId) {
    _pendingLoadId = gameId;
    // If already mounted, load immediately
    if (replayBoard) {
        loadGameHistory(false);
        loadReplay(gameId);
        _pendingLoadId = null;
    }
}

export function mount(container) {
    container.innerHTML = TEMPLATE;

    replayBoard = new BoardRenderer({
        prefix: 'replay',
        isReplay: true,
        renderMoveStats: replayRenderMoveStats,
        renderCardAnnotations: () => updateMovesHighlight(),
        onRequestStep: () => {},
        // Rejoué : ramassage de pli 3× plus rapide, et pause écourtée sur la
        // frame « 4 cartes sur la table ».
        flushDuration: 533,
        flushHoldDelay: 333,
    });

    replayBoard.bindTransport();
    replayBoard.bindKeyboard();

    initCfnBox('replay-cfn');

    // Search
    document.getElementById('replay-search-btn').addEventListener('click', () => {
        const input = document.getElementById('replay-search-input');
        const id = input.value.trim().toLowerCase();
        if (id.length >= 1) loadReplay(id);
    });

    document.getElementById('replay-search-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const id = e.target.value.trim().toLowerCase();
            if (id.length >= 1) loadReplay(id);
        }
    });

    // Bug report
    document.getElementById('replay-report-btn').addEventListener('click', openBugReport);

    // Register WS handlers
    onOpen(flushPendingLoad);
    onMessage('replay_loaded', handleReplayLoaded);
    onMessage('agent_review_start', handleAgentReviewStart);
    onMessage('agent_review_move', handleAgentReviewMove);
    onMessage('agent_review_done', handleAgentReviewDone);
    onMessage('agent_review_error', handleAgentReviewError);
    onMessage('replay_bid_quick_start', handleQuickStart);
    onMessage('replay_bid_quick_update', handleQuickUpdate);
    onMessage('replay_bid_quick_done', handleQuickDone);
    onMessage('replay_bid_quick_error', handleQuickError);

    // Partie et coup demandés par l'URL — c'est ce que produit un Retour depuis
    // une page d'analyse, ou un lien partagé. Le coup est mis de côté : il ne
    // peut être appliqué qu'une fois la partie chargée (`handleReplayLoaded`).
    const params = new URLSearchParams(window.location.search);
    const urlGame = (params.get('game') || '').trim().toLowerCase();
    const urlIdx = parseInt(params.get('i'), 10);
    _pendingJumpIdx = Number.isFinite(urlIdx) ? urlIdx : null;

    // Load history; if pending load from another view, use that
    if (_pendingLoadId) {
        loadGameHistory(false);
        loadReplay(_pendingLoadId);
        _pendingLoadId = null;
    } else if (urlGame) {
        loadGameHistory(false);
        loadReplay(urlGame);
    } else {
        loadGameHistory(true);
    }
}

export function unmount() {
    offOpen(flushPendingLoad);
    offMessage('replay_loaded', handleReplayLoaded);
    offMessage('agent_review_start', handleAgentReviewStart);
    offMessage('agent_review_move', handleAgentReviewMove);
    offMessage('agent_review_done', handleAgentReviewDone);
    offMessage('agent_review_error', handleAgentReviewError);
    offMessage('replay_bid_quick_start', handleQuickStart);
    offMessage('replay_bid_quick_update', handleQuickUpdate);
    offMessage('replay_bid_quick_done', handleQuickDone);
    offMessage('replay_bid_quick_error', handleQuickError);
    _quickBid.clear();

    if (replayBoard) {
        replayBoard.stopAutoPlay();
        replayBoard.unbindKeyboard();
        replayBoard.active = false;
        replayBoard = null;
    }
    _analysisByIdx = null;
    _analysisSummary = null;
    _bidsByIdx = null;
    _oracleBids = null;
    _curve = null;
    _initialHands = null;
    _agentsByIdx = null;
    _agentsPending = false;
    _agentsDone = 0;
    _agentsTotal = 0;
    _currentGameId = null;
    _wantedGameId = null;
    _pendingJumpIdx = null;
    _gameCfn = null;
    _match = null;
}
