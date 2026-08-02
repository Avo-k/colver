// Compter les points — les plis défilent, on tient le compte.
//
// L'exercice de la page est celui d'une vraie table : les plis se ramassent
// dans l'ordre où ils ont été joués, et on suit le total dans sa tête pendant
// que la donne avance. D'où trois choix qui portent tout le reste :
//
//   1. Les plis viennent de donnes RÉELLEMENT JOUÉES (bots, ou les parties du
//      joueur). Des cartes tirées au hasard donneraient des plis qu'on ne voit
//      jamais à une table — on s'entraînerait sur la mauvaise distribution.
//   2. On compte PENDANT, pas après : le décompte se fait au fil des plis, et
//      la question ne tombe qu'à la fin. C'est pour ça qu'il n'y a aucun
//      compteur à l'écran, et que les tas ne montrent que leur nombre de plis.
//   3. Les plis PARTENT VERS LES QUATRE SIÈGES. La direction est
//      l'information : à une table, on sait à qui est un pli parce qu'on l'a vu
//      partir de son côté, pas parce qu'une étiquette le dit. L'axe porte le
//      camp — vertical pour Nord-Sud, horizontal pour Est-Ouest — et le sens
//      dit lequel des deux partenaires a ramassé.
//
// Un seul aller-retour serveur par séquence (`count_generate` → `count_ready`,
// la donne entière) : la correction est locale, donc instantanée. Un curieux
// peut lire la réponse dans la console — c'est un exercice, la seule personne
// qu'il tromperait est lui-même.

import { send, onMessage, offMessage, onOpen, offOpen } from '../ws.js';
import * as SFX from '../sounds.js';
import {
    PLAIN_POINTS, TRUMP_POINTS, RANKS, SEAT_NAMES_FR,
    cardChipHtml, faceDownCard, renderTrick, contractStr,
} from '../shared/cards.js';
import { suitHtml, SUIT_NAMES_FR } from '../shared/suits.js';
import { teamClass, teamName } from '../shared/seats.js';

// `teamClass` prend un SIÈGE ; ici on raisonne parfois en camps.
const TEAM_CLASS = ['team-ns', 'team-ew'];

// Le pli vole vers le siège qui l'a ramassé — indexé comme les sièges du
// moteur (0=N, 1=E, 2=S, 3=O), donc l'axe vertical est Nord-Sud et l'axe
// horizontal Est-Ouest, ce qui garde le camp lisible d'un coup d'œil.
const SEAT_DIR = ['n', 'e', 's', 'w'];
const FLY_CLASSES = SEAT_DIR.map(d => `pc-fly-${d}`);

const K_CFG = 'colver:compter:cfg';
const K_STATS = 'colver:compter:stats';
const K_SEEN = 'colver:compter:seen';

const SEEN_MAX = 20;

// Le dernier pli est tenu à l'écran avant de demander la réponse : c'est le
// seul qu'on ne verrait jamais autrement, l'écran de saisie le recouvrant dans
// la même image. Même raison que `DEAL_END_HOLD` côté jeu.
const END_HOLD_MS = 1200;
const FLY_MS = 320;

const PRESETS = {
    debutant: { nTricks: 3, speedMs: 1500, method: 'chrono', count: 'un', side: 0, rules: 'simple' },
    confirme: { nTricks: 5, speedMs: 700, method: 'chrono', count: 'un', side: -1, rules: 'simple' },
    expert:   { nTricks: 8, speedMs: 500, method: 'chrono', count: 'deux', side: 0, rules: 'realiste' },
};

const TEMPLATE = `
<div id="pc-wrap" data-phase="config">

  <div id="pc-config" class="pc-panel">
    <div class="pc-head">
      <span class="annonces-title">Compter les points</span>
      <span class="pc-subtitle">Les plis défilent, vous tenez le compte</span>
    </div>
    <p class="pc-intro">Les plis défilent dans l’ordre où ils ont été joués,
      avec l’atout du contrat. Gardez le compte dans votre tête : on vous le
      demandera à la fin.</p>

    <div class="section-title">Niveau
      <span id="pc-preset-note" class="pc-badge-perso hidden"></span></div>
    <div id="pc-presets" class="pc-seg" role="radiogroup" aria-label="Niveau">
      <button type="button" class="pc-seg-btn" role="radio" data-preset="debutant" aria-checked="true">
        Débutant<small>3 plis · 1,5 s · un camp</small></button>
      <button type="button" class="pc-seg-btn" role="radio" data-preset="confirme" aria-checked="false">
        Confirmé<small>5 plis · 0,7 s · un camp</small></button>
      <button type="button" class="pc-seg-btn" role="radio" data-preset="expert" aria-checked="false">
        Expert<small>8 plis · 0,5 s · deux camps · partie entière</small></button>
    </div>

    <button type="button" id="pc-fine-toggle" class="pc-toggle" aria-expanded="false">
      Réglages fins <span class="pc-caret">▾</span></button>
    <div id="pc-fine" class="hidden">
      <div class="pc-row">
        <span class="pc-row-label">Défilement</span>
        <div class="pc-seg" id="pc-method" role="radiogroup" aria-label="Défilement">
          <button type="button" class="pc-seg-btn" role="radio" data-method="carte" aria-checked="false">Carte par carte<small>vous avancez</small></button>
          <button type="button" class="pc-seg-btn" role="radio" data-method="pli" aria-checked="false">Pli par pli<small>vous avancez</small></button>
          <button type="button" class="pc-seg-btn" role="radio" data-method="chrono" aria-checked="true">Chronométré<small>ça défile tout seul</small></button>
        </div>
      </div>
      <div class="pc-row" id="pc-row-speed">
        <span class="pc-row-label">Temps par carte</span>
        <input type="range" id="pc-speed" min="300" max="2000" step="100" value="1500">
        <output id="pc-speed-out">1,5 s</output>
      </div>
      <div class="pc-row" id="pc-row-ntricks">
        <span class="pc-row-label">Nombre de plis</span>
        <input type="range" id="pc-ntricks" min="1" max="8" step="1" value="3">
        <output id="pc-ntricks-out">3</output>
      </div>
      <div class="pc-row">
        <span class="pc-row-label">Comptage</span>
        <div class="pc-seg" id="pc-count" role="radiogroup" aria-label="Comptage">
          <button type="button" class="pc-seg-btn" role="radio" data-count="un" aria-checked="true">Un camp<small>l’autre tas passe, sans compter</small></button>
          <button type="button" class="pc-seg-btn" role="radio" data-count="deux" aria-checked="false">Deux camps<small>deux tas, deux totaux</small></button>
        </div>
      </div>
      <div class="pc-row" id="pc-row-side">
        <span class="pc-row-label">Camp à compter</span>
        <div class="pc-seg" id="pc-side" role="radiogroup" aria-label="Camp à compter">
          <button type="button" class="pc-seg-btn" role="radio" data-side="0" aria-checked="true">Nord-Sud</button>
          <button type="button" class="pc-seg-btn" role="radio" data-side="1" aria-checked="false">Est-Ouest</button>
          <button type="button" class="pc-seg-btn" role="radio" data-side="-1" aria-checked="false">Au hasard</button>
        </div>
      </div>
      <div class="pc-row">
        <span class="pc-row-label">Règles</span>
        <div class="pc-seg" id="pc-rules" role="radiogroup" aria-label="Règles">
          <button type="button" class="pc-seg-btn" role="radio" data-rules="simple" aria-checked="true">Simple<small>cartes seules</small></button>
          <button type="button" class="pc-seg-btn" role="radio" data-rules="realiste" aria-checked="false">Partie entière<small>dix de der + belote</small></button>
        </div>
      </div>
      <div class="pc-row">
        <span class="pc-row-label">Donnes</span>
        <div class="pc-seg" id="pc-source" role="radiogroup" aria-label="Donnes">
          <button type="button" class="pc-seg-btn" role="radio" data-source="auto" aria-checked="true">Nouvelles<small>jouées à l’instant</small></button>
          <button type="button" class="pc-seg-btn" role="radio" data-source="mes" aria-checked="false">Mes parties<small>vos donnes enregistrées</small></button>
        </div>
      </div>
      <div id="pc-fine-note" class="pc-note"></div>
    </div>

    <div class="pc-start-row">
      <button id="pc-start">Commencer</button>
      <span id="pc-loading" class="pc-note hidden">Génération…</span>
    </div>
    <div id="pc-error" class="pc-error hidden" role="alert"></div>
    <div id="pc-stats" class="pc-stats"></div>
  </div>

  <div id="pc-stage">
    <div id="pc-bar">
      <span id="pc-trump-chip" class="pc-trump-chip"></span>
      <span id="pc-contract" class="pc-contract"></span>
      <span id="pc-task" class="pc-task"></span>
      <span id="pc-progress" class="pc-progress"></span>
      <button id="pc-quit" type="button" class="pc-ghost-btn">Abandonner</button>
    </div>

    <div id="pc-board">
      <div class="pc-pile" id="pc-pile-far">
        <div class="pc-pile-label">Est-Ouest</div>
        <div class="pc-pile-stack"></div>
        <div class="pc-pile-count">0 pli</div>
      </div>

      <div id="pc-trick-area" role="button" tabindex="0" aria-label="Avancer le défilement">
        <div class="trick-card" id="pc-trick-n"></div>
        <div class="trick-card" id="pc-trick-w"></div>
        <div class="trick-card" id="pc-trick-e"></div>
        <div class="trick-card" id="pc-trick-s"></div>
        <div id="pc-veil" class="hidden"><span>En pause</span></div>
        <div id="pc-announce" class="hidden"></div>
      </div>

      <div class="pc-pile" id="pc-pile-near">
        <div class="pc-pile-label">Nord-Sud</div>
        <div class="pc-pile-stack"></div>
        <div class="pc-pile-count">0 pli</div>
      </div>
    </div>

    <div id="pc-hint" aria-live="polite"></div>

    <div id="pc-controls">
      <button id="pc-prev" type="button" class="pc-ghost-btn" aria-label="Reculer">◀</button>
      <button id="pc-pause" type="button" class="pc-ghost-btn hidden">Pause</button>
      <button id="pc-next" type="button">Suivant ▶</button>
    </div>
  </div>

  <div id="pc-answer" class="pc-panel">
    <div class="section-title" id="pc-answer-title"></div>
    <form id="pc-answer-form" novalidate>
      <label class="pc-field" id="pc-field-1">
        <span class="pc-field-label">Est-Ouest</span>
        <input id="pc-in-1" type="text" inputmode="numeric" pattern="[0-9]*"
               maxlength="3" autocomplete="off" enterkeyhint="next"></label>
      <label class="pc-field" id="pc-field-0">
        <span class="pc-field-label">Nord-Sud</span>
        <input id="pc-in-0" type="text" inputmode="numeric" pattern="[0-9]*"
               maxlength="3" autocomplete="off" enterkeyhint="done"></label>
      <button id="pc-validate" type="submit">Annoncer</button>
    </form>
    <div id="pc-answer-check" class="pc-note"></div>
  </div>

  <div id="pc-review">
    <div id="pc-verdicts"></div>
    <div id="pc-diag" class="pc-diag"></div>
    <div class="pc-panel">
      <div class="section-title">Décompte pli par pli</div>
      <div id="pc-table-swipe" class="pc-note"></div>
      <div class="pc-table-scroll">
        <table id="pc-table"><thead></thead><tbody></tbody></table>
      </div>
      <div id="pc-table-foot" class="pc-foot"></div>
    </div>
    <div id="pc-actions">
      <button id="pc-again" type="button">Nouvelle séquence</button>
      <button id="pc-replay" type="button" class="pc-ghost-btn">Revoir au ralenti</button>
      <button id="pc-back" type="button" class="pc-ghost-btn">Réglages</button>
    </div>
  </div>
</div>`;

// ===== État de module =======================================================
// Le module est un singleton (import mémoïsé) : tout est remis à zéro dans
// mount(), jamais seulement à la déclaration.

let phase = 'config';
let cfg = null;
let deal = null;       // payload `count_ready`
let win = null;        // les N plis consécutifs montrés (cf. pickWindow)
let side = 0;          // camp à compter en mode « un camp », tiré si side = -1
let pcIdx = 0;         // cartes révélées, 0 .. 4·N
let paused = false;
let pcTimer = null;    // le SEUL timer de la page
let flyTimer = null;
// `pcTimer` porte deux échéances de nature différente : le tick du défilement,
// et le maintien du dernier pli. La pause les tue toutes les deux — il faut
// donc savoir laquelle ré-armer, sinon la reprise relance un tick qui ne peut
// plus avancer (on est à `max`) et la séquence ne se termine JAMAIS.
let endHold = false;
let pending = null;    // requête à rejouer si le socket n'était pas ouvert
let reqId = 0;
// Donnes écartées d'affilée faute de fenêtre montrable (cf. pickWindow).
let dealTries = 0;
const MAX_DEAL_TRIES = 6;
let given = [null, null];
// Tout est ramassé : la croix se vide et les tas sont complets. C'est l'image
// de la fin de donne, celle qu'on a sous les yeux au moment de compter.
let allGathered = false;
// « Revoir au ralenti » force le pas-à-pas ; on rend son mode au joueur ensuite,
// sinon la séquence suivante repartirait au pas alors qu'il a choisi le chrono.
let methodBeforeReplay = null;
// Une relecture n'est pas un essai : la réponse vient d'être affichée juste
// au-dessus. Elle ne repasse donc pas par la saisie et ne compte pas.
let inReplay = false;

const $ = (id) => document.getElementById(id);

// ===== Réglages persistés ===================================================

function loadCfg() {
    const base = { ...PRESETS.debutant, preset: 'debutant', source: 'auto' };
    try {
        const o = JSON.parse(localStorage.getItem(K_CFG) || 'null');
        if (!o || typeof o !== 'object') return base;
        const cfg = { ...base, ...o };
        // Un préréglage est défini par la table, pas par ce qui traîne en
        // localStorage : sinon changer « Débutant » ne changerait rien pour
        // ceux qui y ont déjà joué. Seul `perso` garde ses valeurs.
        if (PRESETS[cfg.preset]) return { ...cfg, ...PRESETS[cfg.preset] };
        return cfg;
    } catch { return base; }
}
function saveCfg() { try { localStorage.setItem(K_CFG, JSON.stringify(cfg)); } catch { /* quota */ } }

function loadStats() {
    try {
        const o = JSON.parse(localStorage.getItem(K_STATS) || '{}');
        return (o && typeof o === 'object') ? o : {};
    } catch { return {}; }
}
function saveStats(s) { try { localStorage.setItem(K_STATS, JSON.stringify(s)); } catch { /* quota */ } }

// Les donnes déjà servies depuis la base. Sans cette mémoire, « Mes parties »
// resservirait la même donne au bout de quelques tirages : un joueur en a
// quelques dizaines, pas quelques milliers.
function loadSeen() {
    try {
        const a = JSON.parse(localStorage.getItem(K_SEEN) || '[]');
        return Array.isArray(a) ? a.filter(x => typeof x === 'string').slice(0, SEEN_MAX) : [];
    } catch { return []; }
}
function pushSeen(id) {
    if (!id) return;
    try {
        const a = [id, ...loadSeen().filter(x => x !== id)].slice(0, SEEN_MAX);
        localStorage.setItem(K_SEEN, JSON.stringify(a));
    } catch { /* quota */ }
}

// Une relecture force `cfg.method = 'carte'` : sans ce repli, le bandeau de
// statistiques changerait de clé en cours de route et afficherait le record
// d'un mode que le joueur n'a pas choisi.
function statsKey() { return `${cfg.preset}|${methodBeforeReplay ?? cfg.method}`; }

// ===== Règles du comptage ===================================================

const cardPoints = (c, trump) => ((c >> 3) === trump ? TRUMP_POINTS : PLAIN_POINTS)[c & 7];

/** Nombre de plis réellement montrés. « Partie entière » impose la donne entière. */
function windowSize() { return cfg.rules === 'realiste' ? 8 : cfg.nTricks; }

/**
 * Les N plis consécutifs à montrer — pas forcément les N premiers.
 *
 * Une fenêtre où le camp à compter n'a presque rien ramassé donne une réponse
 * proche de zéro : l'exercice ne s'est pas posé. On exige donc que **la moitié
 * au moins** des plis montrés lui reviennent (2 sur 3, 3 sur 5), et qu'ils
 * vaillent plus de zéro point — deux plis de 7-8-9 font une réponse aussi vide.
 *
 * La donne entière arrive toujours du serveur, donc la fenêtre peut glisser :
 * commencer au pli n°4 ne change rien à l'exercice (la correction affiche les
 * vrais numéros), et c'est bien moins cher que de redemander une donne. On tire
 * au hasard parmi les fenêtres qui conviennent, sinon la page servirait
 * toujours le même endroit de la donne.
 *
 * Rend `ok: false` avec la **moins mauvaise** fenêtre quand aucune ne convient
 * (un camp qui ne ramasse qu'un pli de la donne n'en offre aucune) : l'appelant
 * redemande une donne, mais si ça échoue encore il vaut mieux montrer ça que
 * les N premiers plis au hasard.
 *
 * À 8 plis il n'y a qu'une fenêtre possible : on ne filtre rien, sinon on
 * rejetterait des donnes entières parfaitement légitimes (un capot subi se
 * compte, il vaut 0).
 */
function pickWindow(tricks, n, teams) {
    if (n >= tricks.length) return { win: tricks.slice(), ok: true };
    // Deux camps à compter : exiger la moitié des plis pour chacun serait
    // contradictoire (3 plis, 2 + 2). Chacun doit juste avoir de quoi compter.
    const need = teams.length > 1 ? 1 : Math.ceil(n / 2);
    const all = [];
    for (let s = 0; s + n <= tricks.length; s++) {
        const w = tricks.slice(s, s + n);
        const share = teams.map((k) => {
            const mine = w.filter((t) => t.winner % 2 === k);
            return [mine.length, mine.reduce((a, t) => a + t.points, 0)];
        });
        all.push({
            w,
            ok: share.every(([c, p]) => c >= need && p > 0),
            // Une fenêtre ne vaut que par son camp le moins servi.
            rank: Math.min(...share.map(([c]) => c)) * 1000
                + Math.min(...share.map(([, p]) => p)),
        });
    }
    const good = all.filter((x) => x.ok);
    if (good.length) {
        return { win: good[Math.floor(Math.random() * good.length)].w, ok: true };
    }
    return { win: all.reduce((a, b) => (b.rank > a.rank ? b : a)).w, ok: false };
}

/**
 * Ce que le joueur doit annoncer, camp par camp.
 *
 * En « simple » ce sont les points cartes des plis montrés, rien d'autre. En
 * « partie entière » s'y ajoutent le dix de der — acquis au camp qui ramasse
 * la 8e levée — et la belote, qui n'est PAS un point carte : les deux tas font
 * 152 avec ou sans elle, les 20 se posent par-dessus.
 */
function expectedTotals() {
    const cards = [0, 0];
    for (const t of win) cards[t.winner % 2] += t.points;
    if (cfg.rules !== 'realiste') return { cards, der: [0, 0], belote: [0, 0], total: cards };
    const der = [0, 0];
    der[deal.der.team] = deal.der.value;
    const belote = deal.belote.slice();
    return { cards, der, belote, total: [0, 1].map(k => cards[k] + der[k] + belote[k]) };
}

// ===== Rendu du défilement ==================================================
//
// `pcIdx` = nombre de cartes révélées, et `renderAt` rejoue TOUT depuis zéro.
// Trente-deux cartes à replacer ne coûtent rien, et reculer n'a alors aucun
// état incrémental à défaire : c'est ce qui rend la flèche gauche exacte.
// C'est aussi pourquoi on n'utilise ni `detectTrickCompletion` (état global de
// module, ne détecte qu'un sens) ni `animateTrickFlush` (qui vise la main du
// gagnant — il n'y a pas de mains ici).

/**
 * L'état de la table à `pcIdx` cartes révélées.
 *
 * `pcIdx = 4·ti + k` : le pli `ti` a `k` cartes posées. À `k = 0` (et pcIdx > 0)
 * c'est le pli précédent qui est **complet sur la table, pas encore ramassé** —
 * c'est le pas suivant qui l'envoie dans son tas. Un pli n'entre donc dans un
 * tas que lorsque la première carte du suivant tombe, exactement comme à une
 * table où l'on ramasse en entamant.
 */
function tableAt(i) {
    const ti = Math.floor(i / 4);
    const k = i % 4;
    const onTable = (k === 0 && i > 0) ? ti - 1 : ti;
    return { onTable, shown: (k === 0 && i > 0) ? 4 : k, gathered: onTable };
}

function renderAt(withSound = false) {
    $('pc-trick-area').classList.remove(...FLY_CLASSES);
    const { onTable, shown, gathered } = tableAt(pcIdx);

    const trick = [-1, -1, -1, -1];
    let lead = null;
    if (!allGathered && onTable < win.length && shown > 0) {
        const t = win[onTable];
        lead = t.lead;
        for (let j = 0; j < shown; j++) {
            const seat = (t.lead + j) % 4;
            trick[seat] = t.cards[seat];
        }
    }
    renderTrick('pc-trick', trick, lead);

    // Les tas : un dos de carte par pli ramassé. Jamais de points affichés —
    // sinon il n'y a plus d'exercice.
    const upTo = allGathered ? win.length : gathered;
    const taken = [[], []];
    for (let i = 0; i < upTo; i++) taken[win[i].winner % 2].push(win[i]);
    renderPile('pc-pile-near', 0, taken[0]);
    renderPile('pc-pile-far', 1, taken[1]);

    const total = win.length;
    $('pc-progress').textContent = `Pli ${Math.min(total, onTable + 1)}/${total}`;
    renderAnnounce(onTable, shown, withSound);
}

function renderPile(id, team, tricks) {
    const el = $(id);
    const stack = el.querySelector('.pc-pile-stack');
    stack.innerHTML = '';
    for (let i = 0; i < tricks.length; i++) {
        const c = faceDownCard();
        c.style.marginTop = i === 0 ? '0' : 'calc(var(--card-h) * -0.82)';
        c.style.marginLeft = `${i * 5}px`;
        stack.appendChild(c);
    }
    const n = tricks.length;
    el.querySelector('.pc-pile-count').textContent = n > 1 ? `${n} plis` : `${n} pli`;
    el.classList.toggle('pc-pile--ghost', cfg.count === 'un' && team !== side);
}

/**
 * « Belote ! » puis « Rebelote ! », au moment où la carte tombe.
 *
 * En mode simple on ne les montre pas du tout : afficher un bonus qui ne
 * compte pas serait un piège, pas un exercice.
 */
function renderAnnounce(ti, shown, withSound) {
    const box = $('pc-announce');
    box.className = 'hidden';
    if (allGathered || cfg.rules !== 'realiste' || ti < 0 || ti >= win.length || shown < 1) return;
    const t = win[ti];
    const seatPlayed = (t.lead + shown - 1) % 4;
    const a = (t.announces || []).find(x => x.seat === seatPlayed
        && t.cards[seatPlayed] === x.card);
    if (!a) return;
    box.textContent = a.event === 'belote' ? 'Belote !' : 'Rebelote !';
    box.className = teamClass(a.seat);
    if (withSound) SFX.belote();
}

/**
 * Le pli complet part vers le tas de son camp, PUIS la table se repeint.
 *
 * L'ordre compte : repeindre d'abord animerait la première carte du pli
 * suivant. Et la fin du vol est un `setTimeout`, jamais un `transitionend` —
 * `prefers-reduced-motion` neutralise les transitions (tokens.css), l'événement
 * ne partirait alors jamais et la table resterait figée.
 */
function flyOut(ti, done) {
    const t = win[ti];
    if (!t) { done(); return; }
    // Le pli part vers LE SIÈGE qui l'a ramassé, aux quatre points de la croix
    // — comme à une table, où l'on voit le vainqueur tirer les cartes à lui.
    //
    // C'est l'AXE qui porte le camp, et lui seul : vertical pour Nord-Sud,
    // horizontal pour Est-Ouest. Le sens, dans l'axe, désigne lequel des deux
    // partenaires a ramassé — une information de plus, jamais une de moins.
    // Rien ne l'écrit, et c'est le but : à une table on sait à qui est un pli
    // parce qu'on l'a vu partir de son côté.
    const dir = SEAT_DIR[t.winner];
    $('pc-trick-area').classList.add(`pc-fly-${dir}`);
    SFX.trickWon();
    if (flyTimer) clearTimeout(flyTimer);
    flyTimer = setTimeout(() => { flyTimer = null; done(); }, FLY_MS);
}

/**
 * Un pas de défilement, `step` en cartes (négatif = retour en arrière).
 *
 * Le pli `m` part vers son tas quand `pcIdx` franchit `4·(m+1)` : c'est-à-dire
 * quand la première carte du pli suivant tombe, jamais quand la quatrième du
 * sien se pose. Un pli complet reste donc visible un pas entier.
 */
function advance(step) {
    // Les commandes de défilement survivent à l'écran sur la phase de saisie ;
    // sans ce garde, un ◀ suivi d'un ▶ y relance `toAnswer`, qui vide les
    // champs — le total que le joueur venait de taper disparaissait.
    if (phase !== 'run') return;
    const max = win.length * 4;
    const from = pcIdx;
    const next = Math.max(0, Math.min(max, pcIdx + step));
    if (next === from) return;
    pcIdx = next;
    if (next <= from) {
        // Reculer annule le maintien de fin : sinon le minuteur déjà armé
        // arracherait le joueur à sa relecture pour l'emmener à la saisie.
        stopTimer();
        endHold = false;
        renderAt(false);
        return;
    }

    // Le pli `m` est ramassé quand on franchit `4·(m+1)`. Si un pas en franchit
    // plusieurs, on n'anime que le dernier : l'animation est décorative,
    // `renderAt` porte l'état.
    let flown = -1;
    for (let b = 4; b <= max; b += 4) if (from <= b && b < next) flown = b / 4 - 1;
    if (flown >= 0) flyOut(flown, () => { renderAt(true); afterStep(max); });
    else { SFX.cardPlay(); renderAt(true); afterStep(max); }
}

function afterStep(max) {
    if (pcIdx >= max) finishRun();
}

function finishRun() {
    stopTimer();
    // La dernière levée est tenue à l'écran avant que la saisie ne la recouvre :
    // c'est la seule qu'on ne verrait jamais autrement.
    endHold = true;
    pcTimer = setTimeout(closeRun, END_HOLD_MS);
}

/** La fin du maintien : tout se ramasse, puis on demande le total.
 *
 *  Fonction nommée, et pas une lambda dans `finishRun` : la pause doit pouvoir
 *  la ré-armer. Une pause pendant le maintien — le battement où l'on additionne,
 *  donc le moment le plus naturel pour demander une seconde — tuait sinon la
 *  séquence pour de bon.
 */
function closeRun() {
    pcTimer = null;
    endHold = false;
    // Puis tout se ramasse, comme en fin de donne — les tas montrent alors
    // le compte exact de plis sur lequel on interroge.
    flyOut(win.length - 1, () => {
        allGathered = true;
        renderAt(false);
        // Une relecture ne redemande pas le total : elle rend la correction,
        // qui est ce qu'on était en train de lire en la lançant.
        if (inReplay) toReview(); else toAnswer();
    });
}

// ===== Chronomètre ==========================================================

function stopTimer() { if (pcTimer) { clearTimeout(pcTimer); pcTimer = null; } }

/** Coupe tout ce qui est armé — tick, maintien de fin, vol. À appeler à chaque
 *  changement de phase : un vol qui se termine après coup rappelle `renderAt`
 *  sur un plateau masqué, et son callback peut enchaîner sur `toAnswer`. */
function stopAllTimers() {
    stopTimer();
    endHold = false;
    if (flyTimer) { clearTimeout(flyTimer); flyTimer = null; }
}

/** Un pli complet tient un temps de plus : c'est là qu'on additionne, et le
 *  vol a besoin de ce battement pour se jouer avant la carte suivante. */
function nextDelay() {
    return (pcIdx > 0 && pcIdx % 4 === 0) ? cfg.speedMs + FLY_MS : cfg.speedMs;
}

function tick() {
    pcTimer = null;
    if (phase !== 'run' || paused) return;
    advance(1);
    if (pcIdx < win.length * 4) pcTimer = setTimeout(tick, nextDelay());
}

function startTimer() {
    stopTimer();
    if (cfg.method !== 'chrono') return;
    pcTimer = setTimeout(tick, cfg.speedMs);
}

function setPaused(on) {
    if (cfg.method !== 'chrono' || phase !== 'run') return;
    paused = on;
    // La pause MASQUE la table : sinon s'arrêter devant un pli complet est un
    // retour en arrière déguisé, et l'exercice se contourne tout seul.
    $('pc-veil').classList.toggle('hidden', !on);
    // Le voile assombrit, il ne cache pas : à 85 % de noir un As de cœur reste
    // parfaitement lisible. Ce sont les cartes elles-mêmes qu'on retire.
    $('pc-trick-area').classList.toggle('pc-paused', on);
    $('pc-pause').textContent = on ? 'Reprendre' : 'Pause';
    if (on) stopTimer();
    // On ré-arme l'échéance qu'on a interrompue, pas systématiquement un tick :
    // pendant le maintien de fin il n'y a plus rien à faire avancer.
    else if (endHold) pcTimer = setTimeout(closeRun, END_HOLD_MS);
    else pcTimer = setTimeout(tick, nextDelay());
}

function hint(msg) {
    const el = $('pc-hint');
    el.textContent = msg || '';
    el.classList.toggle('pc-hint--on', !!msg);
}

// ===== Phases ===============================================================

function setPhase(p) {
    phase = p;
    $('pc-wrap').dataset.phase = p;
}

function requestDeal() {
    // `send()` jette silencieusement quand le socket n'est pas OPEN : on garde
    // la requête et on la rejoue sur `onOpen`. Invisible en navigation SPA,
    // systématique sur un F5 ou un lien froid.
    pending = {
        type: 'count_generate',
        req_id: ++reqId,
        source: cfg.source,
        seen: loadSeen(),
    };
    send(pending);
    $('pc-start').disabled = true;
    $('pc-loading').classList.remove('hidden');
    $('pc-error').classList.add('hidden');
}

function flushPending() { if (pending) send(pending); }

function onReady(data) {
    if (data.req_id !== reqId) return;   // réponse d'une requête abandonnée
    pending = null;
    $('pc-start').disabled = false;
    $('pc-loading').classList.add('hidden');

    deal = data;
    if (data.source === 'base') pushSeen(data.game_id);
    side = cfg.count === 'deux' ? 0
        : (cfg.side === -1 ? (Math.random() < 0.5 ? 0 : 1) : cfg.side);

    // Aucune fenêtre montrable dans cette donne : on en redemande une. Borné,
    // parce que « Mes parties » puise dans un vivier mince et qu'une page qui
    // redemande sans fin ne dit rien au joueur.
    const pick = pickWindow(deal.tricks, windowSize(),
        cfg.count === 'deux' ? [0, 1] : [side]);
    if (!pick.ok && dealTries < MAX_DEAL_TRIES) {
        dealTries++;
        requestDeal();
        return;
    }
    win = pick.win;
    dealTries = 0;

    pcIdx = 0;
    paused = false;
    endHold = false;
    allGathered = false;
    inReplay = false;
    given = [null, null];

    $('pc-trump-chip').innerHTML = `Atout ${suitHtml(deal.trump)}`;
    $('pc-contract').innerHTML = contractStr(deal.contract);
    $('pc-task').textContent = cfg.count === 'deux'
        ? 'Comptez les deux camps'
        : `Comptez ${teamName(side)}`;
    $('pc-veil').classList.add('hidden');
    $('pc-trick-area').classList.remove('pc-paused');
    $('pc-announce').classList.add('hidden');
    $('pc-pause').classList.toggle('hidden', cfg.method !== 'chrono');
    $('pc-pause').textContent = 'Pause';
    $('pc-prev').disabled = cfg.method === 'chrono';
    $('pc-next').classList.toggle('hidden', cfg.method === 'chrono');
    hint(data.source_degraded
        ? 'Aucune de vos donnes n’était disponible : celle-ci vient d’être jouée.'
        : '');

    setPhase('run');
    renderAt();
    startTimer();
}

/**
 * Une génération qui échoue doit se voir — depuis n'importe quelle phase.
 *
 * Le garde portait sur `phase === 'config'`, or « Nouvelle séquence » part de
 * la correction : le message était jeté et le bouton restait sans effet, sans
 * fin. `#pc-hint` vit dans `#pc-stage` (masqué en config) et `#pc-fine-note`
 * dans le repli des réglages : ni l'un ni l'autre n'est un endroit fiable.
 */
function onError(data) {
    if (!pending) return;              // une erreur qui ne nous concerne pas
    pending = null;
    $('pc-start').disabled = false;
    $('pc-loading').classList.add('hidden');
    stopAllTimers();
    setPhase('config');
    const el = $('pc-error');
    el.textContent = data.msg || 'La génération a échoué. Réessayez.';
    el.classList.remove('hidden');
}

function toAnswer() {
    setPhase('answer');
    const two = cfg.count === 'deux';
    $('pc-field-0').classList.toggle('hidden', !two && side !== 0);
    $('pc-field-1').classList.toggle('hidden', !two && side !== 1);
    $('pc-answer-title').textContent = two
        ? 'Combien pour chaque camp ?'
        : `Combien pour ${teamName(side)} ?`;
    $('pc-in-0').value = '';
    $('pc-in-1').value = '';
    $('pc-answer-check').textContent = '';
    // Au doigt on ne focalise pas : iOS n'ouvre le clavier que sur un geste, et
    // un champ focalisé sans clavier fait défiler la page pour rien. Au clavier,
    // en revanche, il faut le focus — le chronométré amène ici tout seul, et
    // c'est le mode des trois préréglages.
    if (!window.matchMedia('(pointer: coarse)').matches) {
        (two ? $('pc-in-1') : $(`pc-in-${side}`)).focus();
    }
    $('pc-answer').scrollIntoView({ block: 'center', behavior: 'smooth' });
}

// ===== Correction ===========================================================

function submitAnswer(e) {
    if (e) e.preventDefault();
    if (phase !== 'answer') return;
    const exp = expectedTotals();
    const teams = cfg.count === 'deux' ? [0, 1] : [side];
    for (const k of teams) {
        const raw = $(`pc-in-${k}`).value.trim();
        if (!/^\d{1,3}$/.test(raw)) {
            $('pc-answer-check').textContent = 'Il manque un nombre.';
            $(`pc-in-${k}`).focus();
            return;
        }
        given[k] = parseInt(raw, 10);
    }
    // Une relecture ne compte pas : la réponse était affichée juste au-dessus.
    // Sans ce garde, série et record se gonflent d'un essai gagné d'avance.
    if (!inReplay) recordStats(teams, exp);
    renderReview(teams, exp);
    setPhase('review');
    $('pc-review').scrollIntoView({ block: 'start', behavior: 'smooth' });
}

function recordStats(teams, exp) {
    const stats = loadStats();
    const key = statsKey();
    const s = stats[key] || { plays: 0, exact: 0, sumAbsDelta: 0, streak: 0, best: 0 };
    const delta = teams.reduce((acc, k) => acc + Math.abs(given[k] - exp.total[k]), 0);
    s.plays += 1;
    s.sumAbsDelta += delta;
    if (delta === 0) {
        s.exact += 1;
        s.streak += 1;
        s.best = Math.max(s.best, s.streak);
    } else {
        s.streak = 0;
    }
    stats[key] = s;
    saveStats(stats);
    renderStats();
}

function renderStats() {
    const s = loadStats()[statsKey()];
    const el = $('pc-stats');
    if (!s || !s.plays) { el.textContent = ''; return; }
    const pct = Math.round(100 * s.exact / s.plays);
    const avg = (s.sumAbsDelta / s.plays).toFixed(1).replace('.', ',');
    el.innerHTML = `<span>Série <b>${s.streak}</b></span>`
        + `<span>record <b>${s.best}</b></span>`
        + `<span><b>${pct} %</b> justes</span>`
        + `<span>écart moyen <b>${avg}</b> pts</span>`
        + `<span class="pc-stats-n">${s.plays} essai${s.plays > 1 ? 's' : ''}</span>`;
}

const VERDICT_CLASS = (d) => d === 0 ? 'prob-badge-correct'
    : (Math.abs(d) <= 4 ? 'pc-badge-close' : 'prob-badge-wrong');

function renderReview(teams, exp) {
    const box = $('pc-verdicts');
    box.innerHTML = '';
    for (const k of teams) {
        const d = given[k] - exp.total[k];
        const el = document.createElement('div');
        el.className = `prob-badge ${VERDICT_CLASS(d)}`;
        const who = teams.length > 1 ? `${teamName(k)} : ` : '';
        el.textContent = d === 0
            ? `${who}juste — ${exp.total[k]} points`
            : `${who}${given[k]} annoncés, ${exp.total[k]} en réalité (${d > 0 ? '+' : ''}${d})`;
        box.appendChild(el);
    }
    $('pc-diag').innerHTML = diagnose(teams, exp).map(m => `<p>${m}</p>`).join('');
    renderTable(exp);
    SFX[teams.every(k => given[k] === exp.total[k]) ? 'victory' : 'defeat']();
}

/**
 * Pourquoi c'est faux, pas seulement que ça l'est.
 *
 * L'ordre compte : ±20 est ambigu (belote, ou Valet d'atout compté 2), ±10
 * aussi (dix de der, ou un 10). On teste donc du plus spécifique au plus
 * général, et on s'arrête au premier qui explique l'écart.
 */
function diagnose(teams, exp) {
    const out = [];
    // Les deux totaux simplement intervertis : un seul message, pas un par
    // camp — c'est une seule erreur, pas deux.
    if (teams.length === 2 && given[0] === exp.total[1] && given[1] === exp.total[0]
        && exp.total[0] !== exp.total[1]) {
        return ['Les deux totaux sont intervertis : chaque camp a reçu celui de l’autre.'];
    }
    // Les deux totaux qui somment juste ne désignent PAS à eux seuls un pli
    // déplacé : donner le dix de der ou la belote au mauvais camp somme juste
    // aussi. C'est pour ça que ce test ne sert plus qu'à nuancer le rang 5,
    // après que le dix de der et la belote ont eu leur tour.
    const sumsMatch = teams.length === 2
        && given[0] + given[1] === exp.total[0] + exp.total[1];
    for (const k of teams) {
        const d = given[k] - exp.total[k];
        if (d === 0) continue;
        const trump = deal.trump;

        if (given[k] === exp.total[1 - k]) {
            out.push(`${teamName(k)} : c’est le total de l’autre camp — vérifiez quel tas vous comptiez.`);
            continue;
        }
        if (cfg.rules === 'realiste' && k === deal.der.team && d === -deal.der.value) {
            out.push(`Dix de der oublié : +${deal.der.value} au camp qui ramasse la dernière levée (${teamName(k)}).`);
            continue;
        }
        if (cfg.rules === 'realiste' && k !== deal.der.team && d === deal.der.value) {
            out.push(`La 8ᵉ levée est allée à ${teamName(deal.der.team)} : le dix de der part avec.`);
            continue;
        }
        if (cfg.rules === 'realiste' && exp.belote[k] === 20 && d === -20) {
            out.push(`Belote oubliée : +20 au camp du joueur qui a le Roi <b>et</b> la Dame d’atout.`);
            continue;
        }
        if (cfg.rules === 'realiste' && exp.belote[1 - k] === 20 && d === 20) {
            out.push(`La belote a été annoncée par ${teamName(1 - k)} : les 20 points sont à eux.`);
            continue;
        }
        // Un pli entier dans le mauvais tas. La direction fait partie du test :
        // sans elle, un pli de la bonne valeur mais du bon côté ferait accuser
        // le joueur d'une erreur qu'il n'a pas commise.
        const missed = win.filter(t => t.points === Math.abs(d)
            && ((d < 0) === (t.winner % 2 === k)));
        if (missed.length) {
            const t = missed[0];
            // Quand les deux totaux somment juste, rien n'a été oublié : c'est un
            // déplacement, et ça se dit une fois pour les deux camps.
            if (sumsMatch) {
                out.push(`Vos deux totaux somment juste (${given[0] + given[1]}) : aucune carte n’a été oubliée. `
                    + `C’est le pli n°${t.no} (${t.points} pts, ramassé par ${SEAT_NAMES_FR[t.winner]}) qui est allé dans le mauvais tas.`);
                break;
            }
            out.push(`Le pli n°${t.no} vaut exactement ${t.points} points, ramassé par ${SEAT_NAMES_FR[t.winner]}`
                + (missed.length > 1 ? ` (${missed.length} plis de cette valeur).` : '.'));
            continue;
        }
        // `d === -14` et pas `Math.abs(d)` : ces deux messages disent « vous
        // avez compté trop peu ». Sur un écart positif ils accusaient d'une
        // erreur exactement inverse de celle commise.
        const nine = 8 * trump + 2, jack = 8 * trump + 3;
        const inPile = (c) => win.some(t => t.winner % 2 === k && t.cards.includes(c));
        if (d === -14 && inPile(nine)) {
            out.push(`Le 9 de ${SUIT_NAMES_FR[trump]} est atout : <b>14</b> points, pas 0.`);
            continue;
        }
        if (d === -18 && inPile(jack)) {
            out.push(`Le Valet de ${SUIT_NAMES_FR[trump]} est atout : <b>20</b> points, pas 2.`);
            continue;
        }
        out.push(`${teams.length > 1 ? teamName(k) + ' : ' : ''}écart de ${Math.abs(d)} points — `
            + `la colonne Cumul dit après quel pli le compte a décroché.`);
    }
    return out;
}

function renderTable(exp) {
    const trump = deal.trump;
    const thead = $('pc-table').querySelector('thead');
    const tbody = $('pc-table').querySelector('tbody');
    const dimNS = cfg.count === 'un' && side !== 0;
    const dimEW = cfg.count === 'un' && side !== 1;
    thead.innerHTML = `<tr><th>#</th><th>Cartes</th><th>Ramassé par</th><th>Pts</th>`
        + `<th class="${dimNS ? 'pc-dim' : ''}">Cumul N-S</th>`
        + `<th class="${dimEW ? 'pc-dim' : ''}">Cumul E-O</th></tr>`;
    tbody.innerHTML = '';
    const run = [0, 0];
    for (const t of win) {
        run[t.winner % 2] += t.points;
        const cards = [0, 1, 2, 3].map(j => {
            const c = t.cards[(t.lead + j) % 4];
            const p = cardPoints(c, trump);
            const isTrumpBig = (c >> 3) === trump && ((c & 7) === 2 || (c & 7) === 3);
            const title = isTrumpBig
                ? ` title="${RANKS[c & 7]} d’atout : ${p} points"` : '';
            return `<span class="pc-chip${isTrumpBig ? ' pc-chip-trump' : ''}"${title}>`
                + `${cardChipHtml(c)}${p ? `<sup>${p}</sup>` : ''}</span>`;
        }).join(' ');
        const tr = document.createElement('tr');
        tr.className = teamClass(t.winner);
        tr.innerHTML = `<td>${t.no}</td><td class="pc-cards">${cards}</td>`
            + `<td>${SEAT_NAMES_FR[t.winner]}</td><td class="pc-pts">${t.points}</td>`
            + `<td class="${dimNS ? 'pc-dim' : ''}">${run[0]}</td>`
            + `<td class="${dimEW ? 'pc-dim' : ''}">${run[1]}</td>`;
        tbody.appendChild(tr);
    }
    if (cfg.rules === 'realiste') {
        for (const k of [0, 1]) {
            if (exp.der[k]) addBonusRow(tbody, 'dix de der', exp.der[k], k, run);
            if (exp.belote[k]) addBonusRow(tbody, 'belote', 20, k, run);
        }
    }
    // Au téléphone la moitié droite du tableau est hors champ, et c'est
    // précisément la colonne que le diagnostic invite à lire. La CSS ne montre
    // cette ligne qu'en dessous de 640px.
    $('pc-table-swipe').textContent =
        'Faites glisser le tableau vers la gauche pour voir Pts et Cumul.';
    const foot = $('pc-table-foot');
    const lines = [`Points cartes : ${exp.cards[0]} pour Nord-Sud, ${exp.cards[1]} pour Est-Ouest`
        + (win.length === 8 ? ' — 152 en tout.' : '.')];
    if (cfg.rules === 'realiste') {
        const bonus = [];
        if (deal.der.value) bonus.push(`+${deal.der.value} de dix de der pour ${teamName(deal.der.team)}`);
        for (const k of [0, 1]) if (exp.belote[k]) bonus.push(`+20 de belote pour ${teamName(k)}`);
        lines.push(`${bonus.join(', ')} — total à annoncer : `
            + `${exp.total[0]} et ${exp.total[1]}.`);
    }
    if (win.length === 8) {
        lines.push('Les deux tas somment toujours à 152 points cartes : compter un seul '
            + 'camp suffit, l’autre s’en déduit.');
    }
    foot.innerHTML = lines.join('<br>');
}

/** Les bonus vivent SOUS le sous-total cartes : ni le dix de der ni la belote
 *  ne sont des points de carte, les deux tas font 152 sans eux. */
function addBonusRow(tbody, label, value, team, run) {
    run[team] += value;
    const tr = document.createElement('tr');
    tr.className = `${TEAM_CLASS[team]} pc-bonus-row`;
    tr.innerHTML = `<td></td><td colspan="2">+${value} ${label} — ${teamName(team)}</td>`
        + `<td class="pc-pts">+${value}</td>`
        + `<td>${run[0]}</td><td>${run[1]}</td>`;
    tbody.appendChild(tr);
}

// ===== Configuration ========================================================

function applyPreset(name) {
    cfg = { ...cfg, ...PRESETS[name], preset: name };
    syncFine();
    saveCfg();
    renderStats();
}

function setSeg(id, attr, value) {
    const btns = [...$(id).querySelectorAll('.pc-seg-btn')];
    let matched = false;
    for (const btn of btns) {
        const on = btn.dataset[attr] === String(value);
        if (on) matched = true;
        btn.setAttribute('aria-checked', on ? 'true' : 'false');
        // Tabulation roulante : un groupe = un seul arrêt de tabulation, sur
        // l'option choisie. Sans ça un `radiogroup` de trois options coûte
        // trois tabulations et ment sur ce qu'il est.
        btn.tabIndex = on ? 0 : -1;
    }
    // « perso » n'a pas de bouton : sans ce repli le groupe Niveau se retrouve
    // sans AUCUN arrêt de tabulation dès qu'on touche un réglage fin, et comme
    // l'état est persisté il devient injoignable au clavier pour de bon.
    if (!matched && btns.length) btns[0].tabIndex = 0;
}

/** Les flèches parcourent un groupe de segments, et la sélection suit le focus
 *  — c'est ce qu'un `role="radiogroup"` promet. Même mécanique que
 *  `createSuitPicker` (shared/suits.js), qui est le précédent du dépôt. */
function wireSegKeys(id) {
    $(id).addEventListener('keydown', (e) => {
        if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
        const btns = [...$(id).querySelectorAll('.pc-seg-btn')].filter(b => !b.disabled);
        const i = btns.indexOf(document.activeElement);
        if (i < 0) return;
        e.preventDefault();
        e.stopPropagation();
        const step = (e.key === 'ArrowRight' || e.key === 'ArrowDown') ? 1 : -1;
        const next = btns[(i + step + btns.length) % btns.length];
        next.focus();
        next.click();
    });
}

function syncFine() {
    // « Partie entière » impose la donne complète — d'abord, pour que le
    // curseur affiche bien 8 et pas la valeur précédente.
    const realiste = cfg.rules === 'realiste';
    if (realiste) cfg.nTricks = 8;

    setSeg('pc-presets', 'preset', cfg.preset);
    // « perso » n'a pas de bouton : sans ce marqueur les trois niveaux restent
    // éteints et le groupe a l'air d'avoir perdu sa sélection.
    const perso = cfg.preset === 'perso';
    $('pc-preset-note').textContent = perso ? 'réglages personnalisés' : '';
    $('pc-preset-note').classList.toggle('hidden', !perso);
    setSeg('pc-method', 'method', cfg.method);
    setSeg('pc-count', 'count', cfg.count);
    setSeg('pc-side', 'side', cfg.side);
    setSeg('pc-rules', 'rules', cfg.rules);
    setSeg('pc-source', 'source', cfg.source);
    $('pc-speed').value = cfg.speedMs;
    $('pc-speed-out').textContent = (cfg.speedMs / 1000).toFixed(1).replace('.', ',') + ' s';
    $('pc-ntricks').value = cfg.nTricks;
    $('pc-ntricks-out').textContent = String(cfg.nTricks);

    // Le dix de der ne se joue qu'à la 8e levée et la belote ne se juge
    // qu'une fois les deux cartes tombées : on le dit, plutôt que de laisser
    // un réglage sans effet.
    $('pc-ntricks').disabled = realiste;
    $('pc-row-ntricks').classList.toggle('pc-row--off', realiste);
    $('pc-row-speed').classList.toggle('hidden', cfg.method !== 'chrono');
    $('pc-row-side').classList.toggle('hidden', cfg.count === 'deux');
    $('pc-fine-note').textContent = realiste
        ? 'Le dix de der ne se joue qu’à la 8ᵉ levée : la partie entière se compte forcément sur une donne complète.'
        : '';
    if (realiste) cfg.nTricks = 8;
}

function touchFine() {
    cfg.preset = 'perso';
    setSeg('pc-presets', 'preset', 'perso');
    syncFine();
    saveCfg();
    renderStats();
}

// ===== Clavier ==============================================================

function onKeyDown(e) {
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    // Un bouton qui a le focus traite DÉJÀ Espace et Entrée : après un clic à
    // la souris sur « Suivant », le bouton garde le focus et une barre d'espace
    // ferait avancer deux fois. Les flèches, elles, ne veulent rien dire pour
    // un bouton : on les laisse passer.
    if (tag === 'BUTTON' && (e.key === ' ' || e.key === 'Enter')) return;

    if (phase === 'config') {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); startRun(); }
        else if (e.key === '1') applyPreset('debutant');
        else if (e.key === '2') applyPreset('confirme');
        else if (e.key === '3') applyPreset('expert');
        return;
    }
    if (phase === 'run') {
        const chrono = cfg.method === 'chrono';
        const step = cfg.method === 'pli' ? 4 : 1;
        if (e.key === 'Escape') {
            e.preventDefault();
            if (chrono) setPaused(!paused); else toConfig();
        } else if (e.key === ' ' || e.key === 'p' || e.key === 'P') {
            e.preventDefault();
            if (chrono) setPaused(!paused); else advance(step);
        } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
            e.preventDefault();
            if (!chrono) advance(step);
        } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
            e.preventDefault();
            // Reculer en chronométré, ce serait recompter : c'est un autre
            // exercice. On refuse, mais on le dit — une touche inerte passe
            // pour un bug.
            if (chrono) hint('Retour en arrière désactivé en chronométré — c’est l’exercice.');
            else advance(-step);
        }
        return;
    }
    if (phase === 'review') {
        if (e.key === 'Enter' || e.key === ' ' || e.key === 'n' || e.key === 'N') {
            e.preventDefault(); startRun();
        } else if (e.key === 'r' || e.key === 'R') {
            e.preventDefault(); replaySlow();
        } else if (e.key === 'Escape') {
            e.preventDefault(); toConfig();
        }
    }
}

// ===== Navigation de phase ==================================================

function startRun() {
    if (phase === 'run') return;
    stopAllTimers();
    inReplay = false;
    restoreMethod();
    hint('');
    requestDeal();
}

function restoreMethod() {
    if (methodBeforeReplay === null) return;
    cfg = { ...cfg, method: methodBeforeReplay };
    methodBeforeReplay = null;
    syncFine();
}

function toConfig() {
    // `stopTimer` seul laissait le vol en cours : son callback rappelait
    // `renderAt` sur un plateau masqué, puis `toAnswer` — la page de réglages
    // basculait toute seule sur l'écran de saisie une seconde et demie plus tard.
    stopAllTimers();
    inReplay = false;
    restoreMethod();
    setPhase('config');
    renderStats();
}

/** Retour à la correction déjà calculée — le DOM de `#pc-review` est intact. */
function toReview() {
    inReplay = false;
    restoreMethod();
    setPhase('review');
    renderStats();
    $('pc-review').scrollIntoView({ block: 'start', behavior: 'smooth' });
}

/** Rejoue la MÊME donne, au pas, sans rien redemander au serveur. */
function replaySlow() {
    if (!deal) return;
    stopAllTimers();
    pcIdx = 0;
    paused = false;
    allGathered = false;
    inReplay = true;
    if (methodBeforeReplay === null) methodBeforeReplay = cfg.method;
    cfg = { ...cfg, method: 'carte' };
    syncFine();
    $('pc-pause').classList.add('hidden');
    $('pc-next').classList.remove('hidden');
    $('pc-prev').disabled = false;
    $('pc-trick-area').classList.remove('pc-paused');
    setPhase('run');
    renderAt();
    hint('Relecture : avancez avec →, reculez avec ←. La correction revient à la fin.');
}

// ===== Montage ==============================================================

let touchX = null, touchY = null;

export function mount(container) {
    container.innerHTML = TEMPLATE;

    // Le module est un singleton : tout repart de zéro ici.
    phase = 'config'; deal = null; win = null; pcIdx = 0; paused = false;
    allGathered = false; pending = null; given = [null, null];
    endHold = false; inReplay = false; dealTries = 0;
    methodBeforeReplay = null;   // sinon il écraserait le réglage relu ci-dessous
    cfg = loadCfg();
    if (cfg.rules === 'realiste') cfg.nTricks = 8;

    syncFine();
    renderStats();

    $('pc-presets').addEventListener('click', (e) => {
        const b = e.target.closest('.pc-seg-btn');
        if (b) applyPreset(b.dataset.preset);
    });
    for (const [id, attr, cast] of [
        ['pc-method', 'method', String], ['pc-count', 'count', String],
        ['pc-side', 'side', Number], ['pc-rules', 'rules', String],
        ['pc-source', 'source', String],
    ]) {
        $(id).addEventListener('click', (e) => {
            const b = e.target.closest('.pc-seg-btn');
            if (!b || b.disabled) return;
            cfg[attr] = cast(b.dataset[attr]);
            touchFine();
        });
    }
    for (const id of ['pc-presets', 'pc-method', 'pc-count', 'pc-side', 'pc-rules', 'pc-source']) {
        wireSegKeys(id);
    }
    $('pc-speed').addEventListener('input', (e) => {
        cfg.speedMs = Number(e.target.value); touchFine();
    });
    $('pc-ntricks').addEventListener('input', (e) => {
        cfg.nTricks = Number(e.target.value); touchFine();
    });
    $('pc-fine-toggle').addEventListener('click', () => {
        const closed = $('pc-fine').classList.toggle('hidden');
        $('pc-fine-toggle').setAttribute('aria-expanded', closed ? 'false' : 'true');
    });

    $('pc-start').addEventListener('click', startRun);
    $('pc-quit').addEventListener('click', toConfig);
    $('pc-again').addEventListener('click', startRun);
    $('pc-replay').addEventListener('click', replaySlow);
    $('pc-back').addEventListener('click', toConfig);
    $('pc-answer-form').addEventListener('submit', submitAnswer);

    const stepOf = () => (cfg.method === 'pli' ? 4 : 1);
    $('pc-next').addEventListener('click', () => advance(stepOf()));
    $('pc-prev').addEventListener('click', () => advance(-stepOf()));
    $('pc-pause').addEventListener('click', () => setPaused(!paused));

    // La croix est la plus grande cible de l'écran : un tap y avance, ou met
    // en pause en chronométré.
    const area = $('pc-trick-area');
    area.addEventListener('click', () => {
        if (phase !== 'run') return;
        if (cfg.method === 'chrono') setPaused(!paused); else advance(stepOf());
    });
    area.addEventListener('touchstart', (e) => {
        touchX = e.changedTouches[0].clientX;
        touchY = e.changedTouches[0].clientY;
    }, { passive: true });
    area.addEventListener('touchend', (e) => {
        if (touchX === null || phase !== 'run' || cfg.method === 'chrono') return;
        const dx = e.changedTouches[0].clientX - touchX;
        const dy = e.changedTouches[0].clientY - touchY;
        // Seuil vertical strict : sous 640px le body défile, un balayage doit
        // rester distinguable d'un défilement de page.
        if (Math.abs(dx) > 50 && Math.abs(dy) < 40) advance(dx < 0 ? 4 : -4);
        touchX = touchY = null;
    }, { passive: true });

    // Entrée passe d'un champ à l'autre plutôt que de valider un total vide.
    $('pc-in-1').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && cfg.count === 'deux') { e.preventDefault(); $('pc-in-0').focus(); }
    });

    document.addEventListener('keydown', onKeyDown);
    onMessage('count_ready', onReady);
    onMessage('error', onError);
    onOpen(flushPending);
}

export function unmount() {
    stopAllTimers();
    document.removeEventListener('keydown', onKeyDown);
    offMessage('count_ready', onReady);
    offMessage('error', onError);
    offOpen(flushPending);
    reqId += 1;          // les réponses en vol deviennent périmées
    pending = null;
    deal = null;
    win = null;
    pcIdx = 0;
    phase = 'config';
    paused = false;
    allGathered = false;
    methodBeforeReplay = null;
    inReplay = false;
}
