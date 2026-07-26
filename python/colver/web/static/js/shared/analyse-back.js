// Lien « ← Retour à la partie » des pages d'analyse.
//
// Les pages Annonces et Jeu de la carte s'atteignent depuis Rejouer, et on y
// fait souvent l'aller-retour. Le bouton Retour du navigateur suffirait si on
// venait toujours de là, mais ces pages sont aussi partageables et
// bookmarkables : sur un lien reçu, Retour ramène ailleurs, voire hors du site.
// D'où un chemin de retour explicite, porté par l'URL (`from` = id de partie,
// `i` = index du coup) plutôt que par l'historique.

const SHORT = 24;  // au-delà, l'id de partie est tronqué dans le libellé

/**
 * Remplit l'ancre `elId` avec un retour vers la partie, ou la laisse cachée.
 * @param {string} elId  id de l'ancre (classe `analyse-back`, `hidden` au départ)
 * @param {?string} from id de partie transmis par Rejouer
 * @param {?string} idx  index du coup, en chaîne telle que lue dans l'URL
 */
export function renderBackLink(elId, from, idx) {
    const el = document.getElementById(elId);
    if (!el) return;

    const game = (from || '').trim();
    if (!game) {
        el.classList.add('hidden');
        return;
    }

    const q = new URLSearchParams({ game });
    const n = parseInt(idx, 10);
    if (Number.isFinite(n) && n >= 0) q.set('i', String(n));

    el.setAttribute('href', `/analyse/rejouer?${q}`);
    const label = game.length > SHORT ? `${game.slice(0, SHORT)}…` : game;
    el.textContent = Number.isFinite(n) && n >= 0
        ? `← Retour à la partie ${label}, coup ${n + 1}`
        : `← Retour à la partie ${label}`;
    el.classList.remove('hidden');
}
