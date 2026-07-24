// Sièges et équipes — source unique de vérité.
//
// Il existait trois palettes concurrentes : une dans annonces.js, une autre
// dans beliefs.js, et un couple vert/rouge en littéraux dans six feuilles CSS.
// Le vert/rouge posait en plus une collision sémantique : les mêmes teintes
// signalaient « contrat réussi / chuté ». Les équipes passent donc en
// bleu (NS) / violet (EO), et vert/rouge redevient réservé au résultat.
//
// Les valeurs vivent dans tokens.css. On expose ici les *références* aux
// variables CSS, jamais des hexadécimaux : une seule valeur à changer.

export const SEAT_NAMES_FR = ['Nord', 'Est', 'Sud', 'Ouest'];
export const SEAT_INITIALS_FR = ['N', 'E', 'S', 'O'];

/** Ordre moteur : 0=N, 1=E, 2=S, 3=O. */
export const SEAT_COLOR_VARS = [
    'var(--c-seat-n)',
    'var(--c-seat-e)',
    'var(--c-seat-s)',
    'var(--c-seat-w)',
];

export const TEAM_COLOR_VARS = ['var(--c-team-ns)', 'var(--c-team-ew)'];

/** 0 = NS (sièges 0 et 2), 1 = EO (sièges 1 et 3). */
export function teamOf(seat) {
    return seat % 2;
}

/** Classe CSS d'équipe pour un siège — `.team-ns` / `.team-ew` (tokens.css). */
export function teamClass(seat) {
    return teamOf(seat) === 0 ? 'team-ns' : 'team-ew';
}

/** Classe CSS de siège — `.seat-n` / `.seat-e` / `.seat-s` / `.seat-w`. */
export function seatClass(seat) {
    return ['seat-n', 'seat-e', 'seat-s', 'seat-w'][seat];
}

/**
 * Classe d'équipe relative au spectateur : partenaire ou adversaire.
 * `viewer` vaut 2 (Sud) dans le plateau partagé, qui est toujours pivoté.
 */
export function relativeTeamClass(seat, viewer = 2) {
    return teamOf(seat) === teamOf(viewer) ? 'team-partner' : 'team-opponent';
}
