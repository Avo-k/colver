// Symboles de couleur — source unique de vérité.
//
// Avant ce module, ♠♥♦♣ existaient sous trois formes concurrentes :
//   - des glyphes bruts hérités de la couleur du texte ambiant (donc gris),
//   - des emoji ♠️♥️♦️♣️ qui embarquent leur propre palette et ignorent `color`,
//   - des <option> en dur, mono-couleur, dans trois vues différentes.
// Tout passe désormais par ici, et tout sort en rouge ou en noir.

export const SUIT_GLYPHS = ['♠', '♥', '♦', '♣'];
export const SUIT_NAMES_FR = ['Pique', 'Cœur', 'Carreau', 'Trèfle'];
export const SUIT_NAMES_EN = ['spades', 'hearts', 'diamonds', 'clubs'];
export const SUIT_IS_RED = [false, true, true, false];

// Ordre d'affichage : ♠ ♥ ♣ ♦ — alternance noir/rouge.
export const SUIT_DISPLAY_ORDER = [0, 1, 3, 2];

/** Classes CSS du glyphe (voir `.suit` dans tokens.css). */
export function suitClass(suit) {
    return SUIT_IS_RED[suit] ? 'suit suit--red' : 'suit suit--black';
}

/** Glyphe coloré, en HTML. À privilégier partout où on construit une chaîne. */
export function suitHtml(suit, extraClass = '') {
    const cls = suitClass(suit) + (extraClass ? ' ' + extraClass : '');
    return `<span class="${cls}">${SUIT_GLYPHS[suit]}</span>`;
}

/** Glyphe coloré, en élément DOM. */
export function suitEl(suit, extraClass = '') {
    const el = document.createElement('span');
    el.className = suitClass(suit) + (extraClass ? ' ' + extraClass : '');
    el.textContent = SUIT_GLYPHS[suit];
    return el;
}

/** « ♥ Cœur » — glyphe coloré suivi du nom français. */
export function suitLabelHtml(suit) {
    return `${suitHtml(suit)} ${SUIT_NAMES_FR[suit]}`;
}

/**
 * Segmented control de choix de couleur — remplace les <select> dont les
 * <option> ne peuvent pas être colorées de façon fiable selon les navigateurs
 * (d'où le contournement par emoji, qu'on supprime).
 *
 * opts : { value, onChange(suit), withLabels, name }
 * Retour : élément DOM portant `.value` (get/set) et `.disabled` (set).
 */
export function createSuitPicker({ value = 0, onChange = null, withLabels = false, name = 'atout' } = {}) {
    const root = document.createElement('div');
    root.className = 'suit-picker';
    root.setAttribute('role', 'radiogroup');
    root.setAttribute('aria-label', `Couleur d'${name}`);

    let current = value;
    const buttons = [];

    for (const suit of SUIT_DISPLAY_ORDER) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'suit-picker__opt';
        btn.setAttribute('role', 'radio');
        btn.dataset.suit = String(suit);
        btn.title = SUIT_NAMES_FR[suit];
        btn.setAttribute('aria-label', SUIT_NAMES_FR[suit]);
        btn.innerHTML = withLabels ? suitLabelHtml(suit) : suitHtml(suit);
        btn.addEventListener('click', () => {
            setValue(suit);
            if (onChange) onChange(suit);
        });
        root.appendChild(btn);
        buttons.push(btn);
    }

    function setValue(suit) {
        current = suit;
        for (const btn of buttons) {
            const on = Number(btn.dataset.suit) === suit;
            btn.setAttribute('aria-checked', on ? 'true' : 'false');
            btn.tabIndex = on ? 0 : -1;
        }
    }
    setValue(current);

    Object.defineProperty(root, 'value', {
        get: () => current,
        set: (v) => setValue(Number(v)),
    });
    Object.defineProperty(root, 'disabled', {
        get: () => root.getAttribute('aria-disabled') === 'true',
        set: (v) => root.setAttribute('aria-disabled', v ? 'true' : 'false'),
    });

    // Flèches gauche/droite entre les options, comme un vrai radiogroup.
    root.addEventListener('keydown', (e) => {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        e.preventDefault();
        const idx = SUIT_DISPLAY_ORDER.indexOf(current);
        const step = e.key === 'ArrowRight' ? 1 : -1;
        const next = SUIT_DISPLAY_ORDER[(idx + step + 4) % 4];
        setValue(next);
        buttons[SUIT_DISPLAY_ORDER.indexOf(next)].focus();
        if (onChange) onChange(next);
    });

    return root;
}
