// Entry point — bootstraps app, connects WS, inits router

import { connect } from './ws.js';
import * as SFX from './sounds.js';
import { initBugReportModal } from './shared/bug-report.js';
import { init as initRouter } from './router.js';

// Connect WebSocket
connect();

// Init bug report modal (it lives in index.html shell)
initBugReportModal();

// Mobile nav toggle
const navToggle = document.getElementById('nav-toggle');
const nav = document.querySelector('nav');
if (navToggle && nav) {
    const setOpen = (open) => {
        nav.classList.toggle('nav-open', open);
        navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    };
    navToggle.addEventListener('click', () => setOpen(!nav.classList.contains('nav-open')));
    // Close nav when a link is clicked
    nav.addEventListener('click', (e) => {
        const link = e.target.closest('.nav-item, .nav-link');
        if (!link) return;
        setOpen(false);
        // Un lien cliqué garde le focus, donc `.nav-group:focus-within` maintient
        // le menu déroulant ouvert même une fois la souris partie. On relâche le
        // focus pour rendre la main au survol — sauf si l'activation vient du
        // clavier (detail === 0), où garder le focus est le comportement attendu.
        if (e.detail > 0) link.blur();
    });
    // Échap ferme le panneau, et referme aussi un menu déroulant ouvert au
    // clavier (il reste ouvert tant que le focus est dedans).
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        if (nav.classList.contains('nav-open')) setOpen(false);
        const focused = document.activeElement;
        if (focused && nav.contains(focused)) focused.blur();
    });
}

// Sound toggle
const soundBtn = document.getElementById('sound-toggle');
if (soundBtn) {
    function updateIcon() {
        soundBtn.textContent = SFX.isMuted() ? '\u{1F507}' : '\u{1F50A}';
        soundBtn.classList.toggle('muted', SFX.isMuted());
    }
    updateIcon();
    soundBtn.addEventListener('click', () => {
        SFX.toggleMute();
        updateIcon();
    });
}

// Show username in the nav when logged in
(async () => {
    const accountLink = document.getElementById('nav-account');
    if (!accountLink) return;
    try {
        const base = document.querySelector('base')?.getAttribute('href') || '/';
        const resp = await fetch(`${base}api/me`);
        const me = resp.ok ? await resp.json() : { user: null };
        // Ne remplacer que le libellé : le bouton contient aussi le chevron.
        if (me.user) accountLink.childNodes[0].nodeValue = me.user.username;
    } catch { /* stay on the default "Compte" label */ }
})();

// Start router
initRouter();
