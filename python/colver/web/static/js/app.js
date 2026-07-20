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
    navToggle.addEventListener('click', () => {
        nav.classList.toggle('nav-open');
    });
    // Close nav when a link is clicked
    nav.addEventListener('click', (e) => {
        if (e.target.matches('.nav-item, .nav-link')) {
            nav.classList.remove('nav-open');
        }
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
        if (me.user) accountLink.textContent = me.user.username;
    } catch { /* stay on the default "Compte" label */ }
})();

// Start router
initRouter();
