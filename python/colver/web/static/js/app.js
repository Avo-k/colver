// Entry point — bootstraps app, connects WS, inits router

import { connect } from './ws.js';
import * as SFX from './sounds.js';
import { initBugReportModal } from './shared/bug-report.js';
import { init as initRouter } from './router.js';

// Connect WebSocket
connect();

// Init bug report modal (it lives in index.html shell)
initBugReportModal();

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

// Start router
initRouter();
