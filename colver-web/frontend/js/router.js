// Hash-based router with dynamic import, mount/unmount lifecycle

const routes = {
    '/':                  () => import('./views/landing.js'),
    '/jouer/humain':      () => import('./views/play.js'),
    '/jouer/ia':          () => import('./views/watch.js'),
    '/analyse/rejouer':   () => import('./views/replay.js'),
    '/analyse/annonces':  () => import('./views/annonces.js'),
    '/analyse/croyances': () => import('./views/beliefs.js'),
    '/problemes/annonce': () => import('./views/prob-annonce.js'),
    '/problemes/jeu':     () => import('./views/prob-jeu.js'),
    '/about':             () => import('./views/about.js'),
};

// Legacy hash redirects
const legacyRedirects = {
    '#play':         '#/jouer/humain',
    '#watch':        '#/jouer/ia',
    '#replay':       '#/analyse/rejouer',
    '#deal':         '#/jouer/ia',
    '#annonces':     '#/analyse/annonces',
    '#prob-annonce': '#/problemes/annonce',
    '#prob-jeu':     '#/problemes/jeu',
    '#docs':         '#/about',
};

let currentView = null;
let currentPath = null;
const container = () => document.getElementById('app');

function parsePath() {
    const hash = location.hash || '#/';
    // Handle legacy hashes
    if (legacyRedirects[hash]) {
        location.hash = legacyRedirects[hash];
        return null; // will re-trigger hashchange
    }
    return hash.slice(1) || '/'; // remove leading #
}

async function navigate() {
    const path = parsePath();
    if (path === null) return; // redirecting
    if (path === currentPath) return;

    // Unmount current view
    if (currentView && currentView.unmount) {
        currentView.unmount();
    }
    currentView = null;
    currentPath = path;

    const el = container();
    el.innerHTML = '';

    // Load new view
    const loader = routes[path];
    if (!loader) {
        // Unknown route — go to landing
        location.hash = '#/';
        return;
    }

    try {
        const mod = await loader();
        // Check path hasn't changed during load
        if (currentPath !== path) return;
        currentView = mod;
        if (mod.mount) mod.mount(el);
    } catch (err) {
        console.error('Route load error:', err);
        el.innerHTML = '<div style="padding:2rem;color:#ef5350">Erreur de chargement</div>';
    }

    updateNavHighlight(path);
}

function updateNavHighlight(path) {
    // Update dropdown items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.route === path);
    });

    // Update group labels
    document.querySelectorAll('.nav-group').forEach(group => {
        const items = group.querySelectorAll('.nav-item');
        const hasActive = Array.from(items).some(i => i.classList.contains('active'));
        group.classList.toggle('active', hasActive);
    });

    // About link
    const aboutLink = document.querySelector('.nav-link[data-route="/about"]');
    if (aboutLink) aboutLink.classList.toggle('active', path === '/about');
}

export function init() {
    window.addEventListener('hashchange', navigate);
    navigate();
}

export function navigateTo(path) {
    location.hash = '#' + path;
}
