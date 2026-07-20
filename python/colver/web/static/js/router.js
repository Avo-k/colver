// pushState router with dynamic import, mount/unmount lifecycle

const routes = {
    '/':                  () => import('./views/landing.js'),
    '/jouer/humain':      () => import('./views/play.js'),
    '/jouer/ia':          () => import('./views/watch.js'),
    '/analyse/rejouer':   () => import('./views/replay.js'),
    '/analyse/annonces':  () => import('./views/annonces.js'),
    '/analyse/croyances': () => import('./views/beliefs.js'),
    '/problemes/annonce': () => import('./views/prob-annonce.js'),
    '/problemes/jeu':     () => import('./views/prob-jeu.js'),
    '/aide':              () => import('./views/aide.js'),
    '/annoncer':          () => import('./views/annoncer.js'),
    '/score':             () => import('./views/score.js'),
    '/about':             () => import('./views/about.js'),
    '/compte':            () => import('./views/compte.js'),
};

// Legacy hash redirects (old bookmarks still work)
const legacyHashMap = {
    '#play':         '/jouer/humain',
    '#watch':        '/jouer/ia',
    '#replay':       '/analyse/rejouer',
    '#deal':         '/jouer/ia',
    '#annonces':     '/analyse/annonces',
    '#prob-annonce': '/problemes/annonce',
    '#prob-jeu':     '/problemes/jeu',
    '#docs':         '/about',
};

let currentView = null;
let currentPath = null;
const container = () => document.getElementById('app');

// Resolve base href for reverse proxy support (e.g. /colver/)
const base = document.querySelector('base')?.getAttribute('href') || '/';

function parsePath() {
    // Handle legacy hash URLs — redirect to clean path
    const hash = location.hash;
    if (hash) {
        // Legacy short hashes: #play, #watch, etc.
        if (legacyHashMap[hash]) {
            history.replaceState(null, '', base + legacyHashMap[hash].slice(1));
            return legacyHashMap[hash];
        }
        // Old hash-router URLs: #/jouer/humain, #/analyse/annonces, etc.
        if (hash.startsWith('#/')) {
            const path = hash.slice(1); // '#/foo' -> '/foo'
            history.replaceState(null, '', base + path.slice(1));
            return path;
        }
    }
    // Strip base prefix to get the route path
    let path = location.pathname;
    if (base !== '/' && path.startsWith(base)) {
        path = path.slice(base.length - 1); // keep leading /
    }
    return path || '/';
}

async function navigate() {
    const path = parsePath();
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
        navigateTo('/');
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

    // Top-level nav links (Aide, Annoncer, Marquer, À propos, Compte…)
    document.querySelectorAll('.nav-link[data-route]').forEach(link => {
        link.classList.toggle('active', link.dataset.route === path);
    });
}

export function init() {
    // Intercept clicks on internal links to avoid full page reloads
    document.addEventListener('click', (e) => {
        const link = e.target.closest('a[href]');
        if (!link) return;
        const href = link.getAttribute('href');
        // Only handle local paths (not external links, ws://, etc.)
        if (!href || href.startsWith('http') || href.startsWith('//')) return;
        e.preventDefault();
        navigateTo(href);
    });

    window.addEventListener('popstate', navigate);
    navigate();
}

export function navigateTo(path) {
    // Build full URL respecting base href
    const url = base === '/' ? path : base + path.slice(1);
    history.pushState(null, '', url);
    navigate();
}
