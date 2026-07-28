// Compte — login / register / profile / my games

import { navigateTo } from '../router.js';

const base = () => document.querySelector('base')?.getAttribute('href') || '/';

const AUTH_TEMPLATE = `
<div class="compte-page">
    <div class="compte-card">
        <h2 class="compte-title">Mon compte</h2>
        <div class="compte-tabs">
            <button class="compte-tab active" data-tab="login">Connexion</button>
            <button class="compte-tab" data-tab="register">Inscription</button>
        </div>
        <form id="compte-form" class="compte-form">
            <label class="compte-label" for="compte-username">Pseudo</label>
            <input id="compte-username" class="compte-input" type="text"
                   autocomplete="username" maxlength="20" required>
            <label class="compte-label" for="compte-password">Mot de passe</label>
            <input id="compte-password" class="compte-input" type="password"
                   autocomplete="current-password" required>
            <div id="compte-error" class="compte-error hidden"></div>
            <button id="compte-submit" class="compte-submit" type="submit">Se connecter</button>
        </form>
    </div>
</div>`;

const PROFILE_TEMPLATE = `
<div class="compte-page">
    <div class="compte-card">
        <h2 class="compte-title" id="profile-username"></h2>
        <div class="compte-stats" id="profile-stats"></div>
        <button id="compte-logout" class="compte-logout">Se déconnecter</button>
    </div>
    <div class="compte-card hidden" id="compte-open-card">
        <h3 class="compte-subtitle">Parties en cours</h3>
        <div id="compte-open" class="history-list"></div>
    </div>
    <div class="compte-card">
        <h3 class="compte-subtitle">Mes donnes</h3>
        <div id="compte-games" class="history-list compte-games">
            <div class="history-empty">Chargement…</div>
        </div>
    </div>
</div>`;

let mode = 'login';

async function fetchMe() {
    const resp = await fetch(`${base()}api/me`);
    if (!resp.ok) return { user: null };
    return resp.json();
}

function showError(msg) {
    const el = document.getElementById('compte-error');
    el.textContent = msg;
    el.classList.remove('hidden');
}

function mountAuthForms(container) {
    container.innerHTML = AUTH_TEMPLATE;
    mode = 'login';

    const tabs = container.querySelectorAll('.compte-tab');
    tabs.forEach(tab => tab.addEventListener('click', () => {
        mode = tab.dataset.tab;
        tabs.forEach(t => t.classList.toggle('active', t === tab));
        document.getElementById('compte-submit').textContent =
            mode === 'login' ? 'Se connecter' : "Créer le compte";
        document.getElementById('compte-password').setAttribute('autocomplete',
            mode === 'login' ? 'current-password' : 'new-password');
        document.getElementById('compte-error').classList.add('hidden');
    }));

    document.getElementById('compte-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('compte-username').value.trim();
        const password = document.getElementById('compte-password').value;
        const btn = document.getElementById('compte-submit');
        btn.disabled = true;
        try {
            const resp = await fetch(`${base()}api/auth/${mode === 'login' ? 'login' : 'register'}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });
            const data = await resp.json();
            if (!resp.ok) {
                showError(data.error || 'Erreur inconnue');
                btn.disabled = false;
                return;
            }
            // Full reload so the WebSocket reconnects with the session cookie
            // (games started afterwards get attached to the account).
            location.reload();
        } catch {
            showError('Erreur réseau');
            btn.disabled = false;
        }
    });
}

async function mountProfile(container, me) {
    container.innerHTML = PROFILE_TEMPLATE;
    document.getElementById('profile-username').textContent = me.user.username;

    const stats = me.stats || { games: 0, wins: 0 };
    const since = new Date(me.user.created_at);
    const pct = stats.games > 0 ? Math.round(100 * stats.wins / stats.games) : null;
    const elo = stats.elo && stats.elo.games > 0 ? Math.round(stats.elo.elo) : '—';
    document.getElementById('profile-stats').innerHTML = `
        <div class="compte-stat"><span class="compte-stat-val compte-elo">${elo}</span><span class="compte-stat-label">Elo</span></div>
        <div class="compte-stat"><span class="compte-stat-val">${stats.games}</span><span class="compte-stat-label">parties</span></div>
        <div class="compte-stat"><span class="compte-stat-val">${stats.wins}</span><span class="compte-stat-label">victoires</span></div>
        <div class="compte-stat"><span class="compte-stat-val">${pct === null ? '—' : pct + '%'}</span><span class="compte-stat-label">réussite</span></div>
        <div class="compte-stat"><span class="compte-stat-val">${since.toLocaleDateString('fr-FR')}</span><span class="compte-stat-label">membre depuis</span></div>`;

    document.getElementById('compte-logout').addEventListener('click', async () => {
        await fetch(`${base()}api/auth/logout`, { method: 'POST' });
        location.reload();
    });

    const list = document.getElementById('compte-games');
    try {
        const resp = await fetch(`${base()}api/me/games?limit=50`);
        const games = resp.ok ? await resp.json() : [];
        renderGames(list, games);
    } catch {
        list.innerHTML = '<div class="history-empty">Erreur de chargement</div>';
    }

    // Une *partie* (1000 / 2000 points) groupe des *donnes* : elle ne peut pas
    // vivre dans la liste ci-dessus, qui en montre les lignes une à une.
    // Reprendre passe par la page Jouer, qui tient la socket de jeu.
    try {
        const resp = await fetch(`${base()}api/me/matches`);
        const matches = resp.ok ? await resp.json() : [];
        renderOpenMatches(matches);
    } catch { /* la carte reste masquée */ }
}

function renderOpenMatches(matches) {
    if (!matches || matches.length === 0) return;
    document.getElementById('compte-open-card').classList.remove('hidden');
    const list = document.getElementById('compte-open');
    list.innerHTML = '';
    for (const m of matches) {
        const row = document.createElement('div');
        row.className = 'history-row';
        row.addEventListener('click', () => {
            navigateTo(`/jouer/humain?resume=${encodeURIComponent(m.id)}`);
        });

        const id = document.createElement('span');
        id.className = 'history-id';
        id.textContent = m.id;

        const us = (m.human_seat ?? 2) % 2 === 0 ? m.points_ns : m.points_ew;
        const them = (m.human_seat ?? 2) % 2 === 0 ? m.points_ew : m.points_ns;
        const info = document.createElement('span');
        info.className = 'history-info ' + (us >= them ? 'ns-won' : 'ew-won');
        info.textContent = `${us}-${them}`;

        const meta = document.createElement('span');
        meta.className = 'history-date';
        meta.textContent = `/ ${m.target} · ${m.deals} donne${m.deals > 1 ? 's' : ''}`;

        row.appendChild(id);
        row.appendChild(info);
        row.appendChild(meta);
        list.appendChild(row);
    }
}

function renderGames(list, games) {
    list.innerHTML = '';
    if (games.length === 0) {
        list.innerHTML = '<div class="history-empty">Aucune partie — <a href="/jouer/humain">jouez-en une !</a></div>';
        return;
    }
    for (const g of games) {
        const row = document.createElement('div');
        row.className = 'history-row';
        row.addEventListener('click', () => {
            navigateTo('/analyse/rejouer');
            import('./replay.js').then(m => m.loadReplayById(g.id));
        });

        const id = document.createElement('span');
        id.className = 'history-id';
        id.textContent = g.id;

        // Result from the user's team perspective (even seat = NS)
        const userIsNS = (g.user_seat ?? g.human_seat ?? 2) % 2 === 0;
        const userPts = userIsNS ? g.points_ns : g.points_ew;
        const oppPts = userIsNS ? g.points_ew : g.points_ns;
        const info = document.createElement('span');
        info.className = 'history-info';
        info.textContent = `${userPts}-${oppPts}`;
        info.classList.add(userPts > oppPts ? 'ns-won' : 'ew-won');

        const date = document.createElement('span');
        date.className = 'history-date';
        const d = new Date(g.created_at);
        date.textContent = d.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' });
        date.title = d.toLocaleString();

        row.appendChild(id);
        row.appendChild(info);
        if (g.elo_delta !== null && g.elo_delta !== undefined) {
            const delta = document.createElement('span');
            const v = Math.round(g.elo_delta);
            delta.className = 'history-elo-delta ' + (v >= 0 ? 'delta-up' : 'delta-down');
            delta.textContent = (v >= 0 ? '+' : '') + v;
            delta.title = 'Variation Elo';
            row.appendChild(delta);
        }
        row.appendChild(date);
        list.appendChild(row);
    }
}

export async function mount(container) {
    let me;
    try {
        me = await fetchMe();
    } catch {
        me = { user: null };
    }
    if (me.user) {
        await mountProfile(container, me);
    } else {
        mountAuthForms(container);
    }
}

export function unmount() {}
