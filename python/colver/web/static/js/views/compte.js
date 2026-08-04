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
            <!-- L'adresse n'apparaît qu'à l'inscription : elle est facultative,
                 mais c'est le seul recours en cas d'oubli, d'où le rappel. -->
            <div id="compte-email-row" class="hidden">
                <label class="compte-label" for="compte-email">
                    E-mail (facultatif)
                </label>
                <input id="compte-email" class="compte-input" type="email"
                       autocomplete="email" placeholder="pour récupérer un mot de passe oublié">
            </div>
            <div id="compte-error" class="compte-error hidden"></div>
            <button id="compte-submit" class="compte-submit" type="submit">Se connecter</button>
        </form>
        <p class="mdp-foot" id="compte-forgot">
            <a href="/mot-de-passe/oublie">Mot de passe oublié ?</a>
        </p>
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
    <!-- Mes stats : tout ce qui est un *taux* vit ici et pas au classement.
         Sur quelques centaines de donnes un taux porte un intervalle de ±10
         points : il décrit honnêtement un joueur, il n'en ordonne pas deux.
         Chaque ligne affiche donc son n et son intervalle. -->
    <details class="compte-card compte-settings" id="compte-stats-card">
        <summary class="compte-subtitle compte-summary">Mes statistiques</summary>
        <div id="compte-stats-body">
            <div class="history-empty">Chargement…</div>
        </div>
    </details>
    <div class="compte-card">
        <h3 class="compte-subtitle">Mes donnes</h3>
        <div id="compte-games" class="history-list compte-games">
            <div class="history-empty">Chargement…</div>
        </div>
    </div>
    <!-- Réglages du compte : repliés par défaut. On vient ici pour voir ses
         donnes, pas pour changer son mot de passe — mais quand on le cherche,
         il faut le trouver. -->
    <details class="compte-card compte-settings" id="compte-settings">
        <summary class="compte-subtitle compte-summary">Réglages du compte</summary>

        <section class="compte-section">
            <h4 class="compte-section-title">Adresse e-mail</h4>
            <p class="compte-hint" id="compte-email-state"></p>
            <form id="form-email" class="compte-form">
                <input class="compte-hidden-user" type="text" autocomplete="username"
                       data-username tabindex="-1" aria-hidden="true" readonly>
                <label class="compte-label" for="set-email">Adresse</label>
                <input id="set-email" class="compte-input" type="email"
                       autocomplete="email" placeholder="vide = retirer l'adresse">
                <label class="compte-label" for="set-email-pw">Mot de passe</label>
                <input id="set-email-pw" class="compte-input" type="password"
                       autocomplete="current-password" required>
                <div class="compte-error hidden" data-error></div>
                <div class="compte-ok hidden" data-ok></div>
                <button class="compte-submit" type="submit">Enregistrer l'adresse</button>
            </form>
        </section>

        <section class="compte-section">
            <h4 class="compte-section-title">Mot de passe</h4>
            <p class="compte-hint">
                Le changer déconnecte vos autres appareils. Celui-ci reste connecté.
            </p>
            <form id="form-password" class="compte-form">
                <!-- Champ pseudo caché : sans lui, un gestionnaire de mots de
                     passe ne sait pas *quel* compte il enregistre, et Chrome le
                     signale. Rempli par mountSettings(). -->
                <input class="compte-hidden-user" type="text" autocomplete="username"
                       data-username tabindex="-1" aria-hidden="true" readonly>
                <label class="compte-label" for="pw-current">Mot de passe actuel</label>
                <input id="pw-current" class="compte-input" type="password"
                       autocomplete="current-password" required>
                <label class="compte-label" for="pw-new">Nouveau mot de passe</label>
                <input id="pw-new" class="compte-input" type="password"
                       autocomplete="new-password" required minlength="8">
                <label class="compte-label" for="pw-new2">Confirmation</label>
                <input id="pw-new2" class="compte-input" type="password"
                       autocomplete="new-password" required minlength="8">
                <div class="compte-error hidden" data-error></div>
                <div class="compte-ok hidden" data-ok></div>
                <button class="compte-submit" type="submit">Changer le mot de passe</button>
            </form>
        </section>

        <section class="compte-section compte-danger">
            <h4 class="compte-section-title">Supprimer mon compte</h4>
            <p class="compte-hint">
                Définitif. Vos donnes ne sont pas effacées — une donne de salon
                appartient aussi aux trois autres joueurs — mais elles sont
                détachées de vous : votre siège y devient « Invité », et votre
                pseudo, votre classement et votre historique disparaissent.
            </p>
            <form id="form-delete" class="compte-form">
                <input class="compte-hidden-user" type="text" autocomplete="username"
                       data-username tabindex="-1" aria-hidden="true" readonly>
                <label class="compte-label" for="del-confirm">
                    Saisissez votre pseudo pour confirmer
                </label>
                <input id="del-confirm" class="compte-input" type="text"
                       autocomplete="off" required>
                <label class="compte-label" for="del-pw">Mot de passe</label>
                <input id="del-pw" class="compte-input" type="password"
                       autocomplete="current-password" required>
                <div class="compte-error hidden" data-error></div>
                <button class="compte-submit compte-submit-danger" type="submit">
                    Supprimer définitivement
                </button>
            </form>
        </section>
    </details>
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
        // L'adresse ne se saisit qu'à l'inscription ; « mot de passe oublié »
        // n'a de sens qu'à la connexion.
        document.getElementById('compte-email-row')
            .classList.toggle('hidden', mode === 'login');
        document.getElementById('compte-forgot')
            .classList.toggle('hidden', mode !== 'login');
        document.getElementById('compte-error').classList.add('hidden');
    }));

    document.getElementById('compte-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('compte-username').value.trim();
        const password = document.getElementById('compte-password').value;
        const email = document.getElementById('compte-email').value.trim();
        const btn = document.getElementById('compte-submit');
        btn.disabled = true;
        try {
            const body = { username, password };
            if (mode === 'register' && email) body.email = email;
            const resp = await fetch(`${base()}api/auth/${mode === 'login' ? 'login' : 'register'}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
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

    mountStats();
    mountSettings(me.user);
}

// ---- Mes statistiques ------------------------------------------------------

/** Une ligne « libellé — valeur — précision ». `hint` porte le n et l'intervalle. */
function statRow(label, value, hint) {
    if (value === null || value === undefined) return '';
    return '<div class="cs-row">' +
        `<span class="cs-label">${label}</span>` +
        `<span class="cs-val">${value}</span>` +
        `<span class="cs-hint">${hint || ''}</span></div>`;
}

/** Un pourcentage avec son intervalle de confiance, ou un tiret sous 5 obs.
 *
 *  Le seuil n'est pas cosmétique : à n = 3 l'intervalle de Wilson couvre à peu
 *  près tout [0, 100], donc le chiffre n'informe sur rien et sa seule fonction
 *  serait de se faire lire comme une mesure.
 */
function pctRow(label, blob, suffix = '') {
    if (!blob || !blob.n) return '';
    if (blob.n < 5) {
        return statRow(label, '—', `seulement ${blob.n} obs.`);
    }
    const span = `${blob.lo}–${blob.hi} %`;
    return statRow(label, `${blob.pct} %`,
        `n = ${blob.n}${suffix} · IC95 ${span}`);
}

function renderStats(st) {
    if (!st || !st.deals) {
        return '<div class="history-empty">Aucune donne terminée pour l\'instant.</div>';
    }
    const parts = [];

    parts.push('<h4 class="compte-section-title">Résultats</h4>');
    parts.push(pctRow('Donnes gagnées', st.won));
    const m = st.margin || {};
    if (m.n >= 5) {
        parts.push(statRow('Écart moyen par donne',
            `${m.mean > 0 ? '+' : ''}${m.mean}`,
            `n = ${m.n} · ±${m.ci} · médiane ${m.median}`));
    }
    if (st.deals !== st.scored) {
        parts.push(`<p class="compte-hint">${st.deals - st.scored} donne(s) ` +
            'en attente de calcul — exclues des taux ci-dessus.</p>');
    }

    parts.push('<h4 class="compte-section-title">Enchère</h4>');
    const b = st.bidding || {};
    const t = st.takes || {};
    if (t.n) {
        parts.push(statRow('Contrats pris', t.n,
            `${t.per_100} pour 100 donnes · hauteur moyenne ${t.avg_value ?? '—'}`));
    }
    parts.push(pctRow('Contrats tenus',
        { n: t.n, pct: t.held_pct, lo: t.held_lo, hi: t.held_hi }, ' prises'));
    if (b.decisions >= 5) {
        parts.push(statRow('Passes', `${b.pass_pct} %`,
            `sur ${b.decisions} décisions d'enchère`));
    }
    if (b.height_n >= 5) {
        parts.push(statRow('Hauteur annoncée', b.avg_height,
            `moyenne sur ${b.height_n} annonces`));
    }
    // Le trio prises / hauteur / tenus se lit ensemble, et jamais trié : un
    // taux de contrats tenus se maximise en n'annonçant plus que 80.
    parts.push('<p class="compte-hint">Ces trois lignes se lisent ensemble : ' +
        'un taux de contrats tenus très haut accompagné d\'une hauteur basse ' +
        'décrit surtout de la prudence.</p>');

    parts.push('<h4 class="compte-section-title">Défense et faits d\'armes</h4>');
    parts.push(pctRow('Chutes infligées', st.defense, ' défenses'));
    parts.push(statRow('Capots réalisés', st.capots_for, ''));
    parts.push(statRow('Capots subis', st.capots_against, ''));
    if (st.coinches) parts.push(statRow('Coinches annoncées', st.coinches, ''));
    if (st.surcoinches) parts.push(statRow('Surcoinches', st.surcoinches, ''));

    parts.push('<h4 class="compte-section-title">Assiduité</h4>');
    parts.push(statRow('Donnes jouées', st.deals, ''));
    parts.push(statRow('Jours de jeu', st.days,
        st.density ? `${st.density} donnes par jour joué` : ''));
    const tempo = st.tempo || {};
    if (tempo.n >= 5) {
        // Les écarts écartés (interruptions de plus de 15 min) sont dits, pas
        // tus : une troncature silencieuse se lirait comme une mesure complète.
        const dropped = tempo.dropped
            ? ` · ${tempo.dropped} interruption${tempo.dropped > 1 ? 's' : ''} exclue${tempo.dropped > 1 ? 's' : ''}`
            : '';
        parts.push(statRow('Temps par donne', `${tempo.median} s`,
            `médiane sur ${tempo.n} donnes en partie · ${tempo.p25}–${tempo.p75} s${dropped}`));
    }

    return `<div class="compte-stats-list">${parts.filter(Boolean).join('')}</div>`;
}

async function mountStats() {
    const body = document.getElementById('compte-stats-body');
    if (!body) return;
    try {
        const resp = await fetch(`${base()}api/me/stats`);
        body.innerHTML = resp.ok ? renderStats(await resp.json())
            : '<div class="history-empty">Statistiques indisponibles</div>';
    } catch {
        body.innerHTML = '<div class="history-empty">Erreur de chargement</div>';
    }
}

// ---- Réglages du compte ----------------------------------------------------

function feedback(form, { error, ok }) {
    const errEl = form.querySelector('[data-error]');
    const okEl = form.querySelector('[data-ok]');
    if (errEl) {
        errEl.textContent = error || '';
        errEl.classList.toggle('hidden', !error);
    }
    if (okEl) {
        okEl.textContent = ok || '';
        okEl.classList.toggle('hidden', !ok);
    }
}

/** POST JSON + gestion uniforme du bouton et des messages. */
async function submitForm(form, path, body, onSuccess) {
    const btn = form.querySelector('button[type="submit"]');
    btn.disabled = true;
    feedback(form, {});
    try {
        const resp = await fetch(`${base()}api/${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        let data = {};
        try { data = await resp.json(); } catch { /* corps vide */ }
        if (!resp.ok) {
            feedback(form, { error: data.error || 'Erreur inconnue' });
            return;
        }
        onSuccess(data);
    } catch {
        feedback(form, { error: 'Erreur réseau' });
    } finally {
        btn.disabled = false;
    }
}

function renderEmailState(email) {
    const el = document.getElementById('compte-email-state');
    // Sans adresse, il n'y a **aucun** recours en cas d'oubli. C'est la seule
    // chose que cette ligne a besoin de dire, et il faut qu'elle se voie.
    el.classList.toggle('compte-hint-warn', !email);
    el.textContent = email
        ? `Adresse enregistrée : ${email}. C'est par là que passe un mot de passe oublié.`
        : "Aucune adresse enregistrée : si vous perdez votre mot de passe, "
          + "il n'y aura aucun moyen de récupérer ce compte.";
    document.getElementById('set-email').value = email || '';
}

function mountSettings(user) {
    // Le pseudo dans chaque formulaire à mot de passe : c'est lui qui dit au
    // gestionnaire de mots de passe quel compte il met à jour.
    document.querySelectorAll('#compte-settings [data-username]')
        .forEach(el => { el.value = user.username; });
    renderEmailState(user.email);

    const emailForm = document.getElementById('form-email');
    emailForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitForm(emailForm, 'auth/email', {
            email: document.getElementById('set-email').value.trim(),
            password: document.getElementById('set-email-pw').value,
        }, (data) => {
            document.getElementById('set-email-pw').value = '';
            renderEmailState(data.email);
            feedback(emailForm, {
                ok: data.email ? 'Adresse enregistrée.' : 'Adresse retirée.',
            });
        });
    });

    const pwForm = document.getElementById('form-password');
    pwForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const next = document.getElementById('pw-new').value;
        if (next !== document.getElementById('pw-new2').value) {
            feedback(pwForm, { error: 'Les deux mots de passe ne correspondent pas' });
            return;
        }
        submitForm(pwForm, 'auth/password', {
            current_password: document.getElementById('pw-current').value,
            new_password: next,
        }, () => {
            pwForm.reset();
            feedback(pwForm, {
                ok: 'Mot de passe changé. Vos autres appareils ont été déconnectés.',
            });
        });
    });

    const delForm = document.getElementById('form-delete');
    delForm.addEventListener('submit', (e) => {
        e.preventDefault();
        // Le serveur exige déjà le pseudo *et* le mot de passe ; cette
        // confirmation-ci est la dernière marche avant l'irréversible.
        if (!confirm('Supprimer définitivement votre compte ? '
                     + 'Cette action ne peut pas être annulée.')) return;
        submitForm(delForm, 'account/delete', {
            confirm: document.getElementById('del-confirm').value,
            password: document.getElementById('del-pw').value,
        }, () => { location.href = base(); });
    });
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

        // Pas de variation d'Elo sur une ligne de donne : depuis la migration
        // v14 l'unité notée est la partie en 2000 points. Le delta appartient à
        // la partie, et le reporter sur chacune de ses donnes afficherait dix
        // fois le même chiffre.
        row.appendChild(id);
        row.appendChild(info);
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
