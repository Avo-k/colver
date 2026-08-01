// Récupération de compte — deux écrans, un module.
//
//   /mot-de-passe/oublie   demander un lien
//   /mot-de-passe/nouveau  poser un nouveau mot de passe avec le jeton reçu
//
// Ils partagent tout (la carte, le formulaire, la gestion d'erreur) et ne
// diffèrent que par le formulaire montré, d'où un seul fichier. Le routeur
// pointe les deux chemins ici et `mount` lit `location.pathname`.
//
// Deux règles de discipline côté serveur ont leur reflet ici :
//   - la demande **ne dit jamais si un compte existe** (`auth.forgot`), donc
//     l'écran de confirmation ne peut pas promettre qu'un message est parti.
//     Il dit ce qui est vrai : « si un compte correspond, un lien vient de
//     partir » ;
//   - le jeton ne sert qu'une fois, donc un lien déjà utilisé donne la même
//     erreur qu'un lien expiré, et la sortie est la même : en redemander un.

const base = () => document.querySelector('base')?.getAttribute('href') || '/';

const FORGOT_TEMPLATE = `
<div class="compte-page">
    <div class="compte-card">
        <h2 class="compte-title">Mot de passe oublié</h2>
        <p class="mdp-intro">
            Indiquez votre pseudo ou votre adresse e-mail. Si un compte y
            correspond <em>et</em> qu'une adresse y est renseignée, vous
            recevrez un lien pour choisir un nouveau mot de passe.
        </p>
        <form id="mdp-form" class="compte-form">
            <label class="compte-label" for="mdp-id">Pseudo ou e-mail</label>
            <input id="mdp-id" class="compte-input" type="text"
                   autocomplete="username" required>
            <div id="mdp-error" class="compte-error hidden"></div>
            <button id="mdp-submit" class="compte-submit" type="submit">
                Envoyer le lien
            </button>
        </form>
        <p class="mdp-foot"><a href="/compte">← Retour à la connexion</a></p>
    </div>
</div>`;

const SENT_TEMPLATE = `
<div class="compte-page">
    <div class="compte-card">
        <h2 class="compte-title">Vérifiez vos messages</h2>
        <p class="mdp-intro">
            Si un compte correspond à ce que vous avez saisi et qu'une adresse
            e-mail y est renseignée, un lien vient d'y être envoyé. Il est
            valable deux heures et ne fonctionne qu'une fois.
        </p>
        <p class="mdp-intro mdp-muted">
            Rien ne vous parvient ? Le compte n'a peut-être pas d'adresse
            renseignée — dans ce cas il n'y a pas de récupération possible.
        </p>
        <p class="mdp-foot"><a href="/compte">← Retour à la connexion</a></p>
    </div>
</div>`;

const RESET_TEMPLATE = `
<div class="compte-page">
    <div class="compte-card">
        <h2 class="compte-title">Nouveau mot de passe</h2>
        <form id="mdp-form" class="compte-form">
            <label class="compte-label" for="mdp-pw">Nouveau mot de passe</label>
            <input id="mdp-pw" class="compte-input" type="password"
                   autocomplete="new-password" required minlength="8">
            <label class="compte-label" for="mdp-pw2">Confirmation</label>
            <input id="mdp-pw2" class="compte-input" type="password"
                   autocomplete="new-password" required minlength="8">
            <div id="mdp-error" class="compte-error hidden"></div>
            <button id="mdp-submit" class="compte-submit" type="submit">
                Changer le mot de passe
            </button>
        </form>
        <p class="mdp-foot">
            <a href="/mot-de-passe/oublie">Demander un nouveau lien</a>
        </p>
    </div>
</div>`;

function showError(msg) {
    const el = document.getElementById('mdp-error');
    if (!el) return;
    el.textContent = msg;
    el.classList.remove('hidden');
}

async function post(path, body) {
    const resp = await fetch(`${base()}api/${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    let data = {};
    try { data = await resp.json(); } catch { /* corps vide */ }
    return { ok: resp.ok, data };
}

function mountForgot(container) {
    container.innerHTML = FORGOT_TEMPLATE;
    document.getElementById('mdp-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('mdp-submit');
        btn.disabled = true;
        const identifier = document.getElementById('mdp-id').value.trim();
        const { ok, data } = await post('auth/forgot', { identifier });
        if (!ok) {
            // En pratique : seulement le plafond de tentatives — le serveur
            // répond 200 même pour un identifiant inconnu, par construction.
            showError(data.error || 'Erreur réseau');
            btn.disabled = false;
            return;
        }
        container.innerHTML = SENT_TEMPLATE;
    });
}

function mountReset(container) {
    container.innerHTML = RESET_TEMPLATE;
    // Le jeton est lu une fois puis **retiré de l'URL** : un F5, un partage
    // d'écran ou un historique de navigateur ne doivent pas le promener. Même
    // principe que le `?resume=` de la page Jouer.
    const url = new URL(location.href);
    const token = url.searchParams.get('token') || '';
    if (token) {
        url.searchParams.delete('token');
        history.replaceState(null, '', url.pathname + url.search);
    }
    if (!token) {
        showError("Lien incomplet — redemandez-en un.");
        document.getElementById('mdp-submit').disabled = true;
        return;
    }

    document.getElementById('mdp-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const pw = document.getElementById('mdp-pw').value;
        const pw2 = document.getElementById('mdp-pw2').value;
        if (pw !== pw2) {
            showError('Les deux mots de passe ne correspondent pas');
            return;
        }
        const btn = document.getElementById('mdp-submit');
        btn.disabled = true;
        const { ok, data } = await post('auth/reset', { token, new_password: pw });
        if (!ok) {
            showError(data.error || 'Erreur réseau');
            btn.disabled = false;
            return;
        }
        // La réponse ouvre une session : rechargement complet, pour que le
        // WebSocket se reconnecte avec le cookie (comme après une connexion).
        location.href = `${base()}compte`;
    });
}

export function mount(container) {
    if (location.pathname.endsWith('/nouveau')) {
        mountReset(container);
    } else {
        mountForgot(container);
    }
}

export function unmount() {}