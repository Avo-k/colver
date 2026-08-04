// Classement — deux onglets qui ne mesurent pas la même chose.
//
// [Elo] est le classement noble : parties en 2000 points, rien d'autre. Il est
// seul dans son onglet et on n'y ajoute rien — sa valeur vient de ce qu'il ne
// mélange rien.
//
// [Vie du site] ne contient que des **comptes exacts** : capots, jours joués,
// séries, parties. Aucun taux n'y est trié, et c'est délibéré. Sur un corpus de
// quelques centaines de donnes par joueur, un taux de victoire porte un
// intervalle de ±10 points : trier dessus publierait un ordre que les données
// ne soutiennent pas. Les taux vivent sur /compte, avec leur `n` et leur
// intervalle. Voir python/colver/web/stats.py pour le raisonnement complet.

const base = () => document.querySelector('base')?.getAttribute('href') || '/';

const BOT_LABELS = {
    dede: 'Dédé (IS-DD)',
    doudou: 'DouDou50',
    oracle_dd: 'Oracle (DD)',
    random: 'Aléatoire',
    smart: 'Smart (IS-MCTS)',
    naive: 'Naïf (IS-MCTS)',
    heuristic: 'Heuristique',
};

const TABS = [
    { id: 'elo', label: 'Elo', hint: 'le classement officiel' },
    { id: 'vie', label: 'Vie du site', hint: 'qui fait tourner la table' },
];

const TEMPLATE = `
<div class="compte-page classement-page">
    <div class="compte-card">
        <h2 class="compte-title">Classement</h2>
        <div class="pc-seg" id="cl-tabs" role="radiogroup" aria-label="Type de classement">
            ${TABS.map((t, i) => `
            <button class="pc-seg-btn" role="radio" data-tab="${t.id}"
                    aria-checked="${i === 0}">${t.label}<small>${t.hint}</small></button>`).join('')}
        </div>
        <div id="classement-body"><div class="an-loading">Chargement…</div></div>
    </div>
</div>`;

let me = null;

function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

// ---- Onglet Elo ------------------------------------------------------------

function renderElo(rows) {
    if (rows.length === 0) {
        return '<div class="history-empty">Aucune partie classée — ' +
            'lancez une partie en 2000 points !</div>';
    }
    let html = '<p class="salon-desc">Seules les <strong>parties en 2000 points</strong> comptent —' +
        " c'est le format des tournois. Les donnes seules et les parties en 1000 restent" +
        ' libres, mais ne sont pas classées. Il faut 5 parties pour apparaître ici.</p>';
    html += '<div class="cl-scroll"><table class="classement-table">' +
        '<tr><th>#</th><th></th><th class="cl-right">Elo</th><th class="cl-right">Parties</th></tr>';
    rows.forEach((r, i) => {
        const isBot = r.kind === 'bot';
        const isMe = me && !isBot && r.name === me.username;
        const name = isBot ? `${esc(BOT_LABELS[r.ref] || r.ref)} 🤖` : esc(r.name);
        html += `<tr class="${isMe ? 'cl-me' : ''} ${isBot ? 'cl-bot' : 'cl-human'}">` +
            `<td class="cl-rank">${i + 1}</td>` +
            `<td class="cl-name">${name}${isMe ? ' <span class="cl-you">(vous)</span>' : ''}</td>` +
            `<td class="cl-right cl-elo">${Math.round(r.elo)}</td>` +
            `<td class="cl-right cl-games">${r.games}</td></tr>`;
    });
    html += '</table></div>';
    // Un joueur sous le seuil ne se voit pas dans le tableau : sans cette ligne
    // il ne saurait pas pourquoi, et croirait à un bug.
    const st = me && me.stats && me.stats.elo;
    if (st && st.ranked === false) {
        const n = st.remaining;
        html += `<p class="salon-desc">Vous n'êtes pas encore classé : encore ` +
            `<strong>${n}</strong> partie${n > 1 ? 's' : ''} en 2000 points ` +
            `(classement provisoire ${Math.round(st.elo)}).</p>`;
    }
    return html;
}

// ---- Onglet Vie du site ----------------------------------------------------

function table(caption, note, head, rows) {
    if (rows.length === 0) return '';
    return `<h3 class="compte-subtitle">${caption}</h3>` +
        (note ? `<p class="salon-desc">${note}</p>` : '') +
        '<div class="cl-scroll"><table class="classement-table"><tr>' +
        head.map(h => `<th${h.right ? ' class="cl-right"' : ''}>${h.label}</th>`).join('') +
        '</tr>' + rows.join('') + '</table></div>';
}

function row(rank, name, cells) {
    const isMe = me && name === me.username;
    return `<tr class="${isMe ? 'cl-me' : ''} cl-human">` +
        `<td class="cl-rank">${rank}</td>` +
        `<td class="cl-name">${esc(name)}${isMe ? ' <span class="cl-you">(vous)</span>' : ''}</td>` +
        cells.map(c => `<td class="cl-right">${c}</td>`).join('') + '</tr>';
}

function renderVie(rows) {
    if (rows.length === 0) {
        return '<div class="history-empty">Personne n\'a encore joué de donne ' +
            'avec un compte — <a href="/jouer/humain">à vous !</a></div>';
    }

    // Assiduité : trié sur les donnes jouées, l'ordre que le serveur rend déjà.
    const assiduite = rows.map((r, i) => row(i + 1, r.name, [
        r.deals,
        r.days,
        r.streak > 0 ? `${r.streak} j` : '—',
        r.density ?? '—',
        r.matches_1000 + r.matches_2000 || '—',
    ]));

    // Faits d'armes : trié sur les capots réalisés. Un joueur qui n'en a aucun
    // et n'a jamais coinché ni pris n'a rien à y faire — la ligne serait vide.
    const armes = rows
        .filter(r => r.capots_for || r.capots_against || r.coinches || r.takes)
        .sort((a, b) => (b.capots_for - a.capots_for) || (b.takes - a.takes))
        .map((r, i) => row(i + 1, r.name, [
            r.capots_for || '—',
            r.capots_against || '—',
            r.coinches || '—',
            r.takes || '—',
        ]));

    return table(
        'Assiduité',
        'Des comptes, pas des taux : ces colonnes disent qui fait vivre la table, ' +
        'pas qui joue le mieux. Les jours sont comptés en UTC.',
        [{ label: '#' }, { label: '' }, { label: 'Donnes', right: true },
            { label: 'Jours', right: true }, { label: 'Série', right: true },
            { label: 'Donnes/j', right: true }, { label: 'Parties', right: true }],
        assiduite,
    ) + table(
        "Faits d'armes",
        'Capots réalisés et subis, coinches annoncées, contrats pris — au siège ' +
        'exact, jamais au camp : en solo, trois sièges sur quatre sont des bots.',
        [{ label: '#' }, { label: '' }, { label: 'Capots ✓', right: true },
            { label: 'Capots ✗', right: true }, { label: 'Coinches', right: true },
            { label: 'Prises', right: true }],
        armes,
    );
}

// ---- Montage ---------------------------------------------------------------

const cache = {};

async function load(tab) {
    if (cache[tab]) return cache[tab];
    const url = tab === 'elo' ? 'api/leaderboard' : 'api/leaderboard/vie';
    const resp = await fetch(`${base()}${url}`);
    if (!resp.ok) throw new Error('leaderboard');
    cache[tab] = await resp.json();
    return cache[tab];
}

async function show(tab) {
    const body = document.getElementById('classement-body');
    if (!body) return;
    document.querySelectorAll('#cl-tabs .pc-seg-btn').forEach(b => {
        b.setAttribute('aria-checked', String(b.dataset.tab === tab));
    });
    // L'onglet vit dans la query string, pas dans un fragment : le routeur
    // traite `#...` comme une URL héritée et le redirigerait.
    const url = new URL(location.href);
    if (tab === 'elo') url.searchParams.delete('t');
    else url.searchParams.set('t', tab);
    history.replaceState(history.state, '', url);

    body.innerHTML = '<div class="an-loading">Chargement…</div>';
    try {
        const rows = await load(tab);
        body.innerHTML = tab === 'elo' ? renderElo(rows) : renderVie(rows);
    } catch {
        body.innerHTML = '<div class="an-loading">Classement indisponible</div>';
    }
}

export async function mount(container) {
    container.innerHTML = TEMPLATE;
    Object.keys(cache).forEach(k => delete cache[k]);
    try {
        const meResp = await fetch(`${base()}api/me`);
        if (meResp.ok) {
            const blob = await meResp.json();
            me = blob.user ? { ...blob.user, stats: blob.stats } : null;
        }
    } catch { me = null; }

    document.getElementById('cl-tabs').addEventListener('click', (e) => {
        const btn = e.target.closest('.pc-seg-btn');
        if (btn) show(btn.dataset.tab);
    });

    const wanted = new URL(location.href).searchParams.get('t');
    await show(TABS.some(t => t.id === wanted) ? wanted : 'elo');
}

export function unmount() {}
