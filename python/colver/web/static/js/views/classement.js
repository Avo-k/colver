// Classement — l'Elo, seul.
//
// Il n'y a délibérément qu'une chose ici. Une tentative d'onglet « Vie du site »
// (comptes de capots, jours joués, séries) a été retirée : sur trois lignes,
// ordonner des joueurs par leur assiduité n'apprend rien à personne, et poser un
// second tableau à côté de l'Elo invitait à les lire comme deux mesures
// comparables. Ces chiffres-là décrivent bien un joueur, mais un par un — ils
// vivent donc sur /stats.

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

const TEMPLATE = `
<div class="compte-page classement-page">
    <div class="compte-card">
        <h2 class="compte-title">Classement Elo</h2>
        <p class="salon-desc">Seules les <strong>parties en 2000 points</strong> comptent —
        c'est le format des tournois. Les donnes seules et les parties en 1000 restent
        libres, mais ne sont pas classées. Il faut 5 parties pour apparaître ici.</p>
        <div id="classement-body"><div class="an-loading">Chargement…</div></div>
        <p class="salon-desc">Vos propres chiffres — comment vous annoncez, à quel
        point vous jouez près de l'oracle — sont sur <a href="/stats">Mes stats</a>.</p>
    </div>
</div>`;

function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

export async function mount(container) {
    container.innerHTML = TEMPLATE;
    const body = document.getElementById('classement-body');
    try {
        const resp = await fetch(`${base()}api/leaderboard`);
        if (!resp.ok) throw new Error();
        const rows = await resp.json();
        if (rows.length === 0) {
            body.innerHTML = '<div class="history-empty">Aucune partie classée — ' +
                'lancez une partie en 2000 points !</div>';
            return;
        }
        let me = null;
        try {
            const meResp = await fetch(`${base()}api/me`);
            if (meResp.ok) {
                const blob = await meResp.json();
                me = blob.user ? { ...blob.user, stats: blob.stats } : null;
            }
        } catch { /* anonymous */ }

        let html = '<div class="cl-scroll"><table class="classement-table">' +
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
        body.innerHTML = html;
    } catch {
        body.innerHTML = '<div class="an-loading">Classement indisponible</div>';
    }
}

export function unmount() {}
