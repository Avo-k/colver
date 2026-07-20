// Classement — Elo leaderboard mixing humans and bots

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
        <p class="salon-desc">Humains et IA, toutes donnes confondues (parties identifiées uniquement).</p>
        <div id="classement-body"><div class="an-loading">Chargement…</div></div>
    </div>
</div>`;

export async function mount(container) {
    container.innerHTML = TEMPLATE;
    const body = document.getElementById('classement-body');
    try {
        const resp = await fetch(`${base()}api/leaderboard`);
        if (!resp.ok) throw new Error();
        const rows = await resp.json();
        if (rows.length === 0) {
            body.innerHTML = '<div class="history-empty">Aucune partie classée — jouez-en une !</div>';
            return;
        }
        let me = null;
        try {
            const meResp = await fetch(`${base()}api/me`);
            me = meResp.ok ? (await meResp.json()).user : null;
        } catch { /* anonymous */ }

        let html = '<table class="classement-table">' +
            '<tr><th>#</th><th></th><th class="cl-right">Elo</th><th class="cl-right">Donnes</th></tr>';
        rows.forEach((r, i) => {
            const isBot = r.kind === 'bot';
            const isMe = me && !isBot && r.name === me.username;
            const name = isBot ? `${BOT_LABELS[r.ref] || r.ref} 🤖` : r.name;
            html += `<tr class="${isMe ? 'cl-me' : ''} ${isBot ? 'cl-bot' : 'cl-human'}">` +
                `<td class="cl-rank">${i + 1}</td>` +
                `<td class="cl-name">${name}${isMe ? ' <span class="cl-you">(vous)</span>' : ''}</td>` +
                `<td class="cl-right cl-elo">${Math.round(r.elo)}</td>` +
                `<td class="cl-right cl-games">${r.games}</td></tr>`;
        });
        html += '</table>';
        body.innerHTML = html;
    } catch {
        body.innerHTML = '<div class="an-loading">Classement indisponible</div>';
    }
}

export function unmount() {}
