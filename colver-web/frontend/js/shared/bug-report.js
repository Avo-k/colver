// Bug report modal logic

let currentGameId = null;
let currentActionIdx = 0;

export function setGameId(id) { currentGameId = id; }
export function getGameId() { return currentGameId; }
export function setActionIdx(idx) { currentActionIdx = idx; }

export function openBugReport() {
    if (!currentGameId) return;
    document.getElementById('report-game-label').textContent = `Partie : ${currentGameId}`;
    document.getElementById('report-message').value = '';
    document.getElementById('report-status').textContent = '';
    document.getElementById('report-modal').classList.remove('hidden');
    document.getElementById('report-message').focus();
}

export function initBugReportModal() {
    document.getElementById('report-cancel').addEventListener('click', () => {
        document.getElementById('report-modal').classList.add('hidden');
    });

    document.getElementById('report-submit').addEventListener('click', async () => {
        const message = document.getElementById('report-message').value.trim();
        if (!message) return;
        const statusEl = document.getElementById('report-status');
        statusEl.textContent = 'Envoi...';
        try {
            const base = document.querySelector('base')?.getAttribute('href') || '/';
            const resp = await fetch(`${base}api/games/${currentGameId}/report`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, action_idx: currentActionIdx }),
            });
            if (resp.ok) {
                statusEl.textContent = 'Envoye !';
                setTimeout(() => {
                    document.getElementById('report-modal').classList.add('hidden');
                }, 1000);
            } else {
                statusEl.textContent = 'Erreur';
            }
        } catch (e) {
            statusEl.textContent = 'Erreur reseau';
        }
    });

    document.getElementById('report-modal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) {
            document.getElementById('report-modal').classList.add('hidden');
        }
    });
}
