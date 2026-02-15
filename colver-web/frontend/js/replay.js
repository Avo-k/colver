// Replay mode logic

let replayData = null;
let replayStep = 0;
let replayTotal = 0;
let replayActions = [];
let autoPlayTimer = null;

document.getElementById('gen-replay').addEventListener('click', () => {
    send({ type: 'generate_replay', ai: 'naive', time_ms: 20 });
    document.getElementById('gen-replay').disabled = true;
    document.getElementById('gen-replay').textContent = 'Generation...';
});

document.getElementById('replay-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        try {
            const log = JSON.parse(reader.result);
            send({ type: 'load_replay', log });
        } catch (err) {
            alert('Fichier JSON invalide');
        }
    };
    reader.readAsText(file);
});

document.getElementById('replay-start').addEventListener('click', () => seekReplay(0));
document.getElementById('replay-prev').addEventListener('click', () => seekReplay(replayStep - 1));
document.getElementById('replay-next').addEventListener('click', () => seekReplay(replayStep + 1));
document.getElementById('replay-end').addEventListener('click', () => seekReplay(replayTotal));

document.getElementById('replay-auto').addEventListener('click', () => {
    if (autoPlayTimer) {
        clearInterval(autoPlayTimer);
        autoPlayTimer = null;
        document.getElementById('replay-auto').textContent = '\u25B6';
    } else {
        document.getElementById('replay-auto').textContent = '\u23F8';
        autoPlayTimer = setInterval(() => {
            if (replayStep >= replayTotal) {
                clearInterval(autoPlayTimer);
                autoPlayTimer = null;
                document.getElementById('replay-auto').textContent = '\u25B6';
                return;
            }
            seekReplay(replayStep + 1);
        }, 600);
    }
});

function seekReplay(step) {
    step = Math.max(0, Math.min(step, replayTotal));
    replayStep = step;
    send({ type: 'replay_seek', step });
}

function renderReplayState(state, step) {
    document.getElementById('replay-score-ns').textContent = `NS : ${state.points[0]} (${state.tricks_won[0]}P)`;
    document.getElementById('replay-score-ew').textContent = `EO : ${state.points[1]} (${state.tricks_won[1]}P)`;
    document.getElementById('replay-contract-display').textContent = contractStr(state.contract);

    const handEls = {
        0: document.getElementById('replay-hand-north'),
        1: document.getElementById('replay-hand-east'),
        2: document.getElementById('replay-hand-south'),
        3: document.getElementById('replay-hand-west'),
    };
    for (let seat = 0; seat < 4; seat++) {
        renderHand(handEls[seat], state.hands[seat]);
    }

    renderTrick('replay-trick', state.current_trick);

    document.getElementById('replay-step').textContent = `${step} / ${replayTotal}`;
    renderReplayLog(step);
}

function renderReplayLog(currentStep) {
    const el = document.getElementById('replay-action-log');
    el.innerHTML = '';
    const names = ['N', 'E', 'S', 'O'];
    for (let i = 0; i < replayActions.length; i++) {
        const a = replayActions[i];
        const div = document.createElement('div');
        div.className = 'log-entry' + (i === currentStep - 1 ? ' current' : '');
        const name = a.name || actionName(a.action, a.phase);
        div.textContent = `${i + 1}. ${names[a.player]} : ${name}`;
        el.appendChild(div);
    }
    const current = el.querySelector('.current');
    if (current) current.scrollIntoView({ block: 'nearest' });
}

// Message handlers
onMessage('replay_loaded', (data) => {
    replayData = data;
    replayTotal = data.total_steps;
    replayActions = data.actions || [];
    replayStep = 0;
    document.getElementById('replay-transport').classList.remove('hidden');
    document.getElementById('replay-table').classList.remove('hidden');
    document.getElementById('gen-replay').disabled = false;
    document.getElementById('gen-replay').textContent = 'Generer une partie';
    renderReplayState(data.state, 0);
});

onMessage('replay_state', (data) => {
    replayStep = data.step;
    renderReplayState(data.state, data.step);
});

onMessage('generated_replay', (data) => {
    send({ type: 'load_replay', log: data.log });
});
