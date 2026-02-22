// CFN box: click-to-copy widget

export function updateCfnBox(elementId, cfn) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (cfn) {
        el.textContent = cfn;
        el.classList.remove('hidden');
    } else {
        el.classList.add('hidden');
    }
}

export function initCfnBox(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.addEventListener('click', () => {
        const text = el.textContent;
        if (!text) return;
        navigator.clipboard.writeText(text).then(() => {
            el.classList.add('copied');
            const prev = el.textContent;
            el.textContent = 'Copie !';
            setTimeout(() => {
                el.textContent = prev;
                el.classList.remove('copied');
            }, 1000);
        });
    });
}
