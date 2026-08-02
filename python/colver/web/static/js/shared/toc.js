// Sommaire des pages de lecture (/regles, /regles/choix).
//
// Les entrées sont des <button data-goto="id"> et non des <a href="#id"> :
// le routeur intercepte tout lien interne et traite un fragment comme une
// URL héritée, donc une vraie ancre serait avalée sans jamais défiler.

/**
 * Câble les boutons de sommaire d'un conteneur. Rend la fonction de
 * détachement, à appeler dans `unmount`.
 */
export function wireToc(container) {
    const onClick = (e) => {
        const btn = e.target.closest('button[data-goto]');
        if (!btn || !container.contains(btn)) return;
        scrollToSection(container, btn.dataset.goto);
    };
    container.addEventListener('click', onClick);
    return () => container.removeEventListener('click', onClick);
}

/**
 * Défile jusqu'à une section et la souligne. `highlightClass` est retirée
 * de toute autre section d'abord, pour qu'un second clic ne laisse pas
 * deux blocs allumés.
 */
export function scrollToSection(container, id, highlightClass) {
    if (!id) return false;
    const el = container.querySelector(`#${CSS.escape(id)}`);
    if (!el) return false;
    if (highlightClass) {
        container.querySelectorAll('.' + highlightClass)
            .forEach(n => n.classList.remove(highlightClass));
        el.classList.add(highlightClass);
    }
    el.scrollIntoView({ block: 'start', behavior: 'smooth' });
    return true;
}

/**
 * Cible d'arrivée passée en query (`?q=…`) plutôt qu'en fragment, pour la
 * même raison que ci-dessus. Le défilement est instantané : on arrive sur
 * la page, il n'y a rien à suivre des yeux.
 */
export function openQueryTarget(container, param, highlightClass) {
    const id = new URLSearchParams(location.search).get(param);
    if (!id) return;
    const el = container.querySelector(`#${CSS.escape(id)}`);
    if (!el) return;
    if (highlightClass) el.classList.add(highlightClass);
    el.scrollIntoView({ block: 'start' });
}
