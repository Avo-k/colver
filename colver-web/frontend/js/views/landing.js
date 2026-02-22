// Landing page — mode cards hub

import { navigateTo } from '../router.js';

const TEMPLATE = `
<div class="landing">
    <div class="landing-hero">
        <div class="landing-logo">
            <div class="landing-cards-fan">
                <div class="fan-card fan-1"></div>
                <div class="fan-card fan-2"></div>
                <div class="fan-card fan-3"></div>
            </div>
        </div>
        <h2 class="landing-title">Colver</h2>
        <p class="landing-subtitle">Belote Contr\u00e9e \u2014 Moteur IA haute performance</p>
    </div>

    <div class="landing-grid">
        <div class="mode-card" data-route="/jouer/humain">
            <div class="mode-icon">\u2694\uFE0F</div>
            <h3 class="mode-title">Jouer</h3>
            <p class="mode-desc">Affrontez l'IA en Belote Contr\u00e9e. Choisissez vos adversaires et votre partenaire.</p>
            <span class="mode-tag">Humain vs IA</span>
        </div>

        <div class="mode-card" data-route="/jouer/ia">
            <div class="mode-icon">\uD83D\uDC41\uFE0F</div>
            <h3 class="mode-title">Regarder</h3>
            <p class="mode-desc">Observez des parties IA vs IA avec statistiques en temps r\u00e9el et Q-values.</p>
            <span class="mode-tag">IA vs IA</span>
        </div>

        <div class="mode-card" data-route="/analyse/rejouer">
            <div class="mode-icon">\u23EA</div>
            <h3 class="mode-title">Rejouer</h3>
            <p class="mode-desc">Parcourez l'historique et rejouez vos parties pas \u00e0 pas.</p>
            <span class="mode-tag">Analyse</span>
        </div>

        <div class="mode-card" data-route="/analyse/annonces">
            <div class="mode-icon">\uD83C\uDCCF</div>
            <h3 class="mode-title">Annonces</h3>
            <p class="mode-desc">Construisez une main et \u00e9valuez les ench\u00e8res avec le r\u00e9seau de neurones.</p>
            <span class="mode-tag">Outils</span>
        </div>

        <div class="mode-card" data-route="/problemes/annonce">
            <div class="mode-icon">\uD83C\uDFAF</div>
            <h3 class="mode-title">Probl\u00e8mes d'annonce</h3>
            <p class="mode-desc">Entra\u00eenez-vous aux ench\u00e8res avec des probl\u00e8mes g\u00e9n\u00e9r\u00e9s al\u00e9atoirement.</p>
            <span class="mode-tag">Entra\u00eenement</span>
        </div>

        <div class="mode-card" data-route="/problemes/jeu">
            <div class="mode-icon">\u2660</div>
            <h3 class="mode-title">Probl\u00e8mes de jeu</h3>
            <p class="mode-desc">Pratiquez le jeu de la carte face \u00e0 l'Oracle DD et D\u00e9d\u00e9.</p>
            <span class="mode-tag">Entra\u00eenement</span>
        </div>
    </div>

    <div class="landing-footer">
        <a href="#/about" class="landing-about-link">\u00c0 propos de Colver</a>
    </div>
</div>
`;

export function mount(container) {
    container.innerHTML = TEMPLATE;

    // Bind card clicks
    container.querySelectorAll('.mode-card[data-route]').forEach(card => {
        card.addEventListener('click', () => {
            navigateTo(card.dataset.route);
        });
    });
}

export function unmount() {}
