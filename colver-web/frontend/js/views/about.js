// À propos view — static documentation content

const TEMPLATE = `
<div id="docs-content">
    <h2>Colver <span class="docs-subtitle">Moteur de Belote Contr\u00e9e</span></h2>
    <p class="docs-intro">
        Colver est un environnement de Belote Contr\u00e9e haute performance con\u00e7u pour la recherche en apprentissage par renforcement.
        Le moteur est \u00e9crit en Rust, capable de plus d'un million de simulations par seconde.
        Cette interface web permet de jouer contre diff\u00e9rents agents IA, de les observer s'affronter et de cr\u00e9er des donnes personnalis\u00e9es.
    </p>
    <p class="docs-link">
        Code source : <a href="https://github.com/Avo-k/colver" target="_blank" rel="noopener">github.com/Avo-k/colver</a>
    </p>

    <h3>Modes de jeu</h3>

    <div class="docs-section">
        <h4>Jouer</h4>
        <p>
            Jouez en Sud contre des adversaires IA. Choisissez l'IA pour vos adversaires (Est/Ouest) et votre partenaire (Nord) ind\u00e9pendamment.
            La partie suit les r\u00e8gles officielles FFB de la Belote Contr\u00e9e : phase d'ench\u00e8res avec coinche/surcoinche, puis 8 plis de jeu.
        </p>
        <p>
            Le curseur <strong>Pause</strong> (1\u20138s) contr\u00f4le le d\u00e9lai entre les cartes \u2014
            ajustable \u00e0 tout moment pendant la partie. Vos cartes sont jou\u00e9es instantan\u00e9ment au clic ;
            la pause simule le temps de r\u00e9flexion des adversaires.
            Cliquez sur la zone CFN sous la barre de score pour copier la position actuelle (partage ou signalement de bug).
        </p>
    </div>

    <div class="docs-section">
        <h4>Regarder</h4>
        <p>
            Assistez \u00e0 des parties IA contre IA avec visibilit\u00e9 totale sur toutes les mains.
            Assignez un agent diff\u00e9rent \u00e0 chacune des 4 places, puis avancez action par action,
            pli par pli, ou utilisez la lecture automatique. Le panneau de stats affiche les Q-values, scores DD ou \u00e9valuations de main pour chaque d\u00e9cision.
            Collez une cha\u00eene <strong>CFN</strong> pour charger une position sp\u00e9cifique et observer l'IA jouer \u00e0 partir de l\u00e0.
        </p>
    </div>

    <div class="docs-section">
        <h4>Rejouer</h4>
        <p>
            Parcourez et rejouez les parties pass\u00e9es (jou\u00e9es ou observ\u00e9es). L'historique liste les parties r\u00e9centes \u2014
            cliquez sur une entr\u00e9e pour la rejouer pas \u00e0 pas avec les contr\u00f4les de navigation.
            Recherchez par identifiant de partie pour en retrouver une sp\u00e9cifique. Les parties incompl\u00e8tes sont clairement signal\u00e9es.
        </p>
    </div>

    <div class="docs-section">
        <h4>Donner</h4>
        <p>
            Cr\u00e9ez une donne personnalis\u00e9e en glissant les cartes de la palette vers les 4 zones joueurs
            (ou utilisez \u00ab Donne al\u00e9atoire \u00bb pour compl\u00e9ter la distribution). Choisissez le donneur,
            enregistrez la donne, puis cliquez \u00ab Regarder \u00bb pour observer l'IA la jouer en direct.
        </p>
    </div>

    <div class="docs-section">
        <h4>Annonces</h4>
        <p>
            Composez une main de 8 cartes en cliquant sur la palette, choisissez votre position dans le tour d'ench\u00e8res
            (combien de passes ont pr\u00e9c\u00e9d\u00e9 votre tour), puis cliquez \u00ab \u00c9valuer \u00bb pour voir ce que
            <em>Le Bide \u00e0 D\u00e9d\u00e9</em> annoncerait \u2014 avec les Q-values du r\u00e9seau de neurones pour chaque action possible.
        </p>
    </div>

    <h3>Agents IA</h3>
    <p class="docs-intro">
        Tous les agents portent des surnoms fran\u00e7ais. Les ench\u00e8res utilisent un r\u00e9seau de neurones entra\u00een\u00e9 par double-dummy (Le Bide \u00e0 D\u00e9d\u00e9).
        C'est le jeu de la carte qui les diff\u00e9rencie : r\u00e9seau de neurones, solveur exact, ou recherche pond\u00e9r\u00e9e par croyances.
    </p>

    <div class="docs-section">
        <h4>DouDou27 <span class="docs-tag best">Recommand\u00e9</span></h4>
        <p>
            <em>DouDou</em> = le doudou de l'enfant \u2014 parce qu'il apprend en jouant avec lui-m\u00eame.
            <em>35</em> = entra\u00een\u00e9 pendant 35 millions d'\u00e9tapes.
        </p>
        <p>
            R\u00e9seau Q Deep Monte-Carlo entra\u00een\u00e9 par self-play (style DouZero).
            Un MLP \u00e0 3 couches (1024 unit\u00e9s cach\u00e9es, ~2.6M param\u00e8tres) prend une observation de dimension 415
            et produit les Q-values des 32 cartes en une seule passe \u2014 aucun arbre de recherche n\u00e9cessaire.
            L'inf\u00e9rence tourne en Rust pur \u00e0 ~1ms par d\u00e9cision. Agent le plus fort dans l'ensemble.
        </p>
    </div>

    <div class="docs-section">
        <h4>D\u00e9d\u00e9 (IS-DD) <span class="docs-tag">Fort</span></h4>
        <p>
            <em>D\u00e9d\u00e9</em> = diminutif de <em>Double-Dummy</em> (DD \u2192 D\u00e9d\u00e9), un surnom fran\u00e7ais classique.
            IS-DD = Information Set Double-Dummy.
        </p>
        <p>
            Maintient un mod\u00e8le probabiliste de croyances sur les cartes cach\u00e9es, mis \u00e0 jour apr\u00e8s chaque action
            via des contraintes dures (inf\u00e9rence de coupes, plafond d'atout) et des signaux faibles (patterns d'ench\u00e8res, conventions de jeu).
            \u00c9chantillonne des mains adverses plausibles pond\u00e9r\u00e9es par ces croyances, puis r\u00e9sout chaque monde exactement
            avec un solveur alpha-b\u00eata double-dummy \u2014 optimal par d\u00e9terminisation.
            \u00c9crase les anciens agents IS-MCTS (~65% de victoires).
        </p>
    </div>

    <div class="docs-section">
        <h4>Oracle (DD)</h4>
        <p>
            Solveur double-dummy en information parfaite qui voit les 4 mains \u2014 il <em>triche</em>.
            Utilise une recherche alpha-b\u00eata avec tables de transposition, PVS et coups tueurs pour calculer
            la carte optimale exacte en ~7ms (m\u00e9diane). Utile comme borne sup\u00e9rieure pour \u00e9valuer
            la qualit\u00e9 de jeu des autres agents : aucun agent r\u00e9aliste ne devrait battre l'Oracle de mani\u00e8re r\u00e9guli\u00e8re.
        </p>
    </div>

    <div class="docs-footer">
        <p>
            Cr\u00e9\u00e9 par <a href="https://github.com/Avo-k" target="_blank" rel="noopener">Avo-k</a>
            &amp; <a href="https://claude.ai" target="_blank" rel="noopener">Claude Opus 4.6</a>
        </p>
    </div>
</div>
`;

export function mount(container) {
    container.innerHTML = TEMPLATE;
}

export function unmount() {}
