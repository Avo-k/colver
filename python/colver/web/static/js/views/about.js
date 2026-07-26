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

    <h3>Jeu de la carte</h3>

    <div class="docs-section">
        <h4>Oracle (DD)</h4>
        <p>
            Solveur double-dummy en information parfaite qui voit les 4 mains \u2014 il <em>triche</em>.
            Utilise une recherche alpha-b\u00eata avec tables de transposition, PVS et killer moves pour calculer
            la carte optimale exacte en ~7ms (m\u00e9diane). Utile comme borne sup\u00e9rieure pour \u00e9valuer
            la qualit\u00e9 de jeu des autres agents : aucun agent r\u00e9aliste ne devrait battre l'Oracle de mani\u00e8re r\u00e9guli\u00e8re.
        </p>
    </div>

    <div class="docs-section">
        <h4>D\u00e9d\u00e9 (IS-DD) <span class="docs-tag">Fort</span></h4>
        <p>
            Maintient un mod\u00e8le probabiliste de croyances sur les cartes cach\u00e9es, mis \u00e0 jour apr\u00e8s chaque action
            via des contraintes dures (inf\u00e9rence de coupes, plafond d'atout) et des signaux faibles (patterns d'ench\u00e8res, conventions de jeu).
            \u00c9chantillonne des mains adverses plausibles pond\u00e9r\u00e9es par ces croyances, puis r\u00e9sout chaque monde exactement
            avec un solveur alpha-b\u00eata double-dummy \u2014 optimal par d\u00e9terminisation.
            IS-DD se prononce \u00ab is D\u00e9d\u00e9 \u00bb \u2014 d'o\u00f9 le surnom.
        </p>
    </div>

    <div class="docs-section">
        <h4>DouDou50 <span class="docs-tag best">Recommand\u00e9</span></h4>
        <p>
            R\u00e9seau Q Deep Monte-Carlo de type ResNet avec Dueling DQN, entra\u00een\u00e9 par self-play iteratif (Triforge).
            Architecture 411\u21921024\u00b3\u219232 avec skip connections, observation canonique (atout en slot 0, couleurs tri\u00e9es),
            entra\u00een\u00e9 50M pas. L'inf\u00e9rence tourne en Rust pur \u00e0 ~1ms par d\u00e9cision.
            Agent le plus fort dans l'ensemble.
        </p>
    </div>

    <h3>Ench\u00e8res</h3>

    <div class="docs-section">
        <h4>Bid V6 IS-DD <span class="docs-tag best">D\u00e9faut</span></h4>
        <p>
            R\u00e9seau de neurones d'ench\u00e8res utilis\u00e9 par tous les agents.
            Dueling DQN (117\u2192512\u00b3\u219243) avec observation score-aware v3 (scores de match cumul\u00e9s + 4 bits de belote en main),
            entra\u00een\u00e9 75M de pas sur points r\u00e9els IS-DD avec simulation de match compl\u00e8te
            (scores cumul\u00e9s, rotation du donneur, reset \u00e0 2000) et reward sensible \u00e0 la belote (Q+K d'atout en m\u00eame main = +20).
            Stabilisation : reward clipping, EMA des poids (&tau;=0.005), cosine LR decay.
            Succ\u00e8de \u00e0 <em>Bid V5 IS-DD</em> (113-dim, 25M pas) avec un \u00e9cart de +55-65% en match arena.
        </p>
    </div>

    <div class="docs-section">
        <h4>Ench\u00e8res heuristiques</h4>
        <p>
            Syst\u00e8me d'ench\u00e8res \u00e0 base de r\u00e8gles cod\u00e9es \u00e0 la main :
            \u00e9valuation de la main (honneurs d'atout, longueur, as annexes),
            seuils de palier, plafonds d'annonce et conditions de coinche.
            Utilis\u00e9 pendant l'entra\u00eenement des r\u00e9seaux de neurones et disponible comme alternative.
        </p>
    </div>

    <h3>Croyances</h3>

    <div class="docs-section">
        <h4>R\u00e9seau de croyances</h4>
        <p>
            R\u00e9seau de neurones qui pr\u00e9dit, pour chaque carte non visible, la probabilit\u00e9 qu'elle se trouve
            dans chaque main adverse. Utilis\u00e9 par D\u00e9d\u00e9 pour \u00e9chantillonner des distributions de cartes
            plus r\u00e9alistes que le hasard uniforme.
        </p>
    </div>

    <h3>Pages</h3>

    <div class="docs-section">
        <h4>Humain vs IA</h4>
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
        <h4>IA vs IA</h4>
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

    <div class="docs-section" id="doc-annonces">
        <h4>Annonces</h4>
        <p>
            Composez une main de 8 cartes en cliquant sur la palette, choisissez votre position dans le tour d'ench\u00e8res
            (combien de passes ont pr\u00e9c\u00e9d\u00e9 votre tour), puis cliquez \u00ab \u00c9valuer \u00bb pour voir ce que
            <em>Bid V6 IS-DD</em> annoncerait \u2014 avec les Q-values du r\u00e9seau de neurones pour chaque action possible.
            Ce r\u00e9seau a \u00e9t\u00e9 entra\u00een\u00e9 par renforcement sur des millions de donnes.
        </p>
        <p>
            <strong>Facteurs cl\u00e9s</strong> d\u00e9compose la d\u00e9cision en caract\u00e9ristiques lisibles (longueur d'atout,
            cartes \u00e0 points, coupes\u2026) \u00e0 l'aide d'un mod\u00e8le XGBoost distill\u00e9 du r\u00e9seau. Il <em>approxime</em>
            le r\u00e9seau : ces contributions ne sortent <strong>pas</strong> du r\u00e9seau lui-m\u00eame et peuvent diverger
            de sa vraie d\u00e9cision.
        </p>
        <p>
            Deux tableaux \u00e9valuent ensuite la main sur des centaines de distributions tir\u00e9es au hasard \u2014
            vos 8 cartes sont fix\u00e9es, les 24 autres sont redistribu\u00e9es \u00e0 chaque simulation. Ils r\u00e9pondent
            \u00e0 deux questions diff\u00e9rentes\u00a0: <em>ce contrat est-il tenable\u00a0?</em> et <em>que se passe-t-il vraiment\u00a0?</em>
        </p>
        <p>
            Le s\u00e9lecteur <strong>Simulations</strong> fixe le nombre de donnes tir\u00e9es. Plus il est \u00e9lev\u00e9,
            plus les chiffres sont stables \u2014 et plus le calcul est long. \u00ab\u00a0Analyser une autre annonce\u00a0\u00bb
            relance le <em>Jeu r\u00e9el</em> en for\u00e7ant votre annonce\u00a0: le reste de l'ench\u00e8re et tout le jeu
            restent pilot\u00e9s par les r\u00e9seaux, ce qui permet de comparer deux annonces sur un pied d'\u00e9galit\u00e9.
        </p>
    </div>

    <div class="docs-section" id="doc-jeu-parfait">
        <h4>Jeu parfait <span class="docs-tag subtle">Annonces</span></h4>
        <p>
            Chaque distribution tir\u00e9e est r\u00e9solue par l'<em>Oracle</em>, le solveur double-dummy\u00a0:
            il voit les 4 mains et calcule le r\u00e9sultat exact si tout le monde jouait parfaitement.
            Chaque cellule indique le pourcentage de donnes o\u00f9 le contrat est <strong>r\u00e9alisable</strong>.
        </p>
        <p>
            C'est un <strong>plafond th\u00e9orique</strong>, pas une pr\u00e9diction\u00a0: personne ne voit le jeu adverse
            \u00e0 une vraie table, donc le taux de r\u00e9ussite r\u00e9el sera toujours plus bas. Le tableau sert
            \u00e0 jauger le potentiel brut de la main, et ne d\u00e9pend pas de ce que vous annoncez.
        </p>
    </div>

    <div class="docs-section" id="doc-jeu-reel">
        <h4>Jeu r\u00e9el <span class="docs-tag subtle">Annonces</span></h4>
        <p>
            Les m\u00eames distributions, mais jou\u00e9es pour de bon\u00a0: l'ench\u00e8re compl\u00e8te est men\u00e9e par le r\u00e9seau
            d'annonces aux 4 places, puis les 8 plis sont jou\u00e9s par <em>DouDou50</em>. Aucun de ces joueurs
            ne voit les cartes des autres \u2014 ils se trompent, sous-annoncent, chutent. C'est donc
            ce qui arrive <em>vraiment</em> avec cette main, et non ce qui serait possible.
        </p>
        <p>
            Le chiffre en haut est le principal\u00a0: la part des donnes o\u00f9 <strong>Nord-Sud marque plus
            qu'Est-Ouest</strong>, accompagn\u00e9e de l'\u00e9cart de points moyen par donne. Les donnes pass\u00e9es
            (personne ne prend) comptent comme nulles.
        </p>
        <p>
            Le tableau d\u00e9taille, par couleur et par palier, le taux de contrats <strong>r\u00e9ussis</strong>,
            avec dessous le nombre de donnes observ\u00e9es. Les cellules peu observ\u00e9es sont estomp\u00e9es\u00a0:
            leur chiffre est moins fiable. La couleur ne suit pas le pourcentage brut mais la borne
            basse d'un intervalle de confiance (Wilson) \u2014 une cellule \u00e0 100\u00a0% sur 2 donnes reste prudente.
            Le filtre <strong>Contrats pris par</strong> s\u00e9pare vos contrats de ceux de l'adversaire.
        </p>
    </div>

    <div class="docs-section">
        <h4>Croyances</h4>
        <p>
            Visualisez comment <em>Playgen</em> pr\u00e9dit la localisation des cartes au fil d'une partie.
            G\u00e9n\u00e9rez une partie al\u00e9atoire, avancez pas \u00e0 pas, et observez les barres de probabilit\u00e9
            par carte avec marquage de la v\u00e9rit\u00e9 terrain et statistiques de pr\u00e9cision.
            Changez de perspective d'observateur (N/E/S/O) pour voir ce que chaque si\u00e8ge peut d\u00e9duire.
        </p>
    </div>

    <div class="docs-section">
        <h4>Probl\u00e8mes d'annonce</h4>
        <p>
            Probl\u00e8mes d'ench\u00e8res : voyez une main et l'historique des ench\u00e8res, puis trouvez la bonne annonce.
            L'IA \u00e9value votre r\u00e9ponse en comparant avec la recommandation de <em>Bid V6 IS-DD</em>.
        </p>
    </div>

    <div class="docs-section">
        <h4>Probl\u00e8mes de jeu</h4>
        <p>
            Probl\u00e8mes de jeu de la carte : voyez une position en cours de partie et trouvez la meilleure carte.
            Comparez votre choix au jeu optimal du solveur double-dummy.
        </p>
    </div>

    <h3>Remerciements</h3>
    <div class="docs-section">
        <p>
            Merci à <strong>Ronan Guillou</strong>, joueur de coinche aguerri, pour ses conseils avisés sur le jeu
            et pour avoir été le premier testeur — son bon sens a guidé de nombreux choix d'interface.
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

    // ?s=<section> : ancre venue d'un lien « ? » dans l'interface. Un vrai
    // fragment (#jeu-reel) serait avalé par le routeur, qui traite tout hash
    // comme une URL héritée.
    const target = new URLSearchParams(location.search).get('s');
    if (!target) return;
    const el = container.querySelector(`#doc-${CSS.escape(target)}`);
    if (!el) return;
    el.scrollIntoView({ block: 'center' });
    el.classList.add('docs-section--target');
}

export function unmount() {}
