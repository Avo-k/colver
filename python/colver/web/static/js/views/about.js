// À propos view — static documentation content

const TEMPLATE = `
<div id="docs-content">
    <h2>Colver <span class="docs-subtitle">Moteur de Belote Contr\u00e9e</span></h2>
    <p class="docs-intro">
        Colver est un environnement de Belote Contr\u00e9e haute performance con\u00e7u pour la recherche en apprentissage par renforcement.
        Le moteur est \u00e9crit en Rust, capable de plus d'un million de simulations par seconde.
        Cette interface web permet d'y jouer \u2014 seul contre les bots ou \u00e0 plusieurs en salon \u2014,
        de les regarder s'affronter, et surtout de revenir sur une donne pour savoir ce qu'il
        aurait fallu annoncer et jouer.
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
            D\u00e9d\u00e9 tire des donnes compatibles avec ce que son si\u00e8ge peut savoir, r\u00e9sout chacune
            exactement avec un solveur alpha-b\u00eata double-dummy, et joue la carte qui s'en sort le
            mieux en moyenne. Les contraintes dures \u2014 coupes r\u00e9v\u00e9l\u00e9es par le jeu, plafond d'atout,
            cartes d\u00e9j\u00e0 tomb\u00e9es \u2014 sont des faits et s'appliquent toujours.
            IS-DD se prononce \u00ab is D\u00e9d\u00e9 \u00bb \u2014 d'o\u00f9 le surnom.
        </p>
        <p>
            Toute la difficult\u00e9 est dans le tirage : des mains adverses tir\u00e9es au hasard donneraient
            des mondes que personne n'annoncerait ni ne jouerait ainsi. Les mondes viennent donc de
            <em>Playgen</em> (voir plus bas), qui les tire d'une distribution apprise. Sans lui,
            D\u00e9d\u00e9 retombe sur un tirage uniforme sous contraintes \u2014 et le dit dans ses statistiques
            de d\u00e9cision.
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
            \u00e9valuation de la main (grosses cartes d'atout, longueur, as annexes),
            seuils de palier, plafonds d'annonce et conditions de coinche.
            Utilis\u00e9 pendant l'entra\u00eenement des r\u00e9seaux de neurones et disponible comme alternative.
        </p>
    </div>

    <h3>Croyances</h3>

    <div class="docs-section" id="doc-playgen">
        <h4>Playgen</h4>
        <p>
            Un transformer causal entra\u00een\u00e9 \u00e0 <em>prolonger</em> une donne \u00e0 partir du seul pr\u00e9fixe
            visible par un observateur : les annonces entendues, les cartes tomb\u00e9es, sa propre main.
            Le d\u00e9rouler jusqu'au bout r\u00e9v\u00e8le les mains cach\u00e9es \u2014 un d\u00e9roulement est donc une donne
            compl\u00e8te, plausible, tir\u00e9e d'une distribution apprise plut\u00f4t que d'un m\u00e9lange uniforme.
        </p>
        <p>
            C'est de l\u00e0 que viennent les mondes de <em>D\u00e9d\u00e9</em>, et c'est ce que montre la page
            Croyances : en agr\u00e9geant beaucoup de d\u00e9roulements, on obtient pour chaque carte non vue
            la probabilit\u00e9 qu'elle soit dans chaque main. Le mod\u00e8le sait aussi annoncer, ce qui lui
            permet de tirer des donnes en <em>cours</em> d'ench\u00e8re et, accessoirement, de servir
            d'ench\u00e9risseur (son avis appara\u00eet dans la revue d'ench\u00e8re de Rejouer).
        </p>
    </div>

    <h3>Pages</h3>

    <div class="docs-section" id="doc-jouer">
        <h4>Humain vs IA</h4>
        <p>
            Jouez en Sud contre trois bots, aux r\u00e8gles officielles FFB : ench\u00e8res avec
            coinche/surcoinche, puis 8 plis. Deux r\u00e9glages seulement, choisis avant la donne.
        </p>
        <p>
            Le <strong>rythme</strong> d\u00e9signe \u00e0 la fois le tempo d'affichage et le bot assis aux
            quatre places : <em>Standard</em> = D\u00e9d\u00e9, \u2248 40 s la donne ; <em>Rapide</em> = DouDou50,
            \u2248 15 s. Les deux vont ensemble et ce n'est pas cosm\u00e9tique \u2014 une recherche IS-DD co\u00fbte du
            temps r\u00e9el \u00e0 chaque coup, donc un tempo rapide n'est honn\u00eate que derri\u00e8re un bot qui
            r\u00e9pond instantan\u00e9ment. Les quatre si\u00e8ges jouent le m\u00eame bot : une table o\u00f9 le partenaire
            est plus faible que les adversaires ne dirait rien de votre partie.
        </p>
        <p>
            Le <strong>format</strong> est une donne s\u00e8che, ou une partie en 1000 / 2000 points.
            Dans une partie, le score cumul\u00e9 est transmis aux bots \u2014 et l'ench\u00e9risseur le lit :
            il n'annonce pas la m\u00eame chose \u00e0 900-200 qu'\u00e0 0-0. Connect\u00e9, vous pouvez quitter une
            partie en cours et la reprendre plus tard (une donne <em>entam\u00e9e</em>, elle, ne se
            reprend pas : les bots n'ont pas de m\u00e9moire persistante, la donne est conc\u00e9d\u00e9e et le
            score de partie conserv\u00e9).
        </p>
        <p>
            Vos cartes partent instantan\u00e9ment au clic. La pause appartient \u00e0 la position qui
            <em>pr\u00e9c\u00e8de</em> un coup, et le bot r\u00e9fl\u00e9chit dedans plut\u00f4t que par-dessus. Quand passer
            est la seule annonce possible, le serveur passe pour vous ; le dernier pli, o\u00f9 plus
            personne n'a de choix, se d\u00e9roule tout seul ; et la derni\u00e8re lev\u00e9e reste 2 s \u00e0 l'\u00e9cran
            avant le panneau de fin. Pendant le jeu, le contrat au centre du bandeau se d\u00e9plie
            (chevron \u25be) pour revoir toute l'ench\u00e8re. Cliquez la zone CFN sous la barre de score
            pour copier la position (partage ou signalement de bug).
        </p>
    </div>

    <div class="docs-section" id="doc-salon">
        <h4>Salon multijoueur</h4>
        <p>
            Cr\u00e9ez un salon, partagez son code \u00e0 4 caract\u00e8res, et jouez \u00e0 plusieurs humains autour de
            la m\u00eame table \u2014 les si\u00e8ges libres sont tenus par des bots. L'h\u00f4te choisit le rythme et le
            format, et lance la donne quand tout le monde est assis.
        </p>
        <p>
            Chaque joueur ne re\u00e7oit que sa propre main, et la table est pivot\u00e9e pour qu'il soit
            toujours assis en bas : personne ne peut lire le jeu d'un autre, m\u00eame en inspectant les
            messages. Les si\u00e8ges sont li\u00e9s aux comptes, donc une d\u00e9connexion ne co\u00fbte pas la place.
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

    <div class="docs-section" id="doc-rejouer">
        <h4>Rejouer</h4>
        <p>
            Parcourez et rejouez les donnes pass\u00e9es \u2014 jou\u00e9es, observ\u00e9es ou re\u00e7ues par lien \u2014
            coup par coup, avec les quatre mains visibles. L'adresse de la page suit le coup affich\u00e9 :
            un coup pr\u00e9cis se partage tel quel.
        </p>
        <p>
            Deux analyses s'ajoutent par-dessus, calcul\u00e9es s\u00e9par\u00e9ment pour que la lente ne retarde
            pas la rapide. La premi\u00e8re donne le <strong>co\u00fbt exact de chaque carte</strong> (un solve
            double-dummy sur la vraie donne) et une revue de l'ench\u00e8re portant deux avis : celui de
            <em>Bid V6</em>, la r\u00e9f\u00e9rence, et celui de <em>Playgen</em>, qui est un mod\u00e8le du monde
            plut\u00f4t qu'un ench\u00e9risseur. La seconde dit ce que <strong>DouDou50, l'Oracle et D\u00e9d\u00e9</strong>
            auraient jou\u00e9 \u00e0 chaque carte non forc\u00e9e, quel qu'en soit l'auteur ; elle s'affiche au fil
            du calcul.
        </p>
        <p>
            Chaque annonce et chaque carte porte un lien vers sa page d'analyse d\u00e9di\u00e9e, avec le
            chemin du retour vers le coup exact d'o\u00f9 vous veniez.
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
            \u00ab&nbsp;Analyser une autre annonce&nbsp;\u00bb relance le <em>Jeu r\u00e9el</em> en for\u00e7ant l'annonce
            de votre choix&nbsp;: le reste de l'ench\u00e8re et tout le jeu restent pilot\u00e9s par les r\u00e9seaux,
            ce qui permet de comparer deux annonces sur un pied d'\u00e9galit\u00e9. Chaque annonce analys\u00e9e
            ouvre son propre <strong>onglet</strong> au lieu d'\u00e9craser la pr\u00e9c\u00e9dente&nbsp;; le
            <em>Jeu parfait</em>, lui, est partag\u00e9 par tous les onglets, puisque l'Oracle r\u00e9sout les
            quatre couleurs sans rien savoir de ce que vous annoncez. Une seule simulation tourne \u00e0 la
            fois&nbsp;: ouvrir un onglet interrompt celle en cours, qui garde son r\u00e9sultat partiel et
            propose de la relancer.
        </p>
        <p>
            Les mains \u00e9valu\u00e9es s'empilent dans la barre lat\u00e9rale <strong>Mains analys\u00e9es</strong> \u2014
            dans votre navigateur, rien n'est envoy\u00e9. Une entr\u00e9e retient la main <em>et</em> les
            ench\u00e8res qui la pr\u00e9c\u00e9daient&nbsp;: la m\u00eame main apr\u00e8s \u00ab&nbsp;100\u2665&nbsp;\u00bb n'est pas la
            m\u00eame question.
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

    <div class="docs-section" id="doc-jeu-carte">
        <h4>Jeu de la carte <span class="docs-tag subtle">Analyse</span></h4>
        <p>
            Une ligne par carte jouable, à une position précise. La page répond à <strong>deux
            questions différentes</strong>, et tout l'intérêt est de ne pas les confondre.
        </p>
        <p>
            <strong>Les mondes de l'information set</strong> (colonnes de gauche) : on tire des donnes
            compatibles avec ce que le siège qui joue pouvait <em>savoir</em> — sa main, les cartes
            déjà tombées, les coupes que le jeu a révélées — et chacune est résolue en double-dummy.
            <em>Meilleure</em> est la part des mondes où cette carte est la meilleure ; <em>Risque</em>
            la part des mondes où elle perd au moins 10 points. C'est ici qu'on juge la décision.
        </p>
        <p>
            <strong>Le vrai monde</strong> (colonne <em>Coût</em>) : un seul solve, sur la donne telle
            qu'elle était vraiment. C'est exact, et c'est ce que montre déjà Rejouer — mais ça ne dit
            pas si le choix était raisonnable. Une carte deuxième dans la vraie donne mais meilleure
            dans 70 % des mondes était un <strong>bon</strong> coup contre de la malchance.
        </p>
        <p>
            <strong>Jeu réel</strong> : la carte est forcée, puis <em>DouDou50</em> finit la donne aux
            quatre places. L'écart est en points de donne marqués (contrat compris), Nord-Sud moins
            Est-Ouest — <em>pas</em> la même échelle que les points cartes double-dummy des colonnes de
            gauche, les deux ne se soustraient pas. La dernière colonne est lue du côté du siège qui
            joue : en défense, c'est le contrat <em>chuté</em> qui est l'issue favorable.
        </p>
        <p>
            Les cartes équivalentes sont regroupées : départager le 7 et le 8 d'une couleur sans carte
            intermédiaire dehors dépenserait le budget deux fois pour une seule réponse. On arrive sur
            cette page depuis <strong>Rejouer</strong>, où chaque carte porte un lien, ou en collant le
            CFN d'une partie.
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

    <div class="docs-section" id="doc-regles">
        <h4>R\u00e8gles du jeu</h4>
        <p>
            <a href="/regles">Les r\u00e8gles</a> que le moteur applique, en une page&nbsp;: ce qu'on a
            le droit de jouer, ce qui compte pour le contrat, comment le score se calcule.
        </p>
        <p>
            Il n'existe pas de r\u00e8glement officiel unique de la belote contr\u00e9e \u2014 la f\u00e9d\u00e9ration en
            a publi\u00e9 \u00e0 elle seule quatre r\u00e9dactions incompatibles, une f\u00e9d\u00e9ration concurrente
            existe depuis 1997, et le Championnat de France de Cannes ne suit ni l'une ni l'autre.
            Chaque choix a donc \u00e9t\u00e9 tranch\u00e9, et l'est publiquement sur
            <a href="/regles/choix">Pourquoi ces r\u00e8gles</a>&nbsp;: qui dit la m\u00eame chose, qui dit
            le contraire, et \u2014 quand la question se pr\u00eate \u00e0 la mesure \u2014 ce que valent les chiffres.
            Le relev\u00e9 de sources complet vit dans <em>docs/rules-survey</em>.
        </p>
    </div>

    <div class="docs-section">
        <h4>Aide-m\u00e9moire, Guide des annonces, Marquer</h4>
        <p>
            Trois pages de r\u00e9f\u00e9rence, sans IA en marche. L'<strong>aide-m\u00e9moire</strong> donne
            l'ordre de force et la valeur des cartes (\u00e0 l'atout et hors atout), les points de la donne
            et les r\u00e8gles d'ench\u00e8re. Le <strong>guide des annonces</strong> traduit le r\u00e9seau en
            r\u00e8gles tenant sur une page \u2014 poids par carte, seuils selon la position, r\u00e8gle du miroir
            en d\u00e9fense \u2014 obtenues par distillation ML et d'accord avec lui dans 88 \u00e0 94&nbsp;% des cas.
            <strong>Marquer les points</strong> est un compteur pour vos vraies parties&nbsp;: il calcule
            la marque exacte (contrat, coinche, points faits, belote) et affiche la probabilit\u00e9 de
            victoire apr\u00e8s chaque manche. Tout y reste dans votre navigateur.
        </p>
    </div>

    <div class="docs-section" id="doc-classement">
        <h4>Classement</h4>
        <p>
            L'unit\u00e9 not\u00e9e est la <strong>partie en 2000&nbsp;points</strong> et rien d'autre&nbsp;:
            c'est le format des tournois, et c'est aussi celui sur lequel les bots sont \u00e9talonn\u00e9s.
            Une donne isol\u00e9e ou une partie en 1000 reste jouable, analysable et partageable \u2014 elle
            n'est simplement pas class\u00e9e. Il faut cinq parties pour appara\u00eetre, et abandonner vaut
            d\u00e9faite.
        </p>
        <p>
            L'Elo des bots est <em>fig\u00e9</em>&nbsp;: ils sont l'\u00e9chelle, pas des joueurs du
            classement. S'ils d\u00e9rivaient avec la population, l'arriv\u00e9e de joueurs plus faibles
            d\u00e9valuerait en silence tous les inscrits.
        </p>
        <p>
            Il n'y a <strong>qu'un</strong> classement, et c'est d\u00e9lib\u00e9r\u00e9. Sur quelques centaines de
            donnes, un taux de victoire porte une marge d'erreur d'une dizaine de points&nbsp;:
            ordonner des joueurs dessus publierait un classement de hasard. Ces chiffres-l\u00e0 d\u00e9crivent
            tr\u00e8s bien un joueur, mais un par un \u2014 ils sont donc sur <strong>Mes stats</strong>.
        </p>
    </div>

    <div class="docs-section" id="doc-stats">
        <h4>Mes stats</h4>
        <p>
            Votre portrait chiffr\u00e9, en deux moiti\u00e9s.
        </p>
        <p>
            Le haut est <strong>gratuit</strong> et s'affiche toujours&nbsp;: combien de donnes vous
            avez jou\u00e9es, votre s\u00e9rie de jours, \u00e0 quelle fr\u00e9quence vous prenez \u2014 et si c'est plut\u00f4t
            vous ou plut\u00f4t votre partenaire \u2014, votre couleur d'atout de pr\u00e9dilection, votre hauteur
            d'annonce moyenne, les belotes tomb\u00e9es dans votre main, les gens avec qui vous jouez le
            plus. Tout se d\u00e9duit de ce qui est d\u00e9j\u00e0 enregistr\u00e9&nbsp;: aucun calcul, aucune attente.
        </p>
        <p>
            Le bas est <strong>pay\u00e9</strong>, et ne se d\u00e9clenche qu'en appuyant sur le bouton. Chaque
            donne passe alors au solveur double-mort, qui rejoue chaque d\u00e9cision en voyant les quatre
            mains et dit ce que le meilleur coup rapportait. On en tire les points perdus par
            d\u00e9cision et la part de coups <em>sans perte</em>. Comptez environ un quart de seconde par
            donne&nbsp;; le r\u00e9sultat sert aussi \u00e0 Rejouer, qui devient instantan\u00e9.
        </p>
        <p>
            Deux pr\u00e9cautions valent d'\u00eatre connues. \u00ab Sans perte \u00bb veut dire que le solveur ne
            valorise aucune autre carte plus haut, <em>pas</em> que vous avez jou\u00e9 sa carte
            pr\u00e9f\u00e9r\u00e9e&nbsp;: plus d'une position sur deux a plusieurs cartes \u00e9galement bonnes.
            Et les coups o\u00f9 une seule carte \u00e9tait jouable sont exclus du calcul \u2014 ce ne sont pas des
            d\u00e9cisions, et les compter gonflerait le score d'un tiers sans rien dire de vous.
        </p>
        <p>
            La <strong>couverture</strong> est affich\u00e9e en t\u00eate&nbsp;: c'est vous qui choisissez
            quand analyser, donc une moyenne calcul\u00e9e sur un dixi\u00e8me de vos donnes doit se lire comme
            telle.
        </p>
    </div>

    <div class="docs-section">
        <h4>Compte</h4>
        <p>
            Le compte est facultatif&nbsp;: on peut jouer sans. Il sert \u00e0 rattacher vos donnes \u00e0 vous \u2014
            donc \u00e0 les retrouver dans Rejouer, \u00e0 reprendre une partie commenc\u00e9e, \u00e0 garder votre si\u00e8ge
            en salon apr\u00e8s une d\u00e9connexion, et \u00e0 \u00eatre class\u00e9.
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
            &amp; <a href="https://claude.ai" target="_blank" rel="noopener">Claude</a>
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
