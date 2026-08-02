// Pourquoi ces règles — la justification, question par question, de chaque
// choix de la page /regles. Volontairement longue : c'est l'endroit où l'on
// peut être exhaustif, pour que /regles reste lisible.
//
// Les chiffres marqués « mesuré » viennent des mesures du dépôt
// (docs/rules-survey/, docs/measurements/) ; les citations, du relevé de
// sources docs/rules-survey/SOURCES.md.

import { wireToc, openQueryTarget } from '../shared/toc.js';

const TEMPLATE = `
<div class="rg-page">
<div class="rg-inner">

    <div class="rg-head">
        <h2><span class="rg-kicker">Belote contrée</span>Pourquoi ces règles</h2>
        <p class="rg-lede">
            Chaque choix de la page <a href="/regles">Règles du jeu</a> a été tranché, et aucun
            n'allait de soi. Il n'existe pas de règlement officiel unique de la belote
            contrée&nbsp;: la Fédération Française de Belote a publié à elle seule quatre
            rédactions mutuellement incompatibles, une fédération concurrente existe depuis 1997,
            et le tournoi le plus visible de la discipline ne suit ni l'une ni l'autre.
        </p>
        <p class="rg-lede">
            Cette page dit, pour chaque point disputé&nbsp;: ce que Colver fait, qui dit la même
            chose, qui dit le contraire, et — quand la question se prête à la mesure — ce que
            valent les chiffres.
        </p>
        <div class="rg-crosslink">
            <a href="/regles">Les règles, sans les justifications</a>
            <a href="https://github.com/Avo-k/colver/tree/master/docs/rules-survey" target="_blank" rel="noopener">Le relevé de sources complet</a>
        </div>
    </div>

    <div class="rg-toc" role="navigation" aria-label="Sommaire">
        <span class="rg-toc-group">Le cadre</span>
        <button type="button" data-goto="q-pourquoi">Pas de règle officielle&nbsp;?</button>
        <button type="button" data-goto="q-methode">Comment on a tranché</button>

        <span class="rg-toc-group">La marque</span>
        <button type="button" data-goto="q-plus-que-defense">Faire plus que la défense</button>
        <button type="button" data-goto="q-chute">La chute</button>
        <button type="button" data-goto="q-base">162 et pas 160</button>
        <button type="button" data-goto="q-mult">Ce que double la coinche</button>
        <button type="button" data-goto="q-surcoinche">Surcoinche &times;3</button>
        <button type="button" data-goto="q-capot">Le capot</button>
        <button type="button" data-goto="q-belote">La belote prenable</button>
        <button type="button" data-goto="q-arrondi">Aucun arrondi</button>

        <span class="rg-toc-group">Les enchères</span>
        <button type="button" data-goto="q-plafond">Plafond à 160</button>
        <button type="button" data-goto="q-reparler">Reparler, surenchérir</button>
        <button type="button" data-goto="q-couleur">Couleur libre</button>
        <button type="button" data-goto="q-volee">Coinche à la volée</button>
        <button type="button" data-goto="q-passes">Trois passes</button>

        <span class="rg-toc-group">Le jeu</span>
        <button type="button" data-goto="q-pisser">Sous-couper</button>
        <button type="button" data-goto="q-partenaire">Partenaire maître</button>
        <button type="button" data-goto="q-monter">Monter hors atout</button>
        <button type="button" data-goto="q-annonce-belote">Belote automatique</button>

        <span class="rg-toc-group">La partie</span>
        <button type="button" data-goto="q-atteindre">Atteindre ou dépasser</button>

        <span class="rg-toc-group">Les absents</span>
        <button type="button" data-goto="q-annonces">Tierce, carré</button>
        <button type="button" data-goto="q-sata">Sans Atout, Tout Atout</button>

        <span class="rg-toc-group">Divers</span>
        <button type="button" data-goto="q-sens">Le sens du jeu</button>
        <button type="button" data-goto="q-desaccord">« Pas comme ça chez nous »</button>
        <button type="button" data-goto="q-sources">Les sources</button>
    </div>

    <div class="rg-body">

    <!-- ================= LE CADRE ================= -->
    <section class="rg-sec">
        <h3>Le cadre</h3>

        <div class="rg-q" id="q-pourquoi">
            <h4>Il n'y a vraiment pas de règle officielle&nbsp;?</h4>
            <p class="rg-a">Non. Il y a deux fédérations concurrentes, quatre rédactions fédérales incompatibles, un « Championnat de France » qui n'appartient à personne, et un fonds traditionnel plus ancien que tout cela.</p>
            <p>
                La Fédération Française de Belote a publié quatre rédactions incompatibles&nbsp;:
                une contrée vers 2015, une contrée datée du 27.01.2016, une réimpression dite
                « Équipe Ludique », et — sur les mêmes points, avec des réponses différentes — ses
                textes de coinchée et de belote classique. Elles se contredisent sur huit points au
                moins, dont le multiplicateur de la surcoinche, la valeur du capot et le sort de la
                belote en cas de chute. Aucune de ces contradictions n'est signalée par la
                fédération, et les quatre rédactions coexistent en ligne.
            </p>
            <p>
                En face, la <strong>Fédération Française de Coinche</strong> a été fondée à
                Saint-Étienne en 1997, sur un règlement déposé à l'INPI en 1996 et diffusé à
                11&nbsp;000 exemplaires. Elle diverge de la FFB sur l'enchère minimale (82 et non
                80), la distribution (6 cartes, enchères, puis 2), l'obligation de monter à l'atout
                y compris sur son partenaire, et le Sans Atout.
            </p>
            <p>
                Un détail résume la situation. Le règlement FFB contrée de 2016 écrit noir sur
                blanc que la fin de partie « est trop souvent le centre de règles différentes de
                régions en régions, doit-on faire un pli supplémentaire, atteindre les points, les
                dépasser&nbsp;? », annonce homologuer une règle nationale pour y mettre fin — puis
                répond sur l'arrondi, pas sur la question posée.
            </p>
            <p>
                Ce qui est <strong>réellement partagé</strong> tient en un noyau étroit&nbsp;: la
                valeur des cartes, la mécanique des plis, l'obligation de fournir, la belote à 20,
                le dix de der à 10, le total à 162, les enchères de 80 en 80 par pas de 10, et la
                fin des enchères à trois passes. Sur ce noyau, aucune source ne diverge. Tout ce
                qui touche à <em>l'argent</em> — ce que vaut une chute, un contre, un capot, et
                quand la partie s'arrête — est disputé, souvent frontalement.
            </p>
        </div>

        <div class="rg-q" id="q-methode">
            <h4>Alors sur quoi vous êtes-vous appuyés&nbsp;?</h4>
            <p class="rg-a">Sur un relevé de sources fait exprès, avec quatre principes&nbsp;: dédoublonner, préférer les règlements de compétition, mesurer quand c'est mesurable, et assumer le reste.</p>
            <p>
                <strong>Dédoublonner d'abord.</strong> Le web sur ce sujet est un jeu de miroirs.
                Cinq sites très cités recopient la même page fédérale — c'est
                <em>une</em> voix, pas cinq. Un autre est une copie verbatim de Pagat, vérifiée
                caractère par caractère, et un troisième en est une traduction automatique. Compter
                les pages plutôt que les voix donne des majorités entièrement fictives.
            </p>
            <p>
                <strong>Préférer ce qui engage.</strong> Un règlement de compétition est écrit pour
                trancher des litiges réels, avec un arbitre au bout&nbsp;: il vaut mieux qu'une
                page encyclopédique. C'est ce qui a fait pencher plusieurs arbitrages du côté du
                règlement du Championnat de France de Cannes.
            </p>
            <p>
                <strong>Mesurer quand la question s'y prête.</strong> « Faut-il ouvrir les enchères
                au-dessus de 160&nbsp;? » n'est pas une question d'opinion&nbsp;: on peut compter
                combien de mains en auraient besoin. Ces mesures sont signalées par une étiquette
                <span class="rg-tag rg-tag--measured">mesuré</span> dans les réponses ci-dessous.
            </p>
            <p>
                <strong>Et assumer le reste.</strong> Quand deux traditions également légitimes
                s'opposent et qu'aucune mesure ne départage, il faut choisir. On le dit alors
                clairement, plutôt que de faire passer un choix pour une évidence.
            </p>
        </div>
    </section>

    <!-- ================= LA MARQUE ================= -->
    <section class="rg-sec">
        <h3>La marque</h3>

        <div class="rg-q" id="q-plus-que-defense">
            <h4>Faut-il aussi faire plus de points que la défense&nbsp;?</h4>
            <p class="rg-a">Non. Seule l'enchère compte. Un contrat peut être réussi alors que la défense a ramassé plus de points cartes que les preneurs.</p>
            <p>
                Le cas est facile à rencontrer. On annonce 80, on ne ramasse que 70 points de
                plis — mais on a la belote, donc 90, donc le contrat est tenu, alors que la défense
                tient 92 des 162 points de la donne. Ou plus net encore&nbsp;: on annonce 80, on
                fait <strong>80 pile</strong>, et la défense en a 82.
            </p>
            <p>
                Les sources se répondent, et pour une fois elles le font <em>explicitement</em>
                plutôt que par omission&nbsp;:
            </p>
            <blockquote class="rg-quote">
                « Le contrat est réussi si les preneurs obtiennent un total supérieur ou égal à
                l'enchère demandée […] <strong>Ceci est valable même si les défenseurs ont réalisé
                plus de points que les preneurs.</strong> »
                <cite>FFB, règles officielles de la belote contrée, 27.01.2016</cite>
            </blockquote>
            <blockquote class="rg-quote">
                « Le contrat est réussi si <strong>les deux conditions</strong> suivantes sont
                réunies&nbsp;: 1) […] supérieur ou égal à l'enchère demandée […] 2) Les preneurs
                obtiennent un total <strong>supérieur à celui de la défense</strong>. »
                <cite>FFB, règles de la belote coinchée, 2015</cite>
            </blockquote>
            <p>
                Encore une fracture interne à la fédération — mais celle-ci, contrairement aux
                autres, <strong>n'est pas une incohérence</strong>, et c'est ce qui la rend
                éclairante. La coinche a des <button type="button" class="rg-xref" data-goto="q-annonces">annonces</button>
                (tierce, cent, carré), qui peuvent porter la défense au-dessus du preneur
                <em>sans que celui-ci ait mal joué</em>. La condition « faire plus que la défense »
                est donc le corollaire <strong>des annonces</strong>, pas de l'enchère. La contrée
                n'en a pas&nbsp;; elle n'a pas besoin de la condition.
            </p>
            <p>
                Deux indices confirment cette lecture. D'abord, <strong>c'est de là que vient le
                « 82 »</strong> des sources anglophones&nbsp;: 162 / 2 = 81, donc 82 est le
                minimum pour avoir strictement plus que l'adversaire. Pagat le dit lui-même — « <em>It
                is a vestige of this rule that requires a score of at least 82 to win a bid of
                80</em> » — et la FFB, ayant supprimé la règle, a gardé le 80. Ensuite, les
                <strong>quatre implémentations libres</strong> du relevé testent toutes
                « points du preneur ≥ contrat » sans jamais regarder le total adverse, quoi qu'en
                disent les textes qu'elles prétendent suivre.
            </p>
            <p>
                Colver suit donc le camp contrée. Les trois situations, telles que le moteur les
                marque&nbsp;:
            </p>
            <div class="rg-scroll">
                <table class="rg-table">
                    <caption>Contrat de 80, la défense a plus de points cartes</caption>
                    <thead><tr><th>Situation</th><th class="rg-num">Preneurs</th><th class="rg-num">Défense</th></tr></thead>
                    <tbody>
                        <tr><th>70 de plis + belote (défense 92)</th><td class="rg-num">170</td><td class="rg-num">92</td></tr>
                        <tr><th>80 pile (défense 82)</th><td class="rg-num">160</td><td class="rg-num">82</td></tr>
                        <tr><th>80 pile, mais coinché</th><td class="rg-num">322</td><td class="rg-num">0</td></tr>
                    </tbody>
                </table>
            </div>
            <p>
                Le troisième cas mérite qu'on s'y arrête&nbsp;: coincher un 80 qui tombe pile à 80
                fait marquer <strong>322 à 0</strong> aux preneurs, alors que la défense tenait la
                moitié de la donne. C'est le rappel que la belote
                <button type="button" class="rg-xref" data-goto="q-belote">compte pour réaliser le
                contrat</button> — un règlement de concours met en garde en ces termes exacts&nbsp;:
                « attention au moment de contrer&nbsp;! ».
            </p>
            <p class="rg-src">
                <strong>Portée du cas&nbsp;:</strong> pour réussir en ayant moins de points cartes
                que la défense, il faut ramasser au plus 80 points de plis — donc cela ne peut
                arriver <strong>qu'aux contrats 80, 90 et 100</strong>, et <strong>sans belote,
                uniquement à 80 pile</strong>. Au-delà, la valeur du contrat garantit à elle seule
                la majorité des points cartes.
            </p>
        </div>

        <div class="rg-q" id="q-chute">
            <h4>Quand un contrat chute, pourquoi la défense marque-t-elle 162 + le contrat&nbsp;?</h4>
            <p class="rg-a">Parce que c'est la forme la plus cohérente des cinq en circulation, et la seule qu'un règlement de compétition écrit noir sur blanc.</p>
            <p>Cinq barèmes de chute coexistent, tous attestés, tous défendables&nbsp;:</p>
            <div class="rg-scroll">
                <table class="rg-table">
                    <thead><tr><th>La défense marque</th><th>Qui</th></tr></thead>
                    <tbody>
                        <tr><th>160 tout court</th><td>Le fonds traditionnel — « tournoi international », concours de village, plusieurs implémentations libres</td></tr>
                        <tr><th>160 + contrat</th><td>FFB contrée et coinchée, Pagat, la majorité des sites</td></tr>
                        <tr><th>162 + contrat</th><td>Exoty, BelotePoint, maviedesenior, BoardGameArena, le code de <em>ismo009/Coinche</em> — et <strong>Colver</strong></td></tr>
                        <tr><th>162 tout court</th><td>La belote classique&nbsp;: FFB belote, tous les concours de village</td></tr>
                        <tr><th>Le contrat seul</th><td>Les barèmes dits « points annoncés »</td></tr>
                    </tbody>
                </table>
            </div>
            <p>
                Le choix retenu est celui qui reste vrai quand on cesse d'arrondir (voir
                <button type="button" class="rg-xref" data-goto="q-base">162 et pas 160</button>) et qui garde une
                propriété simple&nbsp;: la chute rend à la défense <em>toute</em> la donne plus le
                contrat, sans qu'il faille regarder comment les plis se sont partagés. Le règlement
                du Championnat de France de Cannes l'écrit ainsi&nbsp;:
            </p>
            <blockquote class="rg-quote">
                « Total des points à marquer = points de l'enchère demandée +162 ou 252 + belote
                éventuellement »
                <cite>Règlement de jeu du Championnat de France de Belote Contrée, Cannes, édition 2016</cite>
            </blockquote>
            <p class="rg-src">
                <strong>Réserve&nbsp;:</strong> le parent fédéral le plus proche est la Note 1 du
                règlement FFB contrée 2016 — la seule forme FFB en « base + contrat &times;
                multiplicateur ». Mais cette note est facultative, réservée aux tournois en réel,
                et écrit 160 là où Colver écrit 162.
            </p>
        </div>

        <div class="rg-q" id="q-base">
            <h4>Pourquoi 162 et pas 160&nbsp;? La donne ne vaut-elle pas 160&nbsp;?</h4>
            <p class="rg-a">La donne a toujours valu 162. Le 160 est le résidu d'un arrondi, pas un total.</p>
            <p>
                Les 32 cartes font 152 points, le dix de der en ajoute 10&nbsp;: 162. Personne ne
                conteste ce total — il fait partie du noyau sur lequel toutes les sources sont
                d'accord. Le 160 apparaît uniquement chez celles qui arrondissent la marque à la
                dizaine&nbsp;: tant qu'on arrondit, <span class="rg-formula">162 + 10k</span>
                retombe sur <span class="rg-formula">160 + 10k</span> et la différence ne se voit
                jamais.
            </p>
            <p>
                Le test est net&nbsp;: les sources qui écrivent 162 sont exactement celles qui ne
                marquent pas avec des jetons. Colver n'arrondissant pas (voir
                <button type="button" class="rg-xref" data-goto="q-arrondi">l'arrondi</button>), écrire 160 serait
                importer une contrainte de marquage physique dans un moteur qui somme des entiers.
            </p>
        </div>

        <div class="rg-q" id="q-mult">
            <h4>La coinche double quoi, exactement&nbsp;?</h4>
            <p class="rg-a">La valeur du contrat, et elle seule. Ni les points de plis, ni la belote.</p>
            <p>
                Un contrat de 100 coinché et tenu marque
                <span class="rg-formula">162 + 100 &times; 2 = 362</span>, pas 324 ni 524. C'est la
                forme qu'écrit le règlement de Cannes, qui prend soin de le préciser pour la
                belote&nbsp;:
            </p>
            <blockquote class="rg-quote">
                « Le contre double les points de l'enchère demandée, le surcontre les triple&nbsp;;
                si la belote a été annoncée on ajoute 20 points de bonification qui ne sont ni
                doublés ni triplés. »
                <cite>Règlement de Cannes, 2016</cite>
            </blockquote>
            <p>
                L'autre modèle en circulation remplace la base par un forfait — 320 ou 640 selon le
                niveau de contre. Il donne des scores très différents et ne coïncide avec celui-ci
                nulle part. La FFB écrit les deux&nbsp;: le forfait dans son barème principal, la
                forme « base + contrat &times; multiplicateur » dans la Note 1 du même document.
            </p>
            <p>
                Deuxième conséquence, moins visible&nbsp;: sous une coinche, le camp perdant marque
                <strong>0</strong>. La coinche transforme la donne en pari tout-ou-rien, ce qui est
                précisément ce qui la rend intéressante à jouer.
            </p>
        </div>

        <div class="rg-q" id="q-surcoinche">
            <h4>Pourquoi la surcoinche triple-t-elle&nbsp;? On m'a toujours dit &times;4.</h4>
            <p class="rg-a">Parce que la fracture &times;3 / &times;4 est chronologique à l'intérieur de la FFB, et que les deux seuls règlements de compétition du relevé disent &times;3.</p>
            <div class="rg-scroll">
                <table class="rg-table">
                    <thead><tr><th>&times;4</th><th>&times;3</th></tr></thead>
                    <tbody>
                        <tr>
                            <td>FFB contrée ~2015, FFB contrée 2016, FFB coinchée 2015, Wikipédia, Pagat (règle de base), BoardGameArena, plusieurs implémentations libres</td>
                            <td><strong>FFB « Équipe Ludique »</strong> (la rédaction la plus récente), IsCool, jeu-belote.fr, Pagat (en variante), le Championnat de France de Cannes, le championnat de coinche de Guyane</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <p>
                Ce n'est pas une divergence régionale&nbsp;: c'est la même fédération qui a changé
                d'avis, exactement comme sur l'arrondi. Entre une rédaction ancienne et une
                récente, et entre un texte encyclopédique et un règlement arbitré, Colver suit le
                récent et l'arbitré.
            </p>
            <p class="rg-src">
                <strong>Portée réelle&nbsp;:</strong> la surcoinche est rare, et l'écart entre
                &times;3 et &times;4 ne concerne que ces donnes-là. C'est le choix le plus discutable
                de la liste, et probablement le moins conséquent.
            </p>
        </div>

        <div class="rg-q" id="q-capot">
            <h4>Le capot est-il une prime forfaitaire ou un contrat comme les autres&nbsp;?</h4>
            <p class="rg-a">Un contrat ordinaire, d'une valeur de 250, qui suit le barème général.</p>
            <p>
                Deux modèles s'affrontent, et ils ne coïncident nulle part — ce ne sont pas deux
                façons de dire la même chose&nbsp;:
            </p>
            <ul class="rg-list">
                <li>
                    <strong>Le forfait</strong> — 500 demandé, 1000 contré, 2000 surcontré. C'est
                    la FFB 2016 et le fonds traditionnel.
                </li>
                <li>
                    <strong>Le contrat à 250</strong>, dont la seule particularité est de porter la
                    base cartes de 162 à 252. C'est la FFB « Équipe Ludique », le tableau p.&nbsp;7
                    du règlement FFB 2016, Cannes — et Colver.
                </li>
            </ul>
            <p>
                L'argument décisif est la simplicité du modèle&nbsp;: avec un capot-contrat, il n'y
                a <em>aucun</em> cas particulier dans le calcul du score. La même formule couvre 80
                à pique et le capot, coinché ou non, réussi ou chuté. Le forfait exige au contraire
                une table à part, et c'est précisément là que la FFB se contredit — une même page
                de son site donne 1000/2000 dans une section et 500/1000 dans une autre.
            </p>
            <p>
                Conséquence à connaître&nbsp;: un capot annoncé et réalisé vaut
                <span class="rg-formula">252 + 250 = 502</span>. Et la base passe à 252 dès que les
                preneurs font les huit plis, <em>même sans avoir annoncé capot</em> — c'est le total
                réel des cartes de cette donne-là.
            </p>
            <p class="rg-src">
                <strong>Écart connu&nbsp;:</strong> Cannes chiffre le capot à <strong>270</strong>,
                pas 250, parce que son échelle d'enchères monte de dix en dix jusque-là. Un capot y
                vaut donc 522. C'est le dernier point sur lequel Colver n'a aucune attestation
                exacte — voir <button type="button" class="rg-xref" data-goto="q-plafond">le plafond des enchères</button>.
            </p>
        </div>

        <div class="rg-q" id="q-belote">
            <h4>Pourquoi perd-on sa belote quand on chute&nbsp;?</h4>
            <p class="rg-a">Parce qu'en contrée la belote est prenable. C'est en coinche et en belote classique qu'elle est imprenable.</p>
            <p>
                La fracture suit exactement la frontière entre les deux jeux, y compris à
                l'intérieur de la FFB&nbsp;: ses textes de <em>contrée</em> font passer la belote au
                camp qui marque, ses textes de <em>coinche</em> et de <em>belote classique</em> la
                laissent toujours à qui l'annonce. Colver joue la contrée, donc la belote est
                prenable.
            </p>
            <p>
                À noter&nbsp;: c'est seulement vrai en cas de chute, ou sous une coinche. Sur un
                contrat réussi sans coinche, chaque camp garde sa propre belote — la défense marque
                ses points de plis plus ses 20 points.
            </p>
            <p>
                Dans tous les cas, la belote <strong>compte pour réaliser le contrat</strong> — un
                joueur qui annonce 100, réalise 88 points et sort son Roi-Dame d'atout est à 108 et
                tient son contrat. C'est le point sur lequel un règlement de concours met en garde
                explicitement&nbsp;: attention au moment de coincher.
            </p>
        </div>

        <div class="rg-q" id="q-arrondi">
            <h4>Pourquoi ne pas arrondir les scores à la dizaine&nbsp;? <span class="rg-tag rg-tag--measured">mesuré</span></h4>
            <p class="rg-a">Parce que l'arrondi n'a jamais été une règle du jeu, mais une commodité de marquage physique — et parce qu'il casse l'arithmétique de la donne.</p>
            <p>
                La FFB elle-même le justifie par « des questions de logistiques, <strong>jetons</strong>
                notamment », et le rend explicitement facultatif en belote classique. Plusieurs
                règlements de concours écrivent d'ailleurs « au point, sans arrondir ». Un moteur
                qui additionne des entiers n'a aucune raison d'imiter une contrainte de jetons.
            </p>
            <p>
                Surtout, l'arrondi <strong>ne conserve pas le total</strong>. Les deux camps se
                partagent 162 points&nbsp;; si l'on arrondit chaque camp séparément, la somme
                cesse de valoir 162 dans une bonne partie des cas. Avec la règle courante
                « 5 monte », c'est <strong>48 partages sur 163 (29&nbsp;%)</strong>&nbsp;; avec
                « 6 monte », <strong>16 sur 163 (10&nbsp;%)</strong>. Un partage 75-87 devient
                80-90 = 170 dans le premier cas. Trois corpus étrangers traitent ce total comme un
                invariant à préserver — la coinche suisse en fait carrément un contrôle de donne —
                et presque aucun corpus français ne voit le problème.
            </p>
            <p>
                <strong>L'effet sur les scores, mesuré&nbsp;:</strong> supprimer l'arrondi déplace
                environ <strong>73&nbsp;% des scores de donne</strong>, de <strong>2,4 points en
                moyenne</strong> et de 5 au maximum. C'est visible mais petit&nbsp;: assez pour que
                le compteur affiche 162 + contrat en cas de chute plutôt que 160 + contrat, trop
                peu pour changer l'issue d'une partie autrement que très marginalement.
            </p>
            <p class="rg-src">
                <strong>Écart assumé&nbsp;:</strong> la FFB arrondit à la dizaine (§9.2). Colver ne
                le fait pas. C'est ce qui permet au moteur et au compteur de points d'afficher
                exactement le même chiffre sur chaque donne.
            </p>
        </div>
    </section>

    <!-- ================= LES ENCHÈRES ================= -->
    <section class="rg-sec">
        <h3>Les enchères</h3>

        <div class="rg-q" id="q-plafond">
            <h4>Pourquoi s'arrêter à 160&nbsp;? D'autres règlements montent à 170, 180, jusqu'à 270. <span class="rg-tag rg-tag--measured">mesuré</span></h4>
            <p class="rg-a">Parce que ces paliers ne désignent presque aucune main réelle&nbsp;: 0,35&nbsp;% des donnes en jeu réel, contre 4,8&nbsp;% de capots effectivement réalisés.</p>
            <p>
                Le raisonnement d'abord. Sans capot, le maximum de points de plis est 162 — les
                huit plis moins un concédé à zéro. Donc <strong>170 et 180 ne sont atteignables
                qu'avec la belote</strong>, en concédant respectivement au plus 12 et au plus
                <strong>2</strong> points. Et tout palier à 190 ou au-delà exige le capot, puisque
                le seul total supérieur à 182 est 252&nbsp;: ces paliers-là ne sont pas des
                contrats intermédiaires, ce sont des capots facturés moins cher.
            </p>
            <p>
                Le point décisif est que <strong>le preneur ne contrôle pas la valeur du pli qu'il
                concède</strong>&nbsp;: c'est la défense qui y jette ses cartes, et elle y met ses
                points. Pour qu'un pli concédé vaille 2 points ou moins, il faut que la défense
                n'ait rien à y mettre — c'est-à-dire que le capot passait.
            </p>
            <div class="rg-scroll">
                <table class="rg-table">
                    <caption>Part des donnes où chaque niveau est le meilleur atteignable</caption>
                    <thead><tr><th>Source</th><th class="rg-num">Capot</th><th class="rg-num">180+</th><th class="rg-num">170-179</th><th class="rg-num">170 et 180</th></tr></thead>
                    <tbody>
                        <tr><th>Jeu parfait (solveur)</th><td class="rg-num">16,1&nbsp;%</td><td class="rg-num">0,005&nbsp;%</td><td class="rg-num">0,240&nbsp;%</td><td class="rg-num">0,244&nbsp;%</td></tr>
                        <tr><th>Jeu réel (réseau de neurones)</th><td class="rg-num">4,8&nbsp;%</td><td class="rg-num">0,015&nbsp;%</td><td class="rg-num">0,339&nbsp;%</td><td class="rg-num">0,354&nbsp;%</td></tr>
                        <tr><th>Jeu réel (recherche IS-DD)</th><td class="rg-num">7,0&nbsp;%</td><td class="rg-num">0,013&nbsp;%</td><td class="rg-num">0,340&nbsp;%</td><td class="rg-num">0,353&nbsp;%</td></tr>
                    </tbody>
                </table>
            </div>
            <p>
                Quatre mesures concordantes&nbsp;: 20&nbsp;000 donnes résolues à neuf, un pool de
                5 millions de donnes pré-résolues, et deux couches de scores en jeu réel sur ce
                même pool. Le palier 180 seul concerne <strong>une donne sur 7&nbsp;000</strong>.
                Le rapport avec les capots réellement réalisés est de <strong>1 à 14</strong>.
            </p>
            <p>
                À cela s'ajoute un coût technique&nbsp;: monter jusqu'à 270 ferait passer l'espace
                d'enchères de 43 à 83 actions, presque le double, pour des paliers qu'une politique
                optimale n'emploierait quasiment jamais et qu'il faudrait pourtant apprendre à
                éviter.
            </p>
            <p class="rg-src">
                L'échelle cannoise jusqu'à 270 se lit comme une règle de complétude du texte —
                « toute somme atteignable est une enchère légale » — plutôt que comme un espace
                stratégique. C'est cohérent pour un arbitre, inutile pour un joueur.
            </p>
        </div>

        <div class="rg-q" id="q-reparler">
            <h4>Peut-on vraiment reparler après avoir passé, et surenchérir sur son partenaire&nbsp;?</h4>
            <p class="rg-a">Oui aux deux. C'est l'un des points les plus consensuels de tout le relevé.</p>
            <p>
                Sur « reparler après avoir passé », le relevé compte <strong>treize sources pour et
                aucune contre</strong> — c'est l'axe le plus net du corpus. Sur « surenchérir sur
                son partenaire », aucune source ne l'interdit, et plusieurs en font le cœur du jeu.
            </p>
            <p>
                Il y a une raison de fond&nbsp;: l'enchère est le <em>seul</em> canal par lequel
                deux partenaires échangent de l'information sur leurs mains. Interdire de reparler
                ou de monter sur son partenaire couperait ce canal et transformerait l'enchère en
                simple mise aux enchères. C'est aussi pourquoi les modèles de Colver apprennent à
                annoncer en jouant leurs propres enchères&nbsp;: l'essentiel du signal vient de ce
                que le partenaire a dit, pas de la main qu'on tient.
            </p>
        </div>

        <div class="rg-q" id="q-couleur">
            <h4>Pourquoi peut-on surenchérir avec n'importe quelle couleur&nbsp;?</h4>
            <p class="rg-a">Parce que la hiérarchie entre couleurs est une variante de coinche, pas une règle de contrée — et qu'elle allonge les enchères sans rien apporter.</p>
            <p>
                Certaines tables classent les couleurs (trèfle &lt; carreau &lt; cœur &lt; pique)
                et permettent de surenchérir « à valeur égale, en couleur supérieure ». Pagat la
                donne comme une variante et note qu'elle « conduit à des enchères plus longues ».
                Chez Colver, seule la valeur compte&nbsp;: une annonce doit être strictement
                supérieure en points, et la couleur est entièrement libre — y compris la même que
                la précédente, ce qui est un moyen normal de soutenir son partenaire.
            </p>
        </div>

        <div class="rg-q" id="q-volee">
            <h4>Pourquoi ne peut-on pas coincher à la volée&nbsp;?</h4>
            <p class="rg-a">Parce que coincher hors tour renseigne le partenaire sans passer par une annonce.</p>
            <p>
                La coinche « à la volée » — dès qu'on entend l'annonce, sans attendre son tour —
                est interdite par les sources qui la mentionnent. La raison est la même que celle
                qui pousse deux règlements à encadrer le <em>temps de parole</em> («&nbsp;sans délai
                de réflexion anormalement long&nbsp;»)&nbsp;: le tempo est un canal d'information
                qu'aucune règle d'enchère ne contrôle. Un joueur qui coinche instantanément dit
                quelque chose à son partenaire que la règle ne l'autorisait pas à dire.
            </p>
            <p>
                Sur Colver la question ne se poserait pas techniquement, mais la conserver garde le
                jeu cohérent avec ce qui se pratique en tournoi.
            </p>
        </div>

        <div class="rg-q" id="q-passes">
            <h4>Après une coinche, combien de passes pour terminer&nbsp;?</h4>
            <p class="rg-a">Trois, comme après n'importe quelle annonce. Le décompte repart de zéro au moment de la coinche.</p>
            <p>
                C'est un point sur lequel le relevé est <em>muet</em> plutôt que divisé&nbsp;: une
                seule source dit que le partenaire du coincheur « ne doit pas parler », et ce
                silence rend la règle des trois passes ambiguë après une coinche. Le code de
                BoardGameArena en exige quatre.
            </p>
            <p>
                Colver garde trois passes, ce qui a une conséquence concrète et souhaitable&nbsp;:
                les trois autres joueurs sont consultés après la coinche, donc <strong>le camp
                coinché a toujours l'occasion de surcoincher</strong>, quel que soit celui des deux
                qui parle en premier.
            </p>
        </div>
    </section>

    <!-- ================= LE JEU ================= -->
    <section class="rg-sec">
        <h3>Le jeu de la carte</h3>

        <div class="rg-q" id="q-pisser">
            <h4>Quand je ne peux pas surcouper, dois-je sous-couper ou puis-je me défausser&nbsp;?</h4>
            <p class="rg-a">Au choix. Vous n'êtes obligé de sous-couper que s'il ne vous reste que de l'atout.</p>
            <p>
                C'est la règle dite « on ne pisse pas », et c'est l'une des fractures les plus
                nettes du relevé&nbsp;:
            </p>
            <ul class="rg-list">
                <li><strong>Obligation de sous-couper&nbsp;:</strong> Pagat, Wikipédia Belote, plusieurs règlements de concours, BoardGameArena.</li>
                <li><strong>Libre choix&nbsp;:</strong> FFB contrée et coinchée, Wikipédia Contrée, Cannes, la Coinche stéphanoise, plusieurs sites de référence.</li>
            </ul>
            <p>
                Le fait le plus éclairant est que <strong>la FFB écrit les deux le même jour</strong>&nbsp;:
                son règlement de <em>belote</em> du 27.01.2016 impose le sous-coup, sa <em>contrée</em>
                du 27.01.2016 l'exclut. Le paragraphe est identique mot pour mot dans les deux
                documents jusqu'à la dernière proposition.
            </p>
            <p>
                Colver joue la contrée, donc le camp contrée&nbsp;: choix libre. Et c'est aussi la
                double permission qu'écrit Cannes, qui autorise explicitement les deux.
            </p>
            <p class="rg-src">
                Pagat lit cette fracture comme un partage Nord / Midi — sauf que la contrée
                <em>est</em> le jeu du Midi.
            </p>
        </div>

        <div class="rg-q" id="q-partenaire">
            <h4>Mon partenaire a coupé et il ne me reste que de l'atout&nbsp;: dois-je surcouper&nbsp;? <span class="rg-tag rg-tag--measured">mesuré</span></h4>
            <p class="rg-a">Non. Quand le partenaire est maître, il n'y a aucune obligation — sans exception, y compris celle-là.</p>
            <p>
                C'est le cas de figure le plus subtil de tout le jeu, et Colver s'est trompé
                dessus. Le moteur <em>forçait</em> la surcoupe dans cette situation jusqu'au
                1<sup>er</sup> août 2026.
            </p>
            <p>
                La cause est une source unique. Une réimpression du règlement FFB — celle qui avait
                servi de référence — <strong>supprime le « n'est pas »</strong> de la phrase
                originale et efface la proposition qui l'expliquait. Le texte FFB 2015 dit&nbsp;:
            </p>
            <blockquote class="rg-quote">
                « Il n'est pas obligatoire de couper. On peut se défausser de n'importe quelle carte
                sans exception (y compris un atout inférieur au sien) » — et plus loin, « le seul
                cas de figure, plutôt rare, où il est permis de jouer un atout inférieur ».
                <cite>FFB, règles de la belote contrée, édition 2015</cite>
            </blockquote>
            <p>
                FFB contrée 2015, FFB contrée 2016, FFB belote 2016, Wikipédia Belote et le
                règlement de Cannes disent tous la même chose. Une seule réimpression dit
                l'inverse, et c'est celle-là qui avait été suivie.
            </p>
            <p>
                <strong>Ampleur du bug, mesurée&nbsp;:</strong> sur 20&nbsp;000 donnes aléatoires
                (640&nbsp;000 décisions), le cas se présente <strong>485 fois (0,076&nbsp;%)</strong>,
                et l'ancienne règle retirait effectivement une option dans seulement
                <strong>91 d'entre elles</strong> — 0,014&nbsp;% des décisions, soit environ une
                donne sur 220. Le reste du temps, il n'y avait de toute façon pas d'atout inférieur
                à jouer.
            </p>
            <p class="rg-src">
                <strong>Réserve sur la mesure&nbsp;:</strong> elle est faite en jeu aléatoire, qui
                répartit les coupes autrement qu'un jeu de bon niveau. C'est un ordre de grandeur,
                pas un chiffre exact.
            </p>
        </div>

        <div class="rg-q" id="q-monter">
            <h4>Pourquoi n'y a-t-il aucune obligation de monter hors atout&nbsp;?</h4>
            <p class="rg-a">Parce que l'obligation de monter ne concerne que l'atout. C'est unanime dans tout le relevé.</p>
            <p>
                Fournir la couleur demandée suffit&nbsp;: si l'on a du cœur et que le cœur est
                demandé, n'importe quel cœur convient, y compris le 7 sur un As. L'obligation de
                monter existe uniquement quand la couleur jouée est <strong>l'atout</strong>, parce
                que c'est l'atout qui détermine qui coupe et qui surcoupe.
            </p>
            <p>
                Une seule source du relevé s'en écarte, et elle appartient à l'autre
                fédération&nbsp;: la Coinche stéphanoise impose de monter à l'atout « y compris sur
                son partenaire ».
            </p>
        </div>

        <div class="rg-q" id="q-annonce-belote">
            <h4>Pourquoi ne faut-il pas annoncer la belote&nbsp;?</h4>
            <p class="rg-a">Parce que la déclaration est une règle de table physique&nbsp;: elle sert à informer les autres joueurs de ce que le moteur voit déjà.</p>
            <p>
                À une vraie table, dire « belote » puis « rebelote » est obligatoire, et l'oublier
                coûte les 20 points — non pas parce que les cartes changent, mais parce que
                personne d'autre ne peut vérifier que le Roi et la Dame étaient dans la même main.
                Sur Colver, ce contrôle est gratuit&nbsp;: le bonus est acquis dès que le même
                joueur a joué les deux cartes.
            </p>
            <p>
                Ce qui est conservé, en revanche, c'est la condition de fond&nbsp;: les deux cartes
                doivent être <strong>jouées</strong>. Un Roi d'atout gardé en main jusqu'à la fin ne
                rapporte rien — comme à une vraie table, où l'on n'annonce rebelote qu'en posant la
                seconde carte.
            </p>
        </div>
    </section>

    <!-- ================= LA PARTIE ================= -->
    <section class="rg-sec">
        <h3>La partie</h3>

        <div class="rg-q" id="q-atteindre">
            <h4>Faut-il atteindre la cible ou la dépasser&nbsp;?</h4>
            <p class="rg-a">L'atteindre suffit, à condition de ne pas être à égalité. C'est la position des quatre rédactions FFB.</p>
            <p>
                C'est la question que le règlement FFB 2016 pose lui-même — « doit-on faire un pli
                supplémentaire, atteindre les points, les dépasser&nbsp;? » — avant de ne pas y
                répondre. Les positions&nbsp;:
            </p>
            <ul class="rg-list">
                <li><strong>Atteindre suffit&nbsp;:</strong> les quatre rédactions FFB, Pagat, plusieurs implémentations libres.</li>
                <li><strong>Il faut dépasser&nbsp;:</strong> Maison des Essarts, BoardGameArena (« must be strictly higher », dans le code).</li>
                <li><strong>Contourner&nbsp;:</strong> décaler le seuil à 1010, 1995 ou 2001. Ce n'est pas un bricolage de village — le <strong>Championnat de France de Cannes joue toutes ses parties en 2001 points</strong>, et sa finale en 4 &times; 2001 + 1 &times; 2501. Le tournoi le plus visible de la discipline a réglé la question en ne la posant pas.
                </li>
            </ul>
            <p>
                Colver s'arrête donc à « cible atteinte, et pas d'égalité ». Le cas d'égalité n'est
                pas théorique&nbsp;: les deux camps marquent à chaque donne et peuvent franchir la
                barre ensemble. On rejoue alors une donne.
            </p>
            <p>
                Un détail de vocabulaire du corpus étranger mérite d'être signalé&nbsp;: la belote
                bulgare est le seul corpus à <em>nommer</em> les deux options — «&nbsp;until
                enough&nbsp;» contre «&nbsp;until passed&nbsp;». Aucun texte français ne dispose de
                ce couple de mots, ce qui explique en partie pourquoi le débat n'aboutit jamais.
            </p>
            <p class="rg-src">
                <strong>Écart connu&nbsp;:</strong> les rédactions FFB récentes ajoutent deux
                conditions que Colver n'applique pas — ne pas finir sur une belote seule, et ne pas
                finir en étant capot. Le règlement historique en pose trois cumulées. Ce sont des
                règles de tournoi en réel, pensées pour éviter les fins de partie contestables.
            </p>
        </div>
    </section>

    <!-- ================= LES ABSENTS ================= -->
    <section class="rg-sec">
        <h3>Les variantes absentes</h3>

        <div class="rg-q" id="q-annonces">
            <h4>Pourquoi pas de tierce, cinquante, cent, carré&nbsp;?</h4>
            <p class="rg-a">Parce que les annonces appartiennent à la coinche, pas à la contrée. C'est en grande partie ce qui sépare les deux jeux.</p>
            <p>
                Tous les documents de contrée du relevé — les quatre FFB, le règlement du tournoi
                international, Wikipédia — sont <em>muets</em> sur les annonces, et les règlements
                de concours qui abordent le sujet écrivent&nbsp;:
            </p>
            <blockquote class="rg-quote">
                « On joue sans annonce (tierce, carré, etc.) hormis la belote (et re)… La belote
                (et re) permet de remplir le contrat&nbsp;: attention au moment de contrer&nbsp;! »
                <cite>Règlement de concours ASCEE 2A</cite>
            </blockquote>
            <p>
                À l'inverse, la FFB écrit pour la <em>coinche</em>&nbsp;: « les annonces
                <strong>AIDENT</strong> à faire le contrat. Si un joueur demande 100, qu'il réalise
                83 points mais a une tierce, il a donc 103 points. » Deux jeux, deux réponses, et
                la même fédération pour les écrire.
            </p>
            <p>
                Ce n'est pas un détail de barème&nbsp;: les annonces changent le plafond utile de
                l'enchère. C'est parce qu'elles comptent que la page coinche de la FFB affiche un
                plafond de 650 quand sa page contrée affiche 160 — même site, même semaine.
            </p>
        </div>

        <div class="rg-q" id="q-sata">
            <h4>Pourquoi pas de Sans Atout ni de Tout Atout&nbsp;?</h4>
            <p class="rg-a">Pour la même raison&nbsp;: c'est la définition même de la différence entre coinche et contrée.</p>
            <p>Les deux articles de Wikipédia se répondent mot pour mot&nbsp;:</p>
            <blockquote class="rg-quote">
                « La coinche… se distingue de la belote contrée <strong>par la présence</strong> des
                enchères “tout atout” et “sans atout”. »<br>
                « [La contrée] se distingue de la coinche <strong>par l'absence</strong> des
                enchères “tout atout” et “sans atout”. »
                <cite>Wikipédia, articles Coinche et Belote contrée</cite>
            </blockquote>
            <p>
                La FFB en fait une <em>option d'organisateur</em> (§11 de son règlement 2016), et
                les sites qui la proposent prennent soin de prévenir&nbsp;: « Important&nbsp;: le
                Sans Atout / Tout Atout est une option. Il n'est pas appliqué dans tous les
                tournois. » Une option d'organisateur n'est pas une règle du jeu.
            </p>
            <p>
                Il y a une seconde raison, technique&nbsp;: le Sans Atout et le Tout Atout
                changent la valeur des cartes et donc le total de la donne (120 et 248 au lieu de
                152), ce qui oblige à rééchelonner tout le barème. Les sources qui le font ne le
                font pas de la même façon, et l'une d'elles se trompe même d'arithmétique.
            </p>
        </div>
    </section>

    <!-- ================= DIVERS ================= -->
    <section class="rg-sec">
        <h3>Divers</h3>

        <div class="rg-q" id="q-sens">
            <h4>Dans quel sens tourne le jeu&nbsp;?</h4>
            <p class="rg-a">Toujours vers votre voisin de gauche, enchères comprises.</p>
            <p>
                C'est le seul point de cette page qui n'a aucune conséquence sur le jeu&nbsp;: les
                quatre sièges sont symétriques, et inverser le sens revient à renommer les joueurs.
                La tradition française joue plutôt « à droite »&nbsp;; ce qui compte est que
                l'ordre soit fixe et que le joueur assis après le donneur parle et entame — ce que
                toutes les sources écrivent.
            </p>
        </div>

        <div class="rg-q" id="q-desaccord">
            <h4>Chez nous on ne joue pas comme ça. Qui a raison&nbsp;?</h4>
            <p class="rg-a">Vous, probablement, pour votre table. Il n'y a pas d'autorité au-dessus.</p>
            <p>
                Chaque table tranche pour elle-même, et c'est légitime&nbsp;:
                aucune des deux fédérations ne reconnaît l'autre, et le « Championnat de France »
                est un troisième label, propre à son organisateur. Une règle de belote contrée
                n'est pas vraie ou fausse&nbsp;; elle est celle de la table où l'on joue.
            </p>
            <p>
                Ce que Colver revendique est plus modeste&nbsp;: un jeu de règles <strong>complet,
                cohérent et écrit</strong>, où chaque point disputé a été tranché une fois, pour
                que deux donnes identiques se marquent toujours pareil et qu'une IA entraînée
                dessus mesure bien ce qu'on croit mesurer. Si votre table joue le surcontre à
                &times;4 et le capot à 500, ses parties sont tout aussi valables — simplement, ce
                ne sont pas celles que ce moteur simule.
            </p>
        </div>

        <div class="rg-q" id="q-sources">
            <h4>Où sont les sources&nbsp;?</h4>
            <p class="rg-a">Dans le dépôt, avec les citations, les doublons identifiés et les scripts de mesure.</p>
            <p>
                Le relevé complet vit dans
                <a href="https://github.com/Avo-k/colver/tree/master/docs/rules-survey" target="_blank" rel="noopener">docs/rules-survey</a>&nbsp;:
                une synthèse, la méthode, la liste des sources, et cinq matrices détaillées — une
                par famille de désaccord (l'arrondi, les enchères, le barème, le jeu de la carte,
                la fin de partie). Chaque ligne y porte sa citation et son fichier d'origine.
            </p>
            <p>
                Les mesures citées sur cette page sont reproductibles&nbsp;: les scripts sont dans
                <a href="https://github.com/Avo-k/colver/tree/master/scripts/analysis" target="_blank" rel="noopener">scripts/analysis</a>,
                et chaque exécution est journalisée avec l'empreinte des modèles utilisés — sans
                quoi un résultat n'est plus interprétable six mois plus tard.
            </p>
        </div>
    </section>

    </div><!-- /rg-body -->

    <div class="rg-foot">
        <p>
            Une erreur, une source qui manque&nbsp;? Les corrections sont bienvenues sur
            <a href="https://github.com/Avo-k/colver" target="_blank" rel="noopener">GitHub</a>.
            Retour aux <a href="/regles">règles du jeu</a>.
        </p>
    </div>

</div>
</div>
`;

let detach = null;

export function mount(container) {
    container.innerHTML = TEMPLATE;
    detach = wireToc(container);
    openQueryTarget(container, 'q', 'rg-q--target');
}

export function unmount() {
    if (detach) detach();
    detach = null;
}
