// Règles du jeu — le texte que le moteur applique, sans justification.
// Les « pourquoi » vivent sur /regles/choix, pour que celle-ci reste courte.

import { wireToc, openQueryTarget } from '../shared/toc.js';

const TEMPLATE = `
<div class="rg-page">
<div class="rg-inner">

    <div class="rg-head">
        <h2><span class="rg-kicker">Belote contrée</span>Règles du jeu</h2>
        <p class="rg-lede">
            Voici, en une page, exactement ce que le moteur de Colver applique&nbsp;: ce que vous
            avez le droit de jouer, ce qui compte pour le contrat, et comment le score se calcule.
            Rien n'est laissé à l'appréciation de la table.
        </p>
        <p class="rg-lede">
            Il n'existe pas de règlement officiel unique de la belote contrée — la fédération en a
            publié à elle seule quatre rédactions incompatibles. Chaque choix fait ici est donc
            assumé, et justifié en détail sur la page voisine.
        </p>
        <div class="rg-crosslink">
            <a href="/regles/choix">Pourquoi ces règles&nbsp;?</a>
            <a href="/aide">Aide-mémoire visuel</a>
            <a href="/score">Marquer une vraie partie</a>
        </div>
    </div>

    <div class="rg-toc" role="navigation" aria-label="Sommaire">
        <span class="rg-toc-group">Avant de jouer</span>
        <button type="button" data-goto="s-bref">En bref</button>
        <button type="button" data-goto="s-cartes">Les cartes</button>
        <button type="button" data-goto="s-donne">La donne</button>
        <span class="rg-toc-group">La donne</span>
        <button type="button" data-goto="s-encheres">Les enchères</button>
        <button type="button" data-goto="s-jeu">Le jeu de la carte</button>
        <button type="button" data-goto="s-pli">Le pli</button>
        <button type="button" data-goto="s-belote">La belote</button>
        <span class="rg-toc-group">Compter</span>
        <button type="button" data-goto="s-points">Les points de la donne</button>
        <button type="button" data-goto="s-marque">La marque</button>
        <button type="button" data-goto="s-exemples">Exemples</button>
        <button type="button" data-goto="s-partie">La partie</button>
        <span class="rg-toc-group">Enfin</span>
        <button type="button" data-goto="s-hors">Ce qu'on ne joue pas</button>
    </div>

    <div class="rg-body">

    <!-- ============================================ -->
    <section class="rg-sec" id="s-bref">
        <h3>En bref</h3>
        <p>
            Quatre joueurs, deux équipes face à face&nbsp;: Nord-Sud contre Est-Ouest. Un jeu de
            32 cartes, huit cartes par joueur, et deux temps dans chaque donne.
        </p>
        <ul class="rg-list">
            <li>
                <strong>L'enchère.</strong> Une équipe s'engage sur un nombre de points à réaliser
                et choisit la couleur d'atout. L'autre peut doubler la mise en <em>coinchant</em>.
            </li>
            <li>
                <strong>Le jeu.</strong> Huit plis. On compte les points ramassés, on compare à
                l'engagement, on marque.
            </li>
        </ul>
        <p>
            On redonne, et la première équipe à atteindre la cible de la partie l'emporte.
        </p>
    </section>

    <!-- ============================================ -->
    <section class="rg-sec" id="s-cartes">
        <h3>Les cartes</h3>
        <p>
            L'ordre de force et la valeur des cartes changent selon que la couleur est l'atout ou
            non. C'est la seule chose à savoir par cœur.
        </p>

        <div class="rg-duo">
            <div class="rg-scroll">
                <table class="rg-table">
                    <caption>À l'atout — de la plus forte à la plus faible</caption>
                    <thead><tr><th>Carte</th><th class="rg-num">Points</th></tr></thead>
                    <tbody>
                        <tr><th>Valet</th><td class="rg-num">20</td></tr>
                        <tr><th>9</th><td class="rg-num">14</td></tr>
                        <tr><th>As</th><td class="rg-num">11</td></tr>
                        <tr><th>10</th><td class="rg-num">10</td></tr>
                        <tr><th>Roi</th><td class="rg-num">4</td></tr>
                        <tr><th>Dame</th><td class="rg-num">3</td></tr>
                        <tr><th>8</th><td class="rg-num">0</td></tr>
                        <tr><th>7</th><td class="rg-num">0</td></tr>
                        <tr class="rg-total"><th>Total</th><td class="rg-num">62</td></tr>
                    </tbody>
                </table>
            </div>
            <div class="rg-scroll">
                <table class="rg-table">
                    <caption>Hors atout — de la plus forte à la plus faible</caption>
                    <thead><tr><th>Carte</th><th class="rg-num">Points</th></tr></thead>
                    <tbody>
                        <tr><th>As</th><td class="rg-num">11</td></tr>
                        <tr><th>10</th><td class="rg-num">10</td></tr>
                        <tr><th>Roi</th><td class="rg-num">4</td></tr>
                        <tr><th>Dame</th><td class="rg-num">3</td></tr>
                        <tr><th>Valet</th><td class="rg-num">2</td></tr>
                        <tr><th>9</th><td class="rg-num">0</td></tr>
                        <tr><th>8</th><td class="rg-num">0</td></tr>
                        <tr><th>7</th><td class="rg-num">0</td></tr>
                        <tr class="rg-total"><th>Total</th><td class="rg-num">30</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <p>
            L'atout promeut deux cartes&nbsp;: le Valet devient la plus forte du jeu et le 9 la
            deuxième. Les 32 cartes valent donc toujours
            <span class="rg-formula">62 + 3 &times; 30 = 152</span> points, quelle que soit la
            couleur d'atout.
        </p>
    </section>

    <!-- ============================================ -->
    <section class="rg-sec" id="s-donne">
        <h3>La donne</h3>
        <ul class="rg-list">
            <li>Chaque joueur reçoit <strong>8 cartes</strong>.</li>
            <li>Le donneur change à chaque donne.</li>
            <li>
                Le joueur assis <strong>après le donneur</strong> parle le premier, et c'est lui
                qui entamera le premier pli.
            </li>
            <li>
                Le tour passe toujours au joueur suivant, dans le même sens&nbsp;: sur la table de
                Colver, votre voisin de gauche.
            </li>
        </ul>
    </section>

    <!-- ============================================ -->
    <section class="rg-sec" id="s-encheres">
        <h3>Les enchères</h3>
        <p>À son tour de parole, on a quatre possibilités.</p>
        <ul class="rg-list">
            <li>
                <strong>Passer.</strong>
            </li>
            <li>
                <strong>Annoncer un contrat</strong>&nbsp;: une valeur et une couleur d'atout. Les
                valeurs sont 80, 90, 100, 110, 120, 130, 140, 150, 160, puis <strong>capot</strong>
                (annoncer les huit plis, compté 250).
            </li>
            <li>
                <strong>Coincher</strong> le contrat d'un adversaire, ce qui double sa valeur.
            </li>
            <li>
                <strong>Surcoincher</strong>, ce qui la triple — uniquement si c'est le contrat de
                votre camp qui vient d'être coinché.
            </li>
        </ul>

        <h4>Enchérir</h4>
        <ul class="rg-list">
            <li>
                Toute annonce doit être <strong>strictement supérieure en valeur</strong> à la
                précédente. La couleur est libre&nbsp;: il n'y a pas de hiérarchie entre couleurs,
                et rien n'empêche de reprendre la même.
            </li>
            <li>
                <strong>Passer n'élimine pas</strong>&nbsp;: on peut reparler à un tour suivant.
            </li>
            <li>
                <strong>Surenchérir sur son partenaire est permis</strong> — c'est même le cœur du
                jeu, puisque l'enchère est le seul moment où les deux mains se parlent.
            </li>
            <li>
                Le capot est le sommet de l'échelle&nbsp;: rien ne le surenchérit.
            </li>
        </ul>

        <h4>Coincher</h4>
        <ul class="rg-list">
            <li>
                La coinche se déclare <strong>à son tour</strong>, jamais à la volée, et seulement
                contre un contrat adverse.
            </li>
            <li>
                Elle <strong>gèle le contrat</strong>&nbsp;: plus aucune annonce n'est possible
                ensuite, seulement la surcoinche ou des passes.
            </li>
            <li>
                Seul le camp coinché peut surcoincher, et la surcoinche <strong>termine
                l'enchère immédiatement</strong>.
            </li>
        </ul>

        <h4>Fin de l'enchère</h4>
        <ul class="rg-list">
            <li><strong>Trois passes consécutives</strong> après une annonce&nbsp;: le contrat est fixé.</li>
            <li><strong>Une surcoinche</strong>&nbsp;: fin immédiate.</li>
            <li>
                <strong>Quatre passes</strong> d'entrée, personne n'a parlé&nbsp;: la donne est
                nulle, 0-0, on redonne.
            </li>
        </ul>
        <div class="rg-note">
            <p>
                Il arrive que <strong>passer soit la seule chose possible</strong>&nbsp;: sur le
                capot de votre partenaire, ou quand votre camp a été coinché et que votre
                partenaire a décliné la surcoinche. Il n'y a alors pas de décision à prendre, et
                Colver passe pour vous.
            </p>
        </div>
    </section>

    <!-- ============================================ -->
    <section class="rg-sec" id="s-jeu">
        <h3>Le jeu de la carte</h3>
        <p>
            Le premier pli est entamé par le joueur assis après le donneur, les suivants par le
            vainqueur du pli précédent. L'entameur joue ce qu'il veut&nbsp;; ensuite, une seule
            question se pose à chaque carte, dans cet ordre.
        </p>

        <ol class="rg-steps">
            <li>
                <strong>Avez-vous la couleur demandée&nbsp;?</strong> Alors vous devez la fournir.
                <ul>
                    <li>Si la couleur demandée n'est pas l'atout&nbsp;: n'importe laquelle de vos cartes de cette couleur, aucune obligation de monter.</li>
                    <li>Si c'est l'atout qui a été demandé&nbsp;: vous devez <strong>monter</strong> au-dessus du plus fort atout déjà posé, si vous le pouvez.</li>
                </ul>
            </li>
            <li>
                <strong>Sinon, votre partenaire est-il maître du pli&nbsp;?</strong> Alors vous
                jouez <strong>n'importe quelle carte, sans exception</strong>. On ne coupe pas son
                partenaire, et rien d'autre ne vous est imposé — pas même de monter à l'atout s'il
                ne vous reste que ça.
            </li>
            <li>
                <strong>Sinon, avez-vous de l'atout&nbsp;?</strong> Alors vous devez couper.
                <ul>
                    <li>Un adversaire a déjà coupé&nbsp;: vous devez <strong>surcouper</strong>, si vous le pouvez.</li>
                    <li>
                        Vous ne pouvez pas surcouper&nbsp;: à vous de choisir entre
                        <strong>sous-couper</strong> et <strong>vous défausser</strong> d'une autre
                        couleur. C'est la règle dite « on ne pisse pas ».
                        <ul>
                            <li>Sauf s'il ne vous reste que de l'atout&nbsp;: il faut alors sous-couper.</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li>
                <strong>Pas d'atout non plus&nbsp;?</strong> Vous jouez n'importe quelle carte.
            </li>
        </ol>

        <div class="rg-note">
            <p>
                Les deux pièges classiques&nbsp;: hors atout on ne monte <em>jamais</em> par
                obligation (fournir suffit), et quand le partenaire tient le pli l'obligation
                tombe <em>entièrement</em>, y compris celle de surcouper.
            </p>
        </div>
    </section>

    <!-- ============================================ -->
    <section class="rg-sec" id="s-pli">
        <h3>Qui remporte le pli</h3>
        <ul class="rg-list">
            <li>Le <strong>plus fort atout</strong> posé sur le pli.</li>
            <li>S'il n'y a pas d'atout&nbsp;: la plus forte carte <strong>de la couleur demandée</strong>.</li>
            <li>Une carte d'une autre couleur ne peut pas remporter le pli, quelle que soit sa valeur.</li>
        </ul>
        <p>Le vainqueur ramasse le pli et entame le suivant.</p>
    </section>

    <!-- ============================================ -->
    <section class="rg-sec" id="s-belote">
        <h3>La belote</h3>
        <p>
            Le <strong>Roi et la Dame d'atout dans la même main</strong> valent
            <strong>20&nbsp;points</strong> à leur équipe. Sur Colver il n'y a rien à annoncer&nbsp;:
            le bonus est acquis dès que le même joueur a joué les deux cartes.
        </p>
        <ul class="rg-list">
            <li>Ces 20 points <strong>comptent pour réaliser le contrat</strong> — une belote peut sauver une annonce.</li>
            <li>Ils ne font pas partie des 152 points des cartes&nbsp;: c'est un bonus par-dessus.</li>
            <li>Les deux cartes doivent être <em>jouées</em>. Un Roi d'atout gardé en main jusqu'au bout ne rapporte rien.</li>
            <li>En cas de chute, ou sous une coinche, toutes les belotes vont au camp qui marque.</li>
        </ul>
    </section>

    <!-- ============================================ -->
    <section class="rg-sec" id="s-points">
        <h3>Les points de la donne</h3>
        <p>
            Les 32 cartes valent 152 points. Le dernier pli en rapporte 10 de plus — le
            <strong>dix de der</strong> — ou <strong>100</strong> si la même équipe a fait les huit
            plis (<strong>capot</strong>).
        </p>
        <div class="rg-scroll">
            <table class="rg-table">
                <thead><tr><th>Donne</th><th>Cartes</th><th>Dix de der</th><th class="rg-num">Total</th></tr></thead>
                <tbody>
                    <tr><th>Ordinaire</th><td>152</td><td>+ 10</td><td class="rg-num">162</td></tr>
                    <tr><th>Capot</th><td>152</td><td>+ 100</td><td class="rg-num">252</td></tr>
                </tbody>
            </table>
        </div>
        <p>
            Ce total se partage entre les deux camps. La belote, elle, ne s'y ajoute pas&nbsp;:
            elle vient <em>après</em> le partage, et les deux camps peuvent l'avoir chacun de leur côté.
        </p>
    </section>

    <!-- ============================================ -->
    <section class="rg-sec" id="s-marque">
        <h3>La marque</h3>
        <p>
            Le contrat est <strong>réussi</strong> si les points de plis du preneur, belote
            comprise, atteignent la valeur annoncée. Pour un capot annoncé, il faut les huit plis.
            Sinon c'est une <strong>chute</strong>.
        </p>
        <p>
            C'est la <em>seule</em> condition&nbsp;: il n'est pas nécessaire de faire plus de points
            que la défense. Annoncer 80 et ramasser 80 points de plis pile suffit, même si la
            défense en tient 82. <a href="/regles/choix?q=q-plus-que-defense">Pourquoi&nbsp;?</a>
        </p>

        <div class="rg-scroll">
            <table class="rg-table">
                <caption>Sans coinche</caption>
                <thead><tr><th></th><th>Preneurs</th><th>Défense</th></tr></thead>
                <tbody>
                    <tr class="rg-win">
                        <th>Réussi</th>
                        <td>leurs points + contrat + leur belote</td>
                        <td>leurs points + leur belote</td>
                    </tr>
                    <tr class="rg-lose">
                        <th>Chute</th>
                        <td>0</td>
                        <td>162 + contrat + toutes les belotes</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="rg-scroll">
            <table class="rg-table">
                <caption>Coinché (&times;2) ou surcoinché (&times;3)</caption>
                <thead><tr><th></th><th>Preneurs</th><th>Défense</th></tr></thead>
                <tbody>
                    <tr class="rg-win">
                        <th>Réussi</th>
                        <td>162<sup>*</sup> + contrat &times; mult + toutes les belotes</td>
                        <td>0</td>
                    </tr>
                    <tr class="rg-lose">
                        <th>Chute</th>
                        <td>0</td>
                        <td>162 + contrat &times; mult + toutes les belotes</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <ul class="rg-list">
            <li>
                <sup>*</sup> <strong>252</strong> si les preneurs ont réalisé les huit plis,
                annoncé ou non.
            </li>
            <li>
                Le multiplicateur porte sur la <strong>valeur du contrat seule</strong>&nbsp;: ni
                sur les points de plis, ni sur la belote.
            </li>
            <li>
                En cas de chute, la défense prend le contrat <strong>et la totalité des points de
                la donne</strong>, quel que soit le partage réel des plis. Les preneurs marquent 0.
            </li>
            <li>
                Le <strong>capot est un contrat ordinaire valant 250</strong>, pas une prime
                forfaitaire&nbsp;: les tableaux ci-dessus s'appliquent tels quels. Un capot annoncé
                et réalisé vaut donc <span class="rg-formula">252 + 250 = 502</span>.
            </li>
            <li>
                <strong>Aucun arrondi.</strong> Les scores sont marqués au point près.
            </li>
        </ul>
    </section>

    <!-- ============================================ -->
    <section class="rg-sec" id="s-exemples">
        <h3>Quelques exemples</h3>
        <div class="rg-ex">
            <div class="rg-ex-card">
                <h5>Contrat tenu</h5>
                <p>Annonce 80 à cœur, les preneurs ramassent 92 points de plis.</p>
                <p class="rg-ex-out">Preneurs <strong>172</strong> (92 + 80) &middot; Défense <strong>70</strong></p>
            </div>
            <div class="rg-ex-card">
                <h5>Chute</h5>
                <p>Annonce 100 à pique, les preneurs ne font que 82 points.</p>
                <p class="rg-ex-out">Preneurs <strong>0</strong> &middot; Défense <strong>262</strong> (162 + 100)</p>
            </div>
            <div class="rg-ex-card">
                <h5>La belote sauve le contrat</h5>
                <p>Annonce 100, 88 points de plis, plus la belote&nbsp;: 88 + 20 = 108, c'est tenu.</p>
                <p class="rg-ex-out">Preneurs <strong>208</strong> (88 + 100 + 20)</p>
            </div>
            <div class="rg-ex-card">
                <h5>Contrat coinché et tenu</h5>
                <p>Annonce 80 coinchée, contrat réalisé.</p>
                <p class="rg-ex-out">Preneurs <strong>322</strong> (162 + 80 &times; 2) &middot; Défense <strong>0</strong></p>
            </div>
        </div>
    </section>

    <!-- ============================================ -->
    <section class="rg-sec" id="s-partie">
        <h3>La partie</h3>
        <p>
            Une <strong>donne</strong> se joue seule ou à l'intérieur d'une <strong>partie</strong>
            en 1000 ou 2000 points, au choix. Les deux camps marquent à chaque donne, et le score
            cumulé compte&nbsp;: on n'annonce pas la même chose à 900-200 qu'à 0-0.
        </p>
        <ul class="rg-list">
            <li>La partie s'arrête dès qu'un camp <strong>atteint</strong> la cible.</li>
            <li>
                Les deux camps peuvent la franchir sur la même donne&nbsp;: le plus haut score
                l'emporte. En cas d'égalité parfaite, on rejoue une donne.
            </li>
            <li>Une donne nulle (quatre passes) marque 0-0 et ne compte pas.</li>
        </ul>
    </section>

    <!-- ============================================ -->
    <section class="rg-sec" id="s-hors">
        <h3>Ce qu'on ne joue pas</h3>
        <p>
            Quatre variantes courantes, volontairement absentes. Chacune est justifiée sur la page
            <a href="/regles/choix">Pourquoi ces règles</a>.
        </p>
        <ul class="rg-list">
            <li>
                <strong>Les annonces</strong> (tierce, cinquante, cent, carré)&nbsp;: seule la
                belote compte. C'est ce qui sépare la contrée de la coinche.
            </li>
            <li><strong>Le Sans Atout et le Tout Atout</strong>&nbsp;: quatre couleurs d'atout, pas davantage.</li>
            <li><strong>Les paliers au-dessus de 160</strong>&nbsp;: après 160 vient le capot, directement.</li>
            <li><strong>L'arrondi à la dizaine</strong>&nbsp;: la marque est exacte.</li>
        </ul>
    </section>

    </div><!-- /rg-body -->

    <div class="rg-foot">
        <p>
            Un doute, un désaccord&nbsp;? Tout est justifié et sourcé sur
            <a href="/regles/choix">Pourquoi ces règles</a>.
        </p>
    </div>

</div>
</div>
`;

let detach = null;

export function mount(container) {
    container.innerHTML = TEMPLATE;
    detach = wireToc(container);
    openQueryTarget(container, 's');
}

export function unmount() {
    if (detach) detach();
    detach = null;
}
