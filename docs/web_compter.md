# Compter les points — page d'entraînement

Route `/problemes/compter`, vue [`views/compter.js`](../python/colver/web/static/js/views/compter.js),
feuille [`css/compter.css`](../python/colver/web/static/css/compter.css), session
serveur `CountingSession` dans [`game_manager.py`](../python/colver/web/game_manager.py).

**Implémentée** (2026-08-01). Troisième page de la section « S'entraîner », à
côté des problèmes d'annonce et de jeu.

L'exercice : des plis défilent dans l'ordre où ils ont été joués, avec l'atout
du contrat, et on annonce le total à la fin. On s'entraîne à **compter pendant
la donne**, pas à faire une addition après coup — d'où l'absence de tout
compteur à l'écran et le fait que les tas n'affichent que leur nombre de plis.

## Décisions actées

| | Choix | Pourquoi |
|---|---|---|
| Source des plis | **Donnes réellement jouées** — générées à la volée, ou les parties du joueur | Des cartes tirées au hasard donneraient des plis qu'on ne voit jamais à une table : on s'entraînerait sur la mauvaise distribution |
| Source par défaut | **Génération**, « Mes parties » en option | La base contient quelques dizaines de donnes terminées ; la page en consomme une toutes les vingt secondes |
| Aller-retour | **Un seul** par séquence, la donne entière | La correction est alors locale et instantanée ; un curieux peut lire la réponse dans la console, la seule personne qu'il tromperait est lui-même |
| Ce qu'on compte | Points **cartes** des plis montrés, plus dix de der et belote en mode « partie entière » | Deux niveaux de règle explicites valent mieux qu'un énoncé ambigu |
| Modes de comptage | **un camp** (on ne compte qu'un tas) ou **deux camps** (deux totaux) | Le premier est le geste réel à la table, le second oblige à tenir deux compteurs |

## 1. Les trois façons de faire défiler

Un seul écran, trois méthodes, choisies dans les réglages fins :

| Méthode | Ce qui avance | Clavier | Tactile |
|---|---|---|---|
| `carte` | une carte par pas | `→` `↓` `Espace` avancer, `←` `↑` reculer | tap sur la croix ; balayage = un pli |
| `pli` | quatre cartes par pas | idem | idem |
| `chrono` | tout seul, `speedMs` par carte | `Espace` `P` `Échap` pause/reprise | tap = pause |

**Le retour arrière est refusé en chronométré**, et le refus est dit
(`#pc-hint`) plutôt que silencieux : reculer, ce serait recompter — c'est un
autre exercice. `#pc-prev` porte `disabled` dans ce mode, pour qu'on voie que le
bouton existe et pourquoi il ne répond pas.

**La pause masque la table**. Le voile (`#pc-veil`) ne suffit pas : à
`rgba(10,10,10,.85)` sur une carte blanche, l'As de cœur et le Roi restent
parfaitement lisibles — mesuré, l'exercice se contournait donc tout seul en
s'arrêtant devant un pli complet. Ce sont **les cartes elles-mêmes** qu'on
retire (`#pc-trick-area.pc-paused .trick-card { visibility: hidden }`) ; le
voile ne garde que l'assombrissement et l'étiquette.

**Le maintien de fin est une échéance comme une autre, et la pause doit savoir
la ré-armer.** `pcTimer` porte deux choses de nature différente — le tick du
défilement et le maintien du dernier pli — d'où le drapeau `endHold`. Sans lui,
une pause pendant ce maintien tuait la séquence **pour de bon** : à la reprise
`tick` rappelait `advance(1)` depuis `max`, qui sort avant `afterStep`, donc
`finishRun` n'était plus joignable et l'écran de saisie n'arrivait jamais. Or
c'est le battement « où l'on additionne » : le moment le plus naturel pour
demander une seconde.

## 2. Le moteur de défilement

Tout tient dans un entier, `pcIdx` = nombre de cartes révélées, et une fonction
qui **rejoue tout depuis zéro** :

    pcIdx = 4·ti + k   →   le pli `ti` a `k` cartes posées
    k = 0 (et pcIdx > 0) →  le pli précédent est complet SUR LA TABLE

Un pli n'entre dans son tas qu'au moment où la première carte du suivant tombe
— comme à une table, où l'on ramasse en entamant. Un pli complet reste donc
visible un pas entier : c'est là qu'on additionne.

Rejouer depuis zéro à chaque pas (≤ 32 cartes, coût nul) est ce qui rend le
retour arrière **exact par construction** : il n'y a aucun état incrémental à
défaire. C'est aussi pourquoi la page n'appelle ni `detectTrickCompletion`
(état global de module dans `shared/cards.js`, ne détecte qu'un sens de
transition — le `←` produirait des faux positifs) ni `animateTrickFlush` (qui
envoie les cartes vers *la main du gagnant*, et il n'y a pas de mains ici).

**Un changement de phase coupe TOUT ce qui est armé** (`stopAllTimers` :
tick, maintien de fin, vol). `stopTimer` seul laissait le vol en cours, dont le
callback rappelle `renderAt` puis enchaîne sur `toAnswer` — « Abandonner »
pendant un vol renvoyait aux réglages, puis basculait tout seul sur l'écran de
saisie une seconde et demie plus tard.

**`advance()` ne fait rien hors de la phase `run`.** Les commandes de
défilement sont masquées sur l'écran de saisie (elles n'ont plus rien à faire
avancer, et « Suivant » y rivalisait en or avec « Annoncer »), mais le garde
tient aussi au clavier : sans lui, un ◀ suivi d'un ▶ y relançait `toAnswer`,
qui **vide les champs** — le total qu'on venait de taper disparaissait.

**Le vol précède le repeint.** `flyOut(ti, done)` anime, puis rappelle
`renderAt` : repeindre d'abord animerait la première carte du pli suivant. La
fin du vol est un `setTimeout`, jamais un `transitionend` —
`prefers-reduced-motion` neutralise toutes les transitions (`tokens.css`), et
l'événement ne partirait alors jamais.

En chronométré, **un pli complet tient un temps de plus** (`nextDelay()` rend
`speedMs + FLY_MS`) : c'est le battement pendant lequel on additionne, et c'est
ce qui laisse au vol le temps de se jouer avant la carte suivante. Une donne
entière en Expert (0,5 s) dure donc ≈ 24 s, pas 16.

**La direction est l'information.** Le pli part vers **le siège** qui l'a
ramassé, aux quatre points de la croix — comme à une table, où l'on voit le
vainqueur tirer les cartes à lui. Rien ne l'écrit : on sait à qui est un pli
parce qu'on l'a vu partir de son côté.

L'**axe** porte le camp, et lui seul : vertical pour Nord-Sud, horizontal pour
Est-Ouest. Le **sens**, dans l'axe, désigne lequel des deux partenaires a
ramassé — une information de plus, jamais une de moins. C'est ce qui distingue
ce défilé d'un simple compteur à deux tas : à une vraie table on ne voit pas
« un pli pour Nord-Sud », on voit Nord qui ramasse.

Les quatre distances sont tirées de `--card-h`, pas d'un pourcentage de la
croix : celle-ci est bien plus haute que large, donc un pourcentage de ses
propres dimensions donnerait des départs latéraux deux fois plus courts que les
verticaux — la lisibilité dépendrait de l'axe, ce qui est exactement ce qu'il
ne faut pas.

Les deux tas, eux, restent à gauche et à droite : ils comptent, ils ne sont pas
la destination du vol. Un pli qui part vers l'est ne pointe donc pas son tas, et
c'est sans conséquence — l'étiquette et le compte sont écrits dessus.

## 3. Ce que le joueur doit annoncer

`W` = les `N` premiers plis, `N = 8` en « partie entière », `cfg.nTricks` sinon.

    cards[k] = Σ points des plis de W ramassés par le camp k

- **simple** : `expected[k] = cards[k]`. Les annonces de belote ne sont pas
  affichées du tout — montrer un bonus qui ne compte pas serait un piège.
- **partie entière** (impose `N = 8`) : `expected[k] = cards[k] + der[k] + belote[k]`,
  avec `der` au camp qui ramasse la 8ᵉ levée (10, ou **100** sur un capot
  réalisé) et `belote[k] ∈ {0, 20}`.

Invariant, à `N = 8` : `cards[0] + cards[1] = 152`, quel que soit l'atout
(`card.rs:414-427`, `test_color_points_total`). **La belote n'est jamais un
point carte** : les deux tas font 152 avec ou sans elle, les 20 se posent
par-dessus — c'est pour ça qu'elle est une ligne de pied du tableau, sous le
sous-total cartes.

## 3 bis. Défauts corrigés le 2026-08-02

Repris au navigateur sur les trois formats (grand écran, portable, téléphone).
Les deux premiers rendaient une commande **inatteignable**, le troisième faisait
enseigner une erreur — c'est-à-dire exactement le contraire du but de la page.

| | Symptôme | Cause |
|---|---|---|
| Blocage | Pause pendant le dernier pli (chrono) : plus rien, jamais | `pcTimer` porte deux échéances, la reprise ré-armait la mauvaise — §1 |
| Blocage | « Commencer » sous la découpe en 1280×800, sans ascenseur | `#app` est en `overflow: hidden` au-delà de 640px et la page ne déclarait pas son propre défilement — voir ci-dessous |
| Faux | Dix de der mal attribué diagnostiqué « pli dans le mauvais tas » | ordre de l'échelle — §6 |
| Perte | ◀ sur l'écran de saisie effaçait le total tapé | `advance()` sans garde de phase — §2 |
| Triche | Le voile de pause laissait lire les cartes | 85 % de noir ne cache pas — §1 |
| Triche | La relecture rejouait un essai noté, gagné d'avance | §6 |

**`#app` découpe ce qui dépasse.** `layout.css` le fige à la hauteur de l'écran
avec `overflow: hidden` au-delà de 640px, à charge de chaque vue longue de
déclarer son propre ascenseur — ce que font Regarder et Rejouer. Cette page
dépasse dès que « Réglages fins » est ouvert : sur 1280×800, « Commencer »
tombait 34 px sous la découpe, injoignable à la souris comme à la molette
(seule `Entrée` lançait encore, sans que rien ne le dise) ; en phase correction,
le `scrollIntoView` emportait le bandeau d'atout hors de portée définitivement.
`#app:has(#pc-wrap) { overflow-y: auto }` suffit, et rend au passage sa raison
d'être à la barre de commandes collante.

Réglé en même temps : croix vide masquée en phases `answer`/`review` (≈ 300 px
de cadre pointillé vide, tout le premier écran au téléphone) — **les tas, eux,
restent**, on ne fait pas disparaître ce qu'on demande de compter ; tas qui se
recouvraient sous 390 px (on lisait « 1 plNORD-SUD ») ; barre de commandes
collante qui ne collait à rien et peignait un bandeau `--c-bg-deep` — le fond
des pages *plates* — au milieu du tapis ; curseurs laissés au bleu par défaut de
Chrome ; `:disabled` qui repeignait le ◀ fantôme en pavé gris (même piège que le
`:hover` documenté en tête de `compter.css`) ; champ de réponse non focalisé en
chronométré, c'est-à-dire dans les trois préréglages ; erreur de génération
invisible depuis la correction (`onError` sortait sur `phase !== 'config'`, et
ni `#pc-hint` ni `#pc-fine-note` n'est visible à ce moment-là) ; filet des
statistiques affiché à vide.

## 4. Protocole

Client → serveur, un seul message :

```json
{"type": "count_generate", "req_id": 12, "source": "auto|mes", "seen": ["a1b2"]}
```

`seen` est l'anneau des 20 derniers `game_id` servis, tenu en `localStorage`.
Sans lui, « Mes parties » resservirait la même donne au bout de quelques
tirages : un joueur en a quelques dizaines, pas quelques milliers.

Serveur → client, `count_ready` : le payload de `CountingSession._payload`
aplati — `trump`, `contract`, les **huit** plis (`{no, cards indexées par siège,
lead, winner, points, announces}`), `points` (vérité moteur, dix de der
compris), `card_points`, `der {team, value}`, `belote`, `source`,
`source_degraded`, `game_id`.

On envoie toujours les huit plis, jamais un préfixe : le client décide combien
il en montre selon le niveau, et la correction peut dérouler la donne entière.

### La fenêtre de plis n'est pas forcément les N premiers

`pickWindow` choisit N plis **consécutifs** dans la donne, sous deux conditions
quand N < 8 : le camp à compter doit avoir ramassé **la moitié au moins** des
plis montrés (2 sur 3, 3 sur 5) et plus de zéro point. Sans ça on sert des
séquences dont la réponse est 0 — ce n'est pas un exercice, c'est une donne où
la question ne s'est pas posée. En « deux camps » le seuil tombe à un pli par
camp : en exiger la moitié pour chacun est contradictoire (3 plis, 2 + 2).

La donne entière étant déjà là, faire glisser la fenêtre ne coûte rien et
suffit presque toujours : mesuré sur 2000 donnes réellement jouées, **1,18
donne par séquence à 3 plis, 1,49 à 5 plis**, et jamais de repli. Redemander
une donne au serveur serait le seul autre levier, et il coûte un aller-retour ;
c'est pourquoi il n'intervient qu'en second (`MAX_DEAL_TRIES`), quand aucune
fenêtre ne convient. Après quoi on montre la **moins mauvaise** fenêtre de la
dernière donne, jamais « les N premiers plis » — le repli doit rester le
meilleur exercice disponible, pas un abandon.

À 8 plis on ne filtre rien : il n'y a qu'une fenêtre, et un capot subi se
compte — il vaut 0, légitimement.

`source = "mes"` sans compte, sans donne disponible, ou sur une donne
inexploitable → repli sur la génération avec `source_degraded: true`, **et la
page le dit**. Même politique que `pacing.resolve` avec `mode_degraded`.

## 5. L'assertion qui garde le compte

`_payload` refuse de servir une donne dont le décompte ne tombe pas juste :

```python
sum(cards_pts) == 152  et  der_value ∈ {10, 100}
et  engine_pts[autre camp] == cards_pts[autre camp]
```

Le dix de der est **lu** sur `env.get_points()` (le moteur l'y a déjà versé,
`play.rs:273-283`) au lieu d'être recodé, et l'écart sert du même coup
d'assertion sur le vainqueur du dernier pli — la seule valeur que cette page ne
peut pas se permettre de rater. Une donne générée qui échoue est régénérée ; une
donne de la base qui échoue est signalée dans le journal et remplacée par une
donne générée.

Elle a immédiatement servi : **2 des 17 donnes terminées de la base de dev ont
des `actions` incohérentes avec leurs `hands`** (des cartes jouées deux fois,
d'autres jamais — `env.step()` ne valide pas la légalité, c'est au moteur RL le
comportement attendu). Ces donnes sont aussi fausses dans Rejouer ; c'est un
problème de données antérieur à cette page.

## 6. Correction

Un verdict par camp compté, avec **toujours l'écart**, jamais un juste/faux nu :
juste / « presque » sous 5 points / le chiffre annoncé contre le vrai.

Le diagnostic est une échelle ordonnée, premier match gagnant — l'ordre est
load-bearing, ±20 étant ambigu (belote, ou Valet d'atout compté 2) et ±10 aussi
(dix de der, ou un 10). **Cet ordre a été violé et ça se voyait** : une règle
« vos deux totaux somment juste, c'est un pli qui est allé dans le mauvais tas »
était testée en tête de boucle. Or donner le dix de der ou la belote au mauvais
camp **somme juste aussi**. Sur une donne contenant un pli valant exactement
10 points, un dix de der mal attribué produisait donc : « C'est le pli n°6
(10 pts, ramassé par Nord) qui est allé dans le mauvais tas » — faux deux fois,
puisque ce n'était pas un pli et que celui-là était correctement attribué. Cette
règle n'est plus qu'une **nuance du rang 5**, testée après le dix de der et la
belote.

1. les deux totaux intervertis (un seul message, pas un par camp) ;
2. le total de l'autre camp ;
3. dix de der oublié / donné au mauvais camp ;
4. belote oubliée / attribuée au mauvais camp ;
5. un pli entier oublié, nommé par son numéro et son ramasseur ;
6. le 9 d'atout compté 0, le Valet d'atout compté 2 ;
   — testés sur `d === -14` / `d === -18`, pas sur `Math.abs(d)` : ces deux
   messages disent « vous avez compté trop peu », et sur un écart positif ils
   accusaient d'une erreur exactement inverse de celle commise ;
7. sinon, renvoi à la colonne Cumul.

Les rangs 5 et 6 vérifient **la direction** de l'écart (`(d < 0) === (t.winner
% 2 === k)`) : sans elle, un pli de la bonne valeur mais du bon côté fait
accuser le joueur d'une erreur qu'il n'a pas commise.

Le tableau pli par pli porte la valeur de chaque carte en exposant, et le 9 et
le Valet d'atout un anneau accent : c'est là que l'œil apprend. **Au téléphone
il déborde de plus de la moitié de sa largeur** — les trois colonnes chiffrées,
dont la colonne Cumul que le rang 7 invite justement à lire, sont hors champ.
D'où `#pc-table-swipe`, affiché sous 640px seulement : un conteneur qui défile
sans le dire ne défile pas.

**Une relecture n'est pas un essai.** « Revoir au ralenti » repasse en phase
`run` sur la même donne, dont la réponse vient d'être affichée juste au-dessus :
elle revient donc à la **correction** (`toReview`) et non à la saisie, et
`recordStats` est sauté (`inReplay`). Avant, on pouvait répondre faux, lancer la
relecture, lire la réponse dans le tableau resté à l'écran et la re-saisir :
série et record montaient d'un essai gagné d'avance — sous une clé de
statistiques différente, qui plus est, puisque la relecture force `method =
'carte'`. `statsKey()` retombe sur `methodBeforeReplay` pour la même raison.

## 7. Réglages

Trois préréglages (Débutant 3 plis · 1,5 s, Confirmé 5 plis · 0,7 s, Expert 8 plis
· 0,5 s · deux camps · partie entière) qui **écrivent tous les réglages fins**,
pour qu'on voie ce qu'on a choisi. Toucher un réglage fin bascule sur `perso`,
et les statistiques changent de clé : un record Expert ne se mélange pas à un
réglage sur mesure.

`perso` n'a **pas de bouton**, et c'est un piège à deux têtes. (1) La tabulation
roulante de `setSeg` met `tabIndex = -1` partout où rien n'est coché : le groupe
Niveau se retrouvait sans **aucun** arrêt de tabulation, donc injoignable au
clavier — et comme `preset` est persisté, définitivement. `setSeg` garde
désormais un arrêt sur le premier bouton quand rien ne correspond. (2) Les trois
niveaux éteints donnaient l'impression d'un groupe qui a perdu sa sélection :
d'où la puce `#pc-preset-note`.

`localStorage` : `colver:compter:cfg`, `colver:compter:stats`
(`{"<preset>|<method>": {plays, exact, sumAbsDelta, streak, best}}`),
`colver:compter:seen`.

## 8. À faire — d'autres questions sur la même donne

Le défilé de plis est un **moteur**, pas un exercice : il montre une donne
réelle carte par carte et interroge à la fin. Le total des points n'est qu'une
question parmi d'autres, et c'est la plus dure. Plusieurs questions plus faciles
portent sur exactement la même séquence, sans rien changer au payload — tout est
déjà dans `count_ready` (les huit plis, l'atout, le contrat).

**Questions candidates**, de la plus simple à la plus dure :

| Question | Ce qu'elle entraîne | Réponse |
|---|---|---|
| Combien d'atouts sont tombés ? | le comptage d'atouts, le premier réflexe de table | un nombre 0-8 |
| Combien de cartes dans chaque couleur ? | suivre les couleurs, repérer les coupes | quatre nombres |
| Qui a coupé à quelle couleur ? | la lecture des coupes franches et acquises | un siège × une couleur |
| Quelles cartes sont maîtresses ? | ce qui décide la fin de donne | une sélection de cartes |
| Combien de points ? | l'exercice actuel | un nombre par camp |

Deux niveaux d'ambition, à trancher au moment de le faire :

1. **Un mode = une question**, choisie dans les réglages. Simple, lisible, et
   chaque mode garde ses propres statistiques (`statsKey` porte déjà le
   préréglage et la méthode, il suffit d'y ajouter la question).
2. **Des questions tirées procéduralement à la fin**, une ou plusieurs parmi un
   catalogue, paramétrées par la donne (« combien de piques sont tombés ? »,
   « le Roi de cœur est-il encore maître ? »). C'est ce qui ressemble le plus à
   une vraie table, où l'on ne sait pas d'avance ce qu'il faudra savoir — et ça
   interdit de ne suivre qu'une seule chose.

Ce qu'il faudra décider :
- **Le tirage doit être vérifiable.** Une question générée doit avoir une réponse
  calculable côté client depuis le payload, sinon la correction redevient un
  aller-retour serveur (§8 s'y oppose).
- **Une seule question ou plusieurs ?** Plusieurs, c'est le vrai exercice, mais
  ça change la forme de la correction (un verdict par question) et des
  statistiques (par question, ou par lot ?).
- **« Cartes maîtresses » demande un état de jeu**, pas seulement un décompte :
  maître *à quel moment* — à la fin, ou à un pli donné ? C'est la seule question
  de la liste qui a besoin d'autre chose que les plis déjà tombés.
- **La difficulté ne vient pas de la question mais du débit.** Compter les atouts
  à 0,5 s par carte est plus dur que compter les points à 2 s : les préréglages
  devront se relire à l'aune de la question choisie.

## 9. Hors périmètre, volontairement

| Écarté | Pourquoi |
|---|---|
| Correction côté serveur | Le payload porte déjà la vérité ; un second aller-retour ajouterait de la latence dans la boucle la plus serrée pour empêcher un joueur de se tromper lui-même |
| État dans l'URL | Une donne tirée au hasard n'est pas une adresse. À rouvrir pour partager une donne de la base (`?game=<id>`) |
| Préchargement de la donne suivante | La génération coûte ~20 ms, ça ne se voit pas ; ça imposerait une tâche de fond et donc `wsend` |
| Statistiques serveur, classement | L'entraînement est solitaire ; `/score` établit le précédent d'une page 100 % locale |
| Import d'une donne à compter depuis Rejouer | Utile plus tard ; demande un champ, une validation et un chemin de retour |
