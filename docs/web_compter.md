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
| Modes de comptage | **un camp** (on ne compte qu'un tas) ou **deux camps** (deux totaux, plis dans deux directions) | Le premier est le geste réel à la table, le second oblige à tenir deux compteurs |

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

**La pause masque la table** (`#pc-veil`). Sans ça, s'arrêter devant un pli
complet est un retour en arrière déguisé et l'exercice se contourne tout seul.

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

**Le vol précède le repeint.** `flyOut(ti, done)` anime, puis rappelle
`renderAt` : repeindre d'abord animerait la première carte du pli suivant. La
fin du vol est un `setTimeout`, jamais un `transitionend` —
`prefers-reduced-motion` neutralise toutes les transitions (`tokens.css`), et
l'événement ne partirait alors jamais.

En chronométré, **un pli complet tient un temps de plus** (`nextDelay()` rend
`speedMs + FLY_MS`) : c'est le battement pendant lequel on additionne, et c'est
ce qui laisse au vol le temps de se jouer avant la carte suivante. Une donne
entière en Expert (0,5 s) dure donc ≈ 24 s, pas 16.

**La direction est l'information.** En « deux camps », un pli qui descend est à
Nord-Sud, un pli qui monte est à Est-Ouest ; rien ne l'écrit. C'est ce qu'on lit
à une table, où l'on sait à qui est un pli parce qu'on l'a vu partir de son
côté.

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
(dix de der, ou un 10) :

1. les deux totaux intervertis (un seul message, pas un par camp) ;
2. le total de l'autre camp ;
3. dix de der oublié / donné au mauvais camp ;
4. belote oubliée / attribuée au mauvais camp ;
5. un pli entier oublié, nommé par son numéro et son ramasseur ;
6. le 9 d'atout compté 0, le Valet d'atout compté 2 ;
7. sinon, renvoi à la colonne Cumul.

Le tableau pli par pli porte la valeur de chaque carte en exposant, et le 9 et
le Valet d'atout un anneau accent : c'est là que l'œil apprend.

## 7. Réglages

Trois préréglages (Débutant 3 plis · 1 s, Confirmé 5 plis · 0,7 s, Expert 8 plis
· 0,5 s · deux camps · partie entière) qui **écrivent tous les réglages fins**,
pour qu'on voie ce qu'on a choisi. Toucher un réglage fin bascule sur `perso`,
et les statistiques changent de clé : un record Expert ne se mélange pas à un
réglage sur mesure.

`localStorage` : `colver:compter:cfg`, `colver:compter:stats`
(`{"<preset>|<method>": {plays, exact, sumAbsDelta, streak, best}}`),
`colver:compter:seen`.

## 8. Hors périmètre, volontairement

| Écarté | Pourquoi |
|---|---|
| Correction côté serveur | Le payload porte déjà la vérité ; un second aller-retour ajouterait de la latence dans la boucle la plus serrée pour empêcher un joueur de se tromper lui-même |
| État dans l'URL | Une donne tirée au hasard n'est pas une adresse. À rouvrir pour partager une donne de la base (`?game=<id>`) |
| Préchargement de la donne suivante | La génération coûte ~20 ms, ça ne se voit pas ; ça imposerait une tâche de fond et donc `wsend` |
| Statistiques serveur, classement | L'entraînement est solitaire ; `/score` établit le précédent d'une page 100 % locale |
| Import d'une donne à compter depuis Rejouer | Utile plus tard ; demande un champ, une validation et un chemin de retour |
