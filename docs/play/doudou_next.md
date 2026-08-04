# Prochaine itération de DouDou

Ce qu'il faut avoir décidé **avant** de relancer un entraînement de jeu.

Toute modification de l'observation invalide les poids : la première couche
change de forme, donc on ne reprend pas DouDou50, on repart de zéro pour 50 M de
pas. La liste doit donc être complète le jour où on lance, pas complétée pendant.
C'est la seule raison d'être de cette page.

Une entrée = ce qu'on ajoute, pourquoi, ce que ça coûte, et **ce qu'on n'en
attend pas**.

---

## 1. La belote annoncée (décidé, pas mesuré)

### Le fait

`dmc_obs.rs` ne porte la belote nulle part — ni dans les 415 hérités, ni dans les
411 canoniques (`grep -n belote colver-core/src/dmc/dmc_obs.rs` ne rend rien).
DouDou joue donc sans savoir qu'une belote a été annoncée.

Or c'est de l'**information publique**, au même titre qu'une coupe révélée :
`check_belote` ([play.rs:283](../../colver-core/src/engine/play.rs)) la pose pour
les quatre sièges dès que le premier des deux honneurs d'atout tombe, exactement
comme à une table où l'annonce est obligatoire et à voix haute. `state.belote` /
`state.belote_player` sont lisibles par tout le monde. Les trois autres familles
de joueurs les lisent déjà — IS-DD depuis `play::belote_facts` (2026-08-03),
`CardBeliefs`, et l'obs d'enchère v6 pour sa part de belote. Le réseau de jeu est
le seul à jouer sourd.

### Ce que ça coûte, deux canaux distincts

**(a) Inférence — qui tient le second honneur d'atout.** C'est la demande
d'origine, et elle se lit dans les deux sens
([is_dd.md](is_dd.md#la-belote-et-pourquoi-elle-a-manqué-trois-ans)) :

| | condition | ce qu'on apprend | fréquence¹ |
|---|---|---|---|
| **annonce** | `belote[t] == 1` | l'annonceur tient l'autre carte, et personne d'autre | 5,7 % des positions |
| **silence** | un Roi ou une Dame d'atout tombe sans annonce | son poseur n'a **jamais** l'autre | 20,5 % des positions |

La déduction « silence » est la plus fréquente des deux et la moins évidente.
Elle est *déjà* à portée du réseau — il voit les cartes jouées par siège
(bloc 3) — mais elle lui est inaccessible sans le bit d'annonce : sans lui, un
Roi d'atout tombé ne se distingue pas d'un Roi d'atout tombé en silence.

**(b) Barème — le seuil de réussite.** `scoring.rs` compte la belote dans
`taker_total` pour décider réussi/chute : elle ne s'ajoute pas au bout, elle
**déplace le seuil**. Le réseau voit les points cartes par camp (bloc 6) et la
valeur du contrat (bloc 4) — donc il juge en permanence « où en est le contrat »,
et il se trompe de 20 points du côté du preneur dans les **22,1 % de donnes où
une belote est annoncée**¹. Pire : la récompense d'entraînement DMC est le
verdict binaire gagné/perdu sur le score de donne
([dmc_env.rs:50](../../colver-core/src/dmc/dmc_env.rs)), donc ce basculement de
seuil *est* exactement ce que le réseau est payé pour prédire. Ce n'est pas un
indice manquant, c'est une récompense partiellement inobservable.

Le canal (b) n'était pas dans la demande initiale ; il est ajouté ici parce
qu'il est réel et qu'il est probablement le plus gros des deux.

¹ 1 593 952 positions sur 50 000 donnes du corpus COLVGM01, `bench_belote_facts`
(2026-08-03, `docs/measurements/index.jsonl`).

### Ce qu'on encode

**4 sièges relatifs × 2 flottants = 8**, soit `OBS_DIM_TR` **411 → 419**. Par
siège `s`, dans l'ordre relatif habituel (`me`, `+1`, `+2`, `+3`) :

- `annonce[s]` : 1,0 si `belote_player[team(s)] == s` et `belote[team(s)] >= 1` ;
- `complete[s]` : 1,0 en plus si `belote[team(s)] == 2` (rebelote posée).

La distinction porte tout le canal (a) : tant que `complete == 0`, la seconde
carte est **encore dans la main** de l'annonceur ; une fois posée, la déduction
est éteinte alors que les 20 points, eux, restent acquis.

**Invariant par permutation de couleurs** — la belote est toujours l'atout, et
l'atout est le slot 0 en canonique. Rien à ajouter dans `suit_perm.rs` (au-delà
de la longueur), contrairement aux 4 bits de belote de l'obs d'enchère v6, qui
sont un par couleur *parce qu'à l'enchère l'atout n'est pas encore nommé*.

**Le siège du lecteur mérite ses deux flottants**, malgré le bloc 1 qui donne sa
main. Mes propres cartes jouées ne sont **pas** dans le bloc 3
(`for &seat in &seats[1..]`) : elles ne se retrouvent qu'en soustrayant le bloc 3
des blocs 8/9. Après que j'ai posé mon Roi, « ma belote est annoncée » est au
bout de cette chaîne-là. Deux flottants coûtent moins cher que ce pari.

### Pourquoi surtout pas 415

411 + 4 = 415 = **exactement la taille de fichier de DouDou35** : les deux sont
1024³ dueling, donc `models/dmc_doudou35.bin` (415, hérité) pèse 10 260 612 o et
`models/dmc_50.bin` (411, canonique) 10 244 228 o — l'écart est 4 × 1024 × 4 o.

Or `DmcNet::load` déduit `obs_dim` **de la taille du fichier**
([dmc_net.rs:101](../../colver-core/src/dmc/dmc_net.rs)) et `agent/dmc.rs:43`
branche l'obs canonique sur `obs_dim == OBS_DIM_TR`. Un réseau canonique à 415
serait donc lu comme un DouDou35 hérité et **joué avec la mauvaise obs, sans une
seule erreur** — mêmes symptômes que `set_residual` oublié : ça joue, c'est
légal, c'est beaucoup plus faible. Même famille que `canonical = true` dans le
TOML d'enchère, où la largeur ne trahit pas non plus la convention.

8 flottants (419) évitent la collision au passage. N'importe quel nombre autre
que 4 l'éviterait ; c'est 8 qu'on veut pour d'autres raisons, la collision n'est
qu'une raison de plus de ne pas « économiser » les deux bits du siège lecteur.

### Ce qu'on n'en attend pas

La même déduction, posée en contrainte dure sur les mondes d'IS-DD, change la
carte **8,5 % du temps** pour un gain de **−0,008 ± 0,031 pt DD par décision**
(z = −0,48) : rien. La raison est dans les données — quand elle fait changer
d'avis, les deux cartes sont à 0,51 pt DD l'une de l'autre (médiane) contre
7,0 pts d'étendue à une position : **elle ne déplace la décision que là où la
décision ne pèse presque rien** ([is_dd.md](is_dd.md), 2026-08-03).

Il n'y a pas de raison de croire que le canal (a) rapporte beaucoup plus à un
réseau qui ne cherche pas qu'à un agent qui cherche. Le canal (b), lui, n'a
**jamais été mesuré pour personne** : IS-DD a le seuil correct depuis
`world_belote_for` (2026-08-03) sans que ça lui ait été facturé séparément.

Donc : correctif **gratuit et correct par construction**, à faire au prochain
entraînement parce qu'il ne coûte rien de plus à ce moment-là. **Pas** une raison
de lancer un entraînement.

### La mesure qui vaut la peine, et elle ne demande pas de GPU

Borne haute du canal (b), quelques minutes de CPU sur
`data/training/isdd_games_v1.bin` (43 076 donnes jouées par IS-DD, enchère
réelle) : compter les donnes où la belote **retourne** le verdict, c'est-à-dire
où le camp qui l'annonce est le preneur et où
`taker_pts < contrat ≤ taker_pts + 20`. C'est le sous-ensemble exact sur lequel
les 8 flottants peuvent changer quelque chose au canal (b). S'il est à 1 %, on
saura qu'on n'a ajouté qu'un joli invariant ; s'il est à 5 %, c'est 5 % de donnes
dont le réseau ne pouvait pas prédire l'issue.

L'A/B qui trancherait pour de bon coûte **deux entraînements complets**, donc on
ne le fera pas. On met les bits, et on ne prétend pas savoir ce qu'ils rapportent.

### Ce qu'il faut toucher

| fichier | quoi |
|---|---|
| `dmc/dmc_obs.rs` | `OBS_DIM_TR`, bloc 10 dans `write_observation_tr`, tests de dimension |
| `dmc/dmc_net.rs:101` | `known_dims = [372, 415, 444]` — la désambiguïsation par taille |
| `suit_perm.rs:469,527` | `permute_dmc_obs_tr` : longueur et pas de batch codés en dur à 411 (le nouveau bloc, lui, est invariant) |
| `joint_env.rs` | le chemin d'entraînement canonique. `dmc/dmc_env.rs` est le chemin hérité 415 : il ne bouge pas |
| `train_joint.rs` | `FlexReplayBuffer` (411/32 → 419/32) et l'auto-détection de l'adversaire de checkpoint |
| `agent/dmc.rs`, `agent/models.rs`, `colver-py/src/lib.rs:1067` | branchent sur `OBS_DIM_TR`, donc suivent la constante — à vérifier, pas à changer |
| docs | [dmc.md](dmc.md), section « Observation Layouts » de `CLAUDE.md` |

---

## 2. Autres candidats

Rien de décidé. Cette section existe pour que l'entrée suivante n'ouvre pas une
deuxième page : une itération de DouDou est un événement rare et cher, tout ce
qui doit monter à bord se note ici.
