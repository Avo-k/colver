# Comment annonce Bid v6, en familles de mains

*Écrit le 2026-08-03. Produit par
[bid_rules_table.py](../../../scripts/analysis/bid_rules_table.py) à partir des mesures
de [bid_rule_ceiling.py](../../../scripts/analysis/bid_rule_ceiling.py) ; provenance et
sha256 des poids dans `docs/measurements/index.jsonl`.*

La politique d'annonce de v6, écrite en familles `HandCode`
([hand_classification.md](hand_classification.md)) plutôt qu'en arbre de décision. Une
ligne = une famille, avec **ce que v6 répond** et **à quel point on peut s'y fier**.

Ce qui distingue ces tables de [bid_rules_xgb_v2.md](bid_rules_xgb_v2.md) :
elles portent sur le bidder de référence actuel (v6, pas v2), le vocabulaire est mesuré
au solve DD apparié plutôt que choisi à la main, et surtout **chaque ligne porte son
propre plafond** — voir [rule_ceiling.md](rule_ceiling.md).

---

## Les règles en une page

Ce que les trois tables disent, sans les chiffres. **À lire avec le §Limites** : ce sont
les règles de v6, pas les bonnes règles, et une main n'explique qu'un quart de ce qui
décide de sa valeur.

**Ouverture.** Compter les atouts de sa meilleure couleur, et regarder si le Valet y est.

| main | annonce |
|---|---|
| Valet + 3 atouts ou plus | **90** (110 à quatre atouts) — sans hésiter |
| Valet + 2 atouts | **80**, mais c'est la limite : v6 lui-même hésite |
| Valet + As d'atout, 2 atouts, rien derrière | **passe** — l'As d'atout ne rattrape rien |
| Valet seul | **passe**, sauf longue latérale |
| 9 sans Valet, 3 atouts | **80** ; avec un As d'atout en plus, c'est un tirage au sort |
| ni Valet ni 9 | **passe** |

**Défense, l'adversaire a ouvert.** La question n'est pas ce qu'on a dans *sa* couleur —
c'est ce qu'on a dans **sa meilleure autre couleur à soi**.

| meilleure autre couleur | réponse |
|---|---|
| 3 cartes dont le Valet | **90** — sûr |
| 2 cartes dont le Valet, ou 3 cartes sans le Valet | **passe** |

**Soutien, le partenaire a ouvert.** On soutient presque toujours, à **90**. Avec le
Valet de sa couleur et une carte de plus, monter à **110**.

---

## Lire les colonnes

| colonne | quoi |
|---|---|
| **famille** | `T<atouts>.<gros atouts>.<côté>` — `T3.J9` = trois atouts dont le Valet et le 9 |
| **part / cum.** | fréquence de la famille, et couverture cumulée en descendant |
| **décision** | annoncer ou passer : ce que v6 fait le plus souvent |
| **accord** | part des 24 renommages de couleurs où v6 fait bien ça. C'est la fiabilité **de cette ligne** |
| **plaf.** | ce qu'aucune règle insensible aux couleurs ne peut dépasser ici |
| **niveau / 2ᵉ** | la valeur annoncée la plus fréquente, et la suivante |

**Les deux colonnes de plafond sont la clé de lecture.** Une ligne à 84 % d'accord sous
un plafond de 91 % est *finie* : les 7 points restants sont du bruit de symétrie du
réseau, pas une règle qui manque. La même à 84 % sous un plafond de 98 % dit qu'il reste
une vraie règle à trouver — et le code est trop grossier pour la porter.

L'ancre de la famille est **la couleur qu'on envisagerait** (`argmax evaluate_for_trump`)
à l'ouverture, et **la couleur annoncée** dès qu'une enchère a été faite : sous préfixe,
l'atout n'est plus à choisir.

---

## 1. Ouverture — premier à parler

*120,000 mains × 24 permutations, modèle `models/bid_v6_isdd_resume/bid_nn_final.bin` (sha256 9443671cab1e35bb), run 2026-08-03T00:25:11.*

| # | famille | part | cum. | décision | accord | plaf. | niveau | 2ᵉ | accord | plaf. |
|--:|---|--:|--:|---|--:|--:|---|---|--:|--:|
| 1 | `T3.J.-/-/-` | 6.8 % | 7 % | **annonce** | 99 % | 100 % | **90** | 80 | 65 % | 81 % |
| 2 | `T2.J.-/-/-` | 6.6 % | 13 % | **annonce** | 66 % | 92 % | **80** | passe | 63 % | 90 % |
| 3 | `T3.J9.-/-/-` | 6.1 % | 19 % | **annonce** | 100 % | 100 % | **90** | 100 | 44 % | 83 % |
| 4 | `T3.JT.-/-/-` | 5.5 % | 25 % | **annonce** | 99 % | 100 % | **90** | 80 | 65 % | 79 % |
| 5 | `T3.JA.-/-/-` | 5.1 % | 30 % | **annonce** | 99 % | 99 % | **90** | 80 | 45 % | 77 % |
| 6 | `T3.9T.-/-/-` | 4.7 % | 35 % | **annonce** | 75 % | 93 % | **80** | passe | 64 % | 87 % |
| 7 | `T3.9.-/-/-` | 4.7 % | 39 % | **annonce** | 77 % | 94 % | **80** | passe | 56 % | 85 % |
| 8 | `T2.J9.-/-/-` | 4.1 % | 43 % | **annonce** | 97 % | 99 % | **90** | 80 | 54 % | 79 % |
| 9 | `T3.9A.-/-/-` | 3.8 % | 47 % | **annonce** | 53 % | 94 % | **80** | passe | 51 % | 92 % |
| 10 | `T2.9.-/-/-` | 3.2 % | 50 % | **passe** | 85 % | 97 % | **passe** | 80 | 85 % | 96 % |
| 11 | `T2.JT.-/-/-` | 2.9 % | 53 % | **annonce** | 65 % | 92 % | **80** | passe | 56 % | 88 % |
| 12 | `T4.J9.-/-/-` | 2.4 % | 56 % | **annonce** | 100 % | 100 % | **110** | 120 | 61 % | 85 % |
| 13 | `T4.JT.-/-/-` | 2.3 % | 58 % | **annonce** | 100 % | 100 % | **100** | 90 | 39 % | 85 % |
| 14 | `T4.JA.-/-/-` | 2.2 % | 60 % | **annonce** | 100 % | 100 % | **90** | 100 | 44 % | 89 % |
| 15 | `T2.JA.-/-/-` | 2.2 % | 62 % | **passe** | 52 % | 90 % | **passe** | 80 | 52 % | 89 % |
| 16 | `T4.9T.-/-/-` | 2.1 % | 65 % | **annonce** | 100 % | 100 % | **90** | 80 | 67 % | 81 % |
| 17 | `T4.9A.-/-/-` | 2.1 % | 67 % | **annonce** | 99 % | 100 % | **90** | 80 | 47 % | 77 % |
| 18 | `T3.T.-/-/-` | 1.8 % | 68 % | **passe** | 65 % | 96 % | **passe** | 80 | 65 % | 92 % |
| 19 | `T2.9T.-/-/-` | 1.7 % | 70 % | **passe** | 74 % | 96 % | **passe** | 80 | 74 % | 95 % |
| 20 | `T4.J9A.-/-/-` | 1.6 % | 72 % | **annonce** | 100 % | 100 % | **110** | 100 | 53 % | 87 % |
| 21 | `T4.AT.-/-/-` | 1.6 % | 73 % | **annonce** | 63 % | 94 % | **80** | passe | 46 % | 87 % |
| 22 | `T3.J9A.-/-/-` | 1.6 % | 75 % | **annonce** | 100 % | 100 % | **90** | 100 | 57 % | 81 % |
| 23 | `T4.J9T.-/-/-` | 1.6 % | 77 % | **annonce** | 100 % | 100 % | **110** | 120 | 67 % | 85 % |
| 24 | `T3.J9T.-/-/-` | 1.6 % | 78 % | **annonce** | 100 % | 100 % | **90** | 100 | 45 % | 84 % |
| 25 | `T4.JAT.-/-/-` | 1.5 % | 80 % | **annonce** | 100 % | 100 % | **90** | 100 | 56 % | 84 % |
| 26 | `T4.9AT.-/-/-` | 1.5 % | 81 % | **annonce** | 99 % | 100 % | **80** | 90 | 53 % | 75 % |
| | *47 familles de plus* | 18.9 % | 100 % | | | | | | | |

### Ce que ça dit, en mots

1. **Trois atouts avec le Valet : on annonce, sans hésitation** (`T3.J*`, 17 % des mains,
   accord 99-100 %). Le niveau est 90, avec 80 comme repli — mais le *niveau* n'est sûr
   qu'à 65 %, contre un plafond de 81 %.
2. **Deux atouts avec le Valet : c'est la frontière.** `T2.J` (6,6 %) annonce 80 à 66 %
   seulement, sous un plafond de 92 %. C'est la famille que la règle publiée
   « J + ≥ 2 atouts → ANNONCE » couvrait sans nuance, et c'est aussi la plus incertaine.
3. **`T2.JA` passe.** Valet **et** As d'atout avec seulement deux atouts : v6 passe. Le
   §6 de la distillation v2 avait trouvé l'effet (« l'As d'atout est toxique, J+A est
   pire que J seul ») ; ici il porte un nom de famille et une fréquence — 2,2 % des mains.
4. **Quatre atouts avec le Valet : 100 à 110, systématiquement** (accord 100 %). C'est la
   zone la plus lisible de toute la politique.
5. **Le 9 sans Valet ne suffit pas à trois atouts** : `T3.9` annonce 80 à 77 %, `T3.9A`
   à 53 % — presque un tirage au sort.

### Où le code est trop grossier

Familles dont l'accord est le plus loin de leur plafond : ce sont celles où **le côté
décide**, et où le niveau `trump` le jette.

| famille | part | accord | plafond | manque |
|---|--:|--:|--:|--:|
| `T3.9A.-/-/-` | 3,8 % | 53 % | 94 % | **41 pt** |
| `T2.JA.-/-/-` | 2,2 % | 52 % | 90 % | **38 pt** |
| `T3.T.-/-/-` | 1,8 % | 65 % | 96 % | 31 pt |
| `T4.AT.-/-/-` | 1,6 % | 63 % | 94 % | 31 pt |
| `T2.JT.-/-/-` | 2,9 % | 65 % | 92 % | 28 pt |
| `T2.J.-/-/-` | 6,6 % | 66 % | 92 % | 26 pt |

Détaillée d'un cran, `T2.JA` devient parfaitement lisible :

### `T2.JA.-/-/-` détaillée au niveau `tops`

*2,584 mains de cette famille.*

| # | famille | part | cum. | décision | accord | plaf. | niveau | 2ᵉ | accord | plaf. |
|--:|---|--:|--:|---|--:|--:|---|---|--:|--:|
| 1 | `T2.JA.x3/x2/x1` | 8.9 % | 9 % | **passe** | 84 % | 91 % | **passe** | 80 | 84 % | 91 % |
| 2 | `T2.JA.A3/x2/x1` | 6.6 % | 15 % | **passe** | 73 % | 86 % | **passe** | 80 | 73 % | 86 % |
| 3 | `T2.JA.T3/x2/x1` | 5.2 % | 21 % | **passe** | 78 % | 93 % | **passe** | 80 | 78 % | 93 % |
| 4 | `T2.JA.A2/x2/x2` | 5.1 % | 26 % | **passe** | 78 % | 89 % | **passe** | 80 | 78 % | 89 % |
| 5 | `T2.JA.T2/x2/x2` | 4.9 % | 31 % | **passe** | 97 % | 97 % | **passe** | 80 | 97 % | 97 % |
| 6 | `T2.JA.x2/x2/x2` | 4.6 % | 35 % | **passe** | 99 % | 99 % | **passe** | 80 | 99 % | 99 % |
| 7 | `T2.JA.AT3/x2/x1` | 3.7 % | 39 % | **passe** | 59 % | 85 % | **passe** | 80 | 59 % | 85 % |
| 8 | `T2.JA.A2/T2/x2` | 3.6 % | 43 % | **passe** | 76 % | 88 % | **passe** | 80 | 76 % | 88 % |
| 9 | `T2.JA.x3/T2/x1` | 3.2 % | 46 % | **passe** | 63 % | 86 % | **passe** | 80 | 63 % | 86 % |
| 10 | `T2.JA.x3/A2/x1` | 3.1 % | 49 % | **annonce** | 64 % | 84 % | **80** | passe | 64 % | 84 % |
| | *96 familles de plus* | 51.1 % | 100 % | | | | | | | |

**Valet + As d'atout, deux atouts, et une main plate derrière (2-2-2, aucun As ni Dix de
côté) : v6 passe à 99 %.** C'est la formulation exacte du résultat « l'As d'atout est
toxique » — il l'est *quand rien d'autre ne rattrape la main*.

    uv run python scripts/analysis/bid_rules_table.py --tag v6-opening \
        --refine "T2.JA.-/-/-" --refine-level tops

---

## 2. Défense — l'adversaire a ouvert à 80

Ici l'ancre est **sa** couleur. Au niveau `trump`, les familles sont des pièces jetées :

*120,000 mains × 24 permutations, modèle `models/bid_v6_isdd_resume/bid_nn_final.bin` (sha256 9443671cab1e35bb), run 2026-08-03T00:28:38.*

| # | famille | part | cum. | décision | accord | plaf. | niveau | 2ᵉ | accord | plaf. |
|--:|---|--:|--:|---|--:|--:|---|---|--:|--:|
| 1 | `T1.-.-/-/-` | 13.3 % | 13 % | **annonce** | 68 % | 98 % | **90** | passe | 55 % | 95 % |
| 2 | `T2.-.-/-/-` | 7.7 % | 21 % | **annonce** | 51 % | 98 % | **passe** | 90 | 49 % | 95 % |
| 3 | `T0.-.-/-/-` | 7.0 % | 28 % | **annonce** | 87 % | 99 % | **90** | 110 | 66 % | 93 % |
| 4 | `T2.9.-/-/-` | 5.1 % | 33 % | **passe** | 51 % | 98 % | **passe** | 90 | 51 % | 95 % |
| 5 | `T2.J.-/-/-` | 5.1 % | 38 % | **passe** | 54 % | 97 % | **passe** | 90 | 54 % | 94 % |
| 6 | `T2.A.-/-/-` | 5.1 % | 43 % | **annonce** | 59 % | 98 % | **90** | passe | 51 % | 95 % |
| 7 | `T2.T.-/-/-` | 5.1 % | 48 % | **annonce** | 54 % | 98 % | **passe** | 90 | 46 % | 95 % |
| 8 | `T1.J.-/-/-` | 3.3 % | 52 % | **annonce** | 61 % | 98 % | **90** | passe | 46 % | 94 % |
| 9 | `T1.9.-/-/-` | 3.3 % | 55 % | **annonce** | 67 % | 98 % | **90** | passe | 54 % | 94 % |
| 10 | `T1.T.-/-/-` | 3.2 % | 58 % | **annonce** | 68 % | 98 % | **90** | passe | 58 % | 95 % |
| | *66 familles de plus* | 41.7 % | 100 % | | | | | | | |

Accord 51-68 % sous un plafond de 96-98 % : **30 points d'écart**, contre ~10 à
l'ouverture. Le code ne décrit pas ce qui décide.

La raison est structurelle. Ancré sur la couleur adverse, `hand_code` réduit *mes* trois
autres couleurs à « As / Dix / longueur » — et jette donc le Valet et le 9 de ma
meilleure couleur, c'est-à-dire exactement ce qui dit si je peux contre-annoncer. En
ajoutant le descripteur de ma deuxième couleur (`|3J9` = trois cartes dont Valet et 9),
la même politique devient nette :

**Niveau `trump+2e`** —

*120,000 mains × 24 permutations, modèle `models/bid_v6_isdd_resume/bid_nn_final.bin` (sha256 9443671cab1e35bb), run 2026-08-03T00:28:38.*

| # | famille | part | cum. | décision | accord | plaf. | niveau | 2ᵉ | accord | plaf. |
|--:|---|--:|--:|---|--:|--:|---|---|--:|--:|
| 1 | `T1.-.-/-/-\|2J` | 1.1 % | 1 % | **passe** | 73 % | 96 % | **passe** | 90 | 73 % | 96 % |
| 2 | `T1.-.-/-/-\|3J` | 0.9 % | 2 % | **annonce** | 98 % | 99 % | **90** | passe | 98 % | 98 % |
| 3 | `T2.-.-/-/-\|2J` | 0.8 % | 3 % | **passe** | 82 % | 96 % | **passe** | 90 | 82 % | 96 % |
| 4 | `T1.-.-/-/-\|3J9` | 0.8 % | 4 % | **annonce** | 100 % | 100 % | **90** | 110 | 89 % | 92 % |
| 5 | `T1.-.-/-/-\|3JA` | 0.7 % | 4 % | **annonce** | 90 % | 96 % | **90** | passe | 89 % | 96 % |
| 6 | `T1.-.-/-/-\|3JT` | 0.7 % | 5 % | **annonce** | 98 % | 99 % | **90** | passe | 97 % | 99 % |
| 7 | `T2.9.-/-/-\|2J` | 0.6 % | 6 % | **passe** | 84 % | 97 % | **passe** | 90 | 84 % | 97 % |
| 8 | `T2.A.-/-/-\|2J` | 0.6 % | 6 % | **passe** | 67 % | 93 % | **passe** | 90 | 67 % | 93 % |
| 9 | `T2.T.-/-/-\|2J` | 0.5 % | 7 % | **passe** | 80 % | 95 % | **passe** | 90 | 80 % | 95 % |
| 10 | `T2.J.-/-/-\|2J` | 0.5 % | 7 % | **passe** | 84 % | 97 % | **passe** | 90 | 84 % | 97 % |
| 11 | `T1.-.-/-/-\|39` | 0.5 % | 8 % | **passe** | 71 % | 97 % | **passe** | 90 | 71 % | 97 % |
| 12 | `T2.-.-/-/-\|3J` | 0.5 % | 8 % | **annonce** | 97 % | 99 % | **90** | passe | 96 % | 98 % |
| 13 | `T1.-.-/-/-\|39A` | 0.5 % | 9 % | **passe** | 85 % | 98 % | **passe** | 90 | 85 % | 97 % |
| 14 | `T0.-.-/-/-\|3J9` | 0.4 % | 9 % | **annonce** | 100 % | 100 % | **90** | 110 | 90 % | 93 % |
| 15 | `T2.-.-/-/-\|29` | 0.4 % | 9 % | **passe** | 97 % | 100 % | **passe** | 90 | 97 % | 100 % |
| 16 | `T1.-.-/-/-\|2J9` | 0.4 % | 10 % | **annonce** | 92 % | 97 % | **90** | passe | 90 % | 97 % |
| | *2767 familles de plus* | 90.1 % | 100 % | | | | | | | |

**En défense, ce qui décide n'est pas ma main dans leur couleur, c'est ma meilleure autre
couleur.** Trois cartes avec le Valet ailleurs → contre-annonce à 90, sûr à 98 %. Deux
cartes avec le Valet, ou trois avec le 9 seul → passe.

C'est la même chose que le probe avait vue côté réseau (`opp_best_other_ts`,
[probe_morning_report.md](probe_morning_report.md)), retrouvée par un autre chemin et
cette fois avec sa taille : la table gagne **+22,7 points** en passant de `trump` à
`trump+2e`, et `trump+2e` (2 647 codes) bat `full` (5 028 codes) de **9,7 points** —
moins de familles, plus de pouvoir explicatif.

Nuance qui compte : **l'angle mort est celui du code, pas celui des features**. Les 17
features publiées se calculent sur *ma* meilleure couleur, donc elles la voient déjà ; y
ajouter la deuxième ne vaut que +0,8 point ici. C'est `HandCode`, ancré sur la couleur de
l'adversaire, qui perd l'information — [rule_ceiling.md](rule_ceiling.md) §3.

---

## 3. Soutien — le partenaire a ouvert à 80, l'adversaire a passé

*120,000 mains × 24 permutations, modèle `models/bid_v6_isdd_resume/bid_nn_final.bin` (sha256 9443671cab1e35bb), run 2026-08-03T00:31:46.*

| # | famille | part | cum. | décision | accord | plaf. | niveau | 2ᵉ | accord | plaf. |
|--:|---|--:|--:|---|--:|--:|---|---|--:|--:|
| 1 | `T1.-.-/-/-` | 13.3 % | 13 % | **annonce** | 75 % | 97 % | **90** | passe | 55 % | 92 % |
| 2 | `T2.-.-/-/-` | 7.7 % | 21 % | **annonce** | 90 % | 97 % | **90** | passe | 73 % | 91 % |
| 3 | `T0.-.-/-/-` | 7.0 % | 28 % | **annonce** | 92 % | 98 % | **90** | 110 | 52 % | 90 % |
| 4 | `T2.J.-/-/-` | 5.1 % | 33 % | **annonce** | 98 % | 98 % | **110** | 120 | 62 % | 76 % |
| 5 | `T2.A.-/-/-` | 5.1 % | 38 % | **annonce** | 81 % | 97 % | **90** | passe | 64 % | 92 % |
| 6 | `T2.T.-/-/-` | 5.1 % | 43 % | **annonce** | 88 % | 97 % | **90** | passe | 74 % | 92 % |
| 7 | `T2.9.-/-/-` | 5.1 % | 48 % | **annonce** | 98 % | 99 % | **90** | 110 | 63 % | 87 % |
| 8 | `T1.9.-/-/-` | 3.4 % | 52 % | **annonce** | 83 % | 96 % | **90** | passe | 63 % | 90 % |
| 9 | `T1.T.-/-/-` | 3.3 % | 55 % | **annonce** | 74 % | 97 % | **90** | passe | 56 % | 93 % |
| 10 | `T1.J.-/-/-` | 3.3 % | 58 % | **annonce** | 94 % | 95 % | **90** | 100 | 52 % | 76 % |
| 11 | `T1.A.-/-/-` | 3.2 % | 62 % | **annonce** | 86 % | 97 % | **90** | 110 | 55 % | 90 % |
| 12 | `T3.A.-/-/-` | 2.4 % | 64 % | **annonce** | 100 % | 100 % | **90** | 110 | 58 % | 84 % |
| 13 | `T3.T.-/-/-` | 2.4 % | 66 % | **annonce** | 100 % | 100 % | **90** | 110 | 49 % | 82 % |
| 14 | `T3.9.-/-/-` | 2.4 % | 69 % | **annonce** | 100 % | 100 % | **110** | 120 | 64 % | 79 % |
| | *61 familles de plus* | 31.1 % | 100 % | | | | | | | |

Le régime le plus simple des trois : **v6 soutient presque toujours**, et la question est
le niveau. Une famille se détache — `T2.J` dans la couleur du partenaire (Valet + une
carte) monte à **110**, sûr à 98 %.

---

## Limites

- **Régimes seulement, pas l'enchère entière.** Trois positions figées (ouverture,
  défense sur 80, soutien sur 80). Une enchère réelle en enchaîne davantage.
- **Score 0-0.** v6 est score-aware : à 900-200 il annonce autrement. Tout ce document
  décrit son comportement en début de partie.
- **Le niveau annoncé reste mal expliqué partout** (accord 40-70 % contre des plafonds de
  76-95 %). La décision d'annoncer est lisible ; sa valeur beaucoup moins.
- **Ce sont les règles de v6, pas les bonnes règles.** v6 se contredit sous renommage de
  couleurs dans un quart des cas ([bid_v7_plan.md](../bid_v7_plan.md) §1.1) ; les
  colonnes « plafond » chiffrent cette limite mais ne la corrigent pas. Un bidder
  canonique rendrait ces tables exactes plutôt qu'approchées.
- **Et une main ne décide qu'un quart de sa propre valeur.** Mesuré : la main explique
  **23,5 %** de la variance de sa valeur DD, le reste étant la répartition des 24 autres
  cartes ([hand_classification.md](hand_classification.md) §6.1). Aucune table de ce
  genre — ni aucun évaluateur de main, si parfait soit-il — ne peut promettre davantage.
  C'est aussi pourquoi l'enchère est un jeu de communication : le gros de l'information
  n'est pas dans sa main, il est dans ce que les autres annoncent.

## Reproduction

```bash
uv run python scripts/analysis/bid_rule_ceiling.py --deals 120000 --tag v6-opening
uv run python scripts/analysis/bid_rule_ceiling.py --deals 120000 --prior 80C --tag v6-defense
uv run python scripts/analysis/bid_rule_ceiling.py --deals 120000 --prior "80C P" --tag v6-support

uv run python scripts/analysis/bid_rules_table.py --tag v6-opening --limit 26
uv run python scripts/analysis/bid_rules_table.py --tag v6-defense --level "trump+2e"
uv run python scripts/analysis/bid_rules_table.py --tag v6-opening --weak
```

Les tables ne recalculent rien : elles relisent les payloads de `runlog`, qui portent les
24 réponses par main. Reposer une question sur une mesure ne doit pas coûter la mesure.
