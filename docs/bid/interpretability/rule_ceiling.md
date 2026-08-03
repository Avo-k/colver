# Le plafond d'une règle humaine — et où les règles publiées butent

*Mesuré le 2026-08-03. Scripts : [bid_rule_ceiling.py](../../../scripts/analysis/bid_rule_ceiling.py),
[bid_rules_by_family.py](../../../scripts/analysis/bid_rules_by_family.py). Provenance et
sha256 des poids : `docs/measurements/index.jsonl`, tags `v6-opening` / `v6-defense` /
`v6-support` / `v2-opening`.*

[bid_rules_xgb_v2.md](bid_rules_xgb_v2.md) distille le bidder en règles lisibles et
annonce ~93 % d'accord. Deux choses manquaient à ce chiffre : on ne savait pas **où**
vivaient les 7 %, ni **s'ils** étaient rattrapables. Ce document répond aux deux, en
croisant l'équivariance ([bid_v7_plan.md](../bid_v7_plan.md) §1.1) et la classification
des mains ([hand_classification.md](hand_classification.md)).

Les règles elles-mêmes, écrites famille par famille, sont dans
[bid_rules_v6.md](bid_rules_v6.md). Ici, la métrologie.

---

## 1. Il existe un plafond, et il dépend beaucoup du régime

Une règle écrite pour un humain ne connaît pas le nom des couleurs — « J + 2 atouts →
annonce » vaut pique comme trèfle. Elle est donc **équivariante**, tout comme les 17
features agrégées de l'arbre. Sur l'orbite d'une main sous les 24 renommages, une règle
équivariante ne peut sortir qu'une réponse ; son meilleur choix est le mode. D'où une
borne qui ne dépend d'aucun choix de features :

> **plafond = moyenne sur les mains de (effectif du mode / 24)**

120 000 mains × 24 permutations = 2,88 M annonces par ligne :

| régime | action exacte | annoncer/passer | mains stables |
|---|---:|---:|---:|
| **ouverture** (v6) | **83,5 %** | **97,4 %** | 38,0 % |
| ouverture (v2) | 83,1 % | 96,8 % | 45,5 % |
| **défense** (l'adversaire ouvre à 80) | **93,5 %** | 97,5 % | 73,1 % |
| **soutien** (le partenaire ouvre à 80) | **82,5 %** | 98,1 % | 37,9 % |

*« Mains stables » = les mains dont les 24 renommages donnent tous la même réponse.*

**Lecture.** Annoncer ou passer est presque déterminé partout (97-98 %). L'**action
exacte** ne l'est pas : à l'ouverture un sixième des annonces de v6 est littéralement
indécidable pour une règle qui traite les couleurs à égalité. Le §5 de
`bid_rules_xgb_v2` (« niveaux d'annonce ») décrivait donc une cible dont un sixième est
du bruit d'apprentissage.

La défense fait exception (93,5 %, 73 % de mains stables) — mais pour une raison sans
mérite : on y passe beaucoup, et passer est une action sans couleur, donc insensible au
renommage. Le soutien, où l'on annonce presque toujours, retombe à 37,9 %.

**v2 et v6 plafonnent au même endroit.** Le score-aware, les 75 M de pas et les 117
dimensions d'obs n'ont pas rendu v6 plus cohérent avec lui-même — v2 a même 7 points de
mains stables en plus. C'est un défaut d'architecture, pas d'entraînement, et c'est
exactement ce que la canonicalisation de l'obs (§3.1 du plan v7) supprime.

### La bonne façon de noter une règle

**Accord avec l'orbite** : la fraction des 24 réponses du réseau que la règle retrouve.
C'est la seule note comparable au plafond, quelle que soit la cible d'apprentissage.

Noter contre le **mode** revient à noter contre le réseau *symétrisé*, qui vise 100 % —
une autre échelle. Confondre les deux fait « dépasser » son plafond à une règle, ce qui
est arrivé pendant l'écriture (une table à 97,2 % sous un plafond de 96,8 %) et a servi
de symptôme.

---

## 2. Négatif : entraîner sur le réseau symétrisé n'apporte rien

Le réseau donne jusqu'à 24 réponses par main ; en prendre une au hasard — celle de
l'ordre physique des couleurs, ce que fait le protocole publié — revient à faire
apprendre du bruit à l'arbre. L'idée de viser plutôt le **mode de l'orbite** (le réseau
symétrisé) est naturelle. Elle ne paie pas :

| cible | arbre d5 | XGBoost |
|---|---:|---:|
| annoncer/passer — brute | 90,4 % | 94,3 % |
| annoncer/passer — **mode** | 90,4 % | 94,3 % |
| niveau annoncé — brute | 58,4 % | 76,6 % |
| niveau annoncé — **mode** | 57,0 % | 77,1 % |

Zéro sur la décision, +0,5 point sur le niveau. L'explication est simple une fois vue :
la réponse à l'identité est un **tirage non biaisé** dans l'orbite, et un apprenant qui
voit 84 000 mains moyenne ce bruit tout seul. Symétriser ne fait qu'économiser des
données, pas déplacer un plafond.

⚠️ **Une première version de cette mesure annonçait +7,7 points.** Elle notait la règle
entraînée sur le mode *contre le mode*, et celle entraînée sur la réponse brute *contre
la réponse brute* — deux échelles. C'est le défaut décrit au §1, et il donnait
exactement le résultat qu'on espérait, ce qui est la circonstance où l'on vérifie le
moins.

---

## 3. La deuxième couleur : +1,5 point aux features, +22,7 au code

Aux 17 features publiées, ajouter le `trump_score` et la longueur de la **deuxième
meilleure couleur** (cible = mode, accord à l'orbite, XGBoost) :

| jeu de features | ouverture | défense (décision) | défense (niveau) |
|---|---:|---:|---:|
| publié (17, + 3 relatives sous préfixe) | 92,8 % | 93,1 % | 87,9 % |
| + `opp_best_other_ts` *(le probe)* | — | 93,4 % | 88,4 % |
| **+ 2ᵉ couleur** | **94,3 %** | **93,9 %** | **89,0 %** |
| + les deux | — | 93,9 % | 89,0 % |
| *plafond* | *97,5 %* | *97,5 %* | *94,2 %* |

+1,5 point à l'ouverture, +0,8 en défense ; **rien pour l'arbre de profondeur 5**, qui
n'a pas les niveaux pour s'en servir.

**`opp_best_other_ts` est dominé par `second_trump_score`** : seul il vaut +0,3 à +0,5,
et ajouté par-dessus il ne vaut rien. C'est attendu — il en est exactement
reconstructible (il vaut `trump_score` quand ma meilleure couleur n'est pas celle qu'on
annonce, `second_trump_score` sinon). Le concept trouvé par le probe est le bon, la forme
particulière ne l'est pas.

⚠️ **Ce n'est pas une réfutation du 77 % → 97 % du probe**, qui mesurait autre chose sur
un autre protocole : sa « meilleure couleur » était choisie par le Q du réseau lui-même,
sa cible était la réponse à l'identité, et sa note une exactitude simple. Les deux
mesures ne sont pas sur la même échelle.

**Ce qui est spectaculaire, c'est le même ajout appliqué au *code*, pas aux features** :
+22,7 points en défense (§4). Les deux chiffres coexistent parce que les 17 features
n'ont jamais eu l'angle mort — elles se calculent sur **ma** meilleure couleur, donc
elles la voient. `HandCode`, lui, s'ancre en défense sur la couleur de **l'adversaire**,
et perd alors précisément ce qui décide.

---

## 4. Combien de familles faut-il ? (la table de correspondance)

Une règle par famille, point : `HandCode → une réponse`, apprise sur 70 % des mains et
notée sur les 30 % restantes. C'est la forme la plus littérale qu'un humain puisse
retenir, et sa précision dit combien de familles il faut connaître.

**Ouverture** (accord à l'orbite, décision annoncer/passer / niveau annoncé) :

| niveau | codes | annoncer/passer | niveau annoncé |
|---|---:|---:|---:|
| `length` | 7 | 81,0 % | 38,0 % |
| `trump` | 73 | 87,2 % | 58,2 % |
| `shape` | 252 | 88,5 % | 63,1 % |
| `tops` | 3 095 | 91,9 % | 70,1 % |
| `full` | 3 901 | 92,9 % | 74,7 % |
| `trump` + 2ᵉ couleur | 1 588 | 90,2 % | 66,5 % |
| `full` + 2ᵉ couleur | 10 347 | **94,4 %** | **77,5 %** |
| *plafond* | | *97,4 %* | *85,5 %* |

**73 familles suffisent à retrouver 87 % des décisions annoncer/passer de v6** — et une
table de 10 000 codes n'en gagne que 7 de plus. À titre de comparaison, XGBoost sur 19
features fait 94,3 % : **la table de 73 codes est à 7 points d'un modèle qui n'est pas
lisible du tout.**

**Défense** — et là le classement change complètement :

| niveau | codes | annoncer/passer |
|---|---:|---:|
| `length` | 8 | 61,5 % |
| `trump` | 75 | 64,4 % |
| `shape` | 317 | 75,0 % |
| `full` | 5 028 | 77,4 % |
| **`trump` + 2ᵉ couleur** | **2 647** | **87,1 %** |
| `full` + 2ᵉ couleur | 18 263 | 90,3 % |
| *plafond* | | *97,4 %* |

*Les plafonds du §4 portent sur le jeu de test (30 % des mains), ceux du §1 sur
l'échantillon entier ; d'où 97,4 ici contre 97,5 là pour la défense.*

**`trump+2e` (2 647 codes) bat `full` (5 028 codes) de 9,7 points** — moitié moins de
familles, dix points de mieux. Raffiner le *côté* ne sert presque à rien ; ajouter la
qualité d'atout de **ma** meilleure couleur vaut +22,7 points.

### Ce que `HandCode` ne voit pas, et c'est structurel

`hand_code(main, atout)` décrit une main **avec un atout désigné** : les couleurs de côté
n'y gardent que As / Dix / longueur. C'est justifié pour décrire un *contrat* — le §2 de
[hand_classification.md](hand_classification.md) mesure le Valet de côté à −0,5 point DD.
Mais une décision d'enchère *compare des atouts candidats*, et le Valet d'une couleur de
côté vaut +49 dès qu'on l'y regarde comme atout.

À l'ouverture le manque est modeste (+1,5 point) parce que l'ancre est déjà la meilleure
couleur de la main. En défense l'ancre est **la couleur de l'adversaire** : le code jette
alors précisément ce qui décide, d'où les +22,7 points.

**Conséquence pour `HandCode`** : la suite `length → trump → shape → tops → full`
raffine le côté. Il manque un **axe orthogonal** qui porte les autres atouts possibles.
Ce n'est pas un raffinement du même axe : il faut moins de codes et il explique plus.

**Et ce n'est pas la même hiérarchie que contre le DD.** La même question posée à la
*valeur* plutôt qu'à la *politique* donne un autre classement : `tops` (2 983 codes) y
sature le plafond et `full` n'ajoute rien
([hand_classification.md](hand_classification.md) §6.1). Le résultat qui l'accompagne
mérite d'être retenu à part : **la main n'explique que 23,5 % de la variance de sa propre
valeur DD** — les trois quarts restants sont la répartition des 24 autres cartes. Aucune
évaluation de main ne peut faire mieux, ce qui borne par le haut tout ce que des règles
lisibles peuvent promettre à un joueur.

---

## 5. Ce que ça change pour la suite

- **Ne pas symétriser la cible** (§2) — mesuré nul, et l'idée est assez séduisante pour
  être re-tentée si elle n'est pas consignée.
- **Ajouter la 2ᵉ couleur aux features** de toute distillation future : +1,5 point à
  l'ouverture, bien plus en défense. Gratuit.
- **Publier les règles avec leur domaine** : la colonne « plafond » par famille de
  [bid_rules_v6.md](bid_rules_v6.md), pas un « 93 % » global.
- **Les règles publiées décrivent v2, pas v6.** `distill_bid.rs` ne gère que les obs
  108/110/113 et refuse les 117 de v6 ; les scripts de ce document contournent le binaire
  en passant par le binding Python (~40 µs par annonce, 2,88 M annonces en ~2 min), donc
  l'étendre n'est plus un préalable.
- **Un bidder canonique relève le plafond à 100 % par construction** (§3.1 du plan v7,
  implémentée le 2026-08-02) et rend la §3.6 — la table exhaustive des 472 579 classes —
  bien définie. Ce document devient alors caduc pour le nouveau bidder : les règles se
  lisent sur la table plutôt que sur un proxy.

---

## Reproduction

```bash
uv run python scripts/analysis/bid_rule_ceiling.py --deals 120000 --tag v6-opening
uv run python scripts/analysis/bid_rule_ceiling.py --deals 120000 --prior 80C --tag v6-defense
uv run python scripts/analysis/bid_rule_ceiling.py --deals 120000 --prior "80C P" --tag v6-support
uv run python scripts/analysis/bid_rule_ceiling.py --deals 120000 \
    --bid-model models/bid_v2/bid_nn_final.bin --tag v2-opening
uv run python scripts/analysis/bid_rules_by_family.py --deals 120000 --tag v6-opening
```

~2-3 min pour un plafond, ~15 min pour un run de familles (les ajustements XGBoost).

**Deux pièges rencontrés en écrivant `anchor_suit`**, tous deux silencieux :

1. **L'ancre doit nommer la couleur qu'on envisagerait.** Une clé lexicographique sur
   l'ordre d'atout sur-pondère la plus haute carte : un Valet sec y bat une couleur de
   quatre cartes à l'As, et la famille « `T1.J` » regroupe alors des mains que v6 annonce
   — dans l'autre couleur. Elle reste un ordre total stable au renommage, donc le
   contrôle d'invariance passe et les plafonds restent justes ; seules les **étiquettes**
   mentent. Un plafond juste sous une étiquette fausse est le pire des deux mondes.
   L'ancre est désormais `argmax evaluate_for_trump`.
2. **Dans cette clé, le bit d'une carte est sa force, pas son complément.** Le Valet pèse
   128, pas 1. Inversée, elle désignait la couleur la plus *pauvre* de la main — et rien
   dans la sortie ne le signalait.

Le contrôle qui protège le regroupement vérifie que le **code** survit au renommage (pas
l'indice de couleur : deux couleurs de rangs identiques sont interchangeables et donnent
le même code). Il valide au passage que `evaluate_for_trump` est elle-même équivariante.

**Neuf runs intermédiaires ont été retirés de `docs/measurements/index.jsonl`** (et leurs
payloads de `data/analysis/`) : ils portent les mêmes tags que les runs définitifs, et
leurs chiffres viennent d'un code affecté par l'un des deux défauts ci-dessus ou par la
confusion d'échelle du §1. Le registre est fait pour qu'on relise des mesures sans les
refaire ; y laisser des lignes fausses sous un tag correct est exactement ce qui
transforme cet avantage en piège. Les runs gardés sont horodatés `2026-08-03T00:25` et
après.
