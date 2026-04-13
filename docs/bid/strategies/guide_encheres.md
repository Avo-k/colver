# Guide d'enchères — Le système du NN V2

Règles distillées du réseau de neurones bid V2 (512-hidden, 3 couches, entraîné sur Double Dummy).
Concordance avec le NN : ~93%.

---

## Évaluer sa main

### Étape 1 : Trouver sa meilleure couleur d'atout

Pour chaque couleur, compter :

**Points d'honneur atout :**

| Carte | Valeur |
|-------|--------|
| Valet | **8** |
| 9 | **6** |
| As | 4 |
| 10 | 3 |
| Roi | 1 |
| Dame | 1 |

**Bonus de longueur :** +(nb_atouts − 2) × 2 si 3+ atouts

**Bonus latéraux** (pour chaque autre couleur) : As = +3, Coupe = +3, Singleton = +1

La couleur avec le score le plus élevé est ta couleur candidate.

### Étape 2 : Le diagnostic rapide

Avant de compter, pose-toi trois questions :

1. **Ai-je le Valet ?** → C'est 80% de la décision.
2. **Combien d'atouts ?** → La longueur est reine.
3. **Ai-je des coupes ?** → Une coupe vaut 3 As latéraux.

---

## Annoncer en ouverture (Position 1)

*Le NN annonce 80% du temps.*

### Avec le Valet d'atout

| Main | Action | Niveau |
|------|--------|--------|
| **J + 9** + n'importe quoi (≥ 2 atouts) | **ANNONCE** | 80 à 120 selon longueur |
| **J** + 2+ atouts (sans 9, sans A) | **ANNONCE** | 80 |
| **J** + 1 atout + As latéral ou coupe | Possible (90%) | 80 |
| **J** seul, sans rien | Passe (70%) | — |

**Niveau avec J + 9 :**

| Total atouts | Niveau typique |
|--------------|---------------|
| 2 (J+9) | 80 |
| 3 | 80-90 |
| 4 | **110** |
| 5 | 110-120 |
| 6 | 120 |

**Bonus belote (K+Q d'atout) :** ne change pas la décision, mais autorise **+10 sur le niveau**. Avec J+9+K+Q en 4 atouts → 120 (37% du temps, contre 22% sans belote).

### Avec le 9 sans Valet

| Main | Action | Niveau |
|------|--------|--------|
| **9** + 4+ atouts | **ANNONCE** (98%) | 80 |
| **9** + 3 atouts + coupe | **ANNONCE** (93%) | 80 |
| **9** + 3 atouts sans coupe | Possible (50%) | 80 |
| **9** + 2 atouts | Rarement (10%) | — |
| **9** + As atout + coupe | **ANNONCE** (97%) | 80-90 |
| **9** + As atout, pas de coupe | Possible (78%) | 80 |

Le 9 seul est un honneur fragile. Il a besoin de **longueur + distribution** pour justifier une annonce.

### Sans Valet ni 9

| Main | Action |
|------|--------|
| 5+ atouts | ANNONCE (94%) |
| 4 atouts + coupe | Possible (50%) |
| 4 atouts sans coupe | Rarement |
| 3 atouts ou moins | **PASSE** |
| "Aux As" (3 As, pas de J/9) | **PASSE** |

---

## Annoncer en 2e (après 1 passe adverse)

*Le NN annonce 66% du temps. Plus sélectif qu'en ouverture.*

Le seuil monte légèrement. L'adversaire suivant peut encore enchérir.

| Seuil | Pos1 | Pos2 |
|-------|------|------|
| J + 2 atouts | 93% | 98% (mieux — info de la passe) |
| 9 + 3 atouts | 79% | 95% (beaucoup mieux) |
| Sans J/9, 4 atouts | 50% | 70% |

La passe de l'adversaire donne de l'information : le NN en profite pour monter.

---

## Annoncer en 3e (après 2 passes)

*Situation asymétrique : ton partenaire a passé.*

### Avec le Valet : annonce agressive (95%)

Le NN protège la donne. Même des mains marginales sont annoncées car sinon la donne meurt (4 passes = redistribution).

### Avec le 9 sans Valet : PRUDENCE (65-75%)

**C'est la pire position pour le 9.** Contrairement au J qui protège, le NN sait que le 9 seul ne sécurise pas le contrat.

| Main | Pos1 | Pos3 |
|------|------|------|
| 9 + 2 atouts | 71% | **64%** (pire !) |
| 9 + 3 atouts | 79% | **75%** |
| 9 + 4 atouts | 98% | **97%** |
| 9 + 3 atouts + coupe | 93% | **98%** (ok avec coupe) |

**Règle :** avec le 9 en Pos3, exiger une coupe ou 4+ atouts.

---

## Annoncer en 4e (dernier à parler)

*Le NN annonce 76% du temps — position de sauvetage.*

Tout le monde a passé. Si tu passes aussi, la donne est nulle. Le NN ouvre largement :

| Main | Taux |
|------|------|
| J + 2 atouts | 100% |
| 9 + 2 atouts | 97% |
| 9 + 3 atouts | 99% |
| Sans J/9, 4 atouts | 86% |
| Sans J/9, 3 atouts | 85% |

Le gap entre J et 9 **disparaît quasi-totalement** en position 4.

Le NN monte aussi en niveau : 9 + 4 atouts en Pos4 → avg 95 (vs 83 en Pos1). Il profite de l'information que personne n'a rien.

---

## Répondre au partenaire (il a annoncé 80)

*Le NN surenchérit dans 82% des cas.*

### Dans sa couleur

Si ton partenaire annonce 80♠ et que tu as des ♠ :

| Support | Action |
|---------|--------|
| 3+ cartes dans sa couleur | **90** quasi-systématique |
| J ou 9 de sa couleur | **90** |
| 0-1 cartes | Annonce dans une autre couleur si possible |

### Dans une autre couleur

| Ta main | Action |
|---------|--------|
| J + 2+ atouts dans ta couleur | Annonce ta couleur à 90 |
| Bonne distribution (coupes) | Annonce ta couleur |
| Rien de spécial | Passe (laisse le partenaire jouer) |

Le NN annonce dans **une autre couleur** 87% du temps quand il a 0 cartes chez le partenaire. Il cherche son propre jeu.

---

## Défendre (l'adversaire a annoncé 80)

*Le NN contre-annonce 55%, coinche 17%, passe 28%.*

### Contre-annoncer

La clé : **combien de cartes as-tu dans la couleur de l'adversaire ?**

| Tes cartes dans sa couleur | Action |
|---------------------------|--------|
| **0-1 carte** | Contre-annonce facilement (dans ta meilleure couleur) |
| **2 cartes** | Contre-annonce si bon jeu (score > 14 ou coupe) |
| **3 cartes** | Difficile — passe sauf très bon jeu |
| **4+ cartes** | **PASSE** (ses atouts sont forts contre toi) |

### Coincher

Le NN coinche 17% du temps. Profil type du coinche :
- **3+ cartes dans la couleur de l'adversaire** (tu tiens ses atouts)
- J ou 9 de sa couleur = encore mieux
- Pas besoin d'un gros jeu (score moyen = 12)
- La Position 3 coinche le plus (28%)

---

## Ce que le NN a appris de contre-intuitif

### L'As d'atout est un piège

L'As d'atout a une contribution **négative** à l'annonce. En duo :

| Combo (2 atouts) | Avantage |
|-------------------|----------|
| J + 9 | +0.085 |
| J + 7 | +0.054 |
| **J + A** | **−0.030** |

Un 7 d'atout est meilleur qu'un As d'atout. L'As est battu par J et 9, prend une place sans contrôler le jeu, et empêche d'avoir une coupe ailleurs.

**Exception :** le 9 + As fonctionne bien (le 9 a besoin d'aide, l'As la fournit).

### La hiérarchie vraie des cartes en atout

Pour la décision d'annoncer, la valeur réelle des cartes est :

```
Valet >>>>>>> 9 >> 7 ≈ 8 ≈ Dame ≈ Roi > 10 > As
```

Tout sauf J et 9 est interchangeable. Un 7 de plus est mieux qu'un 10 ou un As.

### Une coupe vaut 3 As latéraux

| Côtés (avec J+9+7 atout) | Avantage |
|---------------------------|----------|
| 0 as, 2 coupes | **+0.222** |
| 0 as, 1 coupe | +0.202 |
| 2 as, 0 coupe | +0.075 |
| 3 as, 0 coupe | +0.044 |

La distribution est plus importante que les honneurs latéraux.

### La belote ne change pas la décision, mais le niveau

K+Q d'atout n'aide pas à décider *si* on annonce (synergie ≈ 0). Mais la belote permet de monter de 10 : elle sécurise 20 points gratuits.

### "Annoncer aux As" ne fonctionne pas

3 As latéraux sans J ni 9 → le NN passe. Les As ne font pas de plis en atout.

### K+Q latéral aide, A+10 latéral nuit

Une paire K+Q dans une couleur latérale est le seul combo de cartes hautes qui aide (+0.025). L'As+10 dans la même couleur latérale nuit (−0.055) : trop de cartes dans une couleur qu'on voudrait courte.

---

## Mémo de table

```
┌─────────────────────────────────────────────────────┐
│  ANNONCER ?                                         │
│                                                     │
│  Valet + 2 atouts .................. OUI (80)       │
│  Valet + 9 ......................... OUI (80-120)   │
│  9 seul + 3 atouts + coupe ........ OUI (80)       │
│  9 seul + 4+ atouts ............... OUI (80)       │
│  Pas de J/9 + 5+ atouts ........... OUI (80)       │
│  Sinon ............................. PASSE           │
│                                                     │
│  AJUSTEMENTS PAR POSITION                           │
│  Pos3 (partner passé) : +agressif avec J            │
│                          +prudent avec 9 seul       │
│  Pos4 (dernier) : ouvrir quasi-tout                 │
│                                                     │
│  NIVEAU                                             │
│  +10 par atout au-delà de 3                         │
│  +10 si belote                                      │
│  +10 si 2+ coupes                                   │
│                                                     │
│  NE PAS COMPTER                                     │
│  As d'atout = neutre (voire négatif)                │
│  As latéraux = faible (coupes >> as)                │
│  10 d'atout = poids mort                            │
│                                                     │
│  COINCHER                                           │
│  3+ cartes dans la couleur adverse → coinche        │
└─────────────────────────────────────────────────────┘
```

---

*Distillé de 1.8M décisions du NN V2 par analyse SHAP et arbres de décision. Voir `docs/BID_RULES.md` pour les données brutes et la méthodologie.*
