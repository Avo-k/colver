# Stratégies d'enchères de Colver

## Contexte du projet

Colver est un moteur de jeu de Belote Contrée optimisé pour l'apprentissage par renforcement (RL). Le coeur est écrit en Rust pour la performance (~1,3M simulations/sec), avec des bindings Python via PyO3. Le moteur implémente les règles officielles FFB (Fédération Française de Belote), en mode de comptage **"points faits + points demandés"**.

Les stratégies d'enchères décrites ici sont des fonctions **déterministes** et très rapides (~200 opérations), utilisées à la fois :
- Comme composant d'agents de recherche Monte-Carlo (IS-MCTS), qui simulent des millions de parties pour choisir la meilleure carte à jouer
- Comme enchérisseur par défaut de notre agent RL (DouZero-style) qui n'apprend que le jeu de la carte
- Comme base pour des tournois automatisés de calibration

---

## 1. Évaluation d'une main

### 1.1 Fonction `evaluate_for_trump(main, atout)`

Toutes les stratégies partagent la même fonction d'évaluation. Pour une main donnée et un atout supposé, on calcule un score composite :

**Honneurs d'atout** (points fixes par carte) :

| Carte  | Valet | 9   | As  | 10  | Roi | Dame | 8   | 7   |
|--------|-------|-----|-----|-----|-----|------|-----|-----|
| Points | 8     | 6   | 4   | 3   | 1   | 1    | 0   | 0   |

**Bonus de longueur d'atout** : `max(0, nombre_atouts - 2) × 2`
- 3 atouts = +2, 4 atouts = +4, 5 atouts = +6, etc.

**Couleurs annexes** (pour chacune des 3 couleurs non-atout) :
- As = +3
- Chicane (0 carte) = +3
- Singleton = +1

**Plage typique** : 0 à 35 environ.

### 1.2 Exemples concrets

| Main                              | Atout  | Calcul                                            | Score |
|-----------------------------------|--------|---------------------------------------------------|-------|
| V♠ 9♠ A♠ 10♠ R♠ 7♥ 7♦ 7♣        | Pique  | (8+6+4+3+1) + (5-2)×2 + 0 + 3 + 3 = 34          | 34    |
| V♥ 9♥ 8♥ 7♠ 8♠ 7♦ 8♦ 7♣         | Coeur  | (8+6) + (3-2)×2 + 0 + 0 + 0 = 16                 | 16    |
| V♦ A♦ 10♦ A♠ 7♥ 8♥ 7♣ 8♣        | Carreau| (8+3+4) + (3-2)×2 + 3 + 0 + 0 = 20               | 20    |
| 7♠ 8♠ 7♥ 8♥ 7♦ 8♦ 7♣ 8♣         | (tout) | 0 partout                                          | 0     |
| V♠ 9♠ A♠ A♥ A♦ A♣ 7♥ 7♦          | Pique  | (8+6+4) + (3-2)×2 + 3 + 3 + 3 = 29               | 29    |

### 1.3 Discussion

Cette évaluation est **statique et simpliste** : elle ne tient compte ni de la position à la table, ni des enchères précédentes (sauf pour le boost partenaire dans `heuristic_bid`), ni de la distribution probable des cartes chez les adversaires.

Les poids ont été choisis empiriquement — le Valet d'atout vaut 8 et le 9 vaut 6, ce qui reflète leur puissance dominante à l'atout. Les bonus de chicane et singleton capturent grossièrement le potentiel de coupe.

---

## 2. Stratégie `heuristic_bid` (agressive)

### 2.1 Philosophie

C'est la stratégie la plus simple : on évalue les 4 couleurs, on prend la meilleure, et on convertit le score directement en palier d'enchère. Pas de filtre de qualité, pas de plafond. Enchérit souvent, enchérit haut.

### 2.2 Fonctionnement détaillé

1. **Évaluer les 4 couleurs** : `evaluate_for_trump(main, couleur)` pour chaque couleur.

2. **Boost partenaire** : si le partenaire a fait la dernière enchère, on ajoute +3 au score de sa couleur (reconnaissance basique de soutien).

3. **Choisir la meilleure couleur** : celle avec le score le plus élevé.

4. **Table de conversion score → enchère** :

   | Score    | Enchère |
   |----------|---------|
   | < 10     | Passe   |
   | 10 – 13  | 80      |
   | 14 – 16  | 90      |
   | 17 – 19  | 100     |
   | 20 – 22  | 110     |
   | 23 – 25  | 120     |
   | 26+      | 130     |

5. **Vérification** : si le palier calculé ne surenchérit pas sur l'enchère en cours → Passe. On vérifie aussi la légalité.

### 2.3 Pas de coinche

`heuristic_bid` ne coinche jamais et ne surcoinche jamais.

### 2.4 Pas de filtre de qualité (quality gate)

Même une couleur avec D-R et rien d'autre (score = 2 en atout) peut être choisie si c'est la meilleure — en pratique le seuil de 10 filtre les mains très faibles.

### 2.5 Résultats en tournoi

- **Taux de prise de contrat** : ~50% des donnes
- **Taux de réussite des contrats** : ~70%
- En tournoi round-robin (100 matchs/paire, 20ms/coup), la version `very_aggressive` (≈ heuristic sans quality gate) termine 2e sur 6 avec 59,5% de victoires

### 2.6 Limites connues

- Surenchérit trop souvent sur des mains fragiles (pas de J ni 9 d'atout)
- Monte trop haut sans plafond, ce qui expose à des chutes coûteuses
- N'exploite pas les coinches pour punir les adversaires

---

## 3. Stratégie `smart_bid` (conservative, conventionnelle)

### 3.1 Philosophie

Inspirée des conventions de jeu humain en Belote Contrée. Repose sur la communication par les enchères entre partenaires : ouvrir à 80 signale un Valet OU un 9 (pas les deux), et le partenaire complète avec l'honneur manquant. C'est l'approche la plus « humaine ».

### 3.2 Ouverture

**Condition stricte** : la couleur doit contenir au moins le Valet OU le 9 d'atout.

- **Valet ET 9** (« le 34 ») :
  - 2+ As annexes ou 4+ atouts → **100**
  - 1 As annexe → **90**
  - Sinon → **80**

- **Valet OU 9** (pas les deux) :
  - 3+ cartes dans la couleur → **80** (signale à mon partenaire que j'ai un honneur mais pas l'autre)
  - Sinon → Passe

- **Fallback « aux As »** : si 2+ As au total mais aucune couleur avec V ou 9 → **80** dans la meilleure couleur par score

### 3.3 Réponse au partenaire

- Si le partenaire a ouvert à **80** : c'est un signal V-ou-9. Si j'ai l'honneur manquant (V ou 9 dans sa couleur) → **90** (confirmation du 34)
- Si le partenaire a ouvert à **90+** : il a déjà le V+9, je ne surenchéris pas → Passe

### 3.4 Surenchère sur adversaire (overcall)

- Seulement si l'enchère adverse est **< 100**
- Je dois avoir V+9 dans une **autre** couleur avec un score ≥ 14
- L'enchère est plafonnée à **100**

### 3.5 Coinche

Conditions pour coincher l'adversaire :
- J'ai **V+9** dans la couleur qu'il a annoncée, OU
- J'ai **4+ atouts** dans sa couleur, OU
- Il a enchéri 120+, j'ai **3+ atouts** dans sa couleur + **1 As annexe**

### 3.6 Résultats en tournoi

- **Taux de prise** : ~10-13% (très conservateur)
- **Taux de réussite** : ~78%
- Passe trop souvent, laisse la main aux adversaires gratuitement
- Ne profite pas des mains moyennement fortes

### 3.7 Limites connues

- Le système V/9 est binaire : ne distingue pas entre un V+9+petits et un V+9+A+10
- Ne sait pas surenchérir de manière compétitive (plafond 100)
- Le fallback "aux As" est trop passif
- La réponse au partenaire est rudimentaire (ne relance qu'à 90, jamais plus haut)

---

## 4. Stratégie `improved_bid` (équilibrée, calibrée par tournoi)

### 4.1 Philosophie

Stratégie hybride : on garde un filtre de qualité pour éviter les enchères aberrantes, mais on utilise une table score→enchère optimisée par tournoi avec des plafonds par situation. C'est la stratégie par défaut du moteur.

### 4.2 Filtre de qualité (quality gate)

Avant d'enchérir, la couleur choisie doit satisfaire **au moins une** condition :
- Contenir le **Valet** d'atout
- Contenir le **9** d'atout
- Contenir l'**As** d'atout
- Contenir le **10** d'atout
- Avoir **3+ cartes** dans la couleur

**Raisonnement** : une couleur sans gros honneur et avec seulement 1-2 cartes ne peut pas servir d'atout de manière viable — même si le score est élevé grâce aux couleurs annexes (chicanes, As).

### 4.3 Table de conversion score → enchère (calibrée par tournoi)

| Score    | Enchère |
|----------|---------|
| < 10     | Passe   |
| 10 – 12  | 80      |
| 13 – 16  | 90      |
| 17 – 19  | 100     |
| 20 – 24  | 110     |
| 25+      | 120     |

**Différences avec `heuristic_bid`** :
- Le palier 80 s'ouvre dès 10 (idem), mais le palier 90 à 13 au lieu de 14 — plus agressif en bas
- Le palier 110 a une plage plus large (20-24 au lieu de 20-22)
- Le palier 120 au lieu de 130 maximum — plus prudent en haut
- Pas de palier 130 (plafond structurel à 120)

### 4.4 Ouverture

1. Évaluer les 4 couleurs → choisir la meilleure
2. Filtre de qualité : si la meilleure couleur ne passe pas → Passe
3. Convertir le score via la table ci-dessus
4. **Plafond d'ouverture : 120** (même avec une main monstre, on n'ouvre pas à 130+)

**Raison du plafond** : en tournoi, ouvrir trop haut a un coût asymétrique — si le contrat échoue, l'adversaire encaisse 160 + valeur du contrat. Ouvrir à 120 au maximum laisse le partenaire relancer si sa main le justifie.

### 4.5 Réponse au partenaire

Le partenaire a ouvert. Je décide si je relance :

1. **Si le partenaire a enchéri 130+** : je ne surenchéris pas → Passe

2. **Soutien dans la couleur du partenaire** : j'évalue ma main dans SA couleur avec la même table score→enchère. Si mon palier est supérieur au sien, je relance dans sa couleur. **Plafond de réponse : 130**.

3. **Changement de couleur** : si je ne peux pas soutenir mais j'ai une couleur alternative avec :
   - Score ≥ 16 (main forte)
   - Qui passe le filtre de qualité
   - Palier supérieur à l'enchère actuelle

   → Je propose ma couleur (plafond 120)

### 4.6 Surenchère sur adversaire (overcall)

1. **Si l'adversaire a enchéri 120+** : je ne compète pas → Passe
2. Sinon, je cherche ma meilleure couleur **hors** la couleur adverse
3. Conditions : score ≥ 13, filtre de qualité passé
4. **Plafond de surenchère : 120**
5. L'enchère doit être strictement supérieure à l'enchère adverse

### 4.7 Coinche

Conditions (identiques au système paramétrique) :
- **V+9 dans la couleur adverse** → Coinche (je tiens leurs deux maîtres atout, le contrat va probablement chuter)
- **4+ atouts dans leur couleur + 1 As annexe** → Coinche (masse d'atout + entrée annexe, je vais étouffer leur jeu)

**Pas de surcoinche** : le bidder déterministe ne surcoinche jamais (trop risqué sans information partenaire fine).

### 4.8 Résumé des plafonds

| Situation   | Plafond |
|-------------|---------|
| Ouverture   | 120     |
| Réponse     | 130     |
| Surenchère  | 120     |

### 4.9 Résultats en tournoi

Dans un tournoi round-robin à 6 stratégies (100 matchs par paire, 20ms/coup, Smart IS-MCTS pour le jeu de la carte) :

| Rang | Stratégie         | Win% | Marge moy. |
|------|-------------------|------|-------------|
| 1    | **balanced** (= improved_bid) | 62,0% | +234 |
| 2    | very_aggressive   | 59,5% | +189       |
| 3    | aggressive        | 55,6% | +112       |
| 4    | moderate          | 54,7% | +113       |
| 5    | conservative      | 45,7% | -71        |
| 6    | ultra_conservative| 22,5% | -577       |

**Observation clé** : courbe en U inversé — trop conservateur perd (trop de passes gratuits), trop agressif perd aussi (enchères trop hautes qui chutent). Le sweet spot est un plafond 120 avec filtre de qualité.

---

## 5. Calibration par tournoi

### 5.1 Système paramétrique

Pour affiner les stratégies, on a construit un **enchérisseur paramétrique** (`BidParams`) qui expose tous les leviers sous forme de paramètres configurables :

```
BidParams {
    thresholds: [u16; 6],      // Score minimum pour chaque palier 80-130
    opening_cap: u8,           // Plafond en ouverture
    overcall_cap: u8,          // Plafond en surenchère
    response_cap: u8,          // Plafond en réponse
    overcall_min_score: u16,   // Score minimum pour surenchérir
    quality_gate: bool,        // Filtre de qualité activé/désactivé
}
```

Cela permet de tester systématiquement des variations par tournoi automatisé.

### 5.2 Tournoi large (6 stratégies)

**Protocole** : 6 presets (`ultra_conservative` à `very_aggressive`), 100 matchs par paire dans les deux sens, jeu de la carte par Smart IS-MCTS (20ms/coup), match en 2000 points.

**Résultats** : voir tableau §4.9 ci-dessus.

**Leçons** :
- Le filtre de qualité est crucial : sans lui, `very_aggressive` (59,5%) perd face à `balanced` (62%)
- Les plafonds protègent : `aggressive` (plafond 130) perd face à `balanced` (plafond 120)
- Trop conservateur est désastreux : `ultra_conservative` (22,5%) passe trop et laisse 70%+ des donnes à l'adversaire sans lutter

### 5.3 Tournoi de fine-tuning (12 variations)

Après avoir identifié `balanced` comme meilleure stratégie, on a exploré 12 micro-variations autour de ses paramètres :

**Protocole** : 12 variantes, 100 matchs par paire, 15ms/coup, Naive IS-MCTS pour le jeu.

| Rang | Variante           | Description                        | Win%  | Marge |
|------|--------------------|------------------------------------|-------|-------|
| 1    | **lo110**          | Seuil 110 abaissé de 21→20        | 52,6% | +51   |
| 2    | resp_120           | Plafond réponse 130→120            | 51,7% | +6    |
| 3    | cap_130            | Plafonds ouverture/surenchère→130  | 51,2% | +8    |
| 4    | no_qg              | Quality gate désactivé             | 50,9% | +17   |
| 5    | thr_loose          | Tous seuils -1                     | 50,7% | +34   |
| 6    | balanced (base)    | Référence                          | 50,6% | +19   |
| 7    | hi90               | Seuil 90 remonté 13→15            | 50,5% | +18   |
| 8    | oc_min11           | Score min surenchère 13→11         | 50,4% | -14   |
| 9    | lo80               | Seuil 80 abaissé 10→8             | 50,3% | +5    |
| 10   | lo90               | Seuil 90 abaissé 13→12            | 50,1% | -14   |
| 11   | thr_tight          | Tous seuils +1                     | 49,2% | -49   |
| 12   | cap_110            | Plafonds ouverture/surenchère→110  | 45,3% | -82   |

**Leçons** :
- Les marges sont serrées (50,1% – 52,6%) : toutes les variantes sont proches de `balanced`
- **Resserrer les seuils (thr_tight) fait toujours perdre** : on passe trop de mains viables
- **Baisser les plafonds (cap_110) est la pire variante** : on plafonne nos bons coups trop bas
- **Assouplir légèrement (lo110, thr_loose) aide un peu** : on capte plus de contrats marginaux
- `lo110` bat `balanced` 54-46% en face-à-face avec +51 de marge → adopté comme nouveau défaut

### 5.4 Paramètres finaux de `improved_bid`

Suite au fine-tuning :

```
Seuils score→enchère :
  10 → 80
  13 → 90
  17 → 100
  20 → 110  (anciennement 21, abaissé après tournoi)
  25 → 120

Plafonds :
  Ouverture  : 120
  Surenchère : 120
  Réponse    : 130

Quality gate : activé
  (V ou 9 ou A ou 10, ou 3+ cartes)

Score min surenchère : 13

Coinche : V+9 dans couleur adverse, ou 4+ atouts + As annexe
```

---

## 6. Rappel des règles de comptage

Pour comprendre les enjeux stratégiques des enchères, voici le comptage (mode "points faits + points demandés", règles FFB section 10.2) :

### Contrat réussi (standard)
- **Preneurs** : points de levées + valeur du contrat + belote → arrondi à la dizaine
- **Défense** : points de levées + belote → arrondi à la dizaine

### Contrat chuté (standard)
- **Preneurs** : 0
- **Défense** : 160 + valeur du contrat + toute belote → arrondi à la dizaine

### Contré (réussi)
- **Preneurs** : 320 + contrat×2 + toute belote
- **Défense** : 0

### Contré (chuté)
- **Preneurs** : 0
- **Défense** : 320 + contrat×2 + toute belote

### Surcontré
- Même formule mais 640 + contrat×4

### Capot (enchère à 250)
- Réussi : 500 (contré: 1000, surcontré: 2000) + belote
- Chuté : mêmes valeurs mais pour la défense

**Le total des points de levées vaut toujours 162** (152 de cartes + 10 de dix de der). En cas de capot : 252 (152 + 100 de dix de der capot).

### Impact sur la stratégie d'enchères

- **Enchérir 80 et réussir** : preneurs gagnent ~(81+80)=161→160, défense ~81→80. Gain net : +80 par rapport à ne pas enchérir (où les deux équipes auraient ~80)
- **Enchérir 80 et chuter** : preneurs 0, défense 160+80=240. Perte nette : ~-240 pour les preneurs (contre ~80 s'ils n'enchérissaient pas)
- **Enchérir 120 et chuter** : preneurs 0, défense 160+120=280. Perte massive.
- **Coincher et réussir** : 320+160=480 à soi seul. Gain énorme.
- **Se faire coincher et chuter** : 320+160=480 pour l'adversaire. Catastrophe.

→ Le coût d'une chute est **toujours supérieur** au gain d'une réussite. Cela justifie un biais conservateur en enchères, et explique pourquoi les stratégies plafonnées gagnent les tournois.

---

## 7. Limites globales et pistes d'amélioration

### 7.1 Ce que nos stratégies ne font PAS

1. **Pas de mémorisation de l'historique** : on ne regarde que l'enchère en cours, pas la séquence complète (qui a passé, dans quel ordre, à quel tour)
2. **Pas d'inférence sur les passes adverses** : si un adversaire passe après une ouverture à 80♥, cela donne de l'information sur sa main — nos bidders l'ignorent
3. **Pas de communication fine entre partenaires** : seul `smart_bid` a un vrai système de signalisation (V/9), les autres ignorent les conventions
4. **Pas de gestion de la position** : être en 1re, 2e, 3e ou 4e position change fondamentalement la stratégie optimale
5. **Pas de prise en compte du score du match** : à 1900-1800 en faveur, la stratégie optimale d'enchères est très différente de 0-0
6. **Pas de bluff** : tout est déterministe, pas de surenchère volontairement haute pour pousser l'adversaire à coincher une main qu'on peut gagner
7. **Pas de distribution probable** : l'évaluation ne modélise pas les mains adverses possibles étant donné les enchères
8. **Pas de capot** : aucune stratégie n'enchérit capot (250), même avec une main de rêve
9. **Pas d'enchères au-delà de 130** (heuristic) ou **120** (improved)

### 7.2 Pistes envisagées

- **Intégrer le score du match** : être plus agressif quand on est mené, plus conservateur quand on mène
- **Position-aware** : modifier les seuils selon qu'on ouvre, répond ou surenchérit (déjà partiellement fait dans improved_bid, mais de manière basique)
- **Apprentissage des enchères par RL** : remplacer les règles fixes par un réseau de neurones (comme on l'a fait pour le jeu de la carte avec DouZero-style DMC)
- **Meilleure coinche** : coincher de manière plus nuancée en intégrant le score du match et la distribution probable

---

## 8. Annexe : détail de l'encodage des enchères

Les enchères sont encodées sur 43 actions (masque u64) :

| Action   | Signification                                    |
|----------|--------------------------------------------------|
| 0        | Passe                                            |
| 1 – 36   | Enchère normale : `(palier × 4) + couleur + 1`  |
|          | Paliers : 80, 90, 100, 110, 120, 130, 140, 150, 160 |
|          | Couleurs : 0=♠, 1=♥, 2=♦, 3=♣                   |
| 37 – 40  | Capot (250) dans chaque couleur                  |
| 41       | Coinche                                          |
| 42       | Surcoinche                                       |

**Règles de légalité** :
- Passe toujours légal
- Une enchère doit être strictement supérieure à la précédente (en valeur)
- Coinche : seulement si l'adversaire a enchéri et pas encore coinché
- Surcoinche : seulement si notre équipe a été coinchée
- **Après coinche, le contrat est gelé** : aucune nouvelle enchère n'est possible, seulement passe ou surcoinche
- Surcoinche met fin aux enchères immédiatement
- 3 passes consécutifs après une enchère → fin des enchères
- 4 passes sans enchère → donne annulée

---

## 9. Annexe : presets du système paramétrique

Voici les 6 presets utilisés dans le tournoi large, pour référence :

| Preset             | Seuils (80→130)         | Plafonds (O/S/R)  | Min surenchère | QG  |
|--------------------|-------------------------|--------------------|----------------|-----|
| ultra_conservative | 12, 18, 24, 30, -, -   | 100 / 90 / 110    | 18             | Oui |
| conservative       | 10, 15, 20, 25, -, -   | 110 / 110 / 120   | 14             | Oui |
| moderate           | 10, 14, 18, 22, 26, -  | 120 / 110 / 120   | 14             | Oui |
| **balanced**       | 10, 13, 17, 20, 25, -  | 120 / 120 / 130   | 13             | Oui |
| aggressive         | 10, 14, 17, 20, 23, 26 | 130 / 120 / 130   | 12             | Oui |
| very_aggressive    | 10, 14, 17, 20, 23, 26 | 130 / 130 / 130   | 10             | Non |

Notation : O = ouverture, S = surenchère, R = réponse. `-` = palier désactivé.
