# L'arrondi — qui arrondit quoi, quand, et dans quel sens

Sources : le corpus de [../README.md](../README.md). Collecte du 2026-08-01.

---

## 1. Cinq familles, pas deux

### R0 — Au point près

| Source | Extrait |
|---|---|
| `federations/ffbelote_org_reglements_de_la_belote_avec_ou_sans_annonce.txt` · [url](https://www.ffbelote.org/reglements-de-la-belote-avec-ou-sans-annonce/) | « A la belote, les points sont comptabilisés **au point près. Il n'y a pas d'arrondi.** Note : en tournoi réel, pour des questions de logistiques (**jetons** notamment), les organisateurs peuvent utiliser la règle de l'arrondi (**pour la marque uniquement**). » |
| `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` §10.3 | « **Chaque tournoi se réserve le droit d'appliquer ou non** la règle de l'arrondi et doit le stipuler clairement en amont. […] Dans le cas où elle n'est pas appliquée, le score de chaque équipe est marqué au point près. » |
| `tournois/web_myassoc_org_…pdf.txt` (Lions Club) | « Art. 13 : Les parties se comptent **au point, sans arrondir**. Le capot compte pour 252 points. » |
| `tournois/geraudotloisirs_free_fr_….txt` | « Art. 3 : Les parties se comptent **au point, sans arrondir**. Le capot vaut 252 points. » |
| `tournois/villeconin_fr_….txt` (fiche FFB) | Reprise mot pour mot de la note « jetons » ci-dessus. |

**À retenir :** pour la FFB, l'arrondi n'est **pas une règle du jeu, c'est une commodité de
marquage** — une conséquence du fait qu'on marque avec des jetons de 10. Dès qu'on marque sur
papier ou sur écran, sa raison d'être disparaît.

### R1 — Dizaine la plus proche, **5 monte** (1-4 ↓, 5-9 ↑)

C'est la famille dominante, et de très loin.

| Source | Extrait |
|---|---|
| `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` §10.3 | « les points de la partie sont arrondis à la dizaine la plus proche. **85 → 90, 84 → 80** » |
| `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` §10.3 | idem |
| `federations/ffbelote_org_wp_content_…BELOTE_COINCHEE_pdf.txt` | idem, pour la coinche |
| `federations/ffbelote_org_belote_contree.txt` + `…regles_coinche.txt` | « arrondis à la dizaine inférieure si le dernier chiffre est compris entre 1 et 4 et à la dizaine supérieure si le dernier chiffre est compris entre 5 et 9 » |
| `divers/belotecontree_free_reglement.txt` + `tournois/ainesruraux_saintsever_com_….txt` (« règlement du tournoi international », texte ancien) | « Les **points faits** sont arrondis à la dizaine inférieure jusqu'à 4, à la dizaine supérieure à partir de 5. » |
| `tournois/cdf_missegre11_com_….txt`, `divers/carafons_fr_…`, `divers/cartesetcie_fr_…`, `divers/jeux_regles_com_…` | copies verbatim du texte FFB |
| `divers/pagat_com_jass_coinche.txt`, `apps-sites/gambiter_com_cards_jass_coinche.txt`, `divers/reglesdejeux_github_io_…` | « Scores are rounded to the nearest 10, with scores ending in **5 rounded upwards**. » |
| `divers/fr_wikipedia_org_wiki_Coinche.txt` | « arrondis à la dizaine la plus proche à partir de 5, par exemple 15 est compté 20 » |
| `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « si le chiffre des unités est compris entre 0 et 4, on arrondit à la dizaine inférieure, et s'il est compris entre 5 et 9, à la supérieure » |
| `open-source/CephaloSophie_kydos_…donneScoring.ts`, `open-source/ismo009_Coinche_main_game.js` | `Math.round(raw/10)*10` — c'est exactement 5-monte |

### R2 — **5 descend, 6 monte** (1-5 ↓, 6-9 ↑)

Minoritaire en nombre de sources, mais **c'est la version FFB la plus récente** et elle est
corroborée hors de France.

| Source | Extrait |
|---|---|
| `tournois/web_archive_org_…REGLEMENT_DE_LA_BELOTE_AUX_ENCHERES_OU_CONTREE2016.txt` — **Championnat de France de Belote Contrée, Cannes** | « Les points marqués par chaque équipe seront **arrondis à 6 (156 valant 160)** » — et « **la dernière donne est comptée au point juste** » |
| `federations/LOCAL_regles_officielles_belote_contree.txt` §9.2 (FFB, éd. Équipe Ludique) | « les points de la partie sont arrondis à la dizaine inférieure **jusqu'à 5**, et supérieure **à partir de 6**. Note : **85 est arrondi à 80** tandis que **86 est arrondi à 90**. » |
| `tournois/rjcv_be_belote_regles_pdf.txt` (club belge) | « L'arrondi à la dizaine supérieure se fait **à partir de 6**. » |
| `divers/fr_wikipedia_org_wiki_Belote.txt` (§ « Alternative pour les arrondis ») | « se terminant par 1, 2, 3, 4, **5** à la dizaine **inférieure** ; par 7, 8, 9 à la **supérieure** ; par **6** : cas particulier » |
| `apps-sites/gambiter_com_cards_Belote.txt`, `apps-sites/officialgamerules_org_…`, `apps-sites/licitum_….txt` (belote bulgare) | La limite d'arrondi **dépend du total du contrat** : 5 à Sans Atout (total 160), **6 à la couleur (total 162)**, 4 à Tout Atout (total 258) |

Corroboré par deux corpus ajoutés en seconde passe (cf. [../COLLECTE-2.md](../COLLECTE-2.md)) :

| Source | Extrait |
|---|---|
| `divers/pagat_com_jass_baloot_html.txt` (baloot) | Conversion des points cartes en *game points* par division par 10 avec arrondi, calibrée pour un **total invariant de 16** (Hokum, 162 pts) ou **26** (Sun, 130 pts) |
| `clubs/coinche_ch_regles_coinche_pdf.txt` + `frtj_kirschmann_reglement_tournois_officiels` (coinche suisse) | Donne à **157** et non 162 ; arrondi à la dizaine « selon les principes commerciaux » ; le total sert de **contrôle de donne** — « les cartes ne peuvent être mélangées que si le total donne 157 » |
| `divers/belot_bg_academy_basics_tochkuvane_v_belota.txt` (bulgare) | La limite d'arrondi **dépend du contrat** : 5 à sans-atout, 6 à la couleur, 4 à tout-atout |

**La belote bulgare est la clé de lecture de cette famille.** Elle ne choisit pas 6 par tradition :
elle le calcule, pour que les deux scores arrondis se somment toujours à la même valeur. C'est le
même raisonnement que Wikipédia FR fait explicitement (« afin que la somme des points hors
annonces soit toujours 160 »), et que la FFB « Équipe Ludique » applique sans le justifier.

### R3 — Arrondi **complémentaire** (on arrondit un camp, l'autre prend le reste)

| Source | Extrait |
|---|---|
| `divers/ange_heureux_free_fr_JeuxDeCartes_La_Coinche.txt` | « Les comptes sont arrondis **à l'avantage de la défense** : on compte les points de la défense, on arrondit (à la dizaine supérieure si on finit par 5) puis on en **déduit les points de l'attaque par soustraction à 160**. » |
| `divers/fr_wikipedia_org_wiki_Belote.txt` (règle principale) | « 0-5 : arrondi inférieur, **l'adversaire a le complément** pour atteindre 160. 6 : arrondi supérieur, complément pour atteindre **170**. 7-9 : arrondi supérieur, complément pour atteindre 160. » |

C'est la seule famille qui **garantit** un total invariant, parce qu'un seul des deux nombres est
arrondi. Elle est aussi la seule qui introduit une asymétrie entre les camps.

### R4 — Arrondi **à 5 points**

| Source | Extrait |
|---|---|
| `divers/fr_wikipedia_org_wiki_Belote.txt` | « se terminant par 1, 2, 3 → dizaine inférieure ; 9 → supérieure ; **4, 6, 7, 8 → 5**. […] on commence par **enlever un point à chaque score** puis on arrondit au multiple de 5 le plus proche. » Exemples : 118-44 → 115-45, 117-45 → 115-45 |

Total toujours 160. Marché en jetons de 5.

---

## 2. Pourquoi ces familles existent : l'arithmétique

Les deux totaux de plis d'une donne se somment à **162**. Les paires de derniers chiffres
possibles sont donc `(0,2) (1,1) (3,9) (4,8) (5,7) (6,6)` et leurs symétriques. Si on arrondit
**chaque camp indépendamment**, la somme des deux scores arrondis ne fait plus 160 dans certains
cas. Vérifié sur les 163 partages possibles :

| Convention | Partages où la somme n'est plus 160 | Paires de chiffres fautives |
|---|---|---|
| **5 monte** (R1, FFB 2016, Pagat) | **48 / 163** (29 %) | (5,7) et (6,6) |
| **6 monte** (R2, FFB Équipe Ludique, Belgique, Bulgarie) | **16 / 163** (10 %) | (6,6) seule |

Autrement dit : `75-87` devient `80-90 = 170` avec la règle 5-monte, et `70-90 = 160` avec la règle
6-monte. **Un tiers des donnes voit son enjeu gonflé de 10 points sous la règle FFB 2016 ; un
dixième seulement sous la règle FFB actuelle.**

Le cas `76-86` est **irréductible** pour toute règle symétrique : 7,6 et 8,6 arrondis chacun au
plus proche ne peuvent pas donner 16. Il faut alors soit une asymétrie explicite (Wikipédia :
« on arrondit le score de l'équipe atout à la dizaine inférieure et l'autre à la supérieure »),
soit un arrondi complémentaire (R3), soit accepter la « casse » — le mot qu'emploie l'auteur de
`open-source/CephaloSophie_kydos_…donneScoring.ts`, qui documente le total 170 comme un
comportement voulu et non comme un bug.

**Conclusion de cette section :** le désaccord 5-monte / 6-monte n'est pas un désaccord de goût.
6-monte est la seule convention symétrique quasi conservative. 5-monte est l'arrondi
mathématiquement « standard », qui ignore l'invariant.

**Et la compétition tranche dans le sens de l'arithmétique.** Le seul règlement de championnat du
corpus qui fixe un arrondi — celui de Cannes — choisit 6-monte (« arrondis à 6, 156 valant 160 »),
contre l'avis des règlements FFB de 2015 et 2016. Il ajoute l'exception qui montre qu'on a compris
le problème : **la dernière donne d'une partie est comptée au point juste**, parce que c'est celle
qui décide et qu'un arrondi n'a rien à y faire.

---

## 3. Trois questions que l'arrondi pose et que presque personne ne traite

### 3.1 Qu'est-ce qu'on arrondit exactement ?

| Position | Sources | Détail |
|---|---|---|
| **Les points faits** (le total de plis d'un camp), le contrat s'ajoute non arrondi | FFB contrée 2016 §10.4 | Exemple littéral : « 1770 et réalise **117 points pour un contrat à 110**. Les 117 points sont arrondis à 120, soit **1770 + 120 + 110** » |
| **La ligne de score entière** | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « on additionne les points réalisés à la valeur du contrat et **on arrondit le tout** » — ex. 78 + 10 + 80 = 168 → **170** |
| **Les deux scores finaux de la donne** | `open-source/ismo009_Coinche_main_game.js` | `scoreNS = roundScore(scoreNS); scoreEO = roundScore(scoreEO)` |
| Le der est ajouté **avant** l'arrondi | `open-source/CephaloSophie_kydos_…donneScoring.ts` | commentaire explicite |

En pratique les deux premières positions coïncident presque toujours (contrat et belote sont des
multiples de 10, donc arrondir avant ou après ne change rien) — mais **pas dans les modes où un
forfait non multiple de 10 entre dans la ligne**, et pas si on arrondit un total qui inclut déjà
le contrat sur une base 162.

### 3.2 L'arrondi peut-il faire réussir un contrat ?

**Non, et c'est un des rares points où tout le monde est d'accord** — quand la question est posée.

> « Cet arrondi **ne permet pas de réaliser le contrat**. Une équipe ayant fait 89 points et ayant
> demandé 90 **chute**. »
> — `federations/ffbelote_org_belote_contree.txt`, `…regles_coinche.txt`,
> `tournois/cdf_missegre11_com_….txt`, `divers/carafons_fr_…`, `divers/cartesetcie_fr_…`

Wikipédia contrée dit la même chose autrement : « si une équipe a annoncé 100 points et qu'elle
fait 99, elle chute. Si elle réalise 104 points, le total des plis est arrondi à 100. » La réussite
se juge **au point près**, l'arrondi n'intervient qu'à la marque.

Les autres sources sont muettes là-dessus, ce qui n'est pas un accord : c'est exactement le genre
d'ambiguïté qui se règle à la table.

### 3.3 L'arrondi peut-il faire gagner la partie ? — **le vrai désaccord**

| Position | Sources | Extrait |
|---|---|---|
| **Oui, l'arrondi joue jusqu'au bout et permet de gagner** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` §10.4 | « Afin d'empêcher tout problème lié à ces fins de parties et d'**uniformiser le règlement au niveau national**, la Fédération homologue la règle suivante : **les points sont arrondis en fin de partie et permettent de gagner.** » (1880 + 117 arrondi à 120 = gagné) |
| idem, belote classique | `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` §10.4.2 | « 910 points et réalise 85 points, elle marque 90, ce qui mène son score à 1000. **Elle remporte la partie.** » |
| **Non, la dernière donne se compte au point près** | `federations/LOCAL_regles_officielles_belote_contree.txt` §9.2-9.3 (FFB, version plus récente) | « Une **exception est faite sur la dernière partie** où les points sont notés **sans arrondis**. » + « Le premier camp atteignant ou dépassant 2000 points (**au point près**) remporte la partie. » |
| **Au choix du tournoi, annoncé d'avance** | `divers/belotecontree_free_reglement.txt` + `tournois/ainesruraux_….txt` | Deux articles distincts : « A/ Fin de partie en points réels » et « B/ Fin de partie *arrondie* » |
| **On déplace la cible pour esquiver le problème** | `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` (grand tournoi ASCEE 2A) | « La manche se fait en **1 010 points minimum**. » |
| idem | `divers/fr_wikipedia_org_wiki_Belote.txt` | « Une partie se joue généralement en 1 000 points, **1 001 points ou 1 010 points si les scores sont arrondis à la dizaine**. » |
| **On compte au point près en fin de partie même si on a arrondi avant** | `divers/fr_wikipedia_org_wiki_Belote.txt` | « D'aucuns arrondissent le décompte des points […] mais **en fin de partie on peut décider de compter au point près** (par exemple 113 au lieu de 110), l'équipe en tête d'un ou plusieurs points ayant gagné. » |

**C'est ici que la FFB se contredit frontalement avec elle-même.** Le PDF de 2016 déclare
homologuer une règle nationale précisément parce que « la fin de partie en belote contrée est trop
souvent le centre de règles différentes de régions en régions » — et la rédaction suivante du même
règlement inverse cette décision. Sur le point que la fédération avait identifié comme le plus
disputé, elle a donc produit deux réponses opposées.

---

## 4. Synthèse en une page

| Question | Réponse dominante | Qui n'est pas d'accord |
|---|---|---|
| Arrondit-on ? | Oui à la contrée / coinche, non à la belote classique | La FFB elle-même rend l'arrondi **optionnel** en belote classique, et **le justifie par les jetons** — pas par le jeu. Plusieurs concours marquent au point |
| Dans quel sens ? | Dizaine la plus proche, **5 monte** | La FFB récente, la Belgique, la Bulgarie et Wikipédia (« alternative ») : **5 descend, 6 monte** |
| Sur quoi ? | Sur les points faits de chaque camp | Wikipédia contrée arrondit la ligne entière ; certains n'arrondissent qu'un camp et déduisent l'autre |
| Ça peut faire réussir un contrat ? | **Non** — unanime chez ceux qui posent la question | Personne, mais la majorité des sources est muette |
| Ça peut faire gagner la partie ? | **Désaccord frontal** | FFB 2016 : oui, règle homologuée. FFB récente : non, dernière donne au point près. Tournois : cible à 1 010 pour contourner |
| La somme des deux scores | 160 en général, **170 dans 10 à 29 % des donnes** selon la convention | Seuls Wikipédia, la belote bulgare et l'auteur de kydos ont l'air d'avoir remarqué |

---

## 5. Ce que ça implique pour Colver

Colver a supprimé `round10` de `compute_deal_score` le 2026-07-31 et marque au point près. Le
corpus dit que **c'est défendable et bien plus cohérent qu'il n'y paraissait** :

1. L'arrondi est présenté par la FFB elle-même comme une commodité de **marquage physique**
   (« questions de logistiques, jetons notamment »). Un moteur qui somme des entiers n'a aucune
   raison de l'imiter. Le règlement de belote classique 2016 rend d'ailleurs l'arrondi
   explicitement optionnel.
2. Les règlements de concours qui tranchent explicitement disent, eux aussi, « au point, sans
   arrondir » (Lions Club, Géraudot Loisirs).
3. L'arrondi **n'a jamais eu le droit** de décider de la réussite d'un contrat. Colver juge la
   réussite au point près : conforme.
4. En revanche, **la base 162 et l'arrondi sont liés historiquement** : tant qu'on arrondissait,
   une base 162 retombait sur 160. C'est exactement ce que note `CLAUDE.md`. Si on arrondissait à
   nouveau un jour, il faudrait le faire en **6-monte** et pas en 5-monte, sinon 29 % des donnes
   gonflent de 10 points.
5. Le point à trancher qui reste ouvert côté Colver n'est **pas** l'arrondi de la donne mais la
   **fin de partie** : « atteindre ou dépasser ». La version FFB la plus récente exige de
   **dépasser strictement** (2000 pile ne gagne pas) et ajoute qu'on ne finit pas sur une belote
   seule ni en étant capot ; le règlement historique impose trois conditions cumulées (atteindre,
   avoir plus que l'adversaire, ne pas être capot ni chuter). Colver s'arrête à « cible atteinte et
   pas d'égalité ». C'est un choix, mais il n'est adossé à aucune des deux traditions.
