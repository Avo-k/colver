# Règlement du Championnat de France de Belote Contrée — Cannes (FIJ)

*Enquête du 2026-08-02. Objet : retrouver le barème de score appliqué aux tournois de contrée
du Festival International des Jeux de Cannes (Palais des Festivals), dont les tournois
qualificatifs du Championnat de France, aujourd'hui proposés par l'association
**BELOTE CONTREE MARALPINE**.*

---

## Résultat en une phrase

**Un règlement écrit existe, il est spécifique aux tournois de contrée de Cannes, et je l'ai
retrouvé — mais dans son édition 2016**, publiée par le FIJ lui-même et signée du responsable
des tournois. Aucune version postérieure n'est publique. Il répond aux quatre questions.

## La source

`REGLEMENT DE LA BELOTE AUX ENCHERES OU CONTREE2016.pdf`, 7 pages, hébergé sur le site
officiel du Festival, archivé deux fois (avril et octobre 2016) :

<https://web.archive.org/web/20160421181912if_/http://www.festivaldesjeux-cannes.com/Documents/REGLEMENT%20DE%20LA%20BELOTE%20AUX%20ENCHERES%20OU%20CONTREE2016.pdf>

Dans le corpus : `data/rules-corpus/tournois/web_archive_org_web_20160421181912if_http_festivaldesjeux_cannes_com_Documents_REGLEMENT_20DE_20LA_20BELOTE_20*`

Il se termine par :

> « LA DIRECTION DE LA SEMEC ET L'ORGANISATEUR DES TOURNOIS DE BELOTE CONTREE VOUS
> SOUHAITENT LES MEILLEURS RESULTATS POSSIBLES POUR CES JEUX 2016 »
> Le responsable des tournois — **Michel ISO**

La SEMEC est l'exploitant du Palais des Festivals de Cannes. Ce n'est donc pas un règlement
générique repris d'ailleurs : c'est *le* document du tournoi.

### Ce qui rattache ce document au tournoi d'aujourd'hui

Ce n'est pas une preuve que le barème 2026 est identique, mais la continuité est forte :

- **Même directeur de tournoi.** Michel ISO signe le règlement 2016 ; en 2019 il est encore
  « Monsieur Iso Michel — Direction des tournois » pour la belote du FIJ
  ([Cartes Grimaud, 01/02/2019](https://web.archive.org/web/2020/https://cartes-grimaud.fr/tournoi-belote-festival-des-jeux-cannes-2019/)) ;
  en janvier 2025 la page Facebook du Championnat le cite nommément
  (« Mention spéciale à mes nouveaux super fans ! Michel Iso, Paul Felicetti… »,
  <https://www.facebook.com/ContreeCannes/>).
- **Même format.** Le règlement 2016 décrit des parties de 2001 points, 40 minutes, un tournoi
  en 5 tours géré par appariement informatique au goal-average. Le FIJ 2026 annonce
  « 5 parties de 2001 points »
  ([évén. 1253](https://www.festivaldesjeux-cannes.com/fr/evenement/1253/Tournois-qualificatifs-pour-le-Championnat-de-France-de-Belote-Contree)).
- **L'association organisatrice est neuve.** BELOTE CONTREE MARALPINE (SIREN 989125521,
  1822 Moyenne Corniche des Pugets, 06700 Saint-Laurent-du-Var) a été **créée le 2025-07-09**
  (API `recherche-entreprises.api.gouv.fr`). Elle a donc repris un tournoi préexistant, pas
  fondé le sien.

**Réserve à garder** : dix ans séparent le document du tournoi actuel, et le porteur juridique a
changé. Ce qui suit est « le barème écrit du tournoi de Cannes, édition 2016 », pas
« le barème de Cannes 2026 ».

---

## Réponses aux quatre questions

Section « III – MARQUE » du règlement, citée littéralement (la coquille « ench ère » etc. vient
de l'extraction PDF) :

> La valeur des cartes correspond à un total de 162 POINTS.
> Les points marqués par chaque équipe seront arrondis à 6 (156 valant 160)
> Total des points à marquer = points de l'enchère demandée +162 ou 252 + belote éventuellement.
> **Contrat réussi** : points de l'enchère demandée + points des plis réalisés par le demandeur
> + belote éventuellement ;
> **Contrat perdu** : points de l'enchère demandée +162 points ou 252 + belote éventuellement car
> « prenable »
> Le contre double les points de l'enchère demandée, le surcontre les triple ; si la belote a été
> annoncée on ajoute 20 points de bonification qui ne sont ni doublés ni triplés.
> Une partie se joue en 2001 points et en 40 minutes à partir du signal de l'arbitre, la dernière
> donne est comptée au point juste (100 demandés + 104 faits = 204, l'équipe adverse marquera 58
> points et on inscrira 204 et 58)

### 1. La chute — **162 + contrat**

La défense marque **le contrat demandé + 162 points de cartes** (252 si capot), plus la belote
si elle est prenable. Le preneur marque 0 : le total marqué sur la donne (`contrat + 162`) est
le même qu'en contrat réussi, où il se répartit entre les deux camps. Ni 160, ni 160 + contrat,
ni forfait.

**C'est le même choix que celui du moteur Colver** (base 162, pas 160), et une confirmation
indépendante que 162 + contrat n'est pas une lubie.

### 2. Le surcontre — **×3**

« Le contre double les points de l'enchère demandée, le surcontre les triple. » Le multiplicateur
ne porte que sur **la valeur du contrat**, pas sur les points de cartes — là encore comme Colver.
La belote (20) n'est ni doublée ni triplée.

### 3. Le capot — **contrat ordinaire, pas de forfait**

Aucun forfait 500 / 1000 / 2000. Le capot est une annonce parmi les autres et il ne change que
**la base de points cartes**, qui passe de 162 à 252 (« +162 ou 252 » aux deux lignes,
réussi comme perdu). Section « 2° Les enchères » :

> Les paroles d'enchère autorisées sont : « le nombre de points suivi immédiatement de la couleur
> soit : « passe », « contre », « surcontre », « **générale ou capot** » ou « **270** » contrat le
> plus élevé.

Lecture littérale : « générale »/« capot » est le nom de l'enchère la plus haute, et sa valeur
chiffrée est **270**. (À noter : la FFB, elle, chiffre le capot à 250.) Ce point est le seul des
quatre où la formulation du PDF laisse place à interprétation — la valeur 270 est explicite,
son rattachement au mot « capot » l'est un peu moins.

### 4. L'arrondi — **à la dizaine, bascule à 6 : 85 → 80**

« Les points marqués par chaque équipe seront **arrondis à 6** (156 valant 160). » Le pivot est
donc le chiffre 6 et non 5 : 156 monte à 160, et par symétrie **85 descend à 80**, 86 monte à 90.

**Exception** : la dernière donne d'une partie (celle qui tombe à la limite des 40 minutes) est
« comptée au point juste », avec l'exemple 204 / 58 à l'appui — ces deux nombres ne sont pas
arrondis. L'arrondi est donc la règle, l'exactitude l'exception.

---

## Réponses aux questions bonus

- **La belote est prenable en cas de chute — oui, explicitement.** « Contrat perdu : … + belote
  éventuellement **car "prenable"** ». Le mot est entre guillemets dans le PDF, comme un terme
  technique du tournoi.
- **Coincher à la volée — interdit.** « On ne contre pas à la volée donc le joueur doit attendre
  son tour de parole pour contrer. Si cela n'est pas respectée l'adversaire décidera de la suite à
  donner : soit on continue la partie en assumant le résultat soit on redonne. »
- **« Ne pisse pas » — oui, pas d'obligation de sous-couper.** Section 3° :
  « c) non-obligation de fournir atout (pisser) sur une coupe maîtresse de l'adversaire si la
  surcoupe est impossible (il n'est pas interdit de pisser si le jeu s'avère plus intéressant en
  pissant) ; d) non-obligation de fournir atout ou de monter sur une coupe maîtresse de son
  partenaire. » En revanche « si l'adversaire est maître, il doit obligatoirement surcouper » et
  « obligation de monter dans le jeu d'atouts pour couper ou surcouper sur l'adversaire ».
- **Partenaire maître qui a coupé, et je n'ai plus que de l'atout : pas d'obligation de monter.**
  C'est la seconde moitié du d) ci-dessus. La première (« fournir atout ») est sans objet quand
  on n'a rien d'autre en main ; c'est « **ni de monter** » qui tranche, et il autorise donc à
  poser un petit atout sous celui du partenaire. Toutes les obligations de montée du règlement
  visent l'adversaire, jamais le partenaire. Cannes rejoint ici la FFB 2015, la FFB 2016 et
  Wikipédia Belote — contre la seule réimpression « Équipe Ludique », qui est ce que Colver
  appliquait jusqu'au correctif du 2026-08-01. Réserve : le texte dit qu'on n'est pas *obligé*
  de monter, il n'écrit pas qu'on aurait le *droit* de sous-couper en ayant mieux ; deux sources
  du corpus (Pagat, casimirdehauteclocque) prennent soin de l'interdire, Cannes reste muet.
- **Le contre fige le contrat** : « Le "contre" bloque l'enchère sauf s'il y a "surcontre". »
  Idem Colver.
- **Fin des enchères** : « La parole est retirée lorsque l'annonce "surcontre" est dite ou après
  que trois (3) joueurs successifs ont dit "passe" ».
- **Enchères de 10 en 10 à partir de 80**, sens antihoraire, premier parleur à droite du donneur.
- **Fausse donne constatée après une annonce adverse** : « pénalité de 160 points + le contrat »
  (noter le 160 ici, contre 162 partout ailleurs — le règlement n'est pas parfaitement homogène).

---

## Pistes épuisées (ne pas les refaire)

| Piste | Résultat |
|---|---|
| **Wayback sur `facebook.com/ContreeCannes`** (+ `m.` / `mbasic.` / `www.`, `matchType=prefix`) | **Zéro capture, jamais.** L'API CDX répond vide alors qu'un contrôle sur `ffbelote.org` renvoie bien des lignes. Inutile d'y revenir. |
| **Page Facebook ouverte au navigateur** (Playwright, sans compte) | Un seul post de fil visible, l'onglet *À propos* ne publie **ni mail ni téléphone** (seul lien : `festivaldesjeux-cannes.com`). Les **8 photos publiques** ont été téléchargées et regardées une à une : 4 mains de contrée posées sur une table (des exercices d'annonce), 3 vues de la tente de tournoi, 1 visuel Meta. **Aucune photo de règlement.** |
| **Post `story.php?story_fbid=544428484459116`** (la piste donnée) | « Résultats du mercredi soir », 23/02/2023, une image de classement, un commentaire sans rapport. Rien sur le barème. |
| **Post du 05/02/2023 (`posts/532470912321540`)** | Intéressant *a contrario* : un joueur y demande publiquement « **C est ou qu on peut voir les regles ?** » et n'obtient aucune réponse visible. Le règlement n'est manifestement pas diffusé en ligne par l'organisateur. |
| **Forum `beloter.com/673/participer-au-championnat-de-france`** | Récupéré (certificat expiré, `curl -k`). Une seule réponse, qui dit juste que le championnat a lieu à Cannes pendant le FIJ. Aucun barème. |
| **Site actuel du FIJ** | Les fiches 1250 / 1253 / 1257 / 387 / 392 donnent format, tarif, dotation, plafond de 148 équipes — **aucun lien vers un règlement**. Les seuls PDF `REGLEMENT_*` du site (2023, 2024, AS_D_OR_2025) concernent le label As d'Or, pas la belote. |
| **Wayback, tout le domaine FIJ, filtré sur `reglement|belote|contree`** | Ne remonte que **trois** PDF, tous de 2016 : contrée (trouvé), belote à la tourne, belote à l'envers. Aucune édition 2014/2015/2017-2020 sous un nom voisin ; le PDF contrée n'a que 2 captures, de digest identique. |
| **Registres d'association** | `recherche-entreprises.api.gouv.fr` donne la fiche complète (voir ci-dessus). L'API RNA `entreprise.data.gouv.fr` ne répond plus. Aucun site web ni dirigeant publié. |
| **Presse / agrégateurs** (`123belote`, `flanerbouger`, `beloteenligne`, `club-belote`, `tournoi-belote.com`, `jeu-belote.fr`, tag `cannes-la-bocca` de la FFB) | Annonces d'événements uniquement. Le tag FFB « Cannes La Bocca » ne concerne qu'un autre club (Club Azur Tarot, 2015). |
| **Recherches web ciblées** (une dizaine de formulations sur le barème de Cannes) | Ne remontent que des règlements *génériques* (FFB, FNASCE, Missègre…), sans valeur probante ici. |

---

## Où ce règlement est exploité dans le survey

- **Barème, arrondi, fin de partie** : [SYNTHESE.md](SYNTHESE.md) §6 « Où tombe Colver » —
  Cannes valide la chute à `162 + contrat`, le surcontre ×3 sur le contrat seul et le capot
  traité comme contrat ordinaire ; il diverge sur la valeur du capot (270 contre 250) et sur
  l'arrondi (à la dizaine, bascule à 6).
- **Détail du barème** : [matrices/bareme.md](matrices/bareme.md), axes 1, 3.1, 3.2, 4.1, 4.2,
  4.3, 5.1, 5.3, 5.4, 5.5, 6, 7 et 7.1, plus la typologie (§9, famille B) et **le §12.2, qui a
  été corrigé** : deux des cinq points « attestés nulle part » sont fermés par Cannes, dont le
  plus gênant — « aucune source ne porte les trois choix de Colver ensemble ». Cannes les porte.
- **Enchères** : [matrices/encheres.md](matrices/encheres.md), axes 1, 2, 2b, 6a, 6d, 7, 9, 9b,
  10, 11a, 13a, 13b et 13c. Deux apports propres : la **valeur 270 du capot** (seule du corpus)
  et la **sanction du contre à la volée**, laissée au choix du camp lésé — personne d'autre
  n'écrit ce qui se passe quand la règle est violée. Deux silences notables : le cas des
  **quatre passes** et le sort du **partenaire du contreur**.
- **Jeu de la carte et distribution** : [matrices/jeu-de-la-carte.md](matrices/jeu-de-la-carte.md),
  sous l'alias **Cannes 2016** — cité aux axes 1, 2, 3, 4, 5, 6, 7, 9, 10, 15, 16, 17 et 18.
  Trois apports qu'aucune autre source ne fournissait :
  1. **axe 9 (« pisser »)** — il double le camp « au choix du joueur », qui n'avait qu'un témoin,
     et il est le seul texte du corpus à *motiver* le sous-coup par la tactique ;
  2. **axe 10 (partenaire maître, je n'ai que de l'atout)** — il est le seul règlement de
     compétition à trancher ce cas, et il tranche pour la liberté ;
  3. **axe 17 (fausse donne)** — il gradue par le moment de la découverte et non par la récidive,
     et il **fait tourner le donneur** (« donne au suivant »), ce que la matrice donnait à tort
     pour impossible.
  Il est en revanche **muet sur l'entame** (axe 5) et sur le cas « l'atout est la couleur
  demandée » (axe 8) ; il n'est compté dans aucune position sur ce dernier.

---

## Si l'on veut la version à jour

Il n'y a **pas de règlement écrit public postérieur à 2016**. Le document existe presque
certainement encore — un tournoi à 148 équipes ne se joue pas sans texte de référence, et le
règlement 2016 renvoie lui-même à une feuille de marque signée par les capitaines — mais il
n'est diffusé nulle part en ligne. Par ordre de promesse :

1. **Michel ISO, direction des tournois** — 445 route des Genêts, 40380 Louer ; **06 19 05 36 64**
   (coordonnées publiées par Cartes Grimaud en 2019, à re-vérifier). C'est l'auteur du règlement
   2016 et il est toujours dans la boucle en 2025. Piste la plus directe.
2. **Messenger de la page** <https://www.facebook.com/ContreeCannes/> — le seul canal de contact
   que l'organisateur publie.
3. **BELOTE CONTREE MARALPINE**, 1822 Moyenne Corniche des Pugets, 06700 Saint-Laurent-du-Var
   (courrier ; aucun mail déclaré).
4. **Le Festival** : `jeux@palaisdesfestivals.com` / +33 4 92 99 33 88. Le FIJ a hébergé le
   règlement sur son propre site en 2016 : il a de bonnes chances d'en détenir la version courante.

Demande à formuler : les quatre points ci-dessus, plus la confirmation que l'arrondi « à 6 » et
le capot à 270 sont toujours en vigueur — ce sont les deux endroits où Cannes s'écarte le plus
nettement de la FFB.
