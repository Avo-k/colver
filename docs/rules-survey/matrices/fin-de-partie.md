# Matrice « qui dit quoi » — fin de partie et format de tournoi

Établie le 2026-08-01 sur le corpus local `data/rules-corpus/` uniquement (aucune recherche web).

**L'arrondi est traité à part, dans [arrondi.md](arrondi.md)** — y compris son interaction
avec la fin de partie (l'arrondi qui permet ou non de gagner, les cibles à 1 001 / 1 010).
Ce document-ci ne le mentionne que là où il est indissociable de la question
« atteindre ou dépasser ».

**Comment lire ce document.** Chaque axe donne un tableau `| Position | Sources | Extrait |`
groupé par position, suivi d'une ligne de verdict. Les sources sont citées par **nom de
fichier** ; leurs **URL** sont dans la [Table des sources](#table-des-sources) en fin de
document (les mettre dans les tableaux les rendait illisibles). Quelques fichiers n'ont pas
d'en-tête `SOURCE:` et sont signalés comme tels.

**Trois règles de lecture, appliquées partout :**

1. **Une source muette n'est pas un accord.** Beaucoup de textes s'arrêtent au barème de la
   donne et ne disent rien de la fin de partie. Ils sont comptés « muets », jamais « d'accord ».
   C'est le cas notamment des pages `ffbelote.org/belote-contree/` et `ffbelote.org/regles-coinche/`,
   qui reprennent le corps du règlement FFB **mais coupent avant le § 10.4**.
2. **Les copies ne votent pas plusieurs fois.** Le § 0 recense les familles de textes identiques.
   Un « consensus » à cinq sources dont quatre sont la même feuille photocopiée est un
   consensus à deux.
3. **La cible et le format de comptage ne se lisent pas séparément.** En « points faits +
   points demandés » un camp marque ~250-320 par donne, en « points annoncés seuls » ~110 et
   pour un seul camp. Une partie en 1 000 et une partie en 3 000 peuvent durer le même nombre
   de donnes. Plusieurs sources le disent explicitement (§ 1.3).

---

## 0. Familles de copies — à établir avant de compter les voix

Vérifié par recherche de phrases-sondes sur le corpus normalisé, pas par confiance dans le README.

| Famille | Fichiers | Statut |
|---|---|---|
| **A — FFB, PDF officiels** | `ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt`, `ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt`, `LOCAL_regles_officielles_belote_contree.txt`, `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` | Les deux PDF de 2015 (contrée et coinchée) ont un § 10.4 **mot pour mot identique**. `LOCAL` est ce même texte + « 2000 points (au point près) » + un exemple chiffré. Le PDF du 27/01/2016 a été **réécrit**. → **2 rédactions**, pas 4. |
| **B — FFB, pages web belote classique** | `ffbelote_org_regles_officielle_belote.txt`, `ffbelote_org_reglements_de_la_belote_avec_ou_sans_annonce.txt`, `ffbelote_org_regles_coinche.txt`, `villeconin_..._Informations_sur_le_jeu.txt` | Même texte (« Le premier camp atteignant le nombre de points fixés au préalable… », « On ne termine jamais une partie sur un capot accompagné d'une belote »). → **1 témoignage**. |
| **C — « règlement du tournoi international »** | `ainesruraux_saintsever_com_belote_BELOTE_20TRADITIONNELLE_pdf.txt` ≡ `belotecontree_free_reglement.txt` | **Vérifié identique** (le second ajoute seulement des commentaires de forum en fin de page). → **1 témoignage**. |
| **D — Pagat** | `pagat_com_jass_coinche_html.txt` ≡ `gambiter_com_cards_jass_coinche_html.txt` ≡ `reglesdejeux_github_io_..._la_coinche_index_html.txt` | Phrases de fin de partie **bit-identiques** entre Pagat et Gambiter ; reglesdejeux est une traduction automatique du même texte. → **1 témoignage**. |
| **E — corps FFB recopié, fin de partie divergente** | `cdf_missegre11_...txt`, `carafons_fr_regles_de_la_coinche.txt`, `ffbelote_org_belote_contree.txt`, `ffbelote_org_regles_coinche.txt` | Partagent le corps FFB (« l'erreur ne doit pas bénéficier à l'équipe qui l'a commise », « Le capot demandé et réalisé, ou chuté, vaut 500 points ») **mais divergent sur la fin de partie** : Missègre écrit sa propre section VIII, Carafons une phrase d'intro, les deux pages ffbelote.org **ne disent rien**. → à compter séparément **sur cet axe seulement**. |
| **F — ASCEE 2A (contrée)** | `fnasce_org_IMG_pdf_reglement_pdf.txt` et `fnasce_org_IMG_pdf_belote_reglement_cle1a43c7_pdf.txt` | Même règlement, deux millésimes, **identiques sauf la dernière ligne** : « 1 010 points minimum » contre « 1 000 points ». → **1 témoignage, contradictoire avec lui-même** (§ 1.4). |
| **G — ASCE(E) 79 / Twirl Danse** | `fnasce_org_IMG_pdf_Belote_Reglement_cle19a7cf_pdf.txt`, `fnasce_org_IMG_pdf_reglement_belote_pdf.txt`, `pontdeclaix_..._rglementconcoursdebelote_pdf.txt` | Articles numérotés identiques (« Le litige se situe à 81 pts… au 12ème et dernier tour »). → **1 témoignage** (le 2ᵉ ajoute le litige à 91 et la procédure à nombre impair d'équipes). |
| **H — « Nous vous remercions d'avoir choisi notre concours »** | `cdfcasson_...pdf.txt`, `lesamisdutempslibrevarennes_...txt`, `sc4e58b2fce8a2e7a_...Belote_20R_C3_A8glement_2.txt` | Même modèle (capote 252, case BEL/CAP, litige 81/81 belote exclue, 162 pts de pénalité). **Mais le format du concours est le champ qu'on édite localement** : 6 parties de 12 donnes chez Casson, 5 parties de 10 donnes à Varennes. → **1 témoignage sur les règles, 2 sur le format**. |
| **I — modèle « nombre de victoires »** | `rjcv_be_belote_regles_pdf.txt`, `cdfcasson_...pdf.txt` | Partagent « classement par nombre de victoires (départagés aux points en cas d'égalité) », « coupe obligatoire (minimum 3 cartes) », « Le valet n'est pas forcé ». Recoupement partiel. |
| **J — modèle associatif « au point, sans arrondir »** | `geraudotloisirs_....txt`, `web_myassoc_org_..._Lions_....txt` | Partagent 4 articles mot pour mot. Le PDF du Lions Club est un **hybride** : moitié modèle Géraudot, moitié modèle ASCE 79 (famille G). Il n'est donc témoin indépendant **ni** de « au point / capot 252 », **ni** du litige à 81. |

> Sont en revanche des témoignages **indépendants** : Maison des Essarts, AIL Manissieux,
> coinche-stéphanoise, JWEBARTEAM, Club La Fontaine de Jouvence, La Grand-Combe, AAPPMA
> Koenigshoffen, TCVB Bruche, FC Plouay, Casimir de Hauteclocque, Ducale, les Wikipédias,
> Pagat, BGA, et les implémentations open source.

---

## 1. La cible de la partie

### 1.1 En contrée / coinche française

| Position | Sources | Extrait |
|---|---|---|
| **2 000** | `ainesruraux_saintsever_...pdf.txt` (fam. C) | « Sauf dispositions particulières indiquées en début de tournoi, la partie se joue en 2000 points. » |
| | `LOCAL_regles_officielles_belote_contree.txt` (fam. A) | « se termine lorsqu'une équipe atteint ou dépasse 2000 points » |
| | `web_archive_..._coinche_stephanoise_..._reglement_coinche_pdf.txt` | « lorsqu'une des équipes atteignent le score de 2000 points (soit 1995) minimum » |
| | `pagat_com_jass_coinche_html.txt` (fam. D) | « The first team that reaches a score of 2000 points or more wins the match. » |
| | `drasill_github_io_bga_coinche_rules_fr_html.txt` + `drasill_bga-coinche_master_coinche.game.php` | « Dès qu'une équipe atteint 2000 points (révolus) » / `if ($gameLength == 1) { return 2000; }` |
| | `adpoker_fr_belote_contree_html.txt` | « Etre la première équipe à totaliser 2 000 points. » |
| | `carafons_fr_regles_de_la_coinche.txt` | « Une partie se déroule en 2000 points » |
| | `clubdejeux_com_belote_coinchee_online_regles.txt` | « La Belote Coinchée de ClubDeJeux va en 2000 points. » |
| | `alhoa_free_fr_ALH_belote_rules_htm.txt` | « Une partie se joue en général en 2000 points. » |
| | `gamerules_com_rules_coinche.txt` | « be the first to score 2000 points » |
| **2 001** (= « il faut passer 2 000 ») | `jwebarteam_wordpress_com_championnat_de_coinche_2016_2017_les_regles.txt` | « Partie en 2001 points (Il faut passer les 2000 pour gagner) » — finales en **3001**. |
| **3 000** | `maisondesessarts_fr_article116_html.txt` | « Les parties se déroulent en 3000 points chacune. » |
| | `data_over_blog_kiwi_..._reglement_table_coinche_pdf.txt` (AIL Manissieux) | « Les parties se déroulent en 3000 points (Système AURARD) » |
| | `fr_wikipedia_org_wiki_Coinche.txt` | « Une partie se joue généralement en 3 000 points. » |
| | `ange_heureux_free_fr_JeuxDeCartes_La_Coinche_html.txt` | « Généralement une partie se joue en 3000 points. » |
| **1 500** | `jeu_belote_fr_..._tournoi_competition_belote.txt` (compétitions premium) | « Les parties se déroulent en 1500 points. » |
| | `sebastien-perpignane_..._ContreeGameConfig.java` | `int DEFAULT_MAX_SCORE = 1_500;` |
| **1 010 / 1 000** | `fnasce_org_IMG_pdf_reglement_pdf.txt` / `..._cle1a43c7_pdf.txt` (fam. F) | « La manche se fait en **1 010 points minimum**. » / « … en **1 000 points**. » |
| **1 000** | `fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « La limite de points qu'on utilise la plupart du temps est de 1000 points » |
| | `iscool_..._coinche_rules.txt` (Belote Mobile) | « être la première équipe à marquer 1000 points » |
| | `jeu_belote_fr_..._tournoi_competition_belote.txt` (tournois quotidiens) | « Les parties se déroulent en 1000 points. » |
| | `ElysiumDisc_belote_master_src_belote_config.py` | `TARGET_SCORE: int = 1000` |
| **501 ou 1 001** | `belotepoint_fr_regles_coinche.txt` | « atteindre un score cible, généralement fixé à 501 ou 1001 points » |
| | `playjoy_com_en_coinche_rules.txt` | « The goal of Coinche is to win 1001 points before the rival team. » |
| **701** | `lemagloisirs_fr_regle_coinche.txt`, `jeux_regles_com_regles_coinche.txt` | « victoire à 701 points » / « être la première équipe à atteindre 701 points » |
| **121 puis 151** (plafonnés) | `clublafontainedejouvence_fr_r_C3_A8glement_coinch_C3_A9e.txt` | « 2 parties de **121 points limitées à 135**, la troisième partie en **151 points limitée à 175** » |
| **Non chiffré — « fixé au préalable »** | `ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt`, `ffbelote_org_..._COINCHEE_pdf.txt`, `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (fam. A) | « le nombre de points fixés au préalable » — **la FFB ne chiffre jamais la cible en contrée.** |
| | `cdf_missegre11_...pdf.txt` | « lorsque l'une des deux équipes atteint **le score ou le temps maximal** fixé par l'organisation du tournoi » |
| **Choix explicite entre plusieurs** | `casimirdehauteclocque_fr_jeux_coinche_pdf.txt` | « un certain score fixé à l'avance (en général **1000, 1500 ou 2000** points) » |
| | `exoty_com_regles_coinche_belote.txt` | « le score global fixé (souvent **1000, 2000 ou 3000** points) » |
| | `bk_jeux_ducale_..._belote_coinchee_pour_joueur_expert_pdf.txt` | « un nombre de points à atteindre : **500, 1000, 1500 points ou plus**. » |
| | `pagat_com_jass_coinche_html.txt` (fam. D) | « without announcements to a target of **1000 or 1500** rather than 2000… with announcements… for example **3000**. » |
| **Muet** | `ffbelote_org_belote_contree.txt`, `ffbelote_org_regles_coinche.txt`, `cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt`, `exoty_com_compter_points_coinche.txt`, `maviedesenior_...txt`, `jeubelote_com_...txt` | aucune mention de fin de partie |

**Divergence maximale du corpus.** La cible va de **121** à **3 001**, soit un facteur 25.
Aucune valeur n'est majoritaire dans l'absolu : **2 000 domine chez les généralistes** (10 témoins
indépendants) mais **aucun des quatre règlements FFB ne la prescrit** — la FFB délègue le
chiffre à l'organisateur, et les concours réels utilisent tout l'éventail. Le seul groupement
robuste est : *si on joue en points faits + demandés et en contrée française, 2 000 est le
défaut de fait.*

### 1.2 Hors contrée — belote classique, belote bulgare, jass suisse

| Position | Sources | Extrait |
|---|---|---|
| **1 000** (belote classique) | `pagat_com_jass_belote_html.txt` | « The winning team is the first to reach a total of 1000 points. » |
| | `ilyesbrh_..._docs_GAME_RULES.md` (source secondaire, cf. README) | « Common target: **1000 points** » |
| **501** | `ibelote_com_en_rules_belote_php.txt` | « The game ends when one of the team reaches a minimum of 501 points. » |
| **701** | `gameduell_..._faq_1054_coinche.txt` | « The first team to reach a minimum of 701 points wins the game. » |
| **501 / 701 / 1 001 selon le nombre de joueurs** | `licitum_board_directory_net_t16_belot_rules.txt` (belot bulgare) | « Player who first reaches 501 points is a winner. » (2 j.) / « 701 » (3 j.) / « Team which first reaches **1001** points is a winner » (équipes) |
| **151** | `officialgamerules_org_game_rules_belote.txt` | « First team to reach **151** match points wins, but a valat round must complete before ending the game. » |
| **101 (une seule manche)** | `rjcv_be_belote_regles_pdf.txt` (Belgique) | « Chaque partie se joue en une seule manche de **101 points**. » |
| **1 000 / 1 500 / 2 500 / 12 tours** | `swisslos_..._le_chibre_html.txt` (chibre suisse) | « Le chibre se joue en **1000 points, 2500 points ou en mode tournoi en 12 tours**. » |
| | `jass_geneve_ch_regleschibre_html.txt` | « Objectif habituel : première équipe à **1000 points**… En pique double… l'objectif passe à **1500 points**. » |
| | `chibre_ch_forum_viewtopic_php_t_638.txt` | « La première équipe à atteindre 1000 points remporte la partie… à "Pique double"… la partie se joue en 1500 points » |

**Divergence**, mais **structurée** : les cibles basses (101, 151, 501, 701, 1 001) appartiennent
aux jeux où **un seul camp marque** ou où le total par donne est faible ; les cibles hautes
(2 000, 3 000) à la contrée en points faits + demandés. Voir § 1.3.

### 1.3 Le format de comptage détermine la cible — qui le dit explicitement

| Position | Sources | Extrait |
|---|---|---|
| **Oui, et c'est un rapport ×2 environ** | `fr_wikipedia_org_wiki_Coinche.txt` | « seuls les points annoncés sont marqués… le score avance moins vite, **les parties se font en 1 000 points**. » (contre 3 000 en comptage standard) |
| | `fr_wikipedia_org_wiki_Coinche.txt` | « coincher… ne double que l'annonce… **dans ce cas, la partie se joue généralement en 2 000 points** » |
| | `drasill_github_io_bga_coinche_rules_fr_html.txt` | « ne compter que l'annonce… Ceci entraîne un jeu plus long (de fait, **souvent joué en 1000 au lieu de 2000 points**). » |
| | `bk_jeux_ducale_..._pdf.txt` | « le total des points monte très vite ! Avec cette règle, il est conseillé de jouer la partie en **1500 ou 2000** points. » |
| | `pagat_com_jass_coinche_html.txt` (fam. D) | « without announcements to a target of 1000 or 1500 rather than 2000. When playing with announcements… **for example 3000**. » |
| | `belotecontree_free_reglement.txt` (commentaire de l'auteur du site, **source secondaire**) | « les parties en PF se jouent en **2000** et les parties en PF+PA souvent en **3000**. » |
| **Le corpus permet la variante mais ne dit rien de la cible** | `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` | « Il existe deux méthodes possibles pour compter les points, au choix de l'organisateur : points faits / points faits + points demandés » — **puis aucune conséquence sur la cible.** |
| | `vareversat_carg_..._coinche_belote_game_setting.dart` | expose `maxPoint` **et** `sumTrickPointsAndContract` comme deux réglages indépendants — le lien n'est pas encodé. |

**Consensus** sur le principe (5 témoins indépendants : Wikipédia, BGA, Ducale, Pagat,
belotecontree.free), **mais pas sur le sens du décalage** : Wikipédia et BGA disent que le
comptage *aux annonces seules* **baisse** la cible (1 000 au lieu de 2 000), tandis que
Ducale et Pagat disent que le comptage *avec annonces / points demandés* la **monte**
(vers 2 000-3 000). Ce sont les deux faces du même fait. La **FFB, seule fédération du
corpus, offre les deux modes de comptage et ne dit rien de leur effet sur la cible** — c'est
le trou le plus net de son règlement sur cet axe.

### 1.4 Une contradiction interne documentée : ASCEE 2A

Deux PDF du même organisateur, texte par ailleurs identique caractère pour caractère :

| Fichier | Dernière ligne |
|---|---|
| `fnasce_org_IMG_pdf_reglement_pdf.txt` (« grand tournoi 2016 ») | « La manche se fait en **1 010 points minimum**. » |
| `fnasce_org_IMG_pdf_belote_reglement_cle1a43c7_pdf.txt` | « La manche se fait en **1 000 points**. » |

Aucun des deux ne justifie le 1 010 ; « minimum » suggère qu'il s'agit d'un seuil à franchir
et non d'une cible atteignable pile (cf. § 2 ; l'hypothèse « arrondi » est traitée dans
[arrondi.md](arrondi.md)). Les deux jouent au **décompte aux points annoncés seuls**, ce qui
est cohérent avec une cible basse (§ 1.3).

---

## 2. Atteindre, ou dépasser ?

| Position | Sources | Extrait |
|---|---|---|
| **Atteindre suffit** (« atteignant *ou* dépassant ») | `ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt`, `ffbelote_org_..._COINCHEE_pdf.txt` (fam. A) | « Le premier camp **atteignant ou dépassant** le nombre de points fixés au préalable remporte la partie. » |
| | `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (fam. A) | « La première équipe **atteignant ou dépassant** le nombre de points fixés au préalable… » |
| | `ffbelote_org_regles_officielle_belote.txt` (fam. B) | « Le premier camp **atteignant** le nombre de points fixé remporte la partie. » |
| | `ainesruraux_saintsever_...pdf.txt` (fam. C) | « Elle **atteint ou dépasse** 2000 points » (1ʳᵉ des trois conditions) |
| | `adpoker_fr_belote_contree_html.txt` | « quand l'une des deux équipes a **atteint ou dépassé** le total de 2 000 points. » |
| | `pagat_com_jass_coinche_html.txt` (fam. D) | « reaches a score of 2000 points **or more** » |
| | `casimirdehauteclocque_fr_jeux_coinche_pdf.txt` | « la première équipe à **atteindre ou dépasser** un certain score » |
| | `valmathieu_ContrAI_main_contree-domain.md` | « The first team to **reach or exceed** the target at the end of a round wins. » |
| | `ismo009_Coinche_main_game.js` | `if (this.scores.ns >= this.targetScore …)` — **`>=`** |
| **Il faut strictement dépasser** | `maisondesessarts_fr_article116_html.txt` | « La partie se termine lorsqu'une équipe **dépasse** les 3000 points. » |
| | `ange_heureux_free_fr_JeuxDeCartes_La_Coinche_html.txt` | « Mais il faut **dépasser** ces 3000 points. » |
| | `jwebarteam_wordpress_com_..._les_regles.txt` | « Partie en 2001 points (**Il faut passer les 2000 pour gagner**) » |
| | `drasill_bga-coinche_master_coinche.game.php` | `// Check if end of game (score must be strictly higher than maxScore)` puis `if ($score > $maxScore)` |
| | `drasill_github_io_bga_coinche_rules_fr_html.txt` | « Dès qu'une équipe atteint 2000 points (**révolus**) » |
| **Contradiction interne** | `LOCAL_regles_officielles_belote_contree.txt` | Phrase : « Le premier camp **atteignant ou dépassant** 2000 points (au point près) remporte la partie. » — Exemple, 6 lignes plus bas : « Nous arrivons à **2000 points pile, nous ne gagnons pas** (encore) la partie car nous ne dépassons pas les 2000 points. » |
| **Seuil décalé pour éviter la question** | `tournois/festivaldesjeux_cannes_com_fr_evenement_1253_….txt` — **Championnat de France de Belote Contrée**, Cannes | « Tournoi libre en **5 parties de 2001 points** » ; finale : « les 4 premières parties de 2001 points et la 5ème en **2501 points** » |
| | `fnasce_org_IMG_pdf_reglement_pdf.txt` | « 1 010 points **minimum** » |
| | `web_archive_..._coinche_stephanoise_...pdf.txt` | « le score de **2000 points (soit 1995) minimum** » |
| | `jwebarteam_...txt` | cible fixée à **2001**, finales à **3001** |
| **La FFB reconnaît le désaccord et tranche… sur autre chose** | `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` | « La fin de partie en belote contrée est trop souvent le centre de règles différentes de régions en régions, **doit-on faire un pli supplémentaire, atteindre les points, les dépasser ?** Afin d'empêcher tout problème… la Fédération homologue la règle suivante : **Les points sont arrondis en fin de partie et permettent de gagner.** » |

**Divergence tranchée, et elle est ancienne.** La FFB pose la question à voix haute puis
répond à côté : sa « règle homologuée » porte sur l'arrondi (cf. [arrondi.md](arrondi.md)),
pas sur atteindre-vs-dépasser. Deux implémentations informatiques du même jeu prennent des
décisions opposées et **explicitement commentées** (`>=` chez ismo009, `> maxScore` chez
BoardGameArena avec le commentaire « score must be strictly higher »).

**L'incohérence de `LOCAL_regles_officielles_belote_contree` est réelle et vérifiée.** Le
document est celui dont Colver s'est inspiré ; sa phrase normative et son exemple chiffré se
contredisent mot pour mot. Trois règlements de concours contournent le problème en décalant
le seuil (1 010, 1 995, 2 001), ce qui est la preuve que la question se pose vraiment à la
table.

**Et le plus gros tournoi de contrée de France fait exactement ça.** Le Championnat de France de
Belote Contrée, disputé au Festival International des Jeux de Cannes et organisé par BELOTE
CONTREE MARALPINE, joue **toutes ses parties en 2001 points** — qualificatifs, tournois libres,
Grand Prix de l'Amitié, Grand Prix de Cannes — et sa **finale en 4 × 2001 + 1 × 2501**, entre les
24 meilleures équipes des trois qualificatifs. Le seuil décalé n'est donc pas un bricolage de
village : c'est le format de l'épreuve la plus visible de la discipline. Personne n'écrit
« il faut dépasser 2000 » ; on écrit 2001 et la question ne se pose plus.

Note sur la 5ᵉ partie à 2501 : c'est le seul mécanisme du corpus qui **augmente la cible de la
dernière manche** pour départager. Aucun règlement ne le théorise.

**Note terminologique utile.** La belote bulgare est le seul corpus qui **nomme** les deux
options : `licitum_board_directory_net_t16_belot_rules.txt` — « In first type of ending
(called **"until enough"**) a player/team that is to reach a limit… the round is played until
the player/team reaches that limit… In second type of ending (called **"until passed"**) a
player/team that is to reach a limit still has to pass current round ». Aucun texte français
du corpus ne dispose d'un tel couple de noms.

---

## 3. Départage quand les deux camps franchissent la cible sur la même donne

| Position | Sources | Extrait |
|---|---|---|
| **Le plus gros total (cumulé)** | `ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt`, `ffbelote_org_..._COINCHEE_pdf.txt`, `LOCAL_...txt`, `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (fam. A) | « c'est celui qui a **le plus de points au-delà** qui remporte la partie » |
| | `ffbelote_org_regles_officielle_belote.txt` (fam. B) | idem |
| | `maisondesessarts_fr_article116_html.txt` | « c'est l'équipe qui **va le plus loin** qui est déclarée gagnante » |
| | `pagat_com_jass_coinche_html.txt` (fam. D) | « the team with the higher score wins » |
| | `fr_wikipedia_org_wiki_Coinche.txt` | « l'équipe qui atteint **le plus gros total** remporte la partie » |
| | `jeux_regles_com_regles_coinche.txt`, `lemagloisirs_fr_regle_coinche.txt`, `ibelote_com_...txt`, `gameduell_..._1054_coinche.txt`, `carafons_...txt`, `drasill_..._rules_fr_html.txt`, `valmathieu_..._contree-domain.md` | même formule |
| **Le plus gros total *de la dernière donne seule*** | `ainesruraux_saintsever_...pdf.txt` (fam. C) | « l'équipe gagnante [est] celle ayant réalisé le plus gros total **en tenant compte des points réels de la dernière mène uniquement**, y compris… la "Belote" » |
| **L'équipe qui a pris le dernier contrat** | `bk_jeux_ducale_..._pdf.txt` | « c'est l'équipe qui **a pris le dernier contrat** qui remporte la partie » |
| | `ilyesbrh_..._coinche_GAME_RULES.md` (source secondaire) | « If both teams cross the target in the same round, **the contracting team wins** » |
| **Les vainqueurs de la dernière donne** | `pagat_com_jass_coinche_html.txt` (fam. D) | « In case of a tie, **the winners of the latest deal** win the game. » *(2ᵉ niveau, après égalité de total)* |
| **On rejoue** | `ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt`, `..._COINCHEE_pdf.txt`, `LOCAL_...txt` (fam. A, 2ᵉ niveau) | « En cas de nouvelle égalité, **une dernière donne** départage les deux camps. » |
| | `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (fam. A, réécriture) | « En cas d'égalité, **une nouvelle donne est jouée jusqu'à ce que l'égalité soit rompue**. » |
| | `fr_wikipedia_org_wiki_Coinche.txt` | « Si leur total est identique, **une donne supplémentaire** est effectuée » |
| | `valmathieu_..._contree-domain.md` | « nobody has won yet: play continues with additional rounds (**sudden death**) » |
| | `ainesruraux_saintsever_...pdf.txt` (fam. C) | « il faut jouer **encore une mène** puisque aucune des deux ne remplit les trois conditions » |
| **Match nul** | `pagat_com_jass_belote_html.txt` *(belote classique)* | « If both sides reach 1000 points on the same hand, **the game is drawn**. » |
| **Ordre de comptabilisation (« premier sorti »)** | `swisslos_..._le_chibre_html.txt` *(chibre)* | « Les règles de sorties sont « **Stöck, Wys, Stich** ». La règle ne s'applique que lorsque les deux équipes atteignent simultanément l'objectif… elle détermine **l'ordre dans lequel les points sont comptabilisés**. » |
| | `amisduchibre_ch_histoire_du_chibre_ou_du_jass.txt` | « "**premier sorti**" : … on suit le décompte de points jusqu'à ce qu'une des équipes passe la barre des 1 000 points. Le stöck compte en premier. Les annonces comptent en second. » |

**Consensus solide au premier niveau** : « le plus gros total » l'emporte — c'est la position
des 4 rédactions FFB, de Pagat, de Wikipédia, de BGA et de 6 sites généralistes, soit une
dizaine de témoins indépendants.

**Divergence au second niveau et sur les exceptions** : que faire en cas d'égalité parfaite
(rejouer une donne — majoritaire ; match nul — Pagat pour la belote classique) et surtout
**trois mécaniques qui ne regardent pas le total cumulé du tout** : la dernière donne seule
(fam. C), le dernier preneur (Ducale, + ilyesbrh en secondaire), et l'ordre de
comptabilisation (chibre suisse), où c'est la *séquence* Stöck → annonces → plis qui désigne
qui a franchi la barre en premier. Cette dernière est la seule solution du corpus qui rende
la question décidable sans arbitrage ni donne supplémentaire.

---

## 4. Conditions supplémentaires pour gagner

| Position | Sources | Extrait |
|---|---|---|
| **Trois conditions cumulatives** | `ainesruraux_saintsever_...pdf.txt` (fam. C) | « Une équipe sera déclarée gagnante si elle remplit **trois** conditions : - Elle atteint ou dépasse 2000 points, - Elle obtient **plus de points que l'équipe adverse**, - Elle **n'est pas capot ou ne chute pas** sur un contrat. » |
| **On ne finit pas sur une belote *si l'on est capot*** | `ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` (fam. A) | « Si une équipe est **capot**, et atteint les points fixés **uniquement grâce aux points d'une belote**, la partie n'est pas encore gagnée… L'équipe aura besoin d'**un pli supplémentaire** pour valider la partie. » |
| | `ffbelote_org_..._COINCHEE_pdf.txt` et `LOCAL_...txt` (fam. A) | même clause, élargie : « Si une équipe est **capot ou chute**… » |
| | `ffbelote_org_reglements_de_la_belote_...txt`, `ffbelote_org_regles_coinche.txt`, `villeconin_...txt` (fam. B) | « **Il est possible de terminer sur une belote. La seule condition est de ne pas être capot.** On ne termine jamais une partie sur un capot accompagné d'une belote… un pli supplémentaire devra être effectué. » |
| | `ainesruraux_saintsever_...pdf.txt` (fam. C) | « L'expression "on ne fini pas que la Belote" signifie simplement que la "Belote seule" ne peut permettre de finir **si l'équipe est capot**. Dans tous les autres cas elle "aide à finir" pour 20 points. » |
| **On ne gagne jamais sur une belote (sans condition)** | `ange_heureux_free_fr_JeuxDeCartes_La_Coinche_html.txt` | « Et l'on **ne gagne jamais sur une belote**. » |
| **Ni sur un capot + une annonce** | `web_archive_..._coinche_stephanoise_...pdf.txt` | « … et que la partie **ne se termine pas sur un capot et une annonce**. Au cas où le possesseur d'une annonce finit sa partie sur cette annonce, alors l'adversaire le met capot, la partie ne se termine pas. Il faut faire **une donne supplémentaire**. » |
| **On peut au contraire sortir sur capot** | `rjcv_be_belote_regles_pdf.txt` | « **Il est permis de sortir sur capot.** » |
| **Il faut finir la donne en cours** | `officialgamerules_org_game_rules_belote.txt` | « but a **valat round must complete** before ending the game » |
| | `licitum_..._belot_rules.txt` | mode « until passed » : « still has to **pass current round** » |
| **La FFB 2016 supprime la clause** | `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` | Le § 10.4 réécrit **ne contient plus** le « cas particulier » de la belote — il a disparu entre les rédactions de 2015 et celle du 27/01/2016, sans remplacement. |
| **Muet** | Pagat/Gambiter (fam. D), Wikipédia coinche, BGA, adpoker, carafons, la quasi-totalité des sites généralistes | aucune condition supplémentaire |

**Consensus étroit mais net sur une clause précise** : *une belote seule ne fait pas gagner
une partie à une équipe capot ; il faut un pli supplémentaire*. Trois familles indépendantes
la portent (FFB 2015, pages web FFB, règlement du tournoi international), avec un
**désaccord sur le périmètre** — capot seulement (contrée 2015, fam. B, fam. C) ou capot **et**
chute (coinchée 2015, LOCAL). La formulation la plus explicite est celle de la famille B :
« Il est possible de terminer sur une belote. La seule condition est de ne pas être capot. »

**Divergence dure** : `rjcv.be` autorise explicitement l'inverse (« sortir sur capot »),
`ange_heureux` durcit sans condition (« on ne gagne jamais sur une belote »), et
**la FFB elle-même a retiré la clause dans sa rédaction de 2016**. La condition « ne pas
chuter sur la dernière donne » n'existe que dans la famille C et, implicitement, dans
LOCAL / coinchée 2015 via « capot ou chute ».

---

## 5. Le litige (égalité parfaite des points cartes, 81-81)

| Position | Sources | Extrait |
|---|---|---|
| **Il existe : la défense marque, les points du preneur sont remis en jeu** | `ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` *(belote classique)* | « En cas d'égalité entre les totaux (**81-81 ou 91-91**), il y a « litige ». Les points du preneur sont **remis en jeu** (81 ou 91) et seront reversés à l'équipe réussissant le contrat de la donne suivante. L'équipe qui n'avait pas pris marque ses points **dès la donne ayant mené au litige**. » |
| | `ffbelote_org_regles_officielle_belote.txt`, `villeconin_...txt` (fam. B) | « la défense marque ses points, mais les points des preneurs sont remis en jeu et seront offerts en bonus aux vainqueurs de la **prochaine prise** » |
| | `fr_wikipedia_org_wiki_Belote.txt` | « l'équipe qui défend marque immédiatement ses points. Ceux de l'équipe du preneur seront acquis par le **vainqueur de la donne suivante**. » |
| | `fnasce_org_IMG_pdf_Belote_Reglement_cle19a7cf_pdf.txt` (fam. G) | « Le litige se situe à 81 pts… Les 81 points restants sont attribués au vainqueur du tour suivant. **Si un litige se produit au 12ème et dernier tour, chaque équipe marquera ses 81 points.** » |
| | `fnasce_org_IMG_pdf_reglement_belote_pdf.txt` (fam. G) | « Le litige se situe à **81 pts (91 points si la belote a été annoncée)** » |
| | `web_myassoc_org_..._Lions_...pdf.txt` (fam. J/G) | idem, mais « au **10ème** et dernier tour » |
| | `cdfcasson_...pdf.txt`, `lesamisdutempslibrevarennes_...txt` (fam. H) | « 81/81, il y a litige, **la belote n'est pas prise en compte** : la défense marque ses points « 81 », mais les points des preneurs « 81 » sont remis en jeu… L'équipe qui a éventuellement la belote marque ses 20 points. » |
| | `lagrandcombe_..._Reglement_belote_2020_pdf.txt` | « ceux-ci sont **pendus** pour l'équipe qui a pris et les points pendus seront marqués par l'équipe qui gagnera la donne suivante. » |
| | `s1_static_footeo_..._Rglement_Concours_...pdf.txt` (FC Plouay) | « l'équipe ayant annoncé la couleur ne marque rien. L'autre équipe marque ses points. Au tour suivant, l'équipe gagnante marque ses po[i]nts plus les points du litige. » |
| | `rjcv_be_belote_regles_pdf.txt` | « l'équipe allant laisse les points **en suspens** tandis que l'équipe adverse marque les points. Les points en suspens seront ajoutés aux points de l'équipe gagnant la mêle suivante. » |
| | `playjoy_com_en_coinche_rules.txt` *(appliqué à la coinche !)* | « **Litige** — In case of tie, the team that chose the trump does not score points and they will be **reserved for the winner of the next round**. » |
| **Il n'existe pas : 81 fait perdre le preneur** | `aappmakoenigshoffen_..._reglementtournoibelote_2_pdf.txt` | « Il n'y a **pas de litige** (81 points perd la partie). » |
| | `tcvb_bruche_..._tournoibelote2012reglement_htm.txt` | « LE PRENEUR TOTALISANT 81 POINTS **PERD LE JEU. LE LITIGE N'INTERVENANT PAS.** L'ADVERSAIRE MARQUE 162 POINTS » |
| **Il n'existe pas : on rejoue la donne** | `jwebarteam_wordpress_com_..._les_regles.txt` *(coinche)* | « En cas d'enchères arrêtées à 80, l'attaque doit impérativement faire 82 points. **S'il y a 81 à 81, il faut refaire la mène. (sans changer le donneur)** » |
| **Option laissée au tournoi** | `ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` | « **Chaque tournoi se réserve le droit d'appliquer ou non la règle du litige** et doit le stipuler clairement en amont. Dans le cas où elle n'est pas appliquée, le tournoi doit définir à l'avance si l'égalité de score provoque **la chute ou la réussite** du contrat. » |
| **Le contrat suffit, la comparaison n'a pas lieu** | `fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « **Contrairement à la belote, il n'est pas nécessaire de faire plus de points que son adversaire** pour remporter la manche. Seule compte la réussite du contrat. » |
| **Muet** | les 4 règlements FFB de **contrée/coinchée** (fam. A), Pagat, BGA, la totalité des sites de contrée | le mot « litige » n'y apparaît pas |

**Consensus fort sur le mécanisme, là où le litige existe** : la défense encaisse ses points
tout de suite, ceux du preneur sont mis en réserve et vont au vainqueur de la donne suivante.
Onze sources le disent, dont **au moins cinq familles indépendantes** (FFB belote, ASCE 79,
modèle « Nous vous remercions », La Grand-Combe, FC Plouay, rjcv.be) — et la formulation est
si stable qu'elle a son folklore lexical : « pendus », « en suspens », « remis en jeu ».
Deux sous-questions font consensus aussi : **au dernier tour, chaque équipe marque ses 81**
(fam. G, Lions, Géraudot), et **la belote reste acquise** (fam. G bis, fam. H).

**Divergence sur l'existence même.** Deux règlements alsaciens l'abolissent explicitement et
donnent 162 points au défenseur ; un championnat de coinche fait **rejouer la donne** ;
et **la FFB en fait une option de tournoi** qu'il faut annoncer à l'avance — en imposant
alors de trancher d'avance entre chute et réussite.

**Le point structurel** : le litige est un objet de la **belote classique**, où la réussite se
juge à la majorité des 162 points. **Aucun des quatre règlements FFB de contrée/coinchée ne
le mentionne**, et Wikipédia explique pourquoi — à la contrée, seule compte la réussite du
contrat. Les exceptions `playjoy` (litige appliqué à la coinche) et `jwebarteam` (81-81 sur
un contrat à 80) montrent que le cas ressurgit dès qu'un règlement exige *aussi* « plus de
points que la défense » — ce que font `regles_com`, `data_over_blog` et
`fr_wikipedia_org_wiki_Coinche`.

---

## 6. Format de concours

### 6.1 Structure (nombre de parties × nombre de donnes)

| Position | Sources | Extrait |
|---|---|---|
| **4 parties de 12 donnes** | `fnasce_org_IMG_pdf_Belote_Reglement_cle19a7cf_pdf.txt`, `fnasce_org_IMG_pdf_reglement_belote_pdf.txt`, `pontdeclaix_...pdf.txt` (fam. G) | « Le concours se déroulera en **4 parties de 12 donnes** comptabilisées. » |
| | `s1_static_footeo_...pdf.txt` (FC Plouay) | « Ce concours se déroule sur **4 parties de 12 levées** et sans annonce » |
| | `bonnafousn_wixsite_com_belote_fourquevaux.txt` | « Le concours se deroule en **4 parties de 12 mènes**. » |
| **4 parties de 10 donnes** | `web_myassoc_org_..._Lions_...pdf.txt` | « en **4 parties de 10 donnes** comptabilisées en changeant d'adversaire à chaque partie » |
| **5 tours de 12 coups** | `geraudotloisirs_...txt` | « Le concours comprend **5 tours de 12 coups de cartes** et l'on change 5 fois d'adversaire. » |
| **5 parties de 10 donnes** | `lesamisdutempslibrevarennes_...txt` (fam. H) | « Le concours se joue en **5 parties de 10 donnes** par équipe de 2 joueurs et sans annonce. » |
| **6 parties de 12 donnes + phases finales** | `cdfcasson_...pdf.txt` (fam. H) | « Le concours se joue en **6 parties de 12 donnes** + éventuellement quarts, demi et finale entre les premières équipes du classement par nombre de victoires » |
| **6 tours + phases finales** | `rjcv_be_belote_regles_pdf.txt` | « Le concours se déroule en **6 tours** + éventuellement quarts, demis et finale » |
| **3 manches de 20 jeux** | `aappmakoenigshoffen_..._pdf.txt` | « Le tournoi se dispute en **3 manches de 20 jeux**. » |
| | `tcvb_bruche_...htm.txt` | « LE CONCOURS SE DÉROULE EN **3 MANCHES DE 20 JEUX** » |
| | `club_belote_com_les_tournois_de_belote_html.txt` | « le Tournoi de Belote de Cannes… se décompose en **20 jeux/3 manches** » |
| **2 manches de 24 jeux** | `club_belote_com_...txt` | « le Challenge de Belote de la Côte d'Azur, qui se déroule selon **deux manches de 24 jeux**. » |
| **3 parties, à la cible et au chrono** | `clublafontainedejouvence_...txt` | « Le concours se déroule en **3 parties**. 2 parties de 121 points limitées à 135, la troisième en 151 limitée à 175… Les 2 premières parties se déroulent en **1h15**, la troisième en **1h30**. » |
| **Matchs en 2 manches gagnantes** | `jwebarteam_...txt` | « Tous les matchs se jouent en **2 manches gagnantes** (une partie, une revanche, une belle éventuelle) » |
| **À la cible ou au chrono, au choix de l'organisation** | `cdf_missegre11_...pdf.txt` | « lorsque l'une des deux équipes atteint le **score ou le temps maximal** fixé par l'organisation… Ce score et ce temps seront établis en fonction du nombre d'équipes afin de respecter au mieux les délais du tournoi. » |
| | `ainesruraux_saintsever_...pdf.txt` (fam. C) | « La durée moyenne d'une partie en 2000 points varie de **40 à 50 minutes**. Dans l'hypothèse où la durée d'une partie excéderait **1 heure**, l'organisation pourra décider de l'interrompre en obligeant les deux équipes à jouer un nombre de mènes que fixera l'organisation. » |
| **Les deux formats sont normés** | `ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` | « Chaque tournoi se réserve le droit de fixer pour chaque partie **une limite de donnes ou une limite de points** et le stipuler clairement en amont. » |
| **Mode tournoi = nombre de tours fixe** | `swisslos_..._le_chibre_html.txt` | « ou en **mode tournoi en 12 tours**… En mode tournoi, la partie s'arrête après 12 tours. » |
| **Muet** | l'intégralité des quatre règlements FFB **de contrée/coinchée**, Pagat, Wikipédia, BGA, tous les sites généralistes | aucun format de concours |

**Consensus de structure, pas de chiffre.** Tous les concours réels du corpus ont la **même
forme** : *N parties de M donnes, N ∈ [3, 6], M ∈ [10, 24], adversaire changé à chaque partie*.
Aucun ne joue « à la cible » comme une partie de salon — la cible, quand elle existe, sert à
finir *une partie du concours*, pas le concours. C'est le résultat le plus robuste de cet axe.

**Divergence sur M et N**, sans logique apparente ; et surtout **le format est le champ qu'on
édite** quand on recopie un règlement : la famille H a la même page à la virgule près avec
« 6 parties de 12 donnes » chez l'un et « 5 parties de 10 donnes » chez l'autre.

**La belote classique est le seul jeu du corpus qui ait un format de concours écrit.** Les
règlements de contrée décrivent le jeu et s'arrêtent — sauf Missègre, La Fontaine de Jouvence,
JWEBARTEAM, AIL Manissieux et Maison des Essarts.

### 6.2 Classement et appariement

| Position | Sources | Extrait |
|---|---|---|
| **Addition des points** | `web_myassoc_org_..._Lions_...pdf.txt` | « **Le classement se fait par addition de points** de chaque équipe. » (art. 8 et 21) |
| | `geraudotloisirs_...txt` | « Le classement se fait par **addition de points**. » |
| | `fnasce_org_..._cle19a7cf_pdf.txt`, `..._reglement_belote_pdf.txt`, `pontdeclaix_...txt` (fam. G) | « Le classement se fait par **addition des points** acquis au cours des 4 parties. » |
| | `bonnafousn_wixsite_...txt` | « Les classements sont calculés au **cumul des points** acquis lors des parties. » |
| **Nombre de victoires, départagé aux points** | `rjcv_be_...pdf.txt`, `cdfcasson_...pdf.txt` (fam. I) | « classement par **nombre de victoires** (départagés aux points en cas d'égalité) » |
| | `s1_static_footeo_...pdf.txt` | « L'équipe vainqueur sera celle qui aura **gagné le plus de parties**… suivant le nombre de victoires, de **points acquis** puis de **points perdus**. » |
| | `data_over_blog_kiwi_...pdf.txt` | « Le classement se fait au **nombre de parties gagnées** puis des **points faits moins les points laissés faire**. » |
| **Points de match (3/1/0)** | `jwebarteam_...txt` | « Victoire : **3 points** / Victoire par forfait : 2 / Défaite : 1 / Défaite par forfait : 0 » — goal-average sur les manches gagnées/perdues. |
| **Classement individuel** | `clublafontainedejouvence_...txt` | « Le classement final est **individuel** et s'établit au nombre de points cumulés dans les trois parties (ex : 41+67+154=262). » |
| **Appariement : tirage au sort au 1ᵉʳ tour, puis par classement** | `web_myassoc_org_..._Lions_...pdf.txt` | « Pour la première partie, les équipes seront réparties par **tirage au sort** sur les tables. Pour les parties suivantes, les tables… dans l'**ordre du classement**. » |
| | `fnasce_org_..._cle19a7cf_pdf.txt` (fam. G) | idem |
| | `cdfcasson_...pdf.txt`, `lesamisdutempslibrevarennes_...txt` (fam. H) | « Au premier tour, selon l'**ordre d'inscription**… À partir du deuxième tour, les équipes ayant le plus grand nombre de points cumulés joueront entre elles » |
| **Tirage au sort à chaque manche** | `aappmakoenigshoffen_...pdf.txt` | « Les places sont **tirées au sort après chaque manche**. » |
| | `clublafontainedejouvence_...txt` | « Les équipes sont distribuées par **tirage au sort pour chaque partie**. » |
| **Rotation mécanique (pairs/impairs)** | `pontdeclaix_...pdf.txt` | « les équipes ayant un numéro **impair restent à leur table**, et les équipes ayant un numéro **pair changeront de table** à chaque partie. » |
| **Procédure pour un nombre impair d'équipes** | `fnasce_org_IMG_pdf_reglement_belote_pdf.txt` | tirage de 4 équipes A/B/C/D, « partie 1 bis », puis chacune se repose un tour à son tour (art. 6, 8 lignes de procédure) |
| **Égalité au classement final** | `aappmakoenigshoffen_...pdf.txt` | « les lots seront **tirés au sort** » |
| | `tcvb_bruche_...htm.txt` | « EN CAS D'ÉGALITÉ LE VAINQUEUR DU CONCOURS SERA **TIRÉ AU SORT**. » |
| | `clublafontainedejouvence_...txt` | « il conviendra de faire référence au **meilleur score obtenu au cours de la 1ère partie** » |

**Divergence franche, et c'est la fracture structurante des concours** : classer à
**l'addition des points** (6 témoins, dont les familles G et J) ou au **nombre de victoires**
(4 témoins : rjcv, Casson, Plouay, AIL Manissieux, plus JWEBARTEAM en points de match).
Les deux ne récompensent pas le même jeu — à l'addition, écraser une équipe faible vaut mieux
que battre de peu une forte. Les deux camps sont de taille comparable dans le corpus.

**Consensus sur l'appariement** : tirage au sort au premier tour, puis **appariement par le
classement** (« les premiers jouent contre les premiers ») — 4 familles indépendantes.
Deux règlements préfèrent le tirage au sort à chaque manche, un troisième une rotation
mécanique pairs/impairs.

---

## 7. Pénalités en points

| Position | Sources | Extrait |
|---|---|---|
| **162 points à l'adversaire — le tarif standard** | `aappmakoenigshoffen_...pdf.txt` | signaler son jeu, ne pas jouer toutes les cartes, fausse donne répétée, et « **Toutes erreurs, fautes ou fraudes** sont également sanctionnées de **162 points** » |
| | `tcvb_bruche_...htm.txt` | signaler son jeu, faute de jeu, jouer avant son tour, plis non couverts → « L'ADVERSAIRE MARQUE **162 POINTS** » |
| | `fnasce_org_..._cle19a7cf_pdf.txt`, `..._reglement_belote_pdf.txt`, `pontdeclaix_...txt` (fam. G) | « maldonne après la mise à l'atout… **162 pts** » ; « irrégularité pendant le jeu… le coup annulé et l'équipe fautive pénalisée de **162 points** » |
| | `s1_static_footeo_...pdf.txt` | barème explicite : maldonne après atout / ne pas fournir / omettre de couper / ne pas monter → **162 points** chacun |
| | `web_myassoc_org_..._Lions_...pdf.txt` | « l'équipe fautive sera alors pénalisée de **162 points** » ; 3ᵉ fausse donne → 162 pts « et cela compte pour un coup de cartes » |
| | `geraudotloisirs_...txt` | « S'il y a récidive, au **3ème essai**, l'équipe adverse marque **162 points** et cela compte pour un coup de cartes. » |
| | `cdfcasson_...pdf.txt`, `lesamisdutempslibrevarennes_...txt` (fam. H) | « Une **deuxième** fausse donne sera pénalisée par **162 points** » ; toute faute de jeu idem |
| | `lagrandcombe_..._pdf.txt` | « l'équipe adverse marque **162 points** et cela autant de fois qu'il y aura de fausses donnes » ; impasse à l'atout → 162 |
| **160 points** | `ainesruraux_saintsever_...pdf.txt` / `cdf_missegre11_...pdf.txt` | « À partir de la **troisième irrégularité** dans la même partie, **160 points** sont donnés à l'équipe [adverse] » |
| | `web_archive_..._coinche_stephanoise_...pdf.txt` | « En cas d'une **deuxième maldonne** on attribuera **160 points** à ses adversaires et le joueur suivant sera chargé d'effectuer la donne. » |
| | `cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` | « la donne est considérée comme perdue pour l'équipe fautive et **160 points** sont donnés à l'équipe adverse. » |
| **Peine graduée, pas seulement chiffrée** | `cdf_missegre11_...pdf.txt` / `ainesruraux_...pdf.txt` | 1ʳᵉ fausse donne : redistribuer. 2ᵉ : « l'équipe ayant commis la faute se verra pénalisée et **interdite de toute prise** sur cette seconde donne ». 3ᵉ : 160 points. |
| | `lagrandcombe_...pdf.txt` | « celle-ci refait la donne et **n'a pas le droit de prendre** » |
| | `web_archive_..._coinche_stephanoise_...pdf.txt` | « En cas de maldonne, le donneur redistribue, mais **il ne pourra pas faire d'enchère**. » |
| **16 points (barème belge)** | `rjcv_be_...pdf.txt` | « Pendant la retouche, c'est pénalisé de **16** plus les annonces et la main passe. » |
| **Forfait / abandon** | `lagrandcombe_...pdf.txt` | « les équipes qui tombent contre l'équipe qui a dû arrêter ou qui est disqualifiée marquent **1100 points sans jouer**. » |
| | `ainesruraux_saintsever_...pdf.txt` (fam. C) | « toute équipe qui ne serait pas en état de jouer **10 minutes** après le début sera purement et simplement considérée comme ayant **perdu cette partie**. » |
| | `jwebarteam_...txt` | « Victoire par forfait : **2 points** / Défaite par forfait : **0** » |
| **Plafonnement des scores** | `lagrandcombe_...pdf.txt` | « À la **troisième et quatrième partie**, les points sont **plafonnés à 1280 points** pour l'équipe gagnante et **664 points** pour l'équipe perdante. » |
| | `clublafontainedejouvence_...txt` | « 2 parties de 121 points **limitées à 135**… la troisième en 151 points **limitée à 175** » |
| **Exclusion** | `aappmakoenigshoffen_...pdf.txt` | « l'organisateur se réserve le droit d'**éliminer le fautif** » |
| | `ainesruraux_...pdf.txt` (fam. C) | « L'organisation pourra décider d'**exclure l'équipe** » / « d'éliminer toute équipe dont le comportement troublerait la bonne marche du tournoi » |
| | `cdfcasson_...pdf.txt` (fam. H) | « Les personnes convaincues de **tricherie** seront **exclues définitivement** de ce concours **et des suivants**. » |
| | `clublafontainedejouvence_...txt` | « Tout joueur qui enfreindrait ce règlement serait **éliminé du classement final**. » |
| | `reset_bar_ch_reglement_tournoi_jass.txt` | « Celui qui interrompt, perturbe ou retarde le jeu de façon injustifiée est **automatiquement exclu du tournoi**. » |
| **Muet** | les quatre règlements FFB, Pagat, Wikipédia, BGA, tous les sites généralistes | aucune pénalité chiffrée |

**Consensus le plus net de tout le document : 162 points à l'adversaire.** Huit règlements de
concours issus d'**au moins cinq familles textuelles distinctes** (Alsace, ASCE 79, modèle
« Nous vous remercions », modèle Géraudot/Lions, FC Plouay, La Grand-Combe) appliquent le
même tarif unique — le total de la donne — quelle que soit la faute. C'est un fait de
pratique, pas de doctrine : **aucun texte fédéral ni encyclopédique du corpus ne mentionne
la moindre pénalité en points**. Les concours ont écrit leur droit pénal tout seuls.

**Divergence mineure mais cohérente** : 162 contre 160 (la famille C/Missègre et la
coinche-stéphanoise, qui sont des textes de **contrée**, où 160 est la valeur du dedans).
La pénalité suit la base de la chute du jeu concerné, donc l'écart n'est pas un désaccord.

**Divergence sur le seuil de déclenchement** : 2ᵉ faute (fam. H, coinche-stéphanoise),
3ᵉ faute (Géraudot, Lions, Missègre), ou immédiate (Alsace, FC Plouay, fam. G).

---

## 8. La donne annulée (quatre passes)

⚠️ **Deux jeux différents.** À la contrée, quatre passes signifient qu'aucune enchère n'a
été faite. À la belote classique, il y a deux tours (carte retournée puis couleur libre) et
« personne ne prend » n'a pas la même conséquence. Les positions ci-dessous ne sont
comparables qu'à l'intérieur de chaque bloc.

### 8.1 Contrée / coinche

| Position | Sources | Extrait |
|---|---|---|
| **Pas de points, le donneur suivant redonne** | `ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt`, `ffbelote_org_..._COINCHEE_pdf.txt`, `LOCAL_...txt` (fam. A) | « Si les 4 joueurs passent sans enchérir, la donne est terminée **sans points marqués**. Le donneur rassemble les cartes et les fait passer au **joueur à sa droite** qui sera le nouveau donneur. » |
| | `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (fam. A) | « …sans points marqués. **Le joueur à la droite du donneur devient le nouveau donneur.** » (même règle, phrase resserrée) |
| | `ainesruraux_saintsever_...pdf.txt` (fam. C) | « Si les quatre joueurs passent après la donne, la mène n'est pas jouée et le jeu passe… au **joueur suivant**. » |
| | `fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « une nouvelle donne est réalisée par le **joueur placé à la droite du précédent donneur**. » |
| | `web_archive_..._coinche_stephanoise_...pdf.txt` | « Si tout le monde passe une nouvelle donne a lieu par le **joueur placé à droite du précédent donneur**. » |
| | `reglesdejeux_github_io_...txt` (fam. D) | « les cartes sont jetées et le **prochain donneur** distribue une nouvelle main. » |
| | `belotepoint_fr_regles_coinche.txt` | « Si les quatre joueurs passent sans enchérir, les cartes sont redistribuées. » *(ne dit pas par qui)* |
| **Reformulé sur le premier parleur (revient au même)** | `casimirdehauteclocque_fr_jeux_coinche_pdf.txt` | « Si tout le monde passe dès le début, on redistribue le jeu en **changeant le premier joueur** (celui qui a passé le premier distribue le jeu) » |
| **La donne annulée ne compte pas dans le quota** | `ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` | « Si les quatre joueurs passent sur les deux tours d'enchères, **on ne décompte pas cette donne**. » |
| **Muet** | Pagat/Gambiter (fam. D, texte anglais), Wikipédia coinche, adpoker, carafons, Ducale, la plupart des concours de contrée | — |

### 8.2 Belote classique (concours)

| Position | Sources | Extrait |
|---|---|---|
| **Le même donneur redonne jusqu'à ce que l'atout soit pris** | `web_myassoc_org_..._Lions_...pdf.txt` (fam. J) | « Si personne ne prend à son tour, **le même joueur redonne** jusqu'à ce que l'atout soit pris. » |
| | `geraudotloisirs_...txt` (fam. J) | formule identique |
| | `fnasce_org_..._cle19a7cf_pdf.txt`, `..._reglement_belote_pdf.txt`, `pontdeclaix_...txt` (fam. G) | « En cas de non prise, **le même donneur refait** jusqu'à mise à l'atout. » |
| **Le donneur suivant redonne** | `ffbelote_org_reglements_de_la_belote_...txt`, `ffbelote_org_regles_officielle_belote.txt`, `villeconin_...txt` (fam. B) | « Si les 4 joueurs passent de nouveau, le donneur rassemble les cartes et les donne au **joueur suivant** qui sera le nouveau donneur. » |
| | `ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` | « la donne est terminée. **Le joueur situé immédiatement à la droite du donneur** devient le nouveau donneur. » |
| **Personne ne redonne : un joueur est forcé** | `rjcv_be_...pdf.txt` | « **3ème tour forcé** : Si tous les joueurs ont passé 2 fois, le joueur à gauche du donneur est **forcé**. Tout le monde reçoit sa retouche et le joueur forcé peut choisir une des 4 couleurs. » |
| **Règle voisine, souvent citée avec** | `aappmakoenigshoffen_...pdf.txt`, `tcvb_bruche_...htm.txt`, `rjcv_be_...pdf.txt`, `cdfcasson_...pdf.txt` | « Il n'y a pas de **valet forcé**. » |

**Consensus en contrée** : quatre passes = donne morte, aucun point, **le donneur tourne**.
Sept sources, dont quatre familles indépendantes (FFB, tournoi international, Wikipédia,
coinche-stéphanoise). Une seule dissidence apparente, Casimir de Hauteclocque, et elle n'en
est pas une : son règlement fait distribuer par « le joueur qui a commencé à parler à la
manche précédente », donc changer de premier parleur *est* changer de donneur.

**Divergence dure en belote classique** : le **même donneur** redonne (fam. G et J — six
règlements de concours, deux familles) contre **le donneur suivant** (fam. B et FFB 2016 —
la fédération). Sur ce point précis, **la pratique des concours contredit frontalement la
fédération**, et rjcv.be invente une troisième voie (forcer un joueur à prendre) qui
supprime la redonne.

---

## 9. Ce que le corpus ne dit pas

- **Aucun règlement FFB de contrée ne chiffre la cible.** Les quatre rédactions écrivent
  « le nombre de points fixés au préalable » ; seul `LOCAL` avance 2 000, et c'est
  précisément l'édition non fédérale (« Équipe Ludique »).
- **Aucun texte fédéral ou encyclopédique ne prévoit de pénalité en points.** Tout le droit
  pénal du jeu (162 points, exclusion, forfait) est d'origine associative.
- **Aucun règlement de contrée ne décrit un format de concours complet**, à cinq exceptions
  près (Missègre, Maison des Essarts, AIL Manissieux, La Fontaine de Jouvence, JWEBARTEAM).
  Les seuls formats détaillés du corpus concernent la belote classique.
- **Personne n'explique pourquoi 1 010, 1 995 ou 2 001.** Ces trois seuils décalés sont
  visiblement des réponses au problème « atteindre ou dépasser », mais aucun texte ne le dit.
- **Le « litige » n'a pas de traitement en contrée**, alors que plusieurs règlements de
  contrée exigent « plus de points que la défense » et créent donc le cas.

---

## Table des sources

Fichiers relatifs à `data/rules-corpus/`. Les entrées marquées ⚠ n'ont pas d'en-tête `SOURCE:`
dans le `.txt`.

### `federations/`
| Fichier | URL |
|---|---|
| `LOCAL_regles_officielles_belote_contree.txt` | ⚠ pas d'en-tête — copie locale du PDF « Règles officielles de la belote contrée » (éd. Équipe Ludique), cf. README |
| `ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` | ⚠ pas d'en-tête — PDF FFB « REGLES OFFICIELLES DE LA BELOTE CONTRÉE » (ffbelote.org) |
| `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` | ⚠ en-tête « Fédération Française de Belote » — PDF FFB du 27/01/2016 (contrée) |
| `ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` | ⚠ en-tête « Fédération Française de Belote » — PDF FFB du 27/01/2016 (belote classique) |
| `ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` | https://www.ffbelote.org/wp-content/uploads/2015/11/REGLES-DE-LA-BELOTE-COINCHEE.pdf |
| `ffbelote_org_regles_officielle_belote.txt` | https://www.ffbelote.org/regles-officielle-belote/ |
| `ffbelote_org_reglements_de_la_belote_avec_ou_sans_annonce.txt` | https://www.ffbelote.org/reglements-de-la-belote-avec-ou-sans-annonce/ |
| `ffbelote_org_regles_coinche.txt` | https://www.ffbelote.org/regles-coinche/ |
| `ffbelote_org_belote_contree.txt` | https://www.ffbelote.org/belote-contree/ |

### `tournois/`
| Fichier | URL |
|---|---|
| `aappmakoenigshoffen_e_monsite_com_medias_files_reglementtournoibelote_2_pdf.txt` | http://aappmakoenigshoffen.e-monsite.com/medias/files/reglementtournoibelote-2.pdf |
| `ainesruraux_saintsever_com_belote_BELOTE_20TRADITIONNELLE_pdf.txt` | http://www.ainesruraux-saintsever.com/belote/BELOTE%20TRADITIONNELLE.pdf |
| `bonnafousn_wixsite_com_belote_fourquevaux.txt` | https://bonnafousn.wixsite.com/belote-fourquevaux |
| `casimirdehauteclocque_fr_jeux_coinche_pdf.txt` | https://casimirdehauteclocque.fr/jeux/coinche.pdf |
| `cdf_missegre11_com_medias_files_belote_contre_e_pdf.txt` | http://www.cdf-missegre11.com/medias/files/belote-contre-e.pdf |
| `cdfcasson_fr_files_ugd_0df194_8c501176abb04e5b9237c65fcf80f584_pdf.txt` | https://www.cdfcasson.fr/_files/ugd/0df194_8c501176abb04e5b9237c65fcf80f584.pdf |
| `club_belote_com_les_tournois_de_belote_html.txt` | https://www.club-belote.com/les-tournois-de-belote.html |
| `clublafontainedejouvence_fr_r_C3_A8glement_coinch_C3_A9e.txt` | https://www.clublafontainedejouvence.fr/r%C3%A8glement/coinch%C3%A9e |
| `data_over_blog_kiwi_com_1_05_17_17_20150128_ob_1f68a4_2015_01_27_reglement_table_coinche_pdf.txt` | http://data.over-blog-kiwi.com/1/05/17/17/20150128/ob_1f68a4_2015-01-27-reglement-table-coinche.pdf |
| `fnasce_org_IMG_pdf_reglement_pdf.txt` | https://www.fnasce.org/IMG/pdf/reglement.pdf |
| `fnasce_org_IMG_pdf_belote_reglement_cle1a43c7_pdf.txt` | https://www.fnasce.org/IMG/pdf/belote_reglement_cle1a43c7.pdf |
| `fnasce_org_IMG_pdf_Belote_Reglement_cle19a7cf_pdf.txt` | https://www.fnasce.org/IMG/pdf/Belote_Reglement_cle19a7cf.pdf |
| `fnasce_org_IMG_pdf_reglement_belote_pdf.txt` | https://www.fnasce.org/IMG/pdf/reglement_belote.pdf |
| `geraudotloisirs_free_fr_index_php_option_com_content_view_article_id_116_I.txt` | http://geraudotloisirs.free.fr/index.php?option=com_content&view=article&id=116&Itemid=102 |
| `jeu_belote_fr_regles_php_part_tournoi_competition_belote.txt` | http://www.jeu-belote.fr/regles.php?part=tournoi-competition-belote |
| `jwebarteam_wordpress_com_championnat_de_coinche_2016_2017_les_regles.txt` | https://jwebarteam.wordpress.com/championnat-de-coinche-2016-2017/les-regles/ |
| `lagrandcombe_fr_wp_content_uploads_2020_01_Reglement_belote_2020_pdf.txt` | https://www.lagrandcombe.fr/wp-content/uploads/2020/01/Reglement-belote-2020.pdf |
| `lesamisdutempslibrevarennes_jimdofree_com_activit_C3_A9s_jeux_concours_concours_belote.txt` | https://lesamisdutempslibrevarennes.jimdofree.com/activit%C3%A9s/jeux-concours/concours-belote/ |
| `maisondesessarts_fr_article116_html.txt` | https://www.maisondesessarts.fr/article116.html |
| `pontdeclaix_fr_sites_default_files_2024_01_65a68d18ab985_rglementconcoursdebelote_pdf.txt` | https://www.pontdeclaix.fr/sites/default/files/2024-01/65a68d18ab985_rglementconcoursdebelote.pdf |
| `rjcv_be_belote_regles_pdf.txt` | https://www.rjcv.be/belote/regles.pdf |
| `s1_static_footeo_com_uploads_fcplouay_Medias_Rglement_Concours_mtqpkb_pdf.txt` | http://s1.static-footeo.com/uploads/fcplouay/Medias/Rglement_Concours__mtqpkb.pdf |
| `sc4e58b2fce8a2e7a_jimcontent_com_download_version_1707646012_module_12973656226_name_Belote_20R_C3_A8glement_2.txt` | https://sc4e58b2fce8a2e7a.jimcontent.com/download/version/1707646012/module/12973656226/name/Belote%20R%C3%A8glement%20%202024.pdf |
| `tcvb_bruche_free_fr_dossomma_tournoibelotte2012_tournoibelote2012reglement_htm.txt` | http://tcvb.bruche.free.fr/dossomma/tournoibelotte2012/tournoibelote2012reglement.htm |
| `villeconin_fr_wp_content_uploads_2017_02_F_C3_A9d_C3_A9ration_fran_C3_A7aise_de_Belote_Informations_sur_le_jeu.txt` | https://villeconin.fr/wp-content/uploads/2017/02/F%C3%A9d%C3%A9ration-fran%C3%A7aise-de-Belote-Informations-sur-le-jeu-de-belote.pdf |
| `web_archive_org_web_2020_http_coinche_stephanoise_com_mesdocuments_reglement_coinche_pdf.txt` | https://web.archive.org/web/2020/http://coinche-stephanoise.com/mesdocuments/reglement_coinche.pdf |
| `web_myassoc_org_img_Lions_E2uYP6mbNv79_2238_8c929f766_medias_c7aa239f63982fbbac73870b2563ec21_pdf.txt` | https://web.myassoc.org/img/Lions_E2uYP6mbNv79/2238_8c929f766/medias/c7aa239f63982fbbac73870b2563ec21.pdf |

### `divers/` et `apps-sites/`
| Fichier | URL |
|---|---|
| `adpoker_fr_belote_contree_html.txt` | https://www.adpoker.fr/belote-contree.html |
| `alhoa_free_fr_ALH_belote_rules_htm.txt` | http://alhoa.free.fr/ALH/belote_rules.htm |
| `ange_heureux_free_fr_JeuxDeCartes_La_Coinche_html.txt` | http://www.ange.heureux.free.fr/JeuxDeCartes/La_Coinche.html |
| `belotecontree_free_reglement.txt` | ⚠ pas d'en-tête — belotecontree.free.fr, « Belote Contrée : le règlement officiel » |
| `belotepoint_fr_regles_belote.txt` | https://www.belotepoint.fr/regles-belote |
| `belotepoint_fr_regles_coinche.txt` | https://www.belotepoint.fr/regles-coinche |
| `bk_jeux_ducale_fr_app_uploads_2022_06_cartes_a_jouer_Ducale_regle_jeu_belote_coinchee_pour_joueur_expert_pdf.txt` | https://bk.jeux-ducale.fr/app/uploads/2022/06/cartes-a-jouer-Ducale-regle-jeu-belote-coinchee-pour-joueur-expert.pdf |
| `carafons_fr_regles_de_la_coinche.txt` | https://carafons.fr/regles-de-la-coinche/ |
| `cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` | https://cartesetcie.fr/regle-du-jeu-la-belote-coinchee/ |
| `cartesetcie_fr_regles_officielles_de_la_belote.txt` | https://cartesetcie.fr/regles-officielles-de-la-belote/ |
| `clubdejeux_com_belote_coinchee_online_regles.txt` | https://www.clubdejeux.com/belote-coinchee-online/regles |
| `drasill_github_io_bga_coinche_rules_fr_html.txt` | https://drasill.github.io/bga-coinche/rules-fr.html |
| `exoty_com_compter_points_coinche.txt` | https://exoty.com/compter-points-coinche |
| `exoty_com_regles_coinche_belote.txt` | https://exoty.com/regles-coinche-belote |
| `fr_wikipedia_org_wiki_Belote.txt` | https://fr.wikipedia.org/wiki/Belote |
| `fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | https://fr.wikipedia.org/wiki/Belote_contr%C3%A9e |
| `fr_wikipedia_org_wiki_Coinche.txt` | https://fr.wikipedia.org/wiki/Coinche |
| `gambiter_com_cards_jass_coinche_html.txt` | https://gambiter.com/cards/jass/coinche.html |
| `gameduell_helpshift_com_hc_en_16_belote_com_belote_coinche_faq_1054_coinche.txt` | https://gameduell.helpshift.com/hc/en/16-belote-com---belote-coinche/faq/1054-coinche/ |
| `gamerules_com_rules_coinche.txt` | https://gamerules.com/rules/coinche/ |
| `ibelote_com_en_rules_belote_php.txt` | https://ibelote.com/en/rules-belote.php |
| `iscool_helpshift_com_hc_fr_10_belote_mobile_faq_157_how_to_play_coinche_coinche_rules.txt` | https://iscool.helpshift.com/hc/fr/10-belote-mobile/faq/157-how-to-play-coinche-coinche-rules/ |
| `jeubelote_com_regle_de_la_belote_html.txt` | https://www.jeubelote.com/regle-de-la-belote.html |
| `jeux_regles_com_regles_coinche.txt` | https://jeux-regles.com/regles-coinche/ |
| `lemagloisirs_fr_regle_coinche.txt` | https://www.lemagloisirs.fr/regle-coinche/ |
| `licitum_board_directory_net_t16_belot_rules.txt` | https://licitum.board-directory.net/t16-belot-rules |
| `maviedesenior_com_loisirs_comment_jouer_a_la_belote_coinchee.txt` | https://maviedesenior.com/loisirs/comment-jouer-a-la-belote-coinchee |
| `officialgamerules_org_game_rules_belote.txt` | https://officialgamerules.org/game-rules/belote/ |
| `pagat_com_jass_belote_html.txt` | https://www.pagat.com/jass/belote.html |
| `pagat_com_jass_coinche_html.txt` | https://www.pagat.com/jass/coinche.html |
| `playjoy_com_en_coinche_rules.txt` | https://playjoy.com/en/coinche/rules/ |
| `regles_com_jeux_cartes_coinche_html.txt` | https://www.regles.com/jeux-cartes/coinche.html |
| `reglesdejeux_github_io_regles_du_jeu_la_coinche_index_html.txt` | https://reglesdejeux.github.io/regles-du-jeu-la-coinche/index.html |

### `clubs/` — jass suisse, cité en comparaison
| Fichier | URL |
|---|---|
| `amisduchibre_ch_histoire_du_chibre_ou_du_jass.txt` | https://amisduchibre.ch/histoire-du-chibre-ou-du-jass/ |
| `chibre_ch_forum_viewtopic_php_t_638.txt` | https://www.chibre.ch/forum/viewtopic.php?t=638 |
| `jass_geneve_ch_regleschibre_html.txt` | https://www.jass-geneve.ch/regleschibre.html |
| `reset_bar_ch_reglement_tournoi_jass.txt` | https://reset-bar.ch/reglement-tournoi-jass/ |
| `swisslos_ch_fr_jass_informations_les_regles_du_jass_le_chibre_html.txt` | https://www.swisslos.ch/fr/jass/informations/les-regles-du-jass/le-chibre.html |

### `open-source/` — code et docs de dépôts, **pas des règlements**
| Fichier | Dépôt |
|---|---|
| `drasill_bga-coinche_master_coinche.game.php` | github.com/drasill/bga-coinche (implémentation BoardGameArena) |
| `ismo009_Coinche_main_game.js` | github.com/ismo009/Coinche |
| `ElysiumDisc_belote_master_src_belote_config.py` | github.com/ElysiumDisc/belote |
| `sebastien-perpignane_cardgame_master_src_main_java_sebastien_perpignane_cardgame_game_contree_ContreeGameConfig.java` | github.com/sebastien-perpignane/cardgame |
| `vareversat_carg_main_lib_models_game_setting_coinche_belote_game_setting.dart` | github.com/vareversat/carg |
| `valmathieu_ContrAI_main_contree-domain.md` | github.com/valmathieu/ContrAI |
| `ilyesbrh_twistedFate-belote_main_docs_GAME_RULES.md`, `..._docs_games_coinche_GAME_RULES.md` | github.com/ilyesbrh/twistedFate-belote — ⚠ **source secondaire produite par un agent** (cf. README) : citée en recoupement seulement, jamais comme règlement |
