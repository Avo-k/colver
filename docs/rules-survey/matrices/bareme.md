# Matrice « qui dit quoi » — le barème de score d'une donne

Portée : **contrée / coinche** (jeu à enchères). La belote classique n'est convoquée que là où
elle éclaire un axe (la famille « 162 / 252 / au point près » vient de là, et c'est celle dont
Colver s'inspire) — elle est alors explicitement étiquetée comme telle.

Chaque source est citée par **nom de fichier** et par l'**URL** de sa première ligne
(`SOURCE:`). Quatre fichiers du corpus n'ont pas de ligne `SOURCE` (PDF déposés à la main) :
c'est signalé au cas par cas.

---

## 0. Précautions de lecture — les faux témoins

Avant toute chose, ce qui n'est **pas** un témoignage indépendant :

| Groupe | Fichiers | Statut |
|---|---|---|
| **Texte FFB (page HTML)** | `federations/ffbelote_org_belote_contree.txt`, `federations/ffbelote_org_regles_coinche.txt`, `divers/carafons_fr_regles_de_la_coinche.txt`, `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt`, `tournois/cdf_missegre11_com_medias_files_belote_contre_e_pdf.txt` | **Copies verbatim** (signature partagée : « *Une équipe ayant fait 89 points et ayant demandé 90 chute* »). **Un seul témoignage.** Deux copies ont toutefois été *modifiées* — voir §2.1 (carafons) et §3.1 (Missègre) : les divergences introduites par les copistes sont, elles, informatives. |
| **Texte Pagat** | `divers/pagat_com_jass_coinche_html.txt`, `apps-sites/gambiter_com_cards_jass_coinche_html.txt` (copie mot pour mot, **vérifiée par `diff`**), `divers/reglesdejeux_github_io_regles_du_jeu_la_coinche_index_html.txt` (traduction automatique du même texte) | **Un seul témoignage** (John McLeod). |
| **Texte « tournoi international »** | `divers/belotecontree_free_reglement.txt` (pas de ligne `SOURCE`) et `apps-sites/belotecontree_free_fr_article_php3_id_article_22.txt` → <http://belotecontree.free.fr/article.php3?id_article=22> ; repris intégralement par `tournois/ainesruraux_saintsever_com_belote_BELOTE_20TRADITIONNELLE_pdf.txt` → <http://www.ainesruraux-saintsever.com/belote/BELOTE%20TRADITIONNELLE.pdf> | **Un seul témoignage**, mais avec une valeur particulière : c'est le seul règlement du corpus qui se réclame d'un tournoi réel et qui porte le barème historique en forfaits. |
| **Modèles secondaires** | `open-source/ilyesbrh_twistedFate-belote_main_docs_games_coinche_GAME_RULES.md` et `..._SOURCES.md` | **Produits par un agent** (2026-05), utilisés ici uniquement en recoupement, jamais comme règlement. |

Et surtout : **« la source est muette » ≠ « la source est d'accord »**. Le §11 recense ce sur quoi
le corpus se tait.

> **Ajout du 2026-08-02 — Cannes 2016.** Le règlement du Championnat de France de contrée du
> Festival International des Jeux (édition 2016) a été retrouvé après l'établissement de cette
> matrice et y est désormais cité sous l'alias **Cannes 2016**
> (`tournois/web_archive_org_web_20160421181912if_http_festivaldesjeux_cannes_com_Documents_REGLEMENT_20DE_20LA_20BELOTE_20.txt`).
> Ce n'est la copie de rien : sa rédaction est originale d'un bout à l'autre, et c'est **le seul
> règlement de compétition du corpus dont le barème soit celui de la famille B**. Il change les
> conclusions du **§12.2** (« ce qui n'est attesté nulle part »), qui a été corrigé en
> conséquence. Réserves : document vieux de dix ans, porteur juridique changé en 2025 — détail
> dans [reglement-cannes.md](../reglement-cannes.md).

---

## 1. Méthode de comptage : points faits / points annoncés / faits + demandés

### Position A — « il y a plusieurs méthodes, l'organisateur choisit »

| Position | Sources | Extrait |
|---|---|---|
| **Deux méthodes au choix : points faits, ou points faits + points demandés** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (pas de ligne `SOURCE`) ; `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` (pas de ligne `SOURCE`) | « Il existe deux méthodes possibles pour compter les points, **au choix de l'organisateur d'un tournoi** : − points faits − points faits + points demandés » |
| **Trois méthodes : faits / annoncés / faits+annoncés** | `federations/ffbelote_org_belote_contree.txt` → <https://www.ffbelote.org/belote-contree/> (+ ses 4 copies) | « Il existe plusieurs manières de compter les points à la contrée. **En points faits, en points annoncés, voire en points faits+annoncés.** » |
| **Trois méthodes, à fixer avant la partie** | `divers/bk_jeux_ducale_fr_app_uploads_2022_06_cartes_a_jouer_Ducale_regle_jeu_belote_coinchee_pour_joueur_expert_pdf.txt` → <https://bk.jeux-ducale.fr/app/uploads/2022/06/cartes-a-jouer-Ducale-regle-jeu-belote-coinchee-pour-joueur-expert.pdf> | « Avant la partie, on choisit si on compte **les points annoncés, les points faits ou les points faits + les points annoncés**. » |
| **Deux méthodes principales, à convenir avant de commencer** | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` → <https://fr.wikipedia.org/wiki/Belote_contr%C3%A9e> | « Ces règles changent selon les joueurs qui doivent, avant chaque début de partie, convenir d'une manière de compter les points. **Le décompte aux points faits, et celui aux points annoncés sont les plus pratiqués.** » |
| **Deux modes implémentés en dur** (option de table) | `open-source/drasill_bga-coinche_master_coinche.game.php` (BoardGameArena) | `$doAddPointsToScore = self::getGameStateValue('scoreType') == 1;` — un booléen de partie qui bascule entre « enchère seule » et « enchère + points faits ». |

### Position B — une seule méthode imposée

| Position | Sources | Extrait |
|---|---|---|
| **Faits + demandés, imposé** | `federations/ffbelote_org_regles_coinche.txt` → <https://www.ffbelote.org/regles-coinche/> ; `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` → <https://www.ffbelote.org/wp-content/uploads/2015/11/REGLES-DE-LA-BELOTE-COINCHEE.pdf> ; `federations/LOCAL_regles_officielles_belote_contree.txt` (pas de ligne `SOURCE`) | FFB coinche HTML : « À la coinche, la manière de compter les points est **toujours** Points faits + points annoncés. » — La version « Équipe Ludique » ne mentionne aucune alternative : « Les preneurs marquent **leur total + le montant du contrat demandé**. » |
| **Faits + demandés, imposé** (suite) | `divers/pagat_com_jass_coinche_html.txt` → <https://www.pagat.com/jass/coinche.html> (+ gambiter, reglesdejeux) ; `divers/fr_wikipedia_org_wiki_Coinche.txt` → <https://fr.wikipedia.org/wiki/Coinche> ; `apps-sites/gamerules_com_rules_coinche.txt` → <https://gamerules.com/rules/coinche/> ; `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` → <https://www.jeu-belote.fr/regles.php?part=regles-jeu-coinche> ; `divers/exoty_com_compter_points_coinche.txt` → <https://exoty.com/compter-points-coinche> ; `divers/maviedesenior_com_loisirs_comment_jouer_a_la_belote_coinchee.txt` → <https://maviedesenior.com/loisirs/comment-jouer-a-la-belote-coinchee> ; `divers/belotepoint_fr_regles_coinche.txt` → <https://www.belotepoint.fr/regles-coinche> ; `divers/regles_com_jeux_cartes_coinche_html.txt` → <https://www.regles.com/jeux-cartes/coinche.html> ; `open-source/valmathieu_ContrAI_main_contree-domain.md` ; **Cannes 2016** | Pagat : « the bidding team scores **the number of points they took plus the number they bid** » — Wikipédia Coinche : « marque le nombre de points effectivement faits […] **auxquels s'ajoute la valeur du contrat annoncé** » — Cannes : « **Contrat réussi : points de l'enchère demandée + points des plis réalisés par le demandeur** + belote éventuellement ». |
| **Points annoncés seuls, imposé** | `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` → <https://www.fnasce.org/IMG/pdf/reglement.pdf> ; `tournois/fnasce_org_IMG_pdf_belote_reglement_cle1a43c7_pdf.txt` → <https://www.fnasce.org/IMG/pdf/belote_reglement_cle1a43c7.pdf> (**copies l'une de l'autre**, ASCEE 2A) ; `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` → <https://casimirdehauteclocque.fr/jeux/coinche.pdf> ; `divers/clubdejeux_com_belote_coinchee_online_regles.txt` → <https://www.clubdejeux.com/belote-coinchee-online/regles> ; `divers/adpoker_fr_belote_contree_html.txt` → <https://www.adpoker.fr/belote-contree.html> ; `open-source/slim0_contree_main_backend_game_scoring.py` | ASCEE 2A : « L'équipe qui annonce doit réaliser au minimum son contrat. Elle marque **le nombre de points annoncés, même si elle en a réalisé plus.** » — slim0 : `preneurs_score = contract_value` (rien d'autre). |
| **Points faits (chaque camp garde ses points), imposé** | `open-source/CephaloSophie_kydos_main_packages_core_src_scoring_donneScoring.ts` | « contrat réussi → **chaque équipe marque ses points arrondis** (chacune garde sa belote) ». |

**Divergence.** La ligne de fracture n'est pas « qui a raison » mais **qui tranche** : la FFB et
les sites généralistes présentent la méthode comme un réglage de tournoi, tandis que les
**règlements de concours réels** (ASCEE 2A, Casimir de Hauteclocque) en imposent un seul — et
c'est presque toujours **points annoncés seuls**, le plus rapide à marquer à la main. Presque :
**Cannes impose le mode faits + demandés**, ce qui en fait le seul règlement de compétition du
corpus à marquer la donne entière. Cohérent avec sa cible de partie, **2001 points** — la plus
haute du corpus, et la seule qui suppose des donnes à ~250 points.
Corollaire : **les trois méthodes produisent des scores d'échelles incomparables** (~100 pts par
donne en annoncés seuls, ~250 en faits+demandés), ce qui explique les cibles de partie qui vont
de 501 à 3000 selon les sources.

---

## 2. Condition de réussite du contrat

### 2.1 Faut-il, en plus de l'enchère, faire plus que la défense ?

| Position | Sources | Extrait |
|---|---|---|
| **NON — l'enchère suffit, explicitement** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` ; `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` ; `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/carafons_fr_regles_de_la_coinche.txt` → <https://carafons.fr/regles-de-la-coinche/> | FFB contrée 2016 : « Le contrat est réussi si les preneurs obtiennent un total supérieur ou égal à l'enchère demandée […] **Ceci est valable même si les défenseurs ont réalisé plus de points que les preneurs.** » — Wikipédia contrée : « **Contrairement à la belote, il n'est pas nécessaire de faire plus de points que son adversaire** […] Seule compte la réussite du contrat. » — Pagat : « (They do not need to take more points than their opponents.) » |
| **OUI — deux conditions** | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/ffbelote_org_regles_coinche.txt` ; `divers/fr_wikipedia_org_wiki_Coinche.txt` ; `divers/exoty_com_compter_points_coinche.txt` ; `divers/regles_com_jeux_cartes_coinche_html.txt` ; `apps-sites/ludicash_com_help_rules_coinche.txt` → <https://www.ludicash.com/help/rules-coinche> ; `apps-sites/contree_org_4_joueurs.txt` → <http://contree.org/4-joueurs/> | FFB coinche PDF : « Le contrat est réussi si **les deux conditions** suivantes sont réunies : 1) […] supérieur ou égal à l'enchère demandée […] 2) Les preneurs obtiennent un total **supérieur à celui de la défense**. » — FFB coinche HTML : « Il y a deux conditions pour chuter […] Ou bien l'équipe adverse réalise plus de points que notre équipe. » |
| **NON, mais avec un plancher à 82** | `divers/pagat_com_jass_coinche_html.txt` ; `apps-sites/gameduell_helpshift_com_hc_en_16_belote_com_belote_coinche_faq_1054_coinche.txt` → <https://gameduell.helpshift.com/hc/en/16-belote-com---belote-coinche/faq/1054-coinche/> ; `apps-sites/regles_de_jeux_com_regle_coinche.txt` → <https://www.regles-de-jeux.com/regle-coinche/> ; `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` | Pagat : « The bid must be for at least **82 points** (by convention, 82 is bid by saying "80") […] It is a vestige of this rule that requires a score of at least 82 to win a bid of 80, since at least 82 points are needed to have more than the opponents. » |
| **Silencieuses / implicitement « enchère seule »** (implémentations) | `open-source/drasill_bga-coinche_master_coinche.game.php` ; `open-source/CephaloSophie_kydos_..._donneScoring.ts` ; `open-source/slim0_contree_main_backend_game_scoring.py` ; `open-source/ismo009_Coinche_main_game.js` | BGA : `$bidSuccessful = $teamPoints[$bidTeam] >= $bid;` — kydos : `const met = takerForContract >= input.contract;` — aucune des quatre implémentations ne teste le total adverse. |

**Divergence — et fracture *interne* à la FFB.** La FFB dit **non** pour la contrée et **oui**
pour la coinche, dans deux documents de la même fédération. Ce n'est pas incohérent : la coinche
a des annonces (tierce, carré) qui peuvent porter la défense au-dessus du preneur sans que
celui-ci ait mal joué. Ce que ça révèle, c'est que **la condition « faire plus que la défense »
est le corollaire des annonces, pas de l'enchère**. Détail savoureux : `carafons.fr`, copie
verbatim de la page FFB coinche, a **réécrit** ce passage en « Il y a **une seule condition**
pour chuter à la coinche : Ne pas réaliser son contrat » — le copiste a corrigé sa source.

Les quatre implémentations open source tranchent **toutes** dans le sens « l'enchère suffit ».
C'est ce qui est effectivement appliqué, quoi qu'en disent les textes.

### 2.2 La belote aide-t-elle à réaliser le contrat ?

| Position | Sources | Extrait |
|---|---|---|
| **OUI — les 20 points entrent dans le total qui valide le contrat** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` ; `federations/LOCAL_regles_officielles_belote_contree.txt` ; `federations/ffbelote_org_belote_contree.txt` (+ 4 copies) ; `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/belotecontree_free_reglement.txt` ; `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` ; `divers/bk_jeux_ducale_..._pdf.txt` ; `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` ; `open-source/CephaloSophie_kydos_..._donneScoring.ts` ; `open-source/slim0_contree_main_backend_game_scoring.py` | FFB 2016 §10.1 : « Chaque camp totalise ses points, à savoir : − le nombre de points contenus dans les plis […] − le dix de der, − **la belote/rebelote**. » — FFB HTML : « **La belote permet de réaliser le contrat.** » — « tournoi international » : « Contrat 100 à cœur : Si l'équipe […] dispose de la "Belote" et fait 76 points, **elle chute puisque le total ne représente que 96 points**. » — ASCEE 2A : « La belote (et re) **permet de remplir le contrat** : attention au moment de contrer ! » — kydos : `const takerForContract = rounded[taker] + beloteOf(taker);` |
| **OUI, formulé à l'envers : la belote abaisse la cible de 20** | `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` | « le **score à atteindre est réduit de 20 points**. […] Il ne peut toutefois pas être inférieur à 82. » — arithmétiquement équivalent, sauf au plancher 82 où le comportement diffère. |
| **NON — le contrat se juge sur les plis seuls** | `apps-sites/gameduell_..._faq_1054_coinche.txt` (Belote.com) ; `apps-sites/regles_de_jeux_com_regle_coinche.txt` | GameDuell : « The bidding team has to fulfil the contract […] scoring a minimum of 82 points **through tricks, without melds and Belote-Rebelote**. » |
| **Muettes mais concluantes en pratique** | `open-source/drasill_bga-coinche_master_coinche.game.php` | La belote y est ajoutée aux `$teamPoints` **avant** le test `$teamPoints[$bidTeam] >= $bid` : elle aide *de fait*, sans que le code le documente. |

**Consensus large**, avec **une exception de plateforme** : Belote.com/GameDuell exclut la belote
(et les annonces) du calcul du contrat. C'est le seul endroit du corpus où l'on trouve cette
règle, et elle est cohérente avec sa cible de partie basse (701).

### 2.3 Les annonces (tierce, cinquante, cent, carré) comptent-elles pour réaliser le contrat ?

| Position | Sources | Extrait |
|---|---|---|
| **OUI** | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/ffbelote_org_regles_coinche.txt` ; `apps-sites/regles_de_jeux_com_regle_coinche.txt` | FFB coinche HTML : « À la coinche, **les annonces AIDENT à faire le contrat**. Si un joueur demande 100, qu'il réalise 83 points mais a une tierce, il a donc 103 points. Son contrat est donc réalisé. » |
| **NON** | `apps-sites/gameduell_..._faq_1054_coinche.txt` | « without melds and Belote-Rebelote » |
| **« Selon les variantes »** | `divers/fr_wikipedia_org_wiki_Coinche.txt` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) | Wikipédia : « Les points des annonces s'ajoutent ensuite à chaque équipe. Ils peuvent **dans certaines variantes** être comptabilisés avant et permettre de réaliser le contrat (ou d'être pris). » |
| **Sans objet — pas d'annonces** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` ; `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` ; `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` ; `tournois/web_myassoc_org_img_Lions_E2uYP6mbNv79_2238_8c929f766_medias_c7aa239f63982fbbac73870b2563ec21_pdf.txt` ; `tournois/geraudotloisirs_free_fr_index_php_option_com_content_view_article_id_116_I.txt` | FFB contrée §8 : « **La belote contrée se joue toujours sans annonces.** » |

**Divergence, mais elle recoupe exactement la frontière contrée/coinche** : c'est la *présence*
des annonces qui sépare les deux jeux (cf. Wikipédia contrée : « Étant donné qu'à la contrée les
annonces ne comptent pas (à part la belote) »). Là où elles existent, le consensus est qu'elles
aident au contrat ; seule Belote.com dit le contraire.

---

## 3. La chute : ce que marque la défense, ce que marque le preneur

### 3.1 Ce que marque la défense (contrat simple, non contré)

| Position | Sources | Extrait |
|---|---|---|
| **160 + le contrat** | `federations/ffbelote_org_belote_contree.txt` (mode faits+annoncés) ; `federations/ffbelote_org_regles_coinche.txt` ; `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/LOCAL_regles_officielles_belote_contree.txt` ; `divers/carafons_fr_regles_de_la_coinche.txt` ; `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` → <https://cartesetcie.fr/regle-du-jeu-la-belote-coinchee/> ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` ; `divers/fr_wikipedia_org_wiki_Coinche.txt` ; `apps-sites/gamerules_com_rules_coinche.txt` ; `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` ; `divers/iscool_helpshift_com_hc_fr_10_belote_mobile_faq_157_how_to_play_coinche_coinche_rules.txt` → <https://iscool.helpshift.com/hc/fr/10-belote-mobile/faq/157-how-to-play-coinche-coinche-rules/> ; `divers/drasill_github_io_bga_coinche_rules_fr_html.txt` → <https://drasill.github.io/bga-coinche/rules-fr.html> ; `apps-sites/ludicash_com_help_rules_coinche.txt` ; `open-source/valmathieu_ContrAI_main_contree-domain.md` | LOCAL : « Les défenseurs marquent **160 points de chute** + la belote + le montant du contrat demandé ». — Wikipédia contrée : « contrat 90 points […] L'équipe adverse obtient **90+160+20 = 270** points. » — Pagat : « Their opponents score **160 plus the amount of the bid**. » |
| **162 + le contrat** | `divers/regles_com_jeux_cartes_coinche_html.txt` ; `divers/jeux_regles_com_regles_coinche.txt` → <https://jeux-regles.com/regles-coinche/> ; `divers/maviedesenior_com_loisirs_comment_jouer_a_la_belote_coinchee.txt` ; `divers/belotepoint_fr_regles_coinche.txt` ; `divers/exoty_com_compter_points_coinche.txt` ; `divers/lemagloisirs_fr_regle_coinche.txt` → <https://www.lemagloisirs.fr/regle-coinche/> ; `apps-sites/regles_de_jeux_com_regle_coinche.txt` ; `apps-sites/contree_org_4_joueurs.txt` ; `open-source/drasill_bga-coinche_master_coinche.game.php` ; `open-source/ismo009_Coinche_main_game.js` ; **Cannes 2016** | Cannes : « **Contrat perdu : points de l'enchère demandée + 162 points ou 252** + belote éventuellement car "prenable" » — maviedesenior : « l'équipe qui défend comptabilise **162 points**, ainsi que les points du contrat annoncés […] ces derniers marquent **242 points (162 + 80)**. » — exoty : « les défenseurs auraient marqué **162 points + 100 (contrat) = 262 points**. » — belotepoint : « L'équipe adverse marque **la totalité des 162 points + la valeur du contrat**. » — BGA (code) : `$teamScores[$defenseTeam] += 162;` — ismo009 (code) : `const totalPoints = 162;` |
| **160 tout court (sans le contrat)** | `tournois/cdf_missegre11_com_medias_files_belote_contre_e_pdf.txt` → <http://www.cdf-missegre11.com/medias/files/belote-contre-e.pdf> ; `divers/belotecontree_free_reglement.txt` ; `tournois/ainesruraux_saintsever_com_belote_BELOTE_20TRADITIONNELLE_pdf.txt` ; `divers/adpoker_fr_belote_contree_html.txt` ; `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` ; `open-source/CephaloSophie_kydos_..._donneScoring.ts` | Missègre : « elle marque 0 point et l'équipe adversaire en marque alors **160**. » (le copiste a **retiré** le « + la valeur du contrat demandé » de sa source FFB). — « tournoi international » : « Si une équipe est "DEDANS" […] l'équipe adverse marque **160 points**, éventuellement augmentée des 20 points de "BELOTE" ». — kydos : « contrat chuté → **défense = 160 FIXE**, preneur = 0. » |
| **162 tout court** | `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` (**belote classique**, pas de ligne `SOURCE`) ; `federations/ffbelote_org_regles_officielle_belote.txt` → <https://www.ffbelote.org/regles-officielle-belote/> ; `federations/ffbelote_org_reglements_de_la_belote_avec_ou_sans_annonce.txt` → <https://www.ffbelote.org/reglements-de-la-belote-avec-ou-sans-annonce/> ; `tournois/villeconin_fr_wp_content_uploads_2017_02_F_C3_A9d_C3_A9ration_fran_C3_A7aise_de_Belote_Informations_sur_le_jeu.txt` → <https://villeconin.fr/wp-content/uploads/2017/02/F%C3%A9d%C3%A9ration-fran%C3%A7aise-de-Belote-Informations-sur-le-jeu-de-belote.pdf> ; `tournois/web_myassoc_org_..._pdf.txt` (Lions Club) ; `tournois/pontdeclaix_fr_sites_default_files_2024_01_65a68d18ab985_rglementconcoursdebelote_pdf.txt` ; `tournois/lagrandcombe_fr_wp_content_uploads_2020_01_Reglement_belote_2020_pdf.txt` ; `tournois/fnasce_org_IMG_pdf_reglement_belote_pdf.txt` | **Belote classique** (il n'y a pas de contrat chiffré à ajouter). FFB : « En cas de chute, les preneurs ne marquent rien […] Leurs adversaires marquent **162 points de chute** + leurs annonces + les annonces des preneurs. » — Lions Club : « **La mise dedans compte pour 162 points.** » |
| **Le contrat seul** | `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` (+ sa copie) ; `divers/adpoker_fr_belote_contree_html.txt` (variante b) ; `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` (§ points annoncés) ; `open-source/slim0_contree_main_backend_game_scoring.py` | ASCEE 2A : « L'équipe qui défend […] Si elle réussit, elle marque **le nombre de points annoncés par l'équipe adverse**. » — slim0 : `defenders_score = contract_value * multiplier`. |

### 3.2 Ce que marque le preneur qui chute

| Position | Sources | Extrait |
|---|---|---|
| **0, sauf sa belote qui est imprenable** | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/ffbelote_org_regles_coinche.txt` ; `divers/carafons_fr_regles_de_la_coinche.txt` ; `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/jeux_regles_com_regles_coinche.txt` ; `apps-sites/gamerules_com_rules_coinche.txt` ; `divers/clubdejeux_com_belote_coinchee_online_regles.txt` ; `open-source/drasill_bga-coinche_master_coinche.game.php` | FFB coinche PDF : « Les preneurs ne marquent rien, à l'exception éventuelle des **20 points de belote qui sont réputés imprenables**. » — BGA (code) : `if ($beloteTeamId === $bidTeam) { … ' + 20 (belote)'; }` dans la branche `Failure`. |
| **0, la belote passe à l'adversaire** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` ; `federations/LOCAL_regles_officielles_belote_contree.txt` ; `federations/ffbelote_org_belote_contree.txt` (+ 4 copies) ; `divers/belotecontree_free_reglement.txt` ; `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` ; `open-source/CephaloSophie_kydos_..._donneScoring.ts` ; **Cannes 2016** | FFB contrée 2016 §7 : « **En cas de chute du contrat, la belote est perdue par les preneurs**, même s'ils n'ont pas dit "rebelote". » — LOCAL : « Les preneurs ne marquent rien. **La belote est prenable.** » — Cannes : « Contrat perdu : […] + belote éventuellement **car "prenable"** » (le mot entre guillemets dans le PDF, comme un terme technique). — kydos : `[defense]: c.dedansBase + (hasBelote ? c.beloteBonus : 0)`. |
| **0 tout court** | `open-source/slim0_contree_main_backend_game_scoring.py` ; `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` | slim0 : `preneurs_score = 0` — Casimir : « La belote **ne sert pas en défense**. » |

**Divergence dure, et de nouveau *interne* à la FFB** : la belote est **prenable** dans les
rédactions **contrée** de la FFB, **imprenable** dans ses rédactions **coinche** et dans ses
règles de **belote classique**. Voir §7.

Sur le montant : la fracture principale est **160 vs 162** (le même tas de cartes, arrondi ou
pas — cf. `matrices/arrondi.md`), et surtout **avec ou sans le contrat**. Le corpus penche
massivement vers « + le contrat » ; le « 160 tout court » est le fait de règlements en **points
faits** (Missègre, tournoi international), où le contrat n'entre jamais dans la marque.

**Cannes 2016 est le seul règlement de compétition à écrire « 162 + le contrat »**, et il le fait
en une ligne symétrique de celle du contrat réussi — même total de donne dans les deux cas,
réparti ou non. Il faut noter qu'il **arrondit ensuite** (à la dizaine, bascule à 6, cf.
`matrices/arrondi.md`) : chez lui, 162 est la base de calcul, pas le nombre inscrit sur la feuille.
C'est exactement l'écart qui subsiste avec Colver — voir §12.2.

---

## 4. Contre / coinche : forfait fixe ou multiplicateur ?

### 4.1 La forme du barème

| Position | Sources | Extrait |
|---|---|---|
| **Forfait fixe 320 / 640** (le score de la donne est *remplacé*) | `divers/belotecontree_free_reglement.txt` + `tournois/ainesruraux_saintsever_com_..._pdf.txt` ; `tournois/cdf_missegre11_com_..._pdf.txt` ; `federations/ffbelote_org_belote_contree.txt` (mode **points faits**) + ses copies ; `divers/adpoker_fr_belote_contree_html.txt` ; `open-source/CephaloSophie_kydos_..._donneScoring.ts` | « tournoi international » : « **Le contre vaut 320 points, le surcontre vaut 640 points.** » — FFB HTML points faits : « Le contre vaut 320 points et le surcontre 640. À cela peut venir s'ajouter la belote en supplément, soit 340 ou 660 points. » — kydos : `const base = input.contre === 'surcontree' ? c.surcontreWin : c.contreWin; // 640 / 320` |
| **Forfait 320/640 *plus* le contrat multiplié** (mode faits + demandés) | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (tableau p. 7, image, relevé à la main) ; `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` | Tableau p. 7 : contré réussi = **320 (640) + Contrat×2 (×4)** + belote ; défense 0. |
| **160 + contrat×mult** (la base reste le tas, seul le contrat est multiplié) — **162 chez Cannes** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` **Note 1** ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/belotepoint_fr_regles_coinche.txt` ; `divers/exoty_com_compter_points_coinche.txt` ; `apps-sites/gamerules_com_rules_coinche.txt` ; `open-source/valmathieu_ContrAI_main_contree-domain.md` ; `open-source/slim0_contree_main_backend_game_scoring.py` (sans base) ; **Cannes 2016** | FFB Note 1 : « les organisateurs peuvent utiliser la comptabilisation suivante en cas de contre (surcontre) : **Contrat x 2 + 160** au lieu de Contrat x 2 + 320 (**Contrat x 4 + 160** au lieu de Contrat x 4 + 640) » — Pagat : « A coinche doubles the score **for the bid only** […] 160 + (100 × 2) = 360. » — belotepoint : « la valeur du contrat (**et non les points de pli**) est multipliée ». — Cannes : « Total des points à marquer = **points de l'enchère demandée + 162 ou 252** + belote » puis « Le contre **double les points de l'enchère demandée**, le surcontre les triple » — mêmes deux termes, avec la base à 162/252 au lieu de 160. |
| **(160 + contrat) × mult — tout multiplié** | `federations/ffbelote_org_regles_coinche.txt` ; `federations/ffbelote_org_belote_contree.txt` (mode faits+annoncés) ; `divers/carafons_fr...` ; `divers/cartesetcie_fr...` ; `divers/clubdejeux_com...` ; `federations/LOCAL_regles_officielles_belote_contree.txt` ; `open-source/drasill_bga-coinche_master_coinche.game.php` ; `open-source/ismo009_Coinche_main_game.js` ; `divers/jeux_regles_com_regles_coinche.txt` ; `divers/fr_wikipedia_org_wiki_Coinche.txt` | FFB HTML : « Si le preneur chute son contrat, le défendeur marque **(160 points + les points demandés) x 2** » — BGA (code) : `$teamScores[$defenseTeam] = $bid; … += 162; … *= $multiplier;` — Wikipédia coinche : « les points marqués par **l'équipe victorieuse sont doublés**. » |
| **Multiplicateur sur le *forfait de chute* seulement, pas sur le contrat** | `tournois/data_over_blog_kiwi_com_1_05_17_17_20150128_ob_1f68a4_2015_01_27_reglement_table_coinche_pdf.txt` → <http://data.over-blog-kiwi.com/1/05/17/17/20150128/ob_1f68a4_2015-01-27-reglement-table-coinche.pdf> ; `tournois/maisondesessarts_fr_article116_html.txt` → <https://www.maisondesessarts.fr/article116.html> | Table coinche 2015 : « En cas de coinche : **160 x 2 + contrat + annonces** / En cas de surcoinche : **160 x 4 + contrat + annonces** » — Essarts : « Elle vaut **160 x 2 = 320 points + les points du contrat + les annonces** ». |
| **Multiplicateur sur les points annoncés seuls** | `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` (+ copie) ; `open-source/slim0_contree_main_backend_game_scoring.py` ; `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` ; `federations/ffbelote_org_belote_contree.txt` (mode **points annoncés**) | ASCEE 2A : « Le contre et surcontre multiplie par 2, respectivement 4, **les points annoncés**. » — FFB HTML, points annoncés : « le contre vaut 2 fois la valeur de l'enchère. (si l'enchère est à 120, le contre rapporte 240 points aux 2 équipes). » |
| **Aucun multiplicateur sur le contrat** (idiosyncrasie) | `apps-sites/contree_org_4_joueurs.txt` | « l'envoyeur qui a réalisé son contrat ajoute les points de la mise […] **Contrairement aux autres ces points ne sont pas multipliables en cas de contre comme de surcontre.** […] Soit **162 * 2 = 324** ; plus la belote multipliée par 2 ; plus l'annonce la plus forte multipliée par 2 également ; plus les points du contrat. » |

### 4.2 Sur quelle assiette porte le multiplicateur ?

| Position | Sources | Extrait |
|---|---|---|
| **Sur le contrat seul** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (**Note 1 seulement**) ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/belotepoint_fr_regles_coinche.txt` ; `divers/exoty_com_compter_points_coinche.txt` ; `apps-sites/gamerules_com_rules_coinche.txt` ; `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` ; `open-source/valmathieu_ContrAI_main_contree-domain.md` ; `open-source/slim0_contree_main_backend_game_scoring.py` ; **Cannes 2016** | gamerules : « This **does not double the total points**. For example, if the bid of "100, hearts" was made then the bid of 100 would be doubled to 200. » — ContrAI : « the winning side […] scores `160 + C × M` ». |
| **Sur tout le score de la donne** | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` ; `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/ffbelote_org_regles_coinche.txt` ; `divers/fr_wikipedia_org_wiki_Coinche.txt` ; `divers/jeux_regles_com_regles_coinche.txt` ; `divers/clubdejeux_com...txt` ; `open-source/drasill_bga-coinche_master_coinche.game.php` ; `open-source/ismo009_Coinche_main_game.js` | FFB contrée ~2015 : « Le **score de la donne** sera multiplié par 2 (hors belote). » — jeux-regles : « les **points totaux**, hors Belote et annonces, sont multipliés par deux ». — clubdejeux (exemple chiffré) : « l'équipe 2 réalise **140*2 + 160 = 400** points ». |
| **Le multiplicateur remplace tout par un forfait** | voir §4.1, ligne « forfait 320/640 » | — |
| **Reconnaissent explicitement les deux usages** | `divers/pagat_com_jass_coinche_html.txt` (+ copies) | « When coinche is said, **some apply the double not just to the bid but to the entire score**, so that for example when a team loses a 100 bid with coinche […] the opponent score **520 = 2 × (100 + 160)**. » |

### 4.3 Surcontre : ×3 ou ×4 ? — **le point de fracture**

| Position | Sources | Extrait |
|---|---|---|
| **×4** | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` ; `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (tableau p. 7 **et** Note 1 : « Contrat x 4 ») ; `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/ffbelote_org_belote_contree.txt` (+ 4 copies) ; `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` (+ copie) ; `divers/fr_wikipedia_org_wiki_Coinche.txt` ; `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies, comme règle de base) ; `apps-sites/gamerules_com_rules_coinche.txt` ; `apps-sites/playjoy_com_en_coinche_rules.txt` → <https://playjoy.com/en/coinche/rules/> ; `divers/belotepoint_fr_regles_coinche.txt` ; `divers/exoty_com_compter_points_coinche.txt` ; `divers/jeux_regles_com_regles_coinche.txt` ; `apps-sites/regles_de_jeux_com_regle_coinche.txt` ; `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` ; `open-source/drasill_bga-coinche_master_coinche.game.php` ; `open-source/slim0_contree_main_backend_game_scoring.py` ; `open-source/CephaloSophie_kydos_..._donneScoring.ts` (`surcontreWin = 640`) ; `open-source/valmathieu_ContrAI_main_contree-domain.md` | FFB coinchée : « Le score de la donne sera multiplié par **4**. » — BGA (code) : `if ($countered == 2) { $multiplier = 4; }` — slim0 (code) : `{Double.NONE: 1, Double.CONTRE: 2, Double.SURCONTRE: 4}` |
| **×3** | `federations/LOCAL_regles_officielles_belote_contree.txt` (version « Équipe Ludique », **la plus récente rédaction FFB du corpus**) ; **Cannes 2016** (**le seul règlement de compétition**) ; `divers/iscool_helpshift_..._coinche_rules.txt` (Belote Mobile) ; `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` ; `open-source/ismo009_Coinche_main_game.js` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies, **comme variante recensée**) | LOCAL : « Le score de l'annonce sera multiplié par **3**. » et « + le montant du contrat demandé […] multiplié par 2 (ou par **3** en cas de surcontre) ». — iscool : « Les points seront multipliés par **3** pour celle qui emporte la manche. » — jeu-belote : « Le coefficient multiplicateur passe à **trois**. » — Cannes : « Le contre double les points de l'enchère demandée, **le surcontre les triple** ». — ismo009 (code) : `if (this.contract.surcoinched) multiplier = 3;` — Pagat : « Some play that a surcoinche does not double the score again, but **only increases the multiplier from 2× to 3×**. » |
| **×3 sous forme de forfait (160×3 = 480)** | `tournois/maisondesessarts_fr_article116_html.txt` | « La surcoinche vaut **160 x 3 = 480 points** + les points du contrat + les annonces ». |
| **×4 sous forme de forfait (160×4)** | `tournois/data_over_blog_kiwi_com_..._reglement_table_coinche_pdf.txt` | « En cas de surcoinche : **160 x 4** + contrat + annonces ». |

**Divergence, et c'est la plus tranchée du corpus.** Le ×4 est très largement majoritaire (FFB
2015 et 2016, Wikipédia, Pagat en règle de base, BoardGameArena, la plupart des implémentations).
Le ×3 est **minoritaire mais pas marginal**, et surtout il est porté par :

1. **la rédaction FFB la plus récente** (`LOCAL_regles_officielles_belote_contree`, éditée par
   Équipe Ludique) — donc la FFB elle-même a **changé d'avis** entre 2016 et cette version ;
2. **le règlement du Championnat de France de Cannes** (ajout du 2026-08-02) — c'est-à-dire **le
   seul règlement de compétition du corpus qui tranche cette question**, et il tranche pour ×3 ;
3. **deux plateformes grand public** (Belote Mobile / iscool, jeu-belote.fr) ;
4. **une implémentation** (ismo009) ;
5. **Pagat**, qui le recense explicitement comme usage réel.

La lecture raisonnable du corpus : le ×4 est le barème *historique* (cohérent avec le forfait
640 = 320×2), le ×3 est une **correction moderne** destinée à ce qu'un surcontre ne décide pas à
lui seul d'une partie — exactement la motivation que la FFB écrit noir sur blanc dans sa Note 1
de 2016 (« afin que l'issue de la partie ne soit pas définie par une simple donne »), même si la
Note 1 elle-même conserve le ×4. **Cannes est la preuve que cette correction est appliquée en
compétition** : un tournoi de 148 équipes à 2001 points, dont le règlement écrit ×3 sur le contrat
seul, et pas un forfait. C'est le témoignage qui manquait — le ×3 n'est pas seulement une variante
d'éditeur d'application.

### 4.4 Que devient le score de la défense sur un contrat contré réussi ?

| Position | Sources | Extrait |
|---|---|---|
| **0 — le camp perdant ne marque rien** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (tableau p. 7) ; `federations/LOCAL_regles_officielles_belote_contree.txt` ; `divers/belotecontree_free_reglement.txt` ; `divers/adpoker_fr_belote_contree_html.txt` ; `open-source/drasill_bga-coinche_master_coinche.game.php` ; `open-source/CephaloSophie_kydos_..._donneScoring.ts` ; `open-source/valmathieu_ContrAI_main_contree-domain.md` | LOCAL : « les défenseurs **ne marquent rien**. » — BGA (code) : `if ($countered) { $teamScores[$defenseTeam] = 0; }` — ContrAI : « the **losing side scores 0** — the defense never keeps its own card points once it has doubled. » |
| **0 sauf la belote imprenable** | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `divers/clubdejeux_com...txt` | FFB coinche PDF : « les défenseurs ne marquent rien, **à l'exception éventuelle des 20 points de belote** qui sont réputés imprenables. » |

**Consensus.** Contrer, c'est passer en « le gagnant prend tout ». Seule la belote résiste, et
seulement là où elle est déclarée imprenable.

---

## 5. Le capot

### 5.1 Capot non demandé mais réalisé

| Position | Sources | Extrait |
|---|---|---|
| **250 (forfait qui remplace le tas de cartes)** | `federations/ffbelote_org_belote_contree.txt` (+ copies) ; `federations/LOCAL_regles_officielles_belote_contree.txt` ; `divers/belotecontree_free_reglement.txt` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/fr_wikipedia_org_wiki_Coinche.txt` ; `apps-sites/gamerules_com_rules_coinche.txt` ; `divers/clubdejeux_com...txt` ; `divers/bk_jeux_ducale_..._pdf.txt` ; `open-source/drasill_bga-coinche_master_coinche.game.php` ; `open-source/valmathieu_ContrAI_main_contree-domain.md` ; `divers/exoty_com_compter_points_coinche.txt` | Pagat : « If the bidding team wins all the tricks (capot) they score **250 points plus the amount of the bid**. » — BGA (code) : `if ($teamPoints[0] == 0) { $teamPoints[1] = 250; $isCapot = true; }` — ContrAI : « the trick pile (152 + 10 = 162) is **replaced by a flat 250 substitute**. » |
| **252 (= 152 cartes + dix de der à 100)** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` §9.2 ; `federations/LOCAL_regles_officielles_belote_contree.txt` §8.2 ; `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` §9.2 ; `federations/ffbelote_org_regles_officielle_belote.txt` ; `tournois/villeconin_fr_..._pdf.txt` ; `tournois/web_myassoc_org_..._pdf.txt` ; `tournois/geraudotloisirs_free_fr_...txt` → <http://geraudotloisirs.free.fr/index.php?option=com_content&view=article&id=116&Itemid=102> ; `tournois/tcvb_bruche_free_fr_dossomma_tournoibelotte2012_tournoibelote2012reglement_htm.txt` → <http://tcvb.bruche.free.fr/dossomma/tournoibelotte2012/tournoibelote2012reglement.htm> ; `tournois/pontdeclaix_fr_..._pdf.txt` ; `tournois/cdfcasson_fr_files_ugd_0df194_8c501176abb04e5b9237c65fcf80f584_pdf.txt` ; `tournois/sc4e58b2fce8a2e7a_jimcontent_com_..._Belote_20R_C3_A8glement_2.txt` ; `open-source/ismo009_Coinche_main_game.js` ; `apps-sites/contree_org_4_joueurs.txt` ; **Cannes 2016** | FFB 2016 : « le dix de der vaut **100 points**, portant ainsi le total à **252 points**. » — Lions Club : « Le capot compte alors pour **252 points**. » — Cannes : « Total des points à marquer = points de l'enchère demandée **+162 ou 252** + belote éventuellement » (la même alternative revient aux deux lignes, contrat réussi et contrat perdu ; le 252 y est la base cartes, pas un forfait). — ismo009 (code) : `// Capot non annoncé: 252 + points de l'annonce` puis `scoreNS = 252 + contractBonus;` |
| **162** | `tournois/aappmakoenigshoffen_e_monsite_com_medias_files_reglementtournoibelote_2_pdf.txt` → <http://aappmakoenigshoffen.e-monsite.com/medias/files/reglementtournoibelote-2.pdf> ; `tournois/lagrandcombe_fr_..._pdf.txt` | Koenigshoffen : « Le capot est marqué **162 points**. » — La Grand-Combe : « Aucune annonce – Dedans : 162 points **- Capot : 162 points**. » (le capot ne vaut rien de plus qu'une mise dedans) |
| **150 + l'annonce** | `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` | « Capot non annoncé mais réalisé : L'attaque marque **150 pts + annonce** » — valeur unique dans le corpus, vraisemblablement une coquille pour 250, mais publiée telle quelle. |
| **270 avec la belote** | `divers/belotecontree_free_reglement.txt` + `tournois/ainesruraux_saintsever_com_..._pdf.txt` | « Le capot vaut **250 points, 270 avec la "BELOTE"**. » |
| **Ne compte pas** | `divers/bk_jeux_ducale_..._pdf.txt` (mode **points annoncés** uniquement) | « Le capot annoncé rapporte 250 points et **le capot non annoncé n'est pas pris en compte**. » |

**Divergence sur 250 vs 252**, et c'est une divergence de *nature* : **250 est un forfait qui
remplace le tas** (la donne cesse d'avoir un total de 162), **252 est le tas lui-même** recalculé
avec un dix de der à 100. Les deux coexistent **dans le même document FFB** : le règlement
contrée 2016 dit 252 au §9.2 (dix de der) et 250/500 dans son tableau p. 7 (barème). La FFB
n'explique jamais l'articulation.

**Cannes est le texte qui articule le mieux les deux.** Il n'a qu'une formule de marque, valable
pour toutes les donnes, où le seul terme variable est la base cartes : « +**162 ou 252** ». Le
capot n'y est donc ni une prime ni un cas particulier — c'est la même donne, avec un tas plus gros
parce que le dix de der y vaut 100. C'est précisément la construction de Colver
(`TOTAL_PTS = 162` / `CAPOT_PTS = 252`), et le seul règlement du corpus à l'écrire aussi
directement.

### 5.2 Le dix de der vaut-il 100 en cas de capot ?

| Position | Sources | Extrait |
|---|---|---|
| **OUI — 100, total 252** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` ; `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` ; `federations/LOCAL_regles_officielles_belote_contree.txt` ; `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/ffbelote_org_regles_officielle_belote.txt` ; `divers/fr_wikipedia_org_wiki_Coinche.txt` ; `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` ; `apps-sites/ibelote_com_en_rules_belote_php.txt` → <https://ibelote.com/en/rules-belote.php> ; `divers/belotecontree_free_reglement.txt` ; `apps-sites/contree_org_4_joueurs.txt` | Wikipédia coinche : « lorsque l'on fait un capot le dernier pli est primé de **100 points**, le capot rapporte donc **252 points**. » |
| **NON — le dix de der ne compte pas sur un capot** | `divers/ange_heureux_free_fr_JeuxDeCartes_La_Coinche_html.txt` → <http://www.ange.heureux.free.fr/JeuxDeCartes/La_Coinche.html> (et son doublon `apps-sites/ange_heureux_free_fr_Jeux_LaCoinche_html.txt`) | « Le "Dix de der" est marqué par l'équipe qui fait le dernier pli, soit 10 points supplémentaires, **sauf dans le cas de Tout-Atout ou du Capot où il ne compte pas.** » |
| **Muettes** (elles remplacent simplement le score par un forfait) | `divers/pagat_com_jass_coinche_html.txt` (+ copies), `open-source/drasill_bga-coinche_master_coinche.game.php`, `open-source/CephaloSophie_kydos_..._donneScoring.ts`, `open-source/slim0_contree_main_backend_game_scoring.py` | Le forfait 250 rend la question sans objet. |

**Consensus fort sur 100**, une seule dissidence isolée.

### 5.3 Capot demandé et réussi

| Position | Sources | Extrait |
|---|---|---|
| **500** | `federations/ffbelote_org_belote_contree.txt` (modes faits et faits+annoncés) + ses 4 copies ; `divers/belotecontree_free_reglement.txt` + `tournois/ainesruraux_...` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` ; `divers/fr_wikipedia_org_wiki_Coinche.txt` ; `apps-sites/gamerules_com_rules_coinche.txt` ; `divers/clubdejeux_com...txt` ; `divers/adpoker_fr_belote_contree_html.txt` ; `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` (250 + 250 d'annonce) ; `divers/exoty_com_compter_points_coinche.txt` | « Le capot demandé et réalisé, ou chuté, vaut **500 points**. » — Wikipédia contrée : « Une annonce de capot équivaut à 250 points. Un capot réussi rapporte donc **500 points**. » |
| **250 (contrat comme un autre)** | `federations/LOCAL_regles_officielles_belote_contree.txt` ; `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (tableau p. 7 : « 250 points pour un capot demandé » ajouté aux points faits) ; `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/ffbelote_org_belote_contree.txt` (mode **points annoncés**) ; `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` ; `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` ; `divers/bk_jeux_ducale_..._pdf.txt` (points annoncés) ; `apps-sites/ludicash_com_help_rules_coinche.txt` ; `open-source/theosaulus_coinche_main_coinche_utils.py` | LOCAL : « Les preneurs marquent leur total + le montant du contrat demandé (**250 points pour un capot demandé**). » — ASCEE 2A : « **Capot annoncé = 250 points.** » — theosaulus (code) : `return (250, atout_suit) # capot is encoded as 250` |
| **270 (contrat comme un autre, mais à 270 et non 250)** | **Cannes 2016** | « Les paroles d'enchère autorisées sont : le nombre de points suivi immédiatement de la couleur soit : "passe", "contre", "surcontre", "**générale ou capot**" ou "**270**" contrat le plus élevé. » Le capot est nommé comme l'enchère la plus haute et **chiffré 270** ; la marque lui applique ensuite la formule ordinaire (base 252 + contrat). Valeur unique dans le corpus — c'est le seul écart de fond entre le barème de Cannes et celui de Colver. |
| **350** | `divers/bk_jeux_ducale_..._pdf.txt` (modes **points faits** et **faits + annoncés**) | « Le capot non annoncé rapporte 250 points et **le capot annoncé rapporte 350 points**. » — valeur unique dans le corpus (Ducale, le cartier). |
| **500 mais comme *générale*** | `apps-sites/playjoy_com_en_coinche_rules.txt` | « If a single player succeeds in winning all 8 tricks […] adding **490 extra points** to the "last ten" (**500 in total**). » |

**Divergence structurante.** Deux logiques :
- **prime forfaitaire** : le capot demandé « vaut 500 », un nombre décomposable en 250 de capot
  fait + 250 de contrat (comme l'explicite Wikipédia coinche) ;
- **contrat ordinaire** : le capot est une enchère comme 80 ou 160, et le barème normal
  s'applique. C'est la position de **`LOCAL` (la rédaction FFB la plus récente)**, celle de
  **Cannes**, et celle de Colver (§12).

Sur la *forme*, Cannes conforte donc `LOCAL` et Colver — un contrat, pas un forfait. Sur la
*valeur*, il est seul : **270**, contre 250 partout ailleurs. Ducale est seul à 350.

### 5.4 Capot demandé et chuté

| Position | Sources | Extrait |
|---|---|---|
| **500 à la défense** | `federations/ffbelote_org_belote_contree.txt` + copies ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/fr_wikipedia_org_wiki_Coinche.txt` ; `apps-sites/gamerules_com_rules_coinche.txt` ; `divers/adpoker_fr_belote_contree_html.txt` | Pagat : « If a bid of capot is lost, **the opponents score 500**. » |
| **410 (= 160 + 250) à la défense** | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` ; `divers/pagat_com_jass_coinche_html.txt` (**comme variante recensée**) ; `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` | Wikipédia contrée : « un capot chuté rapporte **410 points** à l'équipe adverse (250 points plus 160 points). » — Pagat : « Some give **only 410 points (160+250)** to the opponents if it is bid and fails. » |
| **base + contrat-capot × mult** (barème général appliqué au capot) | `federations/LOCAL_regles_officielles_belote_contree.txt` (250) ; **Cannes 2016** (252 + 270, par composition de ses deux lignes) ; `open-source/drasill_bga-coinche_master_coinche.game.php` ; `open-source/slim0_contree_main_backend_game_scoring.py` | LOCAL : « Les défenseurs marquent 160 points de chute + la belote + le montant du contrat demandé (**250 points pour un capot demandé**) multiplié par 2 (ou par 3 en cas de surcontre). » |

### 5.5 Capot contré / surcontré

| Position | Sources | Extrait |
|---|---|---|
| **1000 / 2000** | `federations/ffbelote_org_belote_contree.txt` (modes **points faits** et **faits+annoncés**) ; `federations/ffbelote_org_regles_coinche.txt` ; `divers/carafons_fr...` ; `divers/cartesetcie_fr...` ; `tournois/cdf_missegre11_com_..._pdf.txt` ; `divers/belotecontree_free_reglement.txt` + `tournois/ainesruraux_...` ; `divers/adpoker_fr_belote_contree_html.txt` | « Le capot contré vaut **1.000** points. Le capot surcontré vaut **2.000** points. » |
| **500 / 1000** | `federations/ffbelote_org_belote_contree.txt` (mode **points annoncés**, **la même page**) | « Le capot demandé et réalisé vaut 250 points. Le capot contré vaut **500** points. Le capot surcontré vaut **1.000** points. » |
| **250 × 2 ou × 3** (barème général) | `federations/LOCAL_regles_officielles_belote_contree.txt` ; **Cannes 2016** (270 × 2 ou × 3) | « + le montant du contrat demandé (250 points pour un capot demandé) **multiplié par 2 (ou 3 en cas de surcontre)** » → 500 / 750 (LOCAL). Cannes n'a pas de clause spéciale : « le contre double les points de l'enchère demandée, le surcontre les triple » s'applique au capot comme à 80 → 540 / 810, plus la base 252. |
| **Un capot ne peut pas être contré** | `divers/pagat_com_jass_coinche_html.txt` (+ copies), règle de base ; `apps-sites/gamerules_com_rules_coinche.txt` | Pagat : « A capot bid **ends the bidding and cannot be doubled**. » — puis, en variante : « Some allow a capot bid to be doubled and redoubled. » |
| **Le capot peut être contré, et même surcontré** | `divers/belotecontree_free_reglement.txt` ; `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` ; **Cannes 2016** (implicite : « contre » et « surcontre » sont des paroles autorisées à tout moment, et la seule fin d'enchère est le surcontre ou trois passes) | « Un capot demandé ne peut être que contré et éventuellement surcontré ». |

**Divergence — et la contradiction interne au site FFB la plus nette.** Voir §10 et §13.

### 5.6 Le cas oublié : le capot réalisé par l'équipe **contrée**

| Position | Sources | Extrait |
|---|---|---|
| **420 (= 320 + 100)** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt`, Note 2 | « **En points faits**, si l'équipe contrée réalise un capot, elle marque 320 points auxquels s'ajoutent les 100 points du dix de der, soit **420 points**. » |
| **250 substitué à 160 dans la formule contrée** | `federations/LOCAL_regles_officielles_belote_contree.txt` ; `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` | « les preneurs marquent 160 points (**ou 250 points si capot réalisé, même non demandé**) + leur belote + le montant du contrat demandé […] multiplié par 2 (ou 3) ». |
| **Muettes** | Toutes les autres sources du corpus. | Aucune autre ne traite ce cas. |

---

## 6. La générale

| Position | Sources | Extrait |
|---|---|---|
| **Existe, vaut 500** | `divers/belotepoint_fr_regles_coinche.txt` ; `divers/maviedesenior_com_...txt` ; `divers/ange_heureux_free_fr_JeuxDeCartes_La_Coinche_html.txt` ; `apps-sites/playjoy_com_en_coinche_rules.txt` ; `open-source/valmathieu_ContrAI_main_contree-domain.md` ; `open-source/ismo009_Coinche_main_game.js` | belotepoint : « un seul joueur s'engage à remporter tous les plis sans l'aide de son coéquipier. Ce dernier ne joue pas de la manche. En cas de réussite, le bonus est de **500 points**. » — ismo009 (code) : `// Générale: 500 points` |
| **Existe, valeur à convenir** | `divers/fr_wikipedia_org_wiki_Coinche.txt` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) | Wikipédia : « celle-ci est primée et **sa valeur est à déterminer en début de partie** ». — Pagat : « If a bid of générale is allowed, **the score for it must be agreed** — for example 1000. » |
| **Le mot désigne le capot, ce n'est pas une enchère distincte** | **Cannes 2016** | « "**générale ou capot**" ou "270" contrat le plus élevé » — les deux mots sont donnés comme synonymes de la même enchère, la plus haute. Cannes ne connaît donc pas de générale au sens « le preneur seul fait les 8 plis », et n'a rien au-dessus du capot. Même usage du mot que `contree.org`, valeur différente. |
| **Existe, = 252 + 250 si réclamée** | `apps-sites/contree_org_4_joueurs.txt` | « "Générale" est le nom de la prime de capot attribuée à l'équipe ayant fait toutes les levées. Elle représente un forfait de **100 + 152 = 252 points**. Aux 100 points de capot s'ajoute une prime de **250 points** dès lors que la générale a été réclamée. » (usage du mot *générale* pour le capot d'équipe — unique dans le corpus) |
| **Existe, contrat solo distinct** | `open-source/slim0_contree_main_backend_game_scoring.py` ; `apps-sites/ibelote_com_en_rules_belote_php.txt` ; `divers/iscool_helpshift_..._coinche_rules.txt` (mais confondue avec le capot) | slim0 (code) : `if r.contract.bid.is_generale: contract_made = all(t.winner == r.contract.bid.position for t in r.tricks)` — **le preneur seul**, pas son équipe. |
| **N'existe pas / interdite** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` ; `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` ; `federations/LOCAL_regles_officielles_belote_contree.txt` ; `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `tournois/maisondesessarts_fr_article116_html.txt` | FFB 2016 : « Il n'existe pas d'enchère supérieure au capot » — Essarts : « Les autres expressions comme 10 au jeu, parole, mon homme verra, capote, **générale**… **ne sont pas autorisées**. » |

**Divergence, mais nette et bien rangée** : **la FFB ne connaît pas la générale**, tous ses
documents s'arrêtent au capot. Elle vit dans la coinche « de table » et dans les plateformes, où
le consensus le plus fréquent est **500** (le double du capot). Quand une source la définit
sérieusement, elle exige que le **preneur seul** fasse les 8 plis (belotepoint, slim0, Wikipédia,
Pagat) ; **trois** sources emploient le mot pour un capot d'équipe (contree.org, iscool, et
**Cannes**, qui écrit « générale ou capot » comme un seul et même nom d'enchère). Le fait qu'un
règlement de championnat range le mot du côté du synonyme conforte la position FFB : en contrée
de compétition, il n'y a **rien au-dessus du capot**.

---

## 7. La belote : prenable ou imprenable ?

| Position | Sources | Extrait |
|---|---|---|
| **Imprenable — toujours à qui l'annonce, même en chute, même en capot** | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/ffbelote_org_regles_coinche.txt` ; `divers/carafons_fr...` ; `divers/cartesetcie_fr...` ; `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` (**belote classique**) ; `federations/ffbelote_org_regles_officielle_belote.txt` ; `federations/ffbelote_org_reglements_de_la_belote_avec_ou_sans_annonce.txt` ; `tournois/villeconin_fr_..._pdf.txt` ; `tournois/web_myassoc_org_..._pdf.txt` ; `tournois/geraudotloisirs_free_fr_...txt` ; `tournois/rjcv_be_belote_regles_pdf.txt` → <https://www.rjcv.be/belote/regles.pdf> ; `tournois/cdfcasson_fr_..._pdf.txt` ; `tournois/sc4e58b2fce8a2e7a_jimcontent_com_...txt` ; `tournois/fnasce_org_IMG_pdf_reglement_belote_pdf.txt` ; `tournois/data_over_blog_kiwi_com_..._pdf.txt` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/jeux_regles_com_regles_coinche.txt` ; `apps-sites/gamerules_com_rules_coinche.txt` ; `divers/clubdejeux_com...txt` ; `apps-sites/contree_org_4_joueurs.txt` ; `open-source/drasill_bga-coinche_master_coinche.game.php` | FFB coinche HTML : « Elle est **imprenable même en cas de chute ou de capot**, le preneur/les défenseurs marqueront ces 20 points. » — RJCV : « Les points de la belote sont **inviolables**. » — Pagat : « The 20 points for Belote […] are scored by the bidding team, **even if the contract fails**. » — BGA : `+ 20 (belote)` dans la branche `Failure` du camp preneur. |
| **Prenable — elle passe à l'adversaire si le preneur chute** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` ; `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` (implicite via le tableau p. 7) ; `federations/LOCAL_regles_officielles_belote_contree.txt` ; `federations/ffbelote_org_belote_contree.txt` (+ 4 copies) ; `divers/belotecontree_free_reglement.txt` + `tournois/ainesruraux_...` ; `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` ; `open-source/CephaloSophie_kydos_..._donneScoring.ts` ; **Cannes 2016** | FFB contrée 2016 §7 : « **En cas de chute du contrat, la belote est perdue par les preneurs**, même s'ils n'ont pas dit "rebelote". » — FFB HTML contrée : « **La belote n'est pas imprenable.** Elle est également marquée par la défense si l'équipe qui a annoncé le contrat chute. » — « tournoi international » art. 7 : « Elle est toujours marquée par l'équipe l'ayant annoncée **sauf si celle-ci chute** dans le contrat qu'elle joue auquel cas elle est marquée par l'équipe adverse. » — kydos : « si le camp qui l'a annoncée PERD (chute simple ou contre) → **les 20 passent à l'adversaire** ». |
| **Ni l'un ni l'autre — la belote n'entre pas dans la marque** | `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` ; `open-source/slim0_contree_main_backend_game_scoring.py` | Casimir : elle réduit la cible de 20, et « La belote **ne sert pas en défense**. » — slim0 (code) : « La belote **ne profite jamais** à la défense » ; commentaire : « belote preneurs : **non comptée dans score final** ». |
| **« Ça change souvent selon la région »** | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) | Wikipédia : « Les 20 points de belote vont aussi à l'adversaire (**cette partie de la règle change très souvent selon la région et les joueurs**). » — Pagat : « **Many also award the points for belote to the opponents** in this case […] it can be in the bidding team's interest to **suppress the belote announcement** when their contract is likely to fail. » |

**Divergence, et elle coupe exactement le corpus le long de la frontière
contrée / coinche + belote classique :**

- **Contrée** → belote **prenable** (les quatre rédactions FFB de contrée, le règlement du
  tournoi international, Wikipédia contrée, kydos, **Cannes** — qui va jusqu'à mettre le mot
  entre guillemets, « + belote éventuellement car "prenable" », signe que c'est un terme
  technique acquis à la table).
- **Coinche et belote classique** → belote **imprenable** (les deux rédactions FFB de coinche,
  les trois de belote classique, tous les règlements de concours de belote, Pagat, BGA).

Autrement dit : **la FFB se contredit sur la belote entre ses documents contrée et ses documents
coinche**, exactement comme sur la condition « faire plus que la défense » (§2.1). Les deux
contradictions vont dans le même sens et pointent la même chose : les deux corps de règles n'ont
pas été rédigés ensemble.

### 7.1 Que devient la belote en cas de contre ?

| Position | Sources | Extrait |
|---|---|---|
| **Elle s'ajoute au forfait, pour le gagnant** | `divers/belotecontree_free_reglement.txt` ; `federations/ffbelote_org_belote_contree.txt` ; `open-source/CephaloSophie_kydos_..._donneScoring.ts` | « tournoi international » : « en cas de capot contré ou surcontré, **les 20 points sont rajoutés aux points du contre ou du surcontre** » — kydos : `[winner]: base + (hasBelote ? c.beloteBonus : 0)`, quel que soit le camp qui l'a annoncée. |
| **Elle reste au camp qui l'a annoncée, hors multiplicateur** | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` ; `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (tableau p. 7) ; `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; **Cannes 2016** | FFB ~2015 : « Le score de la donne sera multiplié par 2 (**hors belote**). » — Cannes : « si la belote a été annoncée on ajoute 20 points de bonification **qui ne sont ni doublés ni triplés** » (la seule source du corpus à l'écrire pour les *deux* multiplicateurs). |
| **Elle est multipliée comme le reste** | `apps-sites/contree_org_4_joueurs.txt` ; `federations/LOCAL_regles_officielles_belote_contree.txt` (formulation ambiguë) | contree.org : « **plus la belote multipliée par 2** ». |

### 7.2 Que devient la belote en cas de capot ?

| Position | Sources | Extrait |
|---|---|---|
| **L'équipe capot conserve la sienne** | `federations/ffbelote_org_belote_contree.txt` (+ copies) ; `divers/belotecontree_free_reglement.txt` ; `tournois/cdf_missegre11_com_..._pdf.txt` | « **L'équipe Capot conserve sa belote quoiqu'il arrive.** L'équipe qui annonce un capot et ne le réalise pas perd sa belote. » |
| **L'équipe qui *subit* le capot garde la sienne (imprenable)** | `tournois/web_myassoc_org_..._pdf.txt` ; `tournois/fnasce_org_IMG_pdf_reglement_belote_pdf.txt` ; `tournois/pontdeclaix_fr_..._pdf.txt` ; `tournois/cdfcasson_fr_..._pdf.txt` | Lions Club : « La belote reste inviolable. **L'équipe qui subit un capot marque 20 points** si elle a annoncé celle-ci. » |
| **On ne finit jamais une partie sur un capot accompagné d'une belote** | `federations/LOCAL_regles_officielles_belote_contree.txt` ; `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` ; `federations/ffbelote_org_regles_coinche.txt` ; `federations/ffbelote_org_reglements_de_la_belote_avec_ou_sans_annonce.txt` | « Si une équipe est capot ou chute, et atteint les points fixés **uniquement grâce aux points d'une belote**, la partie n'est pas encore gagnée […] L'équipe aura besoin d'un pli supplémentaire. » |

---

## 8. Les annonces (tierce, cinquante, cent, carré)

Rappel de périmètre : **la contrée n'en a pas** (FFB contrée §8 : « La belote contrée se joue
toujours sans annonces »), la **coinche** en a. Beaucoup de concours de belote classique les
excluent explicitement (« sans annonce sauf la belote »).

### 8.1 Les valeurs

| Position | Sources | Extrait |
|---|---|---|
| **Carré V 200 / 9 150 / A-10-R-D 100 ; cent 100 ; cinquante 50 ; tierce 20** | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/ffbelote_org_regles_coinche.txt` ; `divers/carafons_fr...` ; `divers/cartesetcie_fr...` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `apps-sites/regles_de_jeux_com_regle_coinche.txt` ; `divers/jeux_regles_com_regles_coinche.txt` | FFB : « 4 Valets 200 / 4 Neufs 150 / 4 As 100 / 4 Dix 100 / 4 Rois 100 / 4 Dames 100 […] Le Cent […] 100 points. Le Cinquante […] 50 points. La Tierce […] 20 points. » — « les carrés de 7 et 8 ne comptent pas. **Un carré de 100 points est plus fort qu'un cent.** » |
| **Même barème, mais carrés re-tarifés en Sans Atout** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (variante SA/TA) ; `federations/ffbelote_org_belote_contree.txt` | « 4 As **200** points / 4 Dix **150** points / 4 Rois 100 / 4 Dames 100 / 4 Valets 100 / 4 Neufs 100 » — l'ordre est inversé parce qu'à SA le valet et le 9 ne valent plus rien. |
| **Muettes / sans objet** | Tous les documents de contrée (FFB ×4, tournoi international, Wikipédia contrée), et les concours « sans annonce sauf la belote » (`tournois/web_myassoc_org_...`, `tournois/geraudotloisirs_...`, `tournois/lagrandcombe_...`, `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt`) | ASCEE 2A : « On joue **sans annonce** (tierce, carré, etc.) hormis la belote (et re) ». |

**Consensus.** Aucune source du corpus ne propose un autre barème d'annonces (à la nuance SA/TA
près). **C'est l'axe le plus consensuel de toute la matrice.**

### 8.2 À qui vont-elles ?

| Position | Sources | Extrait |
|---|---|---|
| **Seul le camp ayant la meilleure annonce marque — et il marque *toutes* les siennes** | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/jeux_regles_com_regles_coinche.txt` ; `apps-sites/playjoy_com_en_coinche_rules.txt` ; `divers/fr_wikipedia_org_wiki_Coinche.txt` | FFB : « les annonces d'un seul camp peuvent être prises en compte : celui qui montre l'annonce la plus haute […] son camp voit **toutes ses annonces validées**, y compris celles qui étaient inférieures à la meilleure annonce adverse. » |
| **Égalité parfaite → personne ne marque** | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/ffbelote_org_regles_coinche.txt` ; `divers/pagat_com_jass_coinche_html.txt` (en signalant l'autre usage) | FFB : « il y a "égalité" et **aucune annonce ne compte**, et ce, même si un camp bénéficie d'autres annonces inférieures. » — Pagat : « Among equal non-trump sequences, **some play that the first to be announced is best**; others play that they annul each other. » |
| **Sur un capot, les annonces de la défense changent de camp** | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `tournois/villeconin_fr_..._pdf.txt` (belote classique) ; `federations/ffbelote_org_regles_officielle_belote.txt` | FFB coinche : « si un capot est réalisé par les preneurs, **les annonces des défenseurs changent de main** et sont marquées par les preneurs, et ce, que le capot ait été demandé lors des enchères ou non. » |
| **Sur une chute, les annonces des preneurs vont à la défense** | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `divers/pagat_com_jass_coinche_html.txt` (+ copies) ; `divers/jeux_regles_com_regles_coinche.txt` ; `apps-sites/regles_de_jeux_com_regle_coinche.txt` ; `federations/ffbelote_org_regles_officielle_belote.txt` (belote classique) | FFB : « Les défenseurs marquent 160 points de chute + leurs annonces + leur belote + **les annonces des preneurs qui changent de main** ». — Pagat : « When a contract fails, the opponents score **not only the bid and the card points, but also the points for announcements if any**. » |
| **Renonce : le camp adverse marque ce que le fautif avait annoncé** | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` ; `federations/ffbelote_org_regles_coinche.txt` | « Si un joueur se révèle incapable de montrer les combinaisons qu'il a annoncées, il y a renonce : **le camp adverse marquera les points que le camp fautif avait annoncés**. » |

**Consensus** sur « un seul camp marque » et sur « elles suivent le gagnant de la donne » ;
**divergence mineure** sur l'égalité parfaite (annulation FFB vs priorité au premier déclarant,
recensée par Pagat).

---

## 9. Les grandes familles de barème

Cinq familles, par ordre chronologique d'apparition dans le corpus.

### Famille A — **Forfaits historiques** : 320 / 640 / 500 / 1000 / 2000

**D'où ça vient.** Le « règlement officiel récupéré lors du tournoi international »
(`divers/belotecontree_free_reglement.txt`, sans date, republié tel quel par
`tournois/ainesruraux_saintsever_com_...`). C'est le plus ancien texte identifiable du corpus, et
la FFB en a hérité le tableau de la p. 7 de son règlement 2016.

**Ce qui la caractérise.** Le contre, le surcontre et le capot ne *modifient* pas le score de la
donne, ils le **remplacent** par un forfait : 320 / 640 pour le contre, 500 / 1000 / 2000 pour le
capot. Chute simple = 160 tout court. La belote est **prenable**. Capot non demandé : 250, ou 270
avec la belote.

**Qui l'applique.** `belotecontree.free.fr` + `ainesruraux-saintsever`,
`tournois/cdf_missegre11_com_...` (Missègre 11, 2015), `divers/adpoker_fr_...`, la section
« points faits » de la page FFB contrée, et — la seule implémentation moderne à le faire —
`open-source/CephaloSophie_kydos_..._donneScoring.ts` (`contreWin: 320`, `surcontreWin: 640`,
`dedansBase: 160`).

### Famille B — **FFB « faits + demandés »** : base + contrat, multiplicateurs

**D'où ça vient.** La rédaction moderne de la FFB (2015-2016), et sa propre **Note 1** de 2016
qui propose de remplacer le forfait 320 par « Contrat×2 + 160 » : c'est la famille A qui
s'auto-corrige.

**Ce qui la caractérise.** Le tas de cartes reste le socle (160 arrondi, ou 162 exact), le contrat
s'y ajoute, un multiplicateur s'applique. La donne redevient *additive* au lieu d'être
forfaitaire. Le capot demandé vaut 250 (contrat comme un autre) ou 500 (prime) selon la
sous-famille.

**Qui l'applique.** Les quatre rédactions FFB de contrée et les deux de coinche ; Pagat et ses
deux copies ; Wikipédia (contrée et coinche) ; `divers/belotepoint_fr_...`, `divers/exoty_com_...`,
`apps-sites/gamerules_com_...`, `divers/maviedesenior_com_...`,
`divers/drasill_github_io_bga_coinche_rules_fr_html.txt`,
`open-source/valmathieu_ContrAI_...`, `open-source/drasill_bga-coinche_...php` (BoardGameArena),
`open-source/ismo009_Coinche_main_game.js`, et — **seul règlement de compétition de la famille** —
**Cannes 2016** (base 162/252, contrat ajouté, ×2 / ×3 sur le contrat seul). **C'est la famille
dominante du corpus** et celle dont Colver dérive (§12).

Ses deux lignes de faille internes : **160 vs 162** (arrondi ou pas), et **assiette du
multiplicateur** — contrat seul (Note 1 FFB, Pagat, belotepoint, gamerules, ContrAI, **Cannes**)
vs score entier (FFB HTML, Wikipédia, BGA, ismo009).

**Cannes est la sous-famille la plus proche de Colver de tout le corpus** : base 162/252 (et non
160), contrat ajouté, multiplicateur sur le contrat seul, surcontre ×3, capot traité comme un
contrat. Quatre choix sur cinq. Les deux écarts restants sont la **valeur du capot** (270 contre
250) et **l'arrondi** (Cannes arrondit à la dizaine, Colver marque au point près).

### Famille C — **Points annoncés seuls** : le contrat est la seule monnaie

**D'où ça vient.** La pratique des concours à la table : c'est le barème le plus rapide à marquer
et celui qui produit les plus petits nombres.

**Ce qui la caractérise.** Personne ne compte les plis pour la marque — on compte seulement pour
savoir si le contrat passe. Le vainqueur de la donne (preneur qui réussit, ou défense qui fait
chuter) marque **le contrat**, ×2 ou ×4 s'il y a eu contre. Le capot annoncé est un contrat à 250.
La chute vaut parfois un forfait 160 au lieu du contrat. La belote ne sert souvent qu'à valider le
contrat, sans jamais être marquée.

**Qui l'applique.** `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` + sa copie (ASCEE 2A, tournoi
2016, partie en **1 010 points**), `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt`,
`divers/clubdejeux_com_...`, `divers/adpoker_fr_...` (variante b), la section « points annoncés »
de la page FFB contrée, `open-source/slim0_contree_main_backend_game_scoring.py`, et le mode
`scoreType != 1` de BoardGameArena.

### Famille D — **162 / 252 / au point près** (héritage de la belote classique)

**D'où ça vient.** Les règles FFB de **belote classique** et les règlements de concours de belote
de village, qui n'ont jamais eu de contrat chiffré à ajouter.

**Ce qui la caractérise.** Aucun arrondi, aucun forfait : la mise dedans vaut le tas entier
(**162**), le capot vaut le tas entier avec le dix de der à 100 (**252**). La belote est
**inviolable** et va toujours à qui l'a annoncée. Un **litige** à 81-81 remet les points en jeu.

**Qui l'applique.** `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt`,
`federations/ffbelote_org_regles_officielle_belote.txt`,
`federations/ffbelote_org_reglements_de_la_belote_avec_ou_sans_annonce.txt`,
`tournois/villeconin_fr_...`, `tournois/web_myassoc_org_...` (Lions Club Bigorre-Isaby),
`tournois/geraudotloisirs_free_fr_...`, `tournois/tcvb_bruche_free_fr_...`,
`tournois/pontdeclaix_fr_...`, `tournois/cdfcasson_fr_...`,
`tournois/sc4e58b2fce8a2e7a_jimcontent_com_...` (Règlement 2024),
`tournois/fnasce_org_IMG_pdf_reglement_belote_pdf.txt`. Sous-variante alsacienne/cévenole
(`tournois/aappmakoenigshoffen_...`, `tournois/lagrandcombe_...`) : **le capot ne vaut que 162**,
soit rien de plus qu'une mise dedans.

C'est de cette famille que viennent la base **162** de Colver et son « aucun arrondi, au point
près ».

### Famille E — **Bricolages de table** : un multiplicateur, mais pas sur ce qu'on croit

**D'où ça vient.** Des règlements de concours qui reconstruisent le barème à partir de « 160 »,
sans passer par la FFB.

**Ce qui la caractérise.** Le multiplicateur porte sur le **forfait de chute**, pas sur le
contrat : `160×2 + contrat + annonces` pour la coinche, `160×3` ou `160×4` pour la surcoinche. Le
contrat n'est jamais multiplié. Effet pratique : contrer un petit contrat rapporte autant que
contrer un gros — l'inverse exact de la logique de la famille B.

**Qui l'applique.** `tournois/maisondesessarts_fr_article116_html.txt` (160×2 = 320,
**surcoinche 160×3 = 480**),
`tournois/data_over_blog_kiwi_com_..._2015_01_27_reglement_table_coinche_pdf.txt` (160×2 puis
160×4), et `apps-sites/contree_org_4_joueurs.txt` dans sa version la plus radicale : enchères en
82-92-102…, `162×2 = 324`, et « ces points [du contrat] **ne sont pas multipliables** ».

### Hors familles — la belote bulgare

`apps-sites/gambiter_com_cards_Belote_html.txt` → <https://gambiter.com/cards/Belote.html>,
`apps-sites/licitum_board_directory_net_t16_belot_rules.txt` → <https://licitum.board-directory.net/t16-belot-rules>,
`apps-sites/officialgamerules_org_game_rules_belote.txt` → <https://officialgamerules.org/game-rules/belote/>
décrivent un jeu au barème **entièrement autre** : les points de donne sont divisés par 10 pour
donner des *match points*, le capot (« valat ») vaut **9 MP** ou **90 points** selon la source, et
la partie va à 151 MP. Aucun de ses chiffres n'est comparable à ceux de la contrée ; ces sources
ne sont **pas** des témoignages sur le barème français et ne sont comptées dans aucun tableau
ci-dessus.

---

## 10. Récapitulatif des contradictions internes à la FFB

| # | Objet | Position A | Position B | Documents |
|---|---|---|---|---|
| 1 | **Surcontre** | ×4 | ×3 | Contrée ~2015 + Contrée 2016 + Coinchée 2015 (×4) **vs** `LOCAL` / Équipe Ludique (×3) |
| 2 | **Belote en chute** | prenable | imprenable | Contrée ×4 rédactions (prenable) **vs** Coinchée ×2 + Belote classique ×3 (imprenable) |
| 3 | **Faire plus que la défense** | non requis | requis | Contrée 2016 (« valable même si les défenseurs ont réalisé plus ») **vs** Coinchée 2015 (« les deux conditions ») |
| 4 | **Capot contré** | 1000 / 2000 | 500 / 1000 | **La même page** <https://www.ffbelote.org/belote-contree/>, sections « points faits » et « faits+annoncés » **vs** sa section « points annoncés » |
| 5 | **Capot demandé** | 500 (prime) | 250 (contrat) | Page HTML contrée + Coinchée 2015 (500) **vs** `LOCAL` + tableau p. 7 du règlement 2016 (250) |
| 6 | **Chute** | 160 | 162 | Contrée ×4 + Coinchée ×2 (160) **vs** Belote classique ×3 + `villeconin` (162) |
| 7 | **Base du contre** | forfait 320/640 | 160 + Contrat×mult | Tableau p. 7 du règlement 2016 **vs** la **Note 1 du même document**, qui propose l'autre |
| 8 | **Arrondi** | 85 → 90 | 85 → 80 | Contrée 2016 + ~2015 + Coinchée **vs** `LOCAL` / Équipe Ludique — et « au point près, pas d'arrondi » en belote classique |

Aucune de ces huit contradictions n'est signalée par la FFB elle-même. Les quatre rédactions
coexistent en ligne. **On ne peut pas parler d'« un » règlement FFB** — seulement d'un document
FFB daté et nommé.

---

## 11. Ce sur quoi le corpus est massivement muet

- **Le capot réalisé par l'équipe *contrée*** : deux sources sur ~50 (FFB 2016 Note 2 → 420 pts ;
  `LOCAL` + Coinchée 2015 → 250 substitué à 160 dans la formule).
- **La belote de la *défense* quand le contrat passe** : presque toujours implicite.
- **Ce que marque la défense sur un contrat contré *réussi* quand elle a des annonces** : seule
  la FFB coinchée 2015 le dit (« rien, à l'exception éventuelle des 20 points de belote »).
- **La générale** : absente de tous les documents FFB, et sans valeur convenue ailleurs.
- **Le partage en cas d'égalité parfaite des points de cartes** : traité seulement par les
  règlements de belote classique (règle du « litige » à 81-81), jamais par ceux de contrée sauf
  `apps-sites/contree_org_4_joueurs.txt` (« L'envoyeur est donc "dedans" à égalité de points »).
- **Les implémentations open source ne documentent presque jamais leur barème.**
  `open-source/gyscos_libcoinche_master_src_points.rs` ne contient **que** les valeurs de cartes
  et les forces — aucun calcul de score de donne ;
  `open-source/theosaulus_coinche_main_coinche_utils.py` ne contient que l'encodage des actions
  (« capot is encoded as 250 »). Seuls `CephaloSophie_kydos`, `drasill_bga-coinche`,
  `ismo009_Coinche` et `slim0_contree` calculent réellement un score de donne — et **ils
  appliquent quatre barèmes différents.**

---

## 12. Où tombe Colver dans cette typologie

Barème effectivement appliqué par le moteur (`colver-core/src/engine/scoring.rs`, constantes
`TOTAL_PTS = 162` et `CAPOT_PTS = 252`) :

| Cas | Preneurs | Défense |
|---|---|---|
| Réussi, standard | points + contrat + **leur** belote | points + **leur** belote |
| Réussi, contré | **162** (ou **252** si capot réalisé) + contrat**×2** + **toute** belote | 0 |
| Réussi, surcontré | idem avec contrat**×3** | 0 |
| Chute (mult 1 / 2 / 3) | 0 | **162** + contrat**×mult** + **toute** belote |
| Capot | contrat ordinaire à **250**, pas de forfait | — |
| Arrondi | **aucun**, au point près | |

**Famille.** Colver relève de la **famille B** (« FFB faits + demandés : base + contrat,
multiplicateur sur le contrat seul »), avec **la base de la famille D** (162 exact, aucun
arrondi) substituée au 160 arrondi de la famille B. Il n'emprunte rien à la famille A (aucun
forfait 320/640/500) ni à la famille C (les points faits sont toujours marqués) ni à la
famille E (le multiplicateur ne porte jamais sur la base).

### 12.1 Ce qui est attesté, et par qui

| Élément du barème Colver | Attesté par |
|---|---|
| **Mode faits + demandés** | Consensus majoritaire du corpus : FFB coinche (imposé), FFB contrée (au choix), Pagat, Wikipédia coinche, gamerules, exoty, belotepoint, maviedesenior, ContrAI, BGA (mode `scoreType == 1`). |
| **Réussi standard = points + contrat / défense = ses propres points** | `federations/LOCAL_regles_officielles_belote_contree.txt` (« leur total + le montant du contrat demandé […] Les défenseurs marquent leur total ») ; `divers/pagat_com_jass_coinche_html.txt` ; `divers/fr_wikipedia_org_wiki_Coinche.txt` ; `open-source/valmathieu_ContrAI_main_contree-domain.md`. |
| **Chute = 162 + contrat** | `divers/exoty_com_compter_points_coinche.txt` (« les défenseurs auraient marqué **162 points + 100 (contrat) = 262 points** ») ; `divers/belotepoint_fr_regles_coinche.txt` (« L'équipe adverse marque la totalité des **162 points + la valeur du contrat** ») ; `divers/maviedesenior_com_...` (« **242 points (162 + 80)** ») ; `divers/jeux_regles_com_regles_coinche.txt` ; `divers/regles_com_jeux_cartes_coinche_html.txt` ; `apps-sites/regles_de_jeux_com_regle_coinche.txt` ; `divers/lemagloisirs_fr_regle_coinche.txt` ; `open-source/drasill_bga-coinche_master_coinche.game.php` ; `open-source/ismo009_Coinche_main_game.js`. **Bien attesté.** |
| **Multiplicateur sur le contrat seul, jamais sur la base** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` **Note 1** (seule forme fédérale) ; `divers/pagat_com_jass_coinche_html.txt` (« doubles the score for the bid only ») ; `divers/belotepoint_fr_regles_coinche.txt` (« la valeur du contrat, **et non les points de pli** ») ; `apps-sites/gamerules_com_rules_coinche.txt` ; `open-source/valmathieu_ContrAI_main_contree-domain.md`. |
| **Surcontre ×3** | `federations/LOCAL_regles_officielles_belote_contree.txt` ; **Cannes 2016** (« le surcontre les triple ») ; `divers/iscool_helpshift_..._coinche_rules.txt` ; `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` ; `open-source/ismo009_Coinche_main_game.js` ; `divers/pagat_com_jass_coinche_html.txt` (variante recensée). |
| **Capot = contrat ordinaire** (250 chez Colver) | `federations/LOCAL_regles_officielles_belote_contree.txt` ; tableau p. 7 du règlement FFB 2016 ; `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` ; `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` ; `apps-sites/ludicash_com_help_rules_coinche.txt` ; `open-source/theosaulus_coinche_main_coinche_utils.py` ; **Cannes 2016** — mais **à 270**, pas 250. La *forme* est attestée en compétition, la *valeur* ne l'est pas. |
| **Base cartes 162, ou 252 sur capot réalisé** | **Cannes 2016** : « points de l'enchère demandée **+162 ou 252** + belote », aux deux lignes (contrat réussi et contrat perdu). Seul règlement du corpus à poser 162/252 comme la base d'un barème *de contrée* avec contrat ajouté. |
| **Dix de der à 100, total 252** | Consensus quasi unanime (§5.2). |
| **Belote prenable en cas de chute** | Les **quatre** rédactions FFB de contrée ; `divers/belotecontree_free_reglement.txt` ; `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` ; `open-source/CephaloSophie_kydos_..._donneScoring.ts`. |
| **Aucun arrondi, au point près** | `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` et les deux pages HTML FFB de belote classique (« A la belote, les points sont comptabilisés **au point près. Il n'y a pas d'arrondi.** ») ; `tournois/villeconin_fr_..._pdf.txt` ; `tournois/web_myassoc_org_..._pdf.txt` ; `tournois/geraudotloisirs_free_fr_...txt`. **Mais aucune de ces sources n'est un règlement de contrée** — voir 12.2. |
| **Contré réussi en « le gagnant prend tout », défense 0** | Tableau p. 7 du règlement FFB 2016 ; `federations/LOCAL_...` ; `open-source/valmathieu_ContrAI_main_contree-domain.md` (« winner-takes-all […] the losing side scores 0 ») ; `open-source/drasill_bga-coinche_...php`. |

**Le plus proche parent du corpus entier** n'est pas un texte fédéral mais
`open-source/ismo009_Coinche_main_game.js` : sa chute vaut `162 + contrat × mult` avec
`mult ∈ {1, 2, 3}` — c'est **exactement** la formule de chute de Colver. Deux écarts subsistent :
ismo009 ne donne que la belote **de la défense** (Colver donne toute belote au gagnant) et il
arrondit le résultat (`Math.round(v/10)*10`).

### 12.2 Ce qui n'est attesté nulle part

> **Révision du 2026-08-02.** Cette section a été écrite avant la découverte du règlement de
> Cannes. Celui-ci **ferme deux des cinq points** et en réduit deux autres. Ce qui suit est la
> version corrigée ; l'état antérieur est rappelé à chaque point pour qu'on voie ce que Cannes
> a changé.

1. ~~**La base 162 appliquée à un barème de *contrée*.**~~ **Fermé par Cannes**, qui écrit
   « points de l'enchère demandée +162 ou 252 » dans un règlement de contrée, avec contrat ajouté.
   Ce qui reste ouvert est **plus étroit** : le couple « base 162 **et aucun arrondi** ». Cannes a
   la base mais arrondit à la dizaine ; les autres sources de contrée qui écrivent « 162 +
   contrat » (exoty, belotepoint, maviedesenior, ismo009) arrondissent aussi. **BGA reste la seule
   source du corpus à marquer 162 sans arrondi** — et c'est du code, pas un règlement.
2. ~~**252 comme base d'un contré réussi sur capot réalisé.**~~ **Attesté par composition chez
   Cannes** : sa formule de marque unique (« +162 ou 252 ») et sa clause de contre (« double les
   points de l'enchère demandée ») donnent, appliquées ensemble, exactement `252 + contrat×2`.
   Aucune phrase de Cannes ne traite ce cas nommément — mais son barème n'a pas de cas
   particuliers, c'est là tout son intérêt. Le seul texte qui traite le cas *explicitement*
   (`federations/LOCAL_...`) dit toujours 250.
3. **Les valeurs numériques exactes** — 322 (contré réussi sur 80), 502 (capot annoncé réussi),
   752 / 1002 (capot contré / surcontré) — **n'apparaissent littéralement dans aucune source**.
   Nuance depuis Cannes : 322 est bien ce que produit son barème (162 + 80×2), et 502 celui de
   `LOCAL` ; les valeurs de capot divergent parce que Cannes chiffre le capot 270 (→ 522, 792,
   1062). C'est la *structure* qui est désormais attestée, pas la table de nombres.
4. ~~**La combinaison des trois choix.**~~ **Cannes porte les trois** : base 162/252, surcontre
   ×3, capot traité comme un contrat ordinaire. C'était le point le plus faible de la position de
   Colver — « chacun attesté séparément, aucune source ne porte les trois » — et il est levé par
   un règlement de championnat. **Deux écarts subsistent** : la valeur du capot (270 vs 250) et
   l'arrondi (Cannes arrondit, Colver non).
5. **Le silence du corpus sur le capot réalisé par l'équipe *contrée*** (§5.6) reste réel : deux
   sources seulement le nomment (FFB 2016 Note 2 → 420 ; `LOCAL` → 250 substitué à 160). Cannes ne
   le nomme pas non plus, mais son barème sans cas particulier le couvre mécaniquement à 252 —
   ce que fait Colver (`contre_base = CAPOT_PTS`).

**En une phrase** : avant Cannes, la position de Colver était une combinaison cohérente mais
inédite, assemblée à partir de sources qui ne la portaient jamais ensemble ; depuis Cannes, elle
est **la position d'un règlement de championnat à deux paramètres près** (capot à 250 plutôt que
270, et pas d'arrondi).

### 12.3 La Note 1 de la FFB 2016 est-elle le plus proche parent fédéral ? — **Oui**

> **À lire avec le §12.2 corrigé.** La Note 1 reste le plus proche parent **fédéral**. Mais le
> plus proche parent *tout court* est désormais **Cannes 2016**, qui porte quatre des cinq choix
> de Colver au lieu de trois, et qui a l'avantage d'être un règlement effectivement appliqué en
> compétition plutôt qu'une note facultative.

> « Lors de tournois organisés en réel, afin que l'issue de la partie ne soit pas définie par une
> simple donne, en mode points fait + points demandés, les organisateurs peuvent utiliser la
> comptabilisation suivante en cas de contre (surcontre) : **Contrat x 2 + 160** au lieu de
> Contrat x 2 + 320 (**Contrat x 4 + 160** au lieu de Contrat x 4 + 640) »
> — `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt`, §10.2 Note 1

**Oui, sans ambiguïté** : c'est **la seule forme fédérale en `base + contrat × mult`**, et donc
le seul texte FFB dont la *structure* est celle de Colver. Tous les autres passages FFB sont soit
en forfait (320/640, tableau p. 7), soit en « tout multiplié » ((160 + contrat) × 2, pages HTML).

Colver = **Note 1, avec deux substitutions** :

| | FFB Note 1 (2016) | **Cannes 2016** | Colver |
|---|---|---|---|
| Base | 160 (arrondi) | **162 / 252** | **162 / 252** (exact) |
| Contre | Contrat × 2 | Contrat × 2 | Contrat × 2 — **identique aux deux** |
| Surcontre | Contrat × **4** | Contrat × **3** | Contrat × **3** — comme Cannes et `LOCAL` |
| Camp perdant | 0 | 0 | 0 — **identique aux deux** |
| Capot | 250 (tableau p. 7) | contrat à **270** | contrat à **250** |
| Arrondi | à la dizaine | à la dizaine (**bascule à 6**) | **aucun**, au point près |

Deux réserves à ne pas perdre de vue :
- la Note 1 est **facultative et réservée aux tournois en réel** (« les organisateurs *peuvent*
  utiliser ») : ce n'est pas le barème par défaut de la FFB, c'est son échappatoire ;
- elle ne couvre que **la branche contrée**. La chute non contrée de Colver (162 + contrat)
  relève, elle, de la formule FFB ordinaire (« 160 + la valeur du contrat demandé »), avec
  162 substitué à 160.

---

## 13. Corrections à l'index (`docs/rules-survey/README.md`)

Trois affirmations du README à amender, sur la base des vérifications faites pour cette matrice.

### 13.1 `jeux-regles.com` et `regles.com` ne sont **pas** des copies FFB

Le README écrit : « `cartesetcie`, `carafons`, `jeux_regles`, `missegre` et les pages
`ffbelote.org` sont des copies du même texte FFB ».

C'est exact pour `cartesetcie`, `carafons` et `missegre` — ils portent tous la signature
« *Une équipe ayant fait 89 points et ayant demandé 90 chute* ». Ce ne l'est **pas** pour
`divers/jeux_regles_com_regles_coinche.txt` ni pour `divers/regles_com_jeux_cartes_coinche_html.txt` :

- ni l'un ni l'autre ne contient cette signature (`grep -c` → 0) ;
- `jeux-regles.com` porte une règle de coinche que **la FFB n'écrit nulle part** (« Lorsqu'un
  contrat est coinché, **les points totaux, hors Belote et annonces**, sont multipliés par deux »
  — là où la FFB écrit `(160 + contrat) × 2`), une chute à **162** (la FFB dit 160), et une cible
  de partie à **701 points** (jamais mentionnée par la FFB) ;
- `regles.com` renvoie explicitement la formule FFB à une autorité extérieure (« la défense
  marque 160 + le contrat **dans la règle fédérale** »), ce qu'une copie ne ferait pas.

**Ce sont deux témoins indépendants**, et ils doivent être comptés comme tels sur tous les axes
de cette matrice.

### 13.2 `gambiter` est une copie verbatim de Pagat, `reglesdejeux.github.io` une traduction automatique

Le README n'en dit rien et range `gambiter` parmi les « applications et plateformes ».
Vérification faite :

- `apps-sites/gambiter_com_cards_jass_coinche_html.txt` est **identique mot pour mot** à
  `divers/pagat_com_jass_coinche_html.txt` — contrôlé par `diff` sur les phrases de barème, y
  compris la ponctuation et les espaces avant virgule (« *A coinche doubles the score for the bid
  only , and a surcoinche doubles it again.* ») ;
- `divers/reglesdejeux_github_io_regles_du_jeu_la_coinche_index_html.txt` est une **traduction
  automatique** du même texte, reconnaissable à ses calques (« s'il y a une enchère au capot ou
  une **pièce de monnaie** » pour *a coinche*, « chaque **costume** se classe comme les atouts »
  pour *each suit ranks*).

Les trois fichiers ne valent donc que pour **un seul témoignage**, celui de John McLeod.
Conséquence pratique : Pagat pèse un, pas trois, dans tous les comptages de cette matrice — ce
qui compte, vu qu'il est la seule source à recenser systématiquement les variantes (surcontre ×3,
capot chuté à 410, double sur le score entier).

### 13.3 La contradiction « capot contré 500 vs 1000 » est **interne à une seule page**

Le README écrit : « la page `ffbelote_org_belote_contree` donne **capot contré = 500 /
surcontré = 1000**, là où les PDF de la même fédération donnent **1000 / 2000**. […] c'est une
contradiction interne au site. »

La contradiction est réelle mais **mal située**. Vérification faite sur
`federations/ffbelote_org_belote_contree.txt` → <https://www.ffbelote.org/belote-contree/> :

- section **A – Points faits** : « Le capot contré vaut **1.000** points. Le capot surcontré vaut
  **2.000** points. »
- section **B – Points annoncés** : « Le capot contré vaut **500** points. Le capot surcontré
  vaut **1.000** points. »
- section **C – Points faits + points annoncés** : « Le capot contré vaut **1.000** points. Le
  capot surcontré vaut **2.000** points. »

Les deux chiffres sont donc **sur la même page, à quinze lignes d'écart**, et la page ne
contredit pas les PDF : elle se contredit elle-même — ou plus exactement, elle ne dit jamais que
1000 = 500×2 (points faits, où le capot vaut 500) et 500 = 250×2 (points annoncés, où le capot
vaut 250). L'articulation est cohérente **si on la reconstruit**, mais elle n'est écrite nulle
part, et un lecteur qui saute d'une section à l'autre lit deux barèmes incompatibles.

Quant aux PDF : le règlement contrée 2016 **ne donne aucun chiffre de capot contré dans son
texte** ; c'est son tableau image de la p. 7 qui donne 1000/2000, et `LOCAL` donne encore autre
chose (250 × 2 ou × 3 = 500 / 750). **Le corpus ne permet pas de trancher**, et il faut le dire
ainsi plutôt que d'opposer « le site » aux « PDF ».
