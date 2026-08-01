# Collecte 2 — élargissement du corpus

Deuxième passe de collecte (2026-08-01), en complément de [README.md](README.md) qui indexe la
première. **364 nouvelles sources web** (fichier original + `.txt` avec `SOURCE:` en première
ligne) et **113 nouveaux fichiers open source**, soit ~1 050 fichiers au total dans
`docs/rules-survey/` pour l'analyse, `data/rules-corpus/` pour les sources brutes.

| Dossier | Avant | Ajouts | Ce qui a été visé |
|---|---:|---:|---|
| `federations/` | 6 | **5** | les rédactions FFB manquantes |
| `tournois/` | 13 | **26** | règlements de concours réellement appliqués (priorité n°1) |
| `clubs/` | 0 | **43** | francophonie hors France : Belgique, Suisse, Québec |
| `divers/` | 20 | **156** | variantes régionales françaises + **belote non française** |
| `apps-sites/` | 14 | **143** | apps et plateformes, surtout leurs **options de règles** |
| `open-source/` | 12 | **113** | code de calcul de score, configs de variantes, tests |

## Fichiers purgés

Trois `.txt` de la première collecte étaient des coquilles sans contenu exploitable et ont été
supprimés (avec leur `.html`) :

- `tournois/amicaleloisirs_free_fr_page_id_79.*` — le serveur renvoie un `Fatal error` PHP
- `tournois/belote_com_articles_belote_comment_organiser_un_tournoi_de_belote.*` — coquille SPA (« Loading: 0% »)
- `apps-sites/sites_google_com_site_beloteandr.*` — page de connexion Google, pas la fiche de l'app

Conservé malgré sa taille : `apps-sites/eryodsoft_com.txt` (266 o utiles) — c'est la seule source
qui chiffre le catalogue d'Eryod Soft (« 10 jeux, **> 100 options de règles**, > 1000 règles d'IA »).

---

## Fédérations — `federations/`

| Fichier | URL | Apport |
|---|---|---|
| `ffbelote_org_wp_content_uploads_2016_01_REGLES_DE_LA_BELOTE_pdf` | ffbelote.org/…/2016/01/REGLES-DE-LA-BELOTE.pdf | **Cinquième rédaction FFB**, non datée dans le nom, distincte des quatre déjà indexées ; belote classique, arrondi présenté comme optionnel |
| `ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_CONTREE_pdf` | ffbelote.org/…/2015/11/REGLES-DE-LA-BELOTE-CONTREE.pdf | Même document que le `ffbelote_REGLES-DE-LA-BELOTE-CONTREE` local, mais **récupéré à son URL canonique** : la provenance est désormais tracée |
| `ffbelote_org_wp_content_uploads_2016_01_regles_officielles_de_la_Belote_Contree_27_01_2016_pdf` | ffbelote.org/…/2016/01/regles-officielles-de-la-Belote-Contree-27-01-2016.pdf | Idem pour la version du 27/01/2016 |

*(les deux autres entrées du dossier sont les fichiers locaux ré-ancrés sur leur URL)*

---

## Tournois et concours — `tournois/` (26 nouvelles)

### Les plus divergents

| Fichier | URL | Nature | Apport / divergence |
|---|---|---|---|
| `jwebarteam_wordpress_com_championnat_de_coinche_2016_2017_les_regles` | jwebarteam.wordpress.com/championnat-de-coinche-2016-2017/les-regles/ | championnat associatif (Guyane) | **Le seul règlement trouvé qui donne surcoinche = ×3** (« points annoncés ×3 + 160 »), comme Colver ; capot annoncé **500 « tout rond »**, générale **600 tout rond**, partie en **2001** points, et la belote ne peut jamais descendre le contrat sous 80 |
| `clublafontainedejouvence_fr_r_C3_A8glement_coinch_C3_A9e` | clublafontainedejouvence.fr/règlement/coinchée | règlement de club (Eure) | Format inédit : **3 parties de 121 points « limitées à 135 », puis 151 « limitée à 175 »**, chronométrées (1 h 15 / 1 h 30) et **classement individuel** (chacun tire sa place) ; « à la coinchée on ne pisse pas » |
| `data_over_blog_kiwi_com_…_reglement_table_coinche_pdf` | data.over-blog-kiwi.com/…/ob_1f68a4_2015-01-27-reglement-table-coinche.pdf | tournoi (AIL Manissieux) | Parties en **3000 points, système AURARD** ; arrondi explicite **84→80 / 85→90** ; **surcoinche = 160 × 4** ; « capote demandée = 250 + contrat (250) » (donc 500 en deux termes) |
| `web_archive_org_web_2020_http_coinche_stephanoise_com_…_reglement_coinche_pdf` | coinche-stephanoise.com/mesdocuments/reglement_coinche.pdf (via Wayback) | règlement « officiel » d'association | **Récupéré, alors que le README le donnait pour perdu** ; barème stéphanois : capot 250 / 500, base 160 et 162 coexistant dans le même texte |
| `muscletacoinche_wixsite_com_muscletacoinche_copie_de_equipes` | muscletacoinche.wixsite.com/muscletacoinche/copie-de-equipes | championnat associatif (Clermont-Ferrand) | Manches en **2001 points** ; « on ne sort pas avec les points défensifs » ; **la belote ne compte qu'à partir de 100 annoncé** ; litige 81-81 → mène rejouée sans changer le donneur |
| `casimirdehauteclocque_fr_jeux_coinche_pdf` | casimirdehauteclocque.fr/jeux/coinche.pdf | règlement de table rédigé | Document long et précis (2000 points) qui **assume la base 152 + 10 de der** et détaille les cas limites d'annonce |
| `pontdeclaix_fr_…_rglementconcoursdebelote_pdf` | pontdeclaix.fr/…/65a68d18ab985_rglementconcoursdebelote.pdf | concours (Twirl Danse) | 4 parties de 12 donnes ; **capot = 252**, mise dedans = 162 ; les équipes paires changent de table à chaque partie |
| `ville_plougastel_bzh_…_2026_Reglement_Tournoi_Belote_pdf` | ville-plougastel.bzh/…/2026_Reglement_Tournoi_Belote.pdf | tournoi municipal 2026 | Tournoi en **36 mènes (3 × 12)** avec objectif **1500 points**, règlement daté 2026 — le plus récent du corpus |
| `ville_plougastel_bzh_…_Reglement_Concours_Belote_pdf` | ville-plougastel.bzh/…/2023/11/Reglement_Concours_Belote.pdf | tournoi municipal 2023 | La version antérieure du même tournoi : utile pour voir ce qu'une commune change d'une édition à l'autre |
| `calameo_com_books_00070473318190a9fb456` | calameo.com/books/00070473318190a9fb456 | règlement de concours (Calaméo) | Concours **en 5 parties, partie à 1001 points** |
| `sc4e58b2fce8a2e7a_jimcontent_com_…_Belote_Règlement_2024_pdf` | jimcontent.com/…/Belote Règlement 2024.pdf | concours 2024 | 5 parties de **10** donnes (et non 12) ; capot 252, mise dedans 162 |
| `cdfcasson_fr_files_ugd_…_pdf` | cdfcasson.fr/_files/ugd/0df194_8c501176abb04e5b9237c65fcf80f584.pdf | comité des fêtes (Casson) | 4 parties de 12 donnes, capot 252 ; barème des pénalités en 162 |
| `lagrandcombe_fr_…_Reglement_belote_2020_pdf` | lagrandcombe.fr/…/Reglement-belote-2020.pdf | Grand Prix municipal | Classement au **cumul de points bruts sur plusieurs parties** (exemples à 1100 / 1280 / 664 points), et non en manches gagnées |
| `bibuloba_animations_fr_…_Bibuloba_regles_belote_pdf` | bibuloba-animations.fr/…/Bibuloba-regles-belote.pdf | prestataire d'animation | Règlement « clé en main » vendu à des organisateurs — intéressant parce qu'il **normalise** un barème pour des dizaines de concours |
| `s1_static_footeo_com_uploads_fcplouay_…_pdf` | s1.static-footeo.com/uploads/fcplouay/Medias/Rglement_Concours__mtqpkb.pdf | club de foot (Plouay) | Court, mais pose « mise dedans = 162 » comme seule règle de score |
| `lesamisdutempslibrevarennes_jimdofree_com_…_concours_belote` | lesamisdutempslibrevarennes.jimdofree.com/activités/jeux-concours/concours-belote/ | club du 3e âge | 5 parties de 10 donnes, capot 252, belote 20 |
| `bonnafousn_wixsite_com_belote_fourquevaux` | bonnafousn.wixsite.com/belote-fourquevaux | concours hebdomadaire | « Le concours se déroule en **4 parties de 12 mènes** », classement au cumul — vocabulaire « mène » et non « donne » |
| `web_archive_org_web_20220516113001_…_pcsgc_fr_…_regles_officielles_de_la_coinche_html` | pcsgc.fr/pages/infos-utiles/regles-officielles-de-la-coinche.html (via Wayback) | club (le README le donnait pour mort) | **Récupéré** : règlement de coinche de club, arrondi à la dizaine décrit explicitement |
| `zpag_net_Jeux_Cartes_Belote_et_Variante_pdf` | zpag.net/Jeux/Cartes/Belote_et_Variante.pdf | livret de règles + variantes | Compile belote et ses variantes avec un barème d'annonces complet (50/100/150/200) et une règle d'arrondi explicite |
| `fnasce_org_2016_concours_de_belote_contree_coinche_a36693_html` | fnasce.org/2016-concours-de-belote-contree-coinche-a36693.html | page-mère des PDF FNASCE déjà collectés | Donne le **contexte de compétition** (phases finales, quarts/demies) qui manque aux PDF seuls |
| `tournois_apps_com_belote` | tournois-apps.com/belote | logiciel de gestion de concours | Inventaire des **formats de tournoi réellement supportés** par un outil utilisé par les organisateurs (mêlée, parties fixes, classement au cumul) |
| `fr_wikipedia_org_wiki_Syst_C3_A8me_Aurard` | fr.wikipedia.org/wiki/Système_Aurard | système d'appariement | Définit le **système AURARD** cité par les règlements de coinche sans jamais y être expliqué |
| `concours_belote_org_blog_organiser_concours_belote` | concours-belote.org/blog/organiser-concours-belote | guide d'organisateur | Donne les **valeurs par défaut de fait** : 5 tours, adversaires tirés au sort, partie à 1000 points, 30-45 min |
| `weezevent_com_fr_blog_organiser_concours_belote` | weezevent.com/fr/blog/organiser-concours-belote/ | guide d'organisateur | Même fonction, cadrage administratif (déclaration mairie, buvette) |
| `beloter_com_351_organiser_un_concours_en_salle` | beloter.com/351/organiser-un-concours-en-salle | fil de forum | Praticiens qui comparent leurs formats de concours entre eux |
| `cartesetcie_fr_les_regles_de_la_coinche` | cartesetcie.fr/les-regles-de-la-coinche/ | site de règles | Page distincte de celle déjà collectée chez le même éditeur, et **pas le même texte** |

---

## Francophonie hors de France — `clubs/` (43 nouvelles)

Le README notait « aucune fédération belge, suisse ou québécoise ne publie de règlement de
contrée ». C'est confirmé pour la *contrée* — mais la recherche a ouvert deux corpus voisins qui
divergent franchement sur le barème, et un corpus belge de *whist* qui est le cousin structurel
de la coinche.

### Suisse — la coinche à 157 points

| Fichier | URL | Apport / divergence |
|---|---|---|
| `coinche_ch_regles_coinche_pdf` | coinche.ch/regles_coinche.pdf | **La trouvaille suisse : le total du jeu est 157 points, pas 162** (jeu de 36 cartes, dernier pli = 5) ; contrat + points **arrondis à la dizaine « selon les principes commerciaux »** ; match = 250, générale = 500, partie en 2000 |
| `coinche_ch_regles_pdf` | coinche.ch/regles.pdf | Version courte du même barème 157 |
| `coinche_ch_regles_tournoi_pdf` | coinche.ch/regles_tournoi.pdf | **Règlement de tournoi suisse** : parties en 2000 points, procédure de fin de partie point par point |
| `coinche_ch_PageCoinche_GuideCoinche_pdf` | coinche.ch/PageCoinche/GuideCoinche.pdf | Guide complet du club (enchères, conventions) |
| `coinche_ch_PageCoinche_annonces_pdf` | coinche.ch/PageCoinche/annonces.pdf | Barème d'annonces suisse |
| `coinche_ch_PageCoinche_Theoremes_Theoremes_pdf` | coinche.ch/PageCoinche/Theoremes/Theoremes.pdf | « Théorèmes » d'enchère — doctrine de jeu, pas règlement |
| `coinche_ch_PageCoinche_coincheRV_pdf` | coinche.ch/PageCoinche/coincheRV.pdf | Variante interne du club |
| `coinche_ch_mode_pdf` | coinche.ch/mode.pdf | Modalités de rencontre |
| `coinche_ch_seminaire_coinche_pdf` | coinche.ch/seminaire_coinche.pdf | Support de séminaire (pédagogie du barème) |

### Suisse — le jass, barème concurrent

| Fichier | URL | Apport |
|---|---|---|
| `frtj_kirschmann_reglement_tournois_officiels_COMPLET` | règlement des tournois officiels de la Fédération romande de jass | **Le règlement fédéral le plus complet du corpus hors FFB** : total **157** contrôlé à chaque donne (« les cartes ne peuvent être mélangées que si le total donne 157 »), match non réussi = 157 à l'adversaire, partie en **2512** points au chibre |
| `swisslos_ch_fr_jass_informations_les_regles_du_jass_bases_du_jass_html` | swisslos.ch (fr) | Barème jass officiel diffusé par la loterie nationale |
| `swisslos_ch_fr_jass_informations_les_regles_du_jass_le_chibre_html` | swisslos.ch (fr) | Le chibre = l'équivalent suisse de la coinche, avec **facteurs multiplicateurs par atout** (et non un contrat en points) |
| `swisslos_ch_fr_jass_championnatsuissedechibre_informations_html` | swisslos.ch | Format du **championnat suisse de chibre** |
| `jassverband_ch_fr` | jassverband.ch (fr) | Fédération suisse de jass — l'organe qui manquait |
| `jass_geneve_ch_regles_html`, `jass_geneve_ch_regleschibre_html` | jass-geneve.ch | Règles de club genevois, jass et chibre |
| `amisduchibre_ch_histoire_du_chibre_ou_du_jass` | amisduchibre.ch | Histoire et filiation chibre / jass / coinche |
| `chibre_ch_forum_viewtopic_php_*` (5 fils) | chibre.ch/forum | **Fils de forum où des joueurs arbitrent des cas de score litigieux** — la matière la plus « vécue » du corpus |
| `reset_bar_ch_reglement_tournoi_jass` | reset-bar.ch | Règlement de tournoi de bar (150 / 300 pts) |
| `jassshop_ch_pi_Jassreglemente_14279_html` | jassshop.ch | Fiche des règlements jass vendus imprimés |
| `kleeblatt_jass_jimdoweb_com_programm_jassregeln` (dans `divers/`) | kleeblatt-jass.jimdoweb.com | Règles d'un club de jass alémanique |
| `regles_com_jeux_cartes_chibre_html`, `regles_com_jeux_cartes_jass_html` | regles.com | Présentation francophone du chibre et du jass |
| `belotejeu_com_belote_suisse` | belotejeu.com/belote-suisse | Comparatif belote française / suisse |

### Belgique

| Fichier | URL | Apport |
|---|---|---|
| `web_archive_org_…_charlemagnrie_be_…_Règlement_Belote_pdf` | charlemagnrie.be (via Wayback) | **Règlement de concours belge**, en 4 manches — le second club belge du corpus après `rjcv.be` |
| `web_archive_org_…_charlemagnrie_be_tournoi_de_belote` | charlemagnrie.be (via Wayback) | La page d'annonce du même tournoi |
| `rja_be_news_concours_de_belote` | rja.be | Concours de belote d'une association belge |
| `lewhist_be_reglement_html` | lewhist.be | **Règlement de whist belge** : le whist à la couleur est le cousin structurel de la coinche (contrat annoncé, chute), avec un barème **entièrement différent** |
| `whist_hainaut_be_Reglement_htm` | whist-hainaut.be | Règlement de ligue de whist du Hainaut |
| `web_be_clubsdewhist_html` | web.be/clubsdewhist.html | Annuaire des clubs de whist belges |
| `whisthub_com_fr_rules` | whisthub.com/fr/rules | Plateforme de whist en ligne : règles + options |
| `adpoker_fr_whist_a_la_couleur_html` | adpoker.fr | Whist à la couleur, version francophone |

### Québec

| Fichier | URL | Apport |
|---|---|---|
| `fadoq_ca_…_regles_combinees_2019_pdf` | fadoq.ca | **Règlement de jeux de cartes de la FADOQ** (le grand réseau des aînés québécois) : parties en **121 points**, barème sans rapport avec la belote française |
| `fadoq_ca_…_rglement_jeux_en_tte_2019_pdf` | fadoq.ca | Règlement « Jeux en tête » : cadre de compétition québécois |
| `fadoq_ca_…_whist_militaire_pdf`, `fadoq_ca_…_reglements_whist_militaire_pdf` | fadoq.ca | **Whist militaire** — le jeu de contrat pratiqué au Québec là où la France joue la coinche |
| `owfstorage_fadoq_ca_uploads_…_pdf` | fadoq.ca (stockage) | Annexe de règlement FADOQ |

### Divers francophones

| Fichier | URL | Apport |
|---|---|---|
| `foss_bridge_org_comment_jouer_a_la_belote_tunisienne` | foss-bridge.org | **Belote tunisienne** décrite en français — pont entre le corpus français et le corpus maghrébin |
| `revue_bancal_fr_explorez_les_variantes_des_regles_de_la_belote_a_travers_le_monde` | revue-bancal.fr | Panorama des variantes nationales |
| `jouer_au_belote_ueuo_com` | jouer-au-belote.ueuo.com | Site amateur de règles |

---

## Variantes françaises et sites de règles — `divers/` (partie française)

### Les plus divergents

| Fichier | URL | Apport / divergence |
|---|---|---|
| `re_belote_fr_belote_sans_atout_tout_atout` | re-belote.fr/belote-sans-atout-tout-atout/ | **Le Tout Atout / Sans Atout chiffré** : à SA les As valent **19** pour ramener le paquet à 162 ; à TA le paquet vaut **258** et les scores sont multipliés par **162/258 ≈ 0,63** — le seul mécanisme de renormalisation du corpus français |
| `club_belote_com_belote_coinchee_stephanoise_regles_particularites_et_positionnement_html` | club-belote.com | **Coinche stéphanoise** : distribution **6+2** au lieu de 5+3, **enchères de 5 en 5** (et non de 10 en 10), annonces qui ne comptent pas dans le contrat |
| `corsicafan_com_cfan_docs_mn_nbcc_pdf` | corsicafan.com/cfan/docs/mn/nbcc.pdf | **Belote contrée corse (1 à 4 joueurs)** : règles « bastiaises » (ponts faits, capot imparable, belote acquise) et **total qui passe de 162 à 182 quand une équipe a la belote** — un total variable, ce que personne d'autre ne fait |
| `corsicafan_com_cfan_download_bca2_pdf` | corsicafan.com/cfan/download/bca2.pdf | **Belote découverte corse** (dite aussi « arménienne ») : 4 cartes cachées + 4 visibles, contrat à 82 |
| `fr_academic_com_dic_nsf_frwiki_201539` | fr-academic.com — *Belote bridgée* | Variante à enchères de type bridge : **le contrat se dit en nombre de plis**, pas en points |
| `beloter_com_33_litige_C3_A0_la_coinche_C3_A9galit_C3_A9_C3_A0_81` · `beloter_com_906_litige_sur_les_81_points` | beloter.com | Deux fils entiers sur le **litige 81-81** : selon les tables, mène rejouée, points bloqués et reportés en bonus, ou défense qui encaisse — trois résolutions incompatibles |
| `trictrac_net_forum_sujet_belote_questions_diverses` | trictrac.net/forum | Fil où des joueurs constatent que « le capot annoncé vaut 250 chez nous » contre 500 ailleurs |
| `regles_du_jeu_net_annonces_ou_pas_dannonces_belote_de_comptoir_vs_belote_de_fede` | regles-du-jeu.net | Pose explicitement l'opposition **belote de comptoir / belote de fédération** comme deux régimes de règles distincts |
| `cartamundi_fr_…_729710_notice_belote_1_pdf` | cartamundi.fr | **Notice imprimée d'un cartier** (Cartamundi), 2e du corpus après Ducale — la règle telle qu'elle est distribuée dans les boîtes |
| `space_villers_fr_…_A4_Belote_Rules_pdf` | space-villers.fr | Fiche A4 de règles affichée en salle |
| `ludicash_com_belote_en_ligne_bases_belote_le_litige_a_la_belote` | ludicash.com | Décrit le **report cumulatif** des points de litige sur les donnes suivantes |
| `coincheenligne_fr_annonces` · `_encheres` · `_regles_coinche` | coincheenligne.fr | Barème d'annonces coinche complet (tierce 20 / 50 / 100 / carrés 200-150-100) |
| `exoty_com_blog_capot_belote_coinche_regles_points_strategie` | exoty.com | Signale des variantes régionales où **le bonus de capot vaut 100 et non 90** |
| `exoty_com_blog_belote_contree_vs_coinche_le_meme_jeu` | exoty.com | Argumente que contrée et coinche ne diffèrent que par le pas d'enchère |
| `chessclubanderlecht_org_les_variantes_de_la_belote_…` | chessclubanderlecht.org | Panorama des règles régionales |
| `concours_de_belote_fr_les_differents_types_de_belote` | concours-de-belote.fr | Typologie classique / coinchée / contrée du point de vue des organisateurs |

### Le reste du volet français

`beloter_com_*` (9 fichiers : `règles-belote`, `règles-coinche-contre`, `questions`,
`questions_la_belote`, `questions_les_autres_variantes`, `les_autres_variantes`, `251/le
règlement officiel`, `673/participer au championnat de France`) — forum Q&R communautaire, la
meilleure source de **désaccords réels** entre joueurs.
`re_belote_fr_jeu_de_belote_avec_annonces`, `belotepoint_fr_belote_rebelote`,
`regles_du_jeu_net_regles_de_la_belote`, `regles_du_jeu_net_belote_a_5_joueurs_…`,
`regles_com_jeux_cartes_belote_html`, `cartesetcie_fr_regle_du_jeu_belote`,
`exoty_com_regles_belote`, `exoty_com_les_regles_de_la_belote_a_3`,
`vipbelote_fr_blog_la_belote_a_3`, `vipbelote_fr_regles`, `commentjouer_fr_jouer_belote_a_trois`,
`jeutexplique_com_belote_a_2`, `belotejeu_fr_variantes_de_la_belote`,
`belotejeu_com_jouer_variante_belote`, `neopoker_fr_neojeux_variantes_belote_13014`,
`club_belote_com_la_belote_coinchee_et_belote_contree`, `coinche_eboaz_com_regles_jeu_coinche`,
`cartesenmain_fr_blogs_infos_coinche_quand_la_belote_se_corse_et_s_arrose`,
`belote_tv_federation_francaise_de_belote`, `belotecontree_free_reglement` — variantes à 2/3/5
joueurs, barèmes d'annonces, et présentations généralistes. Aucune ne contredit la FFB sur le
barème de base ; leur intérêt est la **couverture des formats hors 4 joueurs**.

---

## Belote non française — `divers/` (partie internationale)

C'est le volet où les barèmes divergent le plus, et surtout où **l'arrondi est traité comme un
invariant** et non comme une commodité.

### Bulgarie — l'arrondi dépend du contrat

| Fichier | URL | Apport |
|---|---|---|
| `belot_bg_academy_basics_tochkuvane_v_belota` · `belot_bg_academy_basics_belot_termini` · `belot_bg_belot_rules` | belot.bg | **Le corpus de référence** : conversion points cartes → points marqués par division par 10, avec une **limite d'arrondi qui dépend du contrat** (5 à sans-atout, 6 à la couleur, 4 à tout-atout) — le total marqué reste constant, c'est le point de tout le système |
| `bg_wikipedia_org_wiki_Белот` · `bg_wikipedia_org_wiki_Бриджбелот` | bg.wikipedia.org | Article encyclopédique + **bridge-belot**, variante à enchères de type bridge |
| `cooldown_bg_belot_turning_pravila_pdf` | cooldown.bg | Règlement de tournoi bulgare en PDF |
| `anatoli1606_blog_bg_…_oficialni_pravila_na_igrata_belot` · `denislavsotirov_blogspot_com_…` · `maxbelot_blogspot_com_p_blog_page_html` · `civil2006_blogspot_com_…` · `zpiderland_blogspot_com_…` · `honorofwriting_blogspot_com_…` | blogs bg | Rédactions concurrentes des « règles officielles » bulgares |
| `learnbelot_alle_bg_…` · `svara_bg_com_belot` · `globet_games_game_belot_pravila` · `yellowclub_net_page_karty_online_belot` · `fss_fmi_uni_sofia_bg_p_1028` · `forums_bgdev_org_index_php_showtopic_38477` · `blog_bozho_net_blog_913` · `bglux_org_forum_viewtopic_php_id_43` | divers bg | Sites de jeu, forums de développeurs et un cours universitaire — le barème vu du côté implémentation |
| `belot_md_reguli_php` · `belot_md_reguli_new_php` | belot.md (Moldavie) | Deux rédactions successives du même site — utile pour voir ce qui a bougé |
| `pagat_com_national_bulgaria_html` | pagat.com | Index Pagat des jeux bulgares |

### Golfe — le baloot et sa renormalisation

| Fichier | URL | Apport |
|---|---|---|
| `pagat_com_jass_baloot_html` | pagat.com/jass/baloot.html | **Le baloot saoudien convertit les points cartes en « game points » en divisant par 10 avec arrondi**, pour un total invariant de **16 (Hokum, 162 cartes)** ou **26 (Sun, 130 cartes)** ; partie à **152** game points |
| `en_wikipedia_org_wiki_Baloot` · `gamerules_com_rules_baloot` · `gambiter_com_cards_Baloot_html` | — | Trois rédactions indépendantes du même barème |
| `balootchampionship_com_baloot` | balootchampionship.com | **Championnat officiel de baloot** — la seule fédération non française du corpus qui publie un règlement de compétition |
| `jawaker_com_en_rules_baloot` · `blog_jawaker_com_en_baloot_rules_en` · `vipbaloot_com_ar_…` · `saudigamer_com_blots_card_game_rules` · `lifeinsaudiarabia_net_rules_to_play_baloot_game` | apps et médias | Les options de règles telles que les apps du Golfe les paramètrent |

### Grèce / Chypre — Vida et Pilotta

`el_wikipedia_org_wiki_Βίδα`, `el_wikipedia_org_wiki_Μπουρλότ`, `el_wikipedia_org_wiki_Πιλόττα`,
`pagat_com_jass_pilotta_html`, `pagat_com_jass_mpourloto_html`, `pagat_com_national_greece_html`,
`gambiter_com_cards_national_greece_html`, `pilottacyprus_blogspot_com_p_blog_page_html`,
`erasmusu_com_…_instructions_of_vida_screw_card_game_395446`,
`web_archive_org_…_agrino_org_izapitis_rules_html` — la **pilotta chypriote** est la coinche la
plus proche structurellement (contrat, coinche/« double »), avec un barème et un seuil de partie
propres.

### Klaverjas — deux barèmes concurrents documentés

`nl_wikipedia_org_wiki_Klaverjassen`, `nl_wikibooks_org_wiki_Kaartspel_Klaverjassen`,
`klaverjas_nl_SpelregelsKlaverjassen`, `klaver_live_puntentelling_klaverjassen`,
`tabellenboekje_nl_puntentelling_tabel_klaverjassen_php`, `thuisleven_com_…`,
`partyspellen_nl_…`, `kaartspellen_online_nl_…`, `pagat_com_jass_klaverjassen_html`,
`pagat_com_jass_kruisjassen_html`, `pagat_com_jass_pandoer_html`, `pagat_com_jass_boonaken_html`,
`pagat_com_jass_staekske_rape_html`, `en_wikipedia_org_wiki_Klaverjas`,
`gambiter_com_cards_Klaverjas_html`, `pagat_com_national_netherlands_html` — **le même jeu à 162
points que la belote**, mais avec deux conventions rivales nommées et documentées
(*Rotterdams* / *Amsterdams*), ce qui est exactement le phénomène qu'on cherche à mesurer.

### Balkans, Levant, Turquie, Arménie, jass alémanique

`hr_wikipedia_org_wiki_Belot`, `sr_wikipedia_org_…_Белот_(игра)`, `legalbet_rs_…_kako_se_igra_belot_bela_…`,
`dalmacijaportal_hr_bela_povijest_pravila_i_najvece_posebnosti`, `6yka_com_blogovi_bela_ili_belot`,
`belaibelot_blogspot_com_p_…_pravila_bele_html`, `eivanec_webmaster_com_hr_…_pravila_igre_bela_belot_pdf`,
`pagat_com_jass_bela_html` (Croatie/Serbie/Bosnie — la *bela*) ·
`he_wikipedia_org_wiki_בלוט`, `tahvivim_com_…`, `pagat_com_jass_klabberjass_html`,
`en_wikipedia_org_wiki_Klabberjass`, `pagat_com_jass_clabber_html`, `pagat_com_jass_klaber_html` (Israël / klabberjass) ·
`eksisozluk_com_belot_1389371`, `pagat_com_national_turkey_html`, `pagat_com_jass_tartli_html` (Turquie) ·
`pagat_com_national_armenia_html`, `avagyanp_blogspot_com_…_bazaar_blot_…`, `gambler_ru_Bazar_belote`,
`ru_wikipedia_org_wiki_Белот` (Arménie / Russie — le *bazar-blot*) ·
`ro_wikipedia_org_wiki_Belotă` (Roumanie) ·
`de_wikipedia_org_wiki_Jass`, `en_wikipedia_org_wiki_Jass`, `pagat_com_national_switzerland_html`,
`pagat_com_jass_schieber_html`, `pagat_com_jass_coiffeur_html`, `pagat_com_jass_differenzler_html`,
`pagat_com_jass_mittlere_html`, `pagat_com_jass_handjass_html`, `pagat_com_jass_swjass_html`,
`pagat_com_jass_alsos_html`, `pagat_com_jass_derda_html`, `pagat_com_jass_jojotte_html`,
`pagat_com_jass_sidi_barrani_html`, `pagat_com_jass_thunee_html`, `swisslos_ch_de_…` (×2),
`kleeblatt_jass_jimdoweb_com_…`, `igrakarta_com_belote`, `torofun_com_en_belote_rules`,
`home_belote_club_en`, `favorite_games_com_html*_rules_belot4_php` (×3),
`dimitrovmitko93_wixsite_com_belot_blank_tunt8`, `jouer_au_belote_ueuo_com`.

---

## Applications et plateformes — `apps-sites/` (143 nouvelles)

L'intérêt de ce dossier n'est pas la règle « officielle » mais **la liste des options
paramétrables** : c'est le meilleur inventaire de variantes existant, parce qu'un éditeur ne code
une option que si des joueurs la réclament.

### Les plus riches en options de règles

| Fichier | URL | Apport |
|---|---|---|
| `iscool_helpshift_com_hc_fr_17_belote_facebook_faq_504_game_options` | iscool.helpshift.com — *Options de jeu* | Options d'interface (langue, chat, affichage) et non de règles — utile pour délimiter ce qu'un éditeur laisse **hors** du champ paramétrable |
| `iscool_helpshift_com_hc_fr_17_belote_facebook_faq_701_coinche_with_announces_at_nt` · `…_10_belote_mobile_faq_618_coinche_declarations_at_nt` | iscool.helpshift.com | **Coinche avec annonces + TA/SA** telle qu'un éditeur la code réellement, y compris la renormalisation du paquet |
| `coinche_en_ligne_com_coinche_lyonnaise` | coinche-en-ligne.com | **Coinche lyonnaise : on joue avec un « chien »** (talon), variante régionale nommée qu'aucune autre source du corpus ne décrit |
| `coinche_en_ligne_com_pierre_coinche` | coinche-en-ligne.com | La « pierre » : technique de refus du pli — vocabulaire de table absent de tous les règlements |
| `coinche_en_ligne_com_generale_coinche` · `_capot_coinche` · `_surcoinche` · `_annonce_contree` · `_difference_entre_belote_coinchee_et_belote_contree` · `_regles_jeu_belote_coinchee` | coinche-en-ligne.com | Une page par mécanisme de score : capot, générale, surcoinche — chacune avec ses chiffres |
| `la_coinche_fr_variantes_bresilienne_stephanoise_bridgee_php` | la-coinche.fr | **Coinche brésilienne, stéphanoise et bridgée** en une page |
| `jeudebelote_org_tout_atout_belote` · `jeudebelote_org_belote_a_la_vache` · `belote_tv_belote_vache` · `ludi_com_…_belote_a_la_vache_aspx` | — | **Belote « à la vache » : prise obligatoire du donneur** si tout le monde passe — une variante qui supprime la donne nulle, plus le Tout Atout chiffré |
| `gameduell_helpshift_com_hc_fr_16_…_faq_1056_contree` · `_faq_1053_classic_belote` · `_faq_1055_belote_with_melds` · `gameduell_fr_gd_belote_belote_contree_html` · `_belote_classique_html` | belote.com / GameDuell | **La contrée telle que la plus grosse plateforme francophone la code**, en français cette fois (le corpus n'avait que la version anglaise) |
| `iscool_helpshift_com_hc_fr_10_belote_mobile_faq_159_…_various_tables` | iscool.helpshift.com | Décrit les **types de tables** et donc les régimes de règles proposés en parallèle |
| `ludicash_com_help_rules_belote_contree` · `_coinche` · `_belote` | ludicash.com | Trois règlements distincts chez le même opérateur — la preuve que « coinche » et « contrée » sont deux produits séparés |
| `re_belote_fr_prise_et_contrat` · `_les_annonces` · `_regles_de_la_belote` | re-belote.fr | Barème d'annonces et mécanique de contrat |
| `vareversat`/`StephaneBg` côté code (voir open source) + `play_google_com_…_com_sbgapps_scoreit` · `…_net_aasuited_belotescore` · `…_com_nico_coinche` · `belotescore_com_fr` · `belote_score_fr` · `belotepoint_fr` · `getscory_com_en_score_keeper_belote` · `apps_apple_com_fr_app_compteur_belote_coinche_id6444708615` · `apps_apple_com_us_app_bela_blok_pro_belote_tracker_id1508462578` · `allbestapps_fr_…_coincoinche` · `iphoneaddict_fr_…_coinchette_…` | compteurs de points | **Les compteurs de points sont la meilleure source de barème** : ils doivent trancher chaque cas, et exposent l'arrondi en option de partie |
| `contree_org_4_joueurs` · `coinche_contree_com_les_regles_de_la_contree` · `coinchee_fr_coinche_en_ligne` · `coinchegratuit_fr_…` | sites dédiés contrée | Rédactions indépendantes du barème de contrée |
| `fr_boardgamearena_com_doc_Gamehelpbelote` · `_Gamehelpkqj` · `en_boardgamearena_com_gamepanel_game_coinche` · `_belote` · `forum_boardgamearena_com_viewtopic_php_t_15305` | BoardGameArena | **Les options de table BGA** + un fil de forum où les joueurs discutent des écarts de barème |
| `eryodsoft_com_fr_jeux_belote` · `_en_games_belote` · `_en_games_manille` | eryodsoft.com | Les autres jeux de l'éditeur le plus riche en options ; la **manille coinchée** entre au corpus |
| `cliquojeux_com_belote_contree` · `_belote` · `_belote_duel` · `alkiom_fr_belote_regles_variantes_accessibles_cliquojeux` | cliquojeux.com | Trois régimes de règles chez un même opérateur, dont un duel à 2 |
| `my_belote_fr_documentation_regles_du_jeu` · `_foire_a_questions` | my-belote.fr | Documentation + FAQ d'un opérateur français |
| `gamevelvet_com_coinche_online_rules` · `_belote_online_rules` · `gamesarena_io_belote` · `gametwist_com_en_skill_games_belote` · `solitaireparadise_com_…` (×4) · `solitairebliss_com_belote` · `funbelote_com_en` · `funbridge_com_blog_fr_funbelote_…` · `belote_en_ligne_fr` · `belote_en_ligne_eu_…` · `belote_en_ligne_org_les_variantes_de_la_belote` · `belote_rebelote_fr` + `_regles_html` · `jeudebelote_fr` · `jeu_belote_fr_regles_php` · `web_internet_fr_…` · `torofun_com_en_belote_rules` · `belot_pro_en` · `ibelote_com_en_rules_coinche_php` · `gamerules_com_rules_belote` · `regles_de_jeux_com_regle_coinche` · `exoty_com_regles_contree_belote` · `clubdejeux_com_belote_online_regles` · `vipbelote_fr` + 3 pages blog | plateformes | Couverture large ; barème conforme au consensus, **sauf VIP Belote qui donne explicitement surcoinche = ×3** (« triplés de l'annonce initiale ») |
| Fiches **App Store** (18) et **Google Play** (16) : `apps_apple_com_*`, `play_google_com_*`, plus `apkcombo_com_…`, `amazon_fr_Eryod_Soft_…`, `casualino_…_aptoide_com_app`, `french_coinche_belote_contree_free_ios_soft112_com`, `iphonesoft_fr_…`, `iphoneaddict_fr_apps_jeux_la_coinche_…` | magasins d'apps | Les descriptions **énumèrent les options** (annonces on/off, TA/SA, coinche/surcoinche, seuil de partie 501/1000/2000/3000) — c'est là que se lit l'espace des variantes réellement offert |

---

## Open source — `open-source/` (113 nouveaux fichiers)

Sélection par intérêt sur le **barème** (le dossier complet fait 125 fichiers).

| Fichier | Apport / divergence |
|---|---|
| `ElysiumDisc_belote_master_src_belote_scoring.py` + `_config.py` + `tests_test_official_rules.py` + `tests_test_scoring_totals.py` | **Le mieux testé du corpus** : des tests qui asserent explicitement les totaux de donne — exactement l'invariant que Colver épingle |
| `sebastien-perpignane_cardgame_master_…_DealScoreCalculator.java` + `…DealScoreCalculatorTest.java` + `…ContreeGameConfig.java` + `…ContreeBidValue.java` | Contrée en Java avec **configuration de partie séparée du calcul** : l'inventaire des paramètres est isolé |
| `eddy-geek_coinchounet_main_server_src_rules_scoring.ts` + `tests_scoring.spec.ts` + `specs_rules.md` | Spécification écrite **et** suite de tests de score du même auteur |
| `vareversat_carg_main_lib_models_…` (10 fichiers) | Compteur de points Flutter qui modélise **coinche et contrée comme deux barèmes distincts**, avec `game_setting` paramétrable — le meilleur inventaire d'options côté code |
| `StephaneBg_ScoreIt_master_…_CoincheSolver.kt` + `BeloteSolver.kt` + `CoincheValue.kt` + `BeloteBonusValue.kt` | Idem en Kotlin ; les `Value` énumèrent les contrats admis |
| `ilyesbrh_twistedFate-belote_main_packages_tunisian_core_src_models_scoring.ts` | **Barème tunisien codé séparément du barème français** dans le même projet |
| `ilyesbrh_…_packages_coinche_core_src_models_scoring.ts` + `…_capot-scoring.test.ts` | Le test dédié au capot isole les cas limites |
| `Jcat2b_BeloteCorse_main_src_utils_gameRules.ts` + `…gameRules.test.ts` | **Belote corse implémentée**, avec ses règles propres |
| `ocentra_…_jass_coinche_coincheScoring.asset` + `coincheRules.asset` + `processed-games_jass_{coinche,belote,baloot}.json` | Règles **sérialisées en données** (et non en code) pour trois variantes, dont le baloot — format directement comparable |
| `Omxr-Xg_quantum-bluff_develop_server_src_logic_belote_{scoring,conteeBidding,conteeConstants}.ts` | Constantes de contrée isolées dans un fichier dédié |
| `christophecraig_coinchette_main_.claudefiles_RULES.md` + `_DECISIONS.md` + `lib_…_score.ex` | Un projet qui **documente ses arbitrages de règles** à côté du code (Elixir) |
| `christophecraig_megacoinche_master_base-knowledge_rules.md` | Base de connaissances de règles du même auteur |
| `mondary_PKcards_main_assets_rules_*` (7 fichiers) | **Sept rédactions concurrentes** de règles belote/coinche/manille coinchée rassemblées dans un même dépôt, dont deux issues d'éditeurs papier (`edimag100`, `fetjain32`) |
| `giretra_giretra_main_src_…_ScoringCalculator.cs` + `DealResult.cs` + `docs_RULES.md` | Contrée en C# avec doc de règles |
| `s4mKa7a_Belote.MG_main_…_HandScorer.cs` + `HandScorerTests.cs` + `BeloteRules.cs` | Scoring + tests en C# |
| `pawelangelow_belote_main_packages_engine_src_{scoring,bidding,declarations}.ts` + `scoring.test.ts` + `RULES.md` | Moteur **de belote bulgare** en TypeScript — la conversion en points marqués est dans le code |
| `InCogNiTo124_BelaBot_master_belabot_engine_declarations.py` | **Bela croate** : les déclarations codées |
| `MrUnBaiat_BelotAI_master_Documents_gameRules.md` | Règles de belot roumain |
| `pierremaker1_blot-hayastan_main_README.md` | **Blot arménien** |
| `yoniBenabou_TwoPlayersBelote_main_README.md` · `yuta-yoshinaga_go_trumpcards_…_belote.md` + `BeloteConfig.go` | Belote à 2 joueurs ; config de règles en Go |
| `Ten0_coinche-server-rs_master_src_game_points.rs` + `contract_bid.rs` | **Seconde implémentation Rust** du barème après `libcoinche` — comparaison directe possible |
| `Alounv_coinche-back_master_domain_4-counting.go` + `_test.go` + `2-bidding.go` | Coinche en Go, comptage testé |
| `Oliboy50_coinche_master_…_pointsCounter.ts` + `.test.ts` · `mattr13000_contree_main_shared_scoring.ts` + `server_rules.ts` · `lkeegan_belote_main_worker_src_game_scoring.ts` · `Raphael-Bely_coinche-app_master_server_game_Scoring.js` · `sofian13_cbgames_main_party_games_contree.ts` · `TomBodard_belote-score-analyzer_main_src_utils_beloteUtils.ts` · `Yoan30_belote_main_src_domain_logic_beloteRebelote.ts` | Implémentations JS/TS variées du calcul de donne |
| `coincoinche_…_engine_contracts_{Contract,ContractCapot,ContractGenerale}.java` + `engine_README.md` + `web-ui_…_Rules.tsx` | **Capot et générale modélisés comme des contrats à part entière**, avec leur propre valeur |
| `CephaloSophie_kydos_master_packages_core_src_rules_ContreeRules.ts` | Le pendant « règles » du `donneScoring.ts` déjà collecté |
| `tokou_coinche_master_…_coinche_Bidding.kt` · `SlimSeb_Elsa-Mina_…_Belote{Constants,Scorer}.cs` · `SFScorpio_contree_master_class_hand_score.py` · `iSeaox_SansCoeurCDX_main_services_scores.py` · `pipoteam_pipobot-modules_master_belotebot_game.py` · `MrRaph_matrix-belote_main_belote.py` + `README.md` · `abaivel_belote_game_main_backend_lib_belote.php` · `Hm5s10_site_jeux_main_games_belote_game_logic.php` · `loic-vial_belote_master_…_Belote.java` · `AugustetheAuguste_COINCHE_DL_main_Environnement_coinche.py` · `BellaajMohsen7_Soff_main_streamlit_belote_app.py` · `ElysiumDisc_belote_master_src_belote_rules.py` | Calculs de score et moteurs divers |
| `pfajeau_rom-whist_master_romwhist_contree_{contree,announce}.py` | **Contrée greffée sur un moteur de whist** — filiation whist/coinche visible dans le code |
| `Hawkynt_…_card-games_belote-variant.js` | Un fichier qui énumère des **variantes** de belote |
| `drasill_bga-coinche_master_docs_rules-{fr,en}.md` · `slim0_contree_main_docs_game-rules.md` · `newtondotcom_coinche_main_apps_web_content_rules.md` · `Brawdunoir_coinche_main_README.md` · `LooperSalty_Belotro_main_README.md` · `racettour_Belote-contree-game_main_README.md` · `ebouda33_beloteMobile_main_docs_specifications-belote.md` · `CyberFoxar_CardWeb_master_…_fr_belote.md` | Documents de règles de dépôts |

---

## À ne pas recompter — copies du texte FFB

Ces sources reprennent le règlement FFB mot pour mot ou à la virgule près. Elles attestent de la
diffusion du texte, **pas d'un usage indépendant** :

- `tournois/fnasce_org_2016_concours_de_belote_contree_coinche_a36693_html` (page-mère des PDF FNASCE déjà indexés)
- `tournois/s1_static_footeo_com_uploads_fcplouay_…` (une seule règle de score, reprise FFB)
- `tournois/lesamisdutempslibrevarennes_…`, `tournois/cdfcasson_fr_…`, `tournois/pontdeclaix_fr_…`, `tournois/sc4e58b2fce8a2e7a_jimcontent_com_…` — même squelette de règlement de concours de belote (« mise dedans 162 / capot 252 / belote 20 / sans annonces sauf belote ») ; leurs **formats de partie** diffèrent en revanche réellement (4×12, 5×10) et c'est à ce titre qu'ils sont retenus
- `divers/cartesetcie_fr_les_regles_de_la_coinche`, `divers/regles_com_jeux_cartes_belote_html`, `divers/cartesenmain_fr_…`, `divers/concours_de_belote_fr_…`, `divers/chessclubanderlecht_org_…` — vulgarisation à partir du texte FFB
- `apps-sites/gamerules_com_rules_belote`, `apps-sites/regles_de_jeux_com_regle_coinche`, `apps-sites/solitaireparadise_com_games_list_*`, `apps-sites/solitairebliss_com_belote`, `apps-sites/gamesarena_io_belote` — reprises de plateformes, sans barème propre

## Échecs (à ne pas retenter tels quels)

- `aubignanfaitses24h.fr/…/reglement-officiel.pdf` — le serveur renvoie **520** sur les deux chemins connus et **aucun instantané Wayback n'existe** ; c'est le seul document de la première liste resté hors d'atteinte
- `beloteenligne.com` — **403** systématique (protection anti-bot), sur la racine comme sur `/les-indispensables/organiser-un-tournoi`
- `clubs-de-bridge.com/belote/belote-coinchee/*` — **500** sur toutes les pages de règles
- `club.fft.fr/lefontaniltennis/…/aarglementtournoicoinche2024_v3.pdf` — coquille « JavaScript requis », supprimée
- `regles2jeux.fr`, `123belote.com` — connexion refusée (certificat / timeout)
- `muscletacoinche.wixsite.com` racine et `jouonsavaugines.wixsite.com/lubenjeux/reglement` — coquilles Wix sans contenu (la page de règlement utile a été trouvée ailleurs sur le même site)
