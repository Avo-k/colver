# Matrice « qui dit quoi » — le jeu de la carte et la distribution

Établie le 2026-08-01 sur le corpus local de `data/rules-corpus/`, sans aucune recherche web.
Sujet : **distribution** et **jeu de la carte** uniquement. Les enchères et le décompte des
points relèvent d'autres matrices.

Chaque source est citée par son **nom de fichier** (chemin relatif à `data/rules-corpus/`) et
par son URL (première ligne du `.txt`, quand elle existe).

---

## 0. Préalable : les familles de copies

Le corpus contient beaucoup moins de témoignages indépendants que de fichiers. Ces
regroupements sont vérifiés phrase par phrase et sont **appliqués dans tous les tableaux
ci-dessous** : une famille y compte pour **une voix**, pas pour cinq.

| Famille | Fichiers | Statut |
|---|---|---|
| **FFB « rédaction 2015 »** | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` (contrée) · `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` (coinche) | Section 5.2 **identique mot pour mot**. Une seule voix. |
| **FFB « Équipe Ludique »** | `federations/LOCAL_regles_officielles_belote_contree.txt` | Ré-édition de la rédaction 2015, avec **trois mutations** (mélange, règle 4, arrondi). Voix distincte là où elle diverge. |
| **FFB « 27.01.2016 »** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` (contrée) · `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` (belote classique) | Même squelette, même numérotation — mais **§5.2.2 dit l'inverse d'un fichier à l'autre**. Deux voix, pas une. |
| ↳ copie | `divers/cartesetcie_fr_regles_officielles_de_la_belote.txt` | Copie verbatim de la belote 2016. **Pas un témoignage.** |
| **FFB « pages web »** (rédaction en 9 règles numérotées) | `federations/ffbelote_org_belote_contree.txt` · `federations/ffbelote_org_regles_coinche.txt` · `federations/ffbelote_org_reglements_de_la_belote_avec_ou_sans_annonce.txt` | Même texte à l'identique **sauf** la règle 4/5 (voir axe 9). Les pages contrée et coinche sont une seule voix ; la page belote en est une autre. |
| ↳ copies | `tournois/cdf_missegre11_com_medias_files_belote_contre_e_pdf.txt` · `tournois/villeconin_fr_..._Informations_sur_le_jeu.txt` · `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` · `divers/carafons_fr_regles_de_la_coinche.txt` | **Pas des témoignages indépendants.** Signalés seulement là où ils *mutent* le texte source. |
| **belotecontree.free.fr « tournoi international »** | `divers/belotecontree_free_reglement.txt` · `tournois/ainesruraux_saintsever_com_belote_BELOTE_20TRADITIONNELLE_pdf.txt` | Copie verbatim vérifiée. Une seule voix. |
| **Pagat (John McLeod)** | `divers/pagat_com_jass_coinche_html.txt` · `divers/pagat_com_jass_belote_html.txt` | Deux pages, deux jeux, même auteur — **cohérentes entre elles**, comptées comme une voix. |
| ↳ copies | `apps-sites/gambiter_com_cards_jass_coinche_html.txt` (copie anglaise verbatim) · `divers/reglesdejeux_github_io_regles_du_jeu_la_coinche_index_html.txt` (traduction automatique en français) | **Pas des témoignages.** |
| **Concours ASCEE / DDT 79** | `tournois/fnasce_org_IMG_pdf_Belote_Reglement_cle19a7cf_pdf.txt` · `tournois/fnasce_org_IMG_pdf_reglement_belote_pdf.txt` · `tournois/pontdeclaix_fr_..._rglementconcoursdebelote_pdf.txt` | Articles 6-17 identiques mot pour mot. Une voix. `tournois/web_myassoc_org_..._pdf.txt` (Lions Club Bigorre) en est une reformulation très proche. |
| **Concours ASCEE 2A (coinche)** | `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` · `tournois/fnasce_org_IMG_pdf_belote_reglement_cle1a43c7_pdf.txt` | Deux millésimes du même texte (1 000 → 1 010 points). Une voix. |
| **Concours « coupe à droite / roi en premier »** | `tournois/cdfcasson_fr_..._pdf.txt` · `tournois/sc4e58b2fce8a2e7a_jimcontent_com_..._Belote_20R_C3_A8glement_2.txt` · `tournois/lesamisdutempslibrevarennes_jimdofree_com_...concours_belote.txt` · `tournois/rjcv_be_belote_regles_pdf.txt` (Belgique) | Articles quasi identiques (« La coupe est obligatoire (minimum 3 cartes) et se fait à droite », « obligé de jouer Belote avec le Roi en premier »). Une voix, avec une extension belge. |
| **ange.heureux.free.fr** | `divers/ange_heureux_free_fr_JeuxDeCartes_La_Coinche_html.txt` · `apps-sites/ange_heureux_free_fr_Jeux_LaCoinche_html.txt` | Deux URL du même site, même texte. Une voix. |
| **iscool / Belote Mobile** | `divers/iscool_helpshift_com_..._faq_157_....txt` · `apps-sites/iscool_helpshift_com_hc_fr_17_belote_facebook_faq_497_game_rules_coinche.txt` | Même FAQ éditeur. Une voix. |

**Source secondaire, jamais citée comme règlement** :
`open-source/ilyesbrh_twistedFate-belote_main_docs_games_coinche_SOURCES.md` est une matrice
déjà construite par un agent en 2026-05. Elle n'est mentionnée nulle part ci-dessous comme
autorité ; elle a seulement servi de contrôle de recoupement (et confirme les deux mêmes
lignes de fracture : O5 « monter sur le partenaire » et le statut du sous-coup).

**Rappel de méthode** : les tableaux distinguent *ce qu'une source affirme* de *ce sur quoi
elle est muette*. Une source muette n'entre dans aucune position ; le cas échéant elle est
listée à part.

---

## 1. Sens de la donne et du jeu · rotation du donneur

| Position | Sources | Extrait |
|---|---|---|
| **Antihoraire** (donne, parole, jeu ; donneur suivant = voisin de droite) | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` · famille FFB 2015 · famille FFB pages web (https://www.ffbelote.org/belote-contree/) · `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` (https://fr.wikipedia.org/wiki/Belote_contr%C3%A9e) · `divers/fr_wikipedia_org_wiki_Belote.txt` (https://fr.wikipedia.org/wiki/Belote) · `divers/pagat_com_jass_coinche_html.txt` (https://www.pagat.com/jass/coinche.html) · `divers/belotecontree_free_reglement.txt` · famille ASCEE 2A (https://www.fnasce.org/IMG/pdf/reglement.pdf) · `apps-sites/en_wikipedia_org_wiki_Coinche.txt` (https://en.wikipedia.org/wiki/Coinche) · `apps-sites/gamerules_com_rules_coinche.txt` (https://gamerules.com/rules/coinche/) · `divers/drasill_github_io_bga_coinche_rules_fr_html.txt` · `open-source/valmathieu_ContrAI_main_contree-domain.md` · `tournois/web_archive_org_..._coinche_stephanoise_com_..._reglement_coinche_pdf.txt` · `apps-sites/coinchegratuit_fr_....txt` | « Le donneur distribue 8 cartes à chaque joueur, **dans le sens inverse des aiguilles d'une montre** en commençant par son voisin de droite » (FFB contrée 2016, §3.4). « Tout se fait dans le sens contraire des aiguilles d'une montre (distribution, enchères, jeu) » (Wikipédia contrée). « Deal and play are **anticlockwise** » (Pagat). |
| **Horaire** | `tournois/tcvb_bruche_free_fr_..._tournoibelote2012reglement_htm.txt` (http://tcvb.bruche.free.fr/dossomma/tournoibelotte2012/tournoibelote2012reglement.htm) · `tournois/web_myassoc_org_..._pdf.txt` (Lions Club) · `tournois/geraudotloisirs_free_fr_....txt` · famille ASCEE/DDT 79 · famille « coupe à droite » (cdfcasson, jimcontent, Varennes, rjcv.be) · `divers/alhoa_free_fr_ALH_belote_rules_htm.txt` · `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` (https://casimirdehauteclocque.fr/jeux/coinche.pdf) · `tournois/clublafontainedejouvence_fr_r%C3%A8glement_coinch%C3%A9e.txt` · `apps-sites/gameduell_helpshift_com_..._faq_1056_contree.txt` · famille iscool · `divers/belotepoint_fr_regles_coinche.txt` · `divers/lemagloisirs_fr_regle_coinche.txt` · `divers/exoty_com_regles_coinche_belote.txt` | « **LE SENS DE LA DONNE & DU JEU EST CELUI DES AIGUILLES D'UNE MONTRE** » (TCVB, art. 15). « Le jeu se joue dans le sens des aiguilles d'une montre » (Lions Club, art. 12 ; Géraudot, art. 2). « La distribution des cartes se fait dans le sens des aiguilles d'une montre » (ASCEE 79, art. 6). « La Contrée se joue […] dans le sens des aiguilles d'une montre » (Belote.com). |
| **Les deux, au choix** | `divers/adpoker_fr_belote_contree_html.txt` (https://www.adpoker.fr/belote-contree.html) · `divers/pagat_com_jass_belote_html.txt` · `apps-sites/eryodsoft_com_fr_jeux_coinche.txt` (option de l'app) | « Donne : […] On peut jouer aussi **dans le sens des aiguilles d'une montre**. » (adpoker, section « Variante »). « People often play the whole game clockwise, rather than anticlockwise. » (Pagat belote, « Variations »). « **Sens de jeu** » figure dans la liste des options d'Eryod Soft (`apps-sites/play_google_com_..._eryodsoft_....txt`). |

**Divergence** — mais la ligne de fracture n'est pas géographique, elle est **entre les deux jeux** :
tout ce qui se présente comme *contrée* ou *coinche* dit antihoraire ; le camp horaire est
presque intégralement composé de **règlements de concours de belote classique** (Alsace,
Pyrénées, Aube, Deux-Sèvres, Isère, Belgique) plus une grappe de sites de contenu SEO
(`belotepoint`, `lemagloisirs`, `exoty`) qui décrivent la coinche en recopiant un patron de
belote. Deux exceptions réelles au sein de la coinche : `casimirdehauteclocque` et
`clublafontainedejouvence`, deux règlements de club qui assument l'horaire.

**Attention à un piège de lecture** : `divers/adpoker_fr_belote_contree_html.txt` se contredit
lui-même — il annonce une donne antihoraire « en commençant par celui placé à sa droite »,
puis écrit dans la section « Le jeu » : « Le joueur placé **à la gauche** du donneur entame
la première levée ». C'est une incohérence interne, pas une position.

---

## 2. Mélange des cartes

| Position | Sources | Extrait |
|---|---|---|
| **Obligatoire avant chaque donne** | famille FFB 2015 · famille FFB 27.01.2016 · famille FFB pages web | « Le mélange du jeu est **obligatoire** avant chaque distribution. » (FFB contrée 2016, §3.2). La page web argumente : « Il y a bien longtemps, il était interdit de mélanger […] La recrudescence de ces cas de triche a rendu le mélange non plus optionnel mais **OBLIGATOIRE** ». |
| **Au vouloir du donneur** | `federations/LOCAL_regles_officielles_belote_contree.txt` · `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` · `tournois/cdf_missegre11_com_..._belote_contre_e_pdf.txt` · `tournois/maisondesessarts_fr_article116_html.txt` · `divers/adpoker_fr_belote_contree_html.txt` · `divers/pagat_com_jass_coinche_html.txt` · famille « coupe à droite » (cdfcasson art. 12) | « Le mélange du jeu est **au vouloir du donneur** avant chaque distribution. » (LOCAL/Équipe Ludique, §3.2 — mutation directe du texte 2015). « Les cartes doivent être mélangées ou non, **à la discrétion du donneur**. » (Wikipédia contrée). « À chaque tour, **seul le donneur, s'il le souhaite** peut mélanger les cartes. » (Maison des Essarts, art. 2). « Le mélange des cartes avant chaque distribution est **optionnel**. » (Missègre — mutation de la page FFB qu'il recopie par ailleurs). |
| **Interdit, sauf circonstances énumérées** | `divers/belotecontree_free_reglement.txt` · `tournois/web_myassoc_org_..._pdf.txt` · `tournois/geraudotloisirs_free_fr_....txt` · famille ange.heureux · `apps-sites/en_wikipedia_org_wiki_Coinche.txt` | « Le donneur **n'est pas obligé de mélanger** les cartes **sauf avant la première mène** ou après un incident de jeu […] Chaque fois que les cartes ont été exposées sur le tapis […] le mélange est alors obligatoire. » (belotecontree.free.fr). « Il est **interdit de brasser les cartes** pendant la partie. » (Lions Club, art. 12 ; Géraudot, art. 11). « le jeu **ne doit jamais être mélangé**, hormis avant le début d'une nouvelle manche » (ange.heureux). « The deck is **never shuffled**, but rather cut by the player who precedes the dealer » (Wikipédia EN Coinche). |
| **Recommandé** (mutation) | `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` | Recopie la page FFB coinche mais remplace « OBLIGATOIRE » par « **fortement recommandé** ». |

**Divergence, et c'est la plus ancienne du corpus.** La fracture passe entre **la fédération
(mélange obligatoire, justifié explicitement par l'anti-triche)** et **la tradition de table
(ne pas mélanger, pour pouvoir lire la donne suivante dans les plis de la précédente)**.
Pagat explique le mobile de cette tradition : « If the cards are not shuffled, players may
use their observation of the order in which cards were played in the previous deal […] to
estimate the likely distribution of cards in the current deal. » Fait notable : la FFB
elle-même *raconte* qu'elle a changé d'avis, et l'édition « Équipe Ludique » du même texte
a **retourné la règle** en « au vouloir ».

---

## 3. La coupe : obligatoire ? par qui ? taille minimale des tas ?

| Position | Sources | Extrait |
|---|---|---|
| **Obligatoire, par le joueur à gauche du donneur, minimum 3 cartes** | famille FFB 2015 · famille FFB 27.01.2016 · famille FFB pages web (+ copies Missègre, Villeconin, cartesetcie, carafons) · `apps-sites/en_wikipedia_org_wiki_Coinche.txt` (min. 3, mais par « le joueur qui précède le donneur ») | « Le joueur situé **à la gauche du donneur** effectue une coupe en 2 de manière franche sans compter le nombre de cartes qu'il souhaite couper. **Chaque tas doit contenir au moins 3 cartes.** Le donneur referme la coupe. » (FFB contrée 2016, §3.3). « At least three cards must be cut. » (Wikipédia EN). |
| **Obligatoire, minimum 2 cartes** | `tournois/data_over_blog_kiwi_com_..._reglement_table_coinche_pdf.txt` (AIL Manissieux) · `tournois/web_archive_org_..._coinche_stephanoise_com_..._reglement_coinche_pdf.txt` | « le jeu sera coupé (**avec un minimum de 2 cartes**) par le joueur a gauche du donneur » (AIL Manissieux, art. 4). « Le jou[eur] placé à gauche du donneur devra **impérativement** couper le jeu **avec un minimum de deux cartes**. » (Coinche stéphanoise). |
| **Obligatoire, un seul tas ne peut pas être réduit à 1 carte** | `divers/belotecontree_free_reglement.txt` | « il doit faire couper le joueur situé à sa gauche **qui ne peut refuser** (la coupe est obligatoire). Celui-ci doit séparer le paquet de carte en 2 **sans que l'un ne soit constitué que d'une seule carte**. » |
| **Obligatoire, mais coupée à droite** | famille « coupe à droite » (cdfcasson, jimcontent, Varennes, rjcv.be) | « La coupe est obligatoire (minimum 3 cartes) et **se fait à droite** » (rjcv.be ; cdfcasson art. 3). Cohérent avec le sens horaire que cette même famille impose (axe 1). |
| **Coupe par « l'adversaire qui précède le donneur »** | famille ange.heureux · `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` | « Une fois le jeu **coupé par l'adversaire qui le précède**, le joueur distribue 8 cartes » (ange.heureux). « les fait couper **au joueur avant lui** » (casimir). |
| **Muettes sur la taille minimale** | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt`, `divers/pagat_com_jass_coinche_html.txt`, `divers/adpoker_fr_belote_contree_html.txt`, `divers/lemagloisirs_fr_regle_coinche.txt` | Toutes affirment la coupe obligatoire (« it is obligatory for the dealer's left-hand opponent to cut the cards », Pagat) mais ne fixent aucun seuil. |

**Consensus sur le principe, divergence sur le chiffre.** La coupe est **obligatoire partout**
où le corpus en parle — c'est une des rares règles à ne connaître aucune exception, y compris
chez les sources qui interdisent le mélange (belotecontree : « la coupe est obligatoire »).
La fracture est sur le seuil (**3** chez la FFB et tout ce qui en descend, **2** dans deux
règlements de tournoi rhônalpins, **« pas une seule carte »** chez belotecontree) et sur le
**côté** (gauche partout sauf dans la famille horaire, où elle passe mécaniquement à droite).

Deux précisions qu'une seule source donne chacune, et qui ne sont contredites par personne :
- **remontage** : « Le donneur doit reconstituer un seul paquet en posant **sur le dessus le paquet qui se situait, avant la coupe, au dessous** » (`divers/belotecontree_free_reglement.txt`), soit « en inversant l'ordre de coupe » (`divers/fr_wikipedia_org_wiki_Belote.txt`) ;
- **coupe franche** : « n'a pas le droit de **relâcher les cartes** pour définir exactement le nombre de cartes qu'il souhaite couper » (famille FFB pages web).

---

## 4. Découpage de la donne (3-2-3 / 3-3-2 / 2-3-3 / autres)

| Position | Sources | Extrait |
|---|---|---|
| **Les trois combinaisons de 3 et 2, au choix** | famille FFB 2015 · famille FFB 27.01.2016 · `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` · `divers/pagat_com_jass_coinche_html.txt` · `apps-sites/en_wikipedia_org_wiki_Coinche.txt` · famille ange.heureux · `divers/regles_com_jeux_cartes_coinche_html.txt` · `divers/adpoker_fr_belote_contree_html.txt` (au titre des variantes) | « de l'une des manières suivantes : **3 cartes chacun puis 2 puis 3, ou 3 cartes puis 3 puis 2, ou encore 2 cartes puis 3 puis 3.** » (FFB contrée 2016, §3.4) |
| **3-2-3 seulement** | `divers/clubdejeux_com_belote_coinchee_online_regles.txt` · `divers/bk_jeux_ducale_fr_..._belote_coinchee_pour_joueur_expert_pdf.txt` · `divers/lemagloisirs_fr_regle_coinche.txt` · `tournois/data_over_blog_kiwi_com_..._reglement_table_coinche_pdf.txt` · `tournois/clublafontainedejouvence_fr_....txt` · `apps-sites/gameduell_helpshift_com_..._faq_1056_contree.txt` | « trois cartes à chacun, puis deux cartes, puis trois cartes » (AIL Manissieux, art. 4) |
| **Par 2 ou par 3, toutes variantes** | famille ASCEE 2A | « le donneur distribue les cartes **par groupe de 2 ou 3 (toutes variantes autorisées)**. » |
| **4-4 admis** | `tournois/cdf_missegre11_com_..._belote_contre_e_pdf.txt` · `divers/alhoa_free_fr_ALH_belote_rules_htm.txt` · `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` · `divers/pagat_com_jass_coinche_html.txt` (comme variante) · `open-source/valmathieu_ContrAI_main_contree-domain.md` | « – En 3 fois : 3 cartes chacun puis 2 ou inversement. – **En 2 fois : 4 cartes à la fois.** » (Missègre). « en 4-4 **existe dans le sud-est** » (alhoa). « **Corsica deal**: 4-4 dealing pattern instead of 3-2-3. » (ContrAI). |
| **4 cartes explicitement interdit** | famille FFB pages web | « Il est **strictement interdit** de distribuer 1 ou 4 cartes à la fois. » (https://www.ffbelote.org/belote-contree/) |
| **Par 2 minimum, par 3 maximum, 3 tours maximum** | `divers/belotecontree_free_reglement.txt` | « de manière égale pour chaque joueur, **par deux minimum, par trois au maximum, sans pouvoir excéder 3 tours de donne**. » |
| **Donne en deux temps (6 cartes, enchères, puis 2)** | `tournois/web_archive_org_..._coinche_stephanoise_com_..._reglement_coinche_pdf.txt` · `divers/fr_wikipedia_org_wiki_Coinche.txt` · `divers/pagat_com_jass_coinche_html.txt` (variante) · `apps-sites/en_wikipedia_org_wiki_Coinche.txt` (variante) | « Dans la **variante stéphanoise** (Loire) également appelée **beloinche**, six cartes sont données à chacun, les annonces sont faites, et deux cartes sont ensuite distribuées à chacun. » (Wikipédia Coinche) — et le règlement de la Coinche stéphanoise le confirme de première main : « La distribution des cartes doit se faire par la droite avec la remise de **deux fois trois cartes** à chaque joueur […] A la fin des enchères, le donneur distribue **deux cartes** à chaque joueur. » |

**Divergence de second ordre.** Personne ne conteste que les 32 cartes sont distribuées en
paquets ; la fracture porte sur ce qui est *toléré en plus* du triplet 3-2-3 / 3-3-2 / 2-3-3.
La FFB est la seule à **interdire** explicitement le 4-4, et un des règlements qui recopient
sa page (Missègre) l'autorise pourtant. Le cas stéphanois est à part : ce n'est plus la même
séquence de jeu, la donne encadre l'enchère.

Interdit accessoire, une seule source, non contredite : « Il est **strictement interdit aux
joueurs de ramasser leurs cartes avant la fin de la distribution**. »
(`federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt`, §3.4).

---

## 5. Qui parle en premier · qui entame le premier pli

| Position | Sources | Extrait |
|---|---|---|
| **Le joueur à droite du donneur parle *et* entame** | famille FFB 2015 · famille FFB 27.01.2016 · famille FFB pages web · `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` · `divers/pagat_com_jass_coinche_html.txt` · `divers/belotecontree_free_reglement.txt` · `divers/bk_jeux_ducale_fr_..._pdf.txt` · `open-source/valmathieu_ContrAI_main_contree-domain.md` · `tournois/data_over_blog_kiwi_com_....txt` · `tournois/web_archive_org_..._coinche_stephanoise_....txt` · `apps-sites/en_wikipedia_org_wiki_Coinche.txt` | « Le joueur situé **à droite du donneur** joue la carte de son choix : c'est l'entame. » (FFB contrée 2016, §5.1). « Le joueur placé à la droite du donneur (**c'est-à-dire celui qui a commencé les enchères**) pose la première carte du pli. » (Wikipédia contrée). « Le premier à parler étant le « **premier en carte** » c'est-à-dire celui situé tout de suite à la droite du donneur. » (belotecontree). |
| **Le joueur à gauche du donneur** (corollaire du sens horaire) | famille ASCEE/DDT 79 (implicite) · `divers/alhoa_free_fr_ALH_belote_rules_htm.txt` · `divers/belotepoint_fr_regles_coinche.txt` · `divers/lemagloisirs_fr_regle_coinche.txt` · `divers/exoty_com_regles_coinche_belote.txt` · famille iscool · `apps-sites/gameduell_helpshift_com_..._faq_1054_coinche.txt` · `tournois/clublafontainedejouvence_fr_....txt` · `apps-sites/coinchegratuit_fr_....txt` | « le joueur situé **à la gauche** du distributeur commence les enchères » (alhoa). « Le joueur situé **à gauche du donneur** entame le premier pli » (belotepoint). |
| **Le joueur qui suit le donneur** (formulation neutre) | famille ange.heureux | « Le premier joueur à jouer est **le joueur qui suit le donneur**, sauf en cas d'annonce "Générale", en quel cas c'est l'annonceur qui entame la partie. » |

**Consensus conditionnel.** Aucune source ne dissocie l'entame de l'ouverture des enchères :
**celui qui parle en premier entame**, sans exception dans tout le corpus. Le désaccord sur
gauche/droite n'est donc pas un axe autonome — il est **entièrement dérivé de l'axe 1**
(sens du jeu), et se répartit exactement selon les mêmes camps.

Exception unique et explicite, chez trois sources indépendantes : **l'annonceur d'une
« générale » entame**, quel que soit son siège. « il est autorisé à jouer en premier **même
s'il n'a pas la main** au premier tour. L'ordre normal des joueurs est conservé. »
(`divers/fr_wikipedia_org_wiki_Coinche.txt`) ; idem famille ange.heureux et
`divers/pagat_com_jass_coinche_html.txt` (« The bidder of a générale may have the right to
lead to the first trick »). La générale n'existe pas à la contrée FFB.

---

## 6. Obligation de fournir la couleur demandée

| Position | Sources | Extrait |
|---|---|---|
| **Obligatoire, sans exception** | **Toutes les sources du corpus qui décrivent le jeu de la carte.** Notamment famille FFB (toutes rédactions) · `divers/fr_wikipedia_org_wiki_Belote.txt` · `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` · `divers/pagat_com_jass_coinche_html.txt` · `divers/belotecontree_free_reglement.txt` · `open-source/valmathieu_ContrAI_main_contree-domain.md` · `open-source/drasill_bga-coinche_master_coinche.game.php` · `open-source/ismo009_Coinche_main_game.js` | « **On doit toujours fournir la couleur demandée à l'entame si l'on en possède.** » (FFB, formule identique dans les quatre rédactions). « Players must **follow suit** if they can. » (Pagat). « **Follow suit.** If you have any card in the led suit, you must play one. » (ContrAI). |

**Consensus, le seul total du corpus.** C'est la seule règle de jeu de la carte sur laquelle
aucune source, aucun tournoi, aucune app et aucune implémentation ne prévoit ni exception ni
variante. Un seul raffinement s'y greffe, non contradictoire : **si la couleur demandée est
l'atout, fournir ne suffit pas, il faut monter** — voir axe 8.

---

## 7. On ne peut pas fournir : faut-il couper ? et si le partenaire est maître ?

| Position | Sources | Extrait |
|---|---|---|
| **Obligation de couper, SAUF si le partenaire est maître (on peut alors se défausser)** | famille FFB 2015 · famille FFB 27.01.2016 · famille FFB pages web · `federations/LOCAL_regles_officielles_belote_contree.txt` · `divers/fr_wikipedia_org_wiki_Belote.txt` · `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` · `divers/pagat_com_jass_coinche_html.txt` · `divers/belotecontree_free_reglement.txt` · `tournois/maisondesessarts_fr_article116_html.txt` · `divers/adpoker_fr_belote_contree_html.txt` · `divers/drasill_github_io_bga_coinche_rules_fr_html.txt` · `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` · famille ASCEE 2A · `open-source/valmathieu_ContrAI_main_contree-domain.md` · `open-source/drasill_bga-coinche_master_coinche.game.php` · `open-source/ismo009_Coinche_main_game.js` · `apps-sites/gamerules_com_rules_coinche.txt` · `apps-sites/playjoy_com_en_coinche_rules.txt` · famille « coupe à droite » · `divers/jeux_regles_com_regles_coinche.txt` · famille iscool | « Si l'on ne possède pas de carte dans la couleur demandée, et que **notre partenaire est maître** […] on peut alors jouer n'importe quelle carte ; on se "défausse". On peut également jouer atout si bon nous semble. » puis « Si […] notre partenaire **n'est pas maître ou n'a pas encore joué** : on est **tenu de jouer un atout** si l'on en possède » (FFB 2015, §5.2 2.1–2.2). « If a player is unable to follow suit, and if the highest card in the trick was played by an opponent, he **must** play a trump. […] However, if the highest card in the trick so far was played by his partner, he is **allowed to discard** even if he has a trump. » (Pagat). « **Partner exception.** If your partner is currently winning the trick […] you are *not* obligated to trump or to overtrump. You may discard freely. » (ContrAI). « Il n'est pas obligatoire de couper **si le pli appartient à son partenaire** » (rjcv.be). |
| **Muettes sur l'exception partenaire** (elles n'énoncent que « on doit couper ») | `divers/exoty_com_regles_coinche_belote.txt` · `divers/bk_jeux_ducale_fr_..._pdf.txt` · `apps-sites/gameduell_helpshift_com_..._faq_1056_contree.txt` | « Couper : Si vous n'avez pas la couleur demandée, vous avez l'**obligation** de "couper" en jouant une carte d'Atout (si vous en avez une). » (exoty) — le cas du partenaire maître n'y est pas traité. **Silence, pas désaccord.** |
| **Position contraire explicite : couper même sur son partenaire** | *aucune source du corpus* | — |

**Consensus.** « Couper si l'adversaire tient le pli, se défausser librement si le partenaire
tient le pli » est, avec l'obligation de fournir, l'autre pilier non contesté. La formule
française consacrée que le corpus donne pour cette règle est **« on ne pisse pas sur le
partenaire »** (`divers/pagat_com_jass_belote_html.txt`).

Nuance à ne pas confondre avec l'axe 9 : ce que l'exception « partenaire maître » autorise
est la **défausse** ; savoir si elle autorise en plus le **sous-coup** (jouer un atout
inférieur à celui du partenaire) est une question distincte, traitée à l'axe 10.

---

## 8. Monter à l'atout quand l'atout est joué — y compris par-dessus son partenaire ?

| Position | Sources | Extrait |
|---|---|---|
| **STRICT : on monte toujours, même sur son partenaire** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` · `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` · famille FFB 2015 (règle 3) · `federations/LOCAL_regles_officielles_belote_contree.txt` · `divers/fr_wikipedia_org_wiki_Belote.txt` · `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` · `divers/pagat_com_jass_coinche_html.txt` + `divers/pagat_com_jass_belote_html.txt` · `divers/belotecontree_free_reglement.txt` · famille ASCEE/DDT 79 · `tournois/web_myassoc_org_..._pdf.txt` · `tournois/geraudotloisirs_free_fr_....txt` · `tournois/tcvb_bruche_free_fr_....txt` · `tournois/s1_static_footeo_com_..._Rglement_Concours_mtqpkb_pdf.txt` · famille « coupe à droite » · famille ange.heureux · `tournois/data_over_blog_kiwi_com_....txt` · `tournois/web_archive_org_..._coinche_stephanoise_....txt` · `tournois/maisondesessarts_fr_article116_html.txt` · `divers/adpoker_fr_belote_contree_html.txt` · `apps-sites/gamerules_com_rules_coinche.txt` · `open-source/drasill_bga-coinche_master_coinche.game.php` · `open-source/ismo009_Coinche_main_game.js` | « **Si la couleur demandée est l'atout**, lorsque c'est possible, il faut **en plus monter sur la carte la plus haute déjà posée**. » (FFB contrée 2016, §5.2.1). « if a trump is led each player must if possible beat the highest trump in the trick, **even if that card was played by their partner**. » (Pagat coinche). « Cette règle s'applique dès lors qu'on joue un atout, **même si c'est son partenaire** qui a mis la carte d'atout la plus forte. » (Wikipédia Belote). « A l'atout, les concurrents doivent monter, **même si le partenaire est maître**. » (Lions Club art. 15 ; ASCEE 79 art. 12 ; Pont-de-Claix art. 13). « En atout, on doit toujours **forcer**, même sur son partenaire. » (Géraudot art. 9). « Ne pas monter à l'atout même sur son partenaire : **162 points de pénalité** » (FC Plouay). |
| **SOUPLE : pas d'obligation de monter sur son partenaire** | `divers/belotepoint_fr_regles_coinche.txt` · `tournois/clublafontainedejouvence_fr_....txt` · `apps-sites/playjoy_com_en_coinche_rules.txt` · `apps-sites/en_wikipedia_org_wiki_Coinche.txt` (pour le Tout Atout) · `divers/pagat_com_jass_coinche_html.txt` (mentionné comme **variante**, pas comme règle) | « **Monter sur le partenaire n'est pas obligatoire**, mais monter sur l'adversaire l'est lorsqu'on joue atout. » (belotepoint). « OBLIGATION de monter à la carte sur son ADVERSAIRE, **mais pas OBLIGATOIREMENT sur son PARTENAIRE** » (Club La Fontaine de Jouvence). « If your partner is going to win the trick, **you are not required to play a higher trump**. » (playjoy). « In all trumps players must always play a higher card if possible **unless the partner is winning the trick**. » (Wikipédia EN). Pagat range la souplesse dans les variantes : « Some also relax the rules when a trump is led, and allow a player whose partner is winning the trick to follow suit with any trump. » |

**Consensus large, contesté à la marge.** La règle stricte domine massivement et de façon
**transversale** : la fédération, les deux Wikipédia françaises, Pagat, les concours de
belote du Nord comme du Sud-Ouest, et les deux implémentations open source qui tranchent
(BGA et ismo009) disent tous « on monte même sur son partenaire ». C'est d'ailleurs la
règle que les règlements de concours **assortissent d'une pénalité chiffrée**, ce qui est
le signe qu'elle est appliquée à la table. Le camp souple est composé de quatre sources
sans autorité comparable, et Pagat le classe explicitement comme une variante minoritaire.

Ne pas confondre avec l'axe 9 : cet axe-ci porte sur le cas **« l'atout est la couleur
demandée »**. Le cas « on coupe une couleur ordinaire » est un axe distinct.

---

## 9. « Pisser » / « ne pisse pas » : doit-on sous-couper quand on ne peut pas surcouper ?

C'est **l'axe qui divise le plus le corpus**, et c'est aussi celui où les sources se
contredisent avec le plus d'aplomb. Situation exacte : un adversaire a coupé et tient le
pli ; je n'ai pas la couleur demandée ; j'ai de l'atout, mais aucun qui batte le sien.

| Position | Sources | Extrait |
|---|---|---|
| **A — On PEUT se défausser (« on ne pisse pas »)** | famille **FFB 2015** (contrée + coinche) · `federations/LOCAL_regles_officielles_belote_contree.txt` · **`federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt`** · famille **FFB pages web** *contrée et coinche* (https://www.ffbelote.org/belote-contree/ , https://www.ffbelote.org/regles-coinche/) · `tournois/cdf_missegre11_com_....txt` · `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` · `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` · `tournois/maisondesessarts_fr_article116_html.txt` · `divers/adpoker_fr_belote_contree_html.txt` · `divers/belotecontree_free_reglement.txt` · `tournois/clublafontainedejouvence_fr_....txt` | « **Précision : Si un adversaire a déjà coupé, et qu'il ne nous reste que des atouts inférieurs au sien, il n'est pas obligatoire d'en jouer un (on dit que l'on « ne pisse pas »), on peut se défausser.** » (FFB 2015, §5.2 2.2 — formule reprise mot pour mot en coinche et en contrée). « Si un joueur adverse a déjà coupé et tient le pli, si l'on doit couper, il est obligatoire de fournir un atout plus fort que lui (on dit que l'on « surcoupe »). Si c'est impossible, **on peut se défausser de n'importe quelle carte sans exception**. » (FFB **contrée** 2016, §5.2.2). « Lorsqu'un adversaire coupe […] si nous n'avons pas d'atout supérieur, **nous pouvons nous défausser**. C'est-à-dire que nous pouvons jouer n'importe quelle autre carte, **nous ne sommes pas obligés de jouer atout**. » (page FFB contrée). « Si une couleur est coupée par un adversaire, le joueur qui n'a pas de carte de la couleur demandée, doit couper lui aussi **mais uniquement s'il peut poser une carte d'atout supérieure** ; dans l'hypothèse inverse **il peut défausser la carte de son choix**. » (belotecontree.free.fr). « A la coinchée **"on ne pisse pas"**. » (Club La Fontaine de Jouvence). |
| **B — On DOIT sous-couper (« obligation de pisser »)** | **`federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt`** (belote classique) + sa copie `divers/cartesetcie_fr_regles_officielles_de_la_belote.txt` · **`federations/ffbelote_org_reglements_de_la_belote_avec_ou_sans_annonce.txt`** + sa copie `tournois/villeconin_fr_....txt` · famille **ASCEE 2A** (https://www.fnasce.org/IMG/pdf/reglement.pdf) · famille **ASCEE/DDT 79** · `tournois/data_over_blog_kiwi_com_....txt` (AIL Manissieux) · `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` · famille **ange.heureux** · `divers/pagat_com_jass_coinche_html.txt` + `divers/pagat_com_jass_belote_html.txt` (**règle de base**) · `apps-sites/gamerules_com_rules_coinche.txt` · `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` · `divers/fr_wikipedia_org_wiki_Belote.txt` (**règle de base**) · `open-source/ismo009_Coinche_main_game.js` · `open-source/drasill_bga-coinche_master_coinche.game.php` | « Si un joueur adverse a déjà coupé et tient le pli, si l'on doit couper, il est obligatoire de fournir un atout plus fort que lui (« surcoupe »). Si c'est impossible, **il faut fournir un atout plus faible (on dit que l'on « pisse »)**. » (FFB **belote** 2016, §5.2.2). « 4- Lorsqu'un adversaire coupe et que nous ne possédons pas la couleur demandée, **il est obligatoire de couper également (pisser dans le jargon de la belote), même si l'on est incapable de surmonter son atout.** » (page FFB belote). « **Obligation de « pisser » de l'atout sur son adversaire** : si un joueur ne peut fournir à la couleur d'entame et qu'un adversaire a déjà coupé avant lui, alors il doit surcouper avec une carte plus forte, **sinon il doit poser une carte d'atout plus faible**. » (ASCEE 2A). « Si l'un des joueurs d'un camp coupe une carte, l'adversaire qui n'aura pas la couleur demandée **devra fournir de l'atout, s'il en a**, et monter le cas échéant. » (ASCEE 79 art. 13). « If he has trumps but is unable to overtrump, **he must still play a trump**, although he does not benefit from doing so. This is termed "undertrumping", or "pisser" in French Belote jargon. » (Pagat). « Si on ne peut monter, **on doit tout de même jouer un atout**. » (Wikipédia Belote). « `// On ne peut pas monter, on doit quand même couper (pisser)` » (ismo009, `game.js` l. 228). |
| **C — Au choix du joueur** | `tournois/web_archive_org_..._coinche_stephanoise_com_..._reglement_coinche_pdf.txt` | « Si votre adversaire a coupé vous devez surcouper ; dans le cas où vous ne le pouvez pas, **vous avez la possibilité de vous défausser**, en fournissant une autre couleur. **Mais vous pouvez également sous-couper** pour conserver une autre carte maîtresse à la couleur par exemple. C'est l'expression pour « Pisser à l'atout ». » |
| **D — C'est une convention à fixer avant la partie / une option de logiciel** | `divers/belotecontree_free_reglement.txt` (forum de l'auteur) · `apps-sites/eryodsoft_com_fr_jeux_coinche.txt` + `apps-sites/play_google_com_..._eryodsoft_....txt` · `apps-sites/play_google_com_store_apps_details_id_com_aandrill_belote_hl_fr.txt` · `divers/lemagloisirs_fr_regle_coinche.txt` | « Tu peux aussi décider en début de partie de convenir avec tes adversaires **d'autoriser ou non à "pisser"**. » (Hervé, belotecontree.free.fr). « **Obligation ou non de « pisser » à l'Atout.** » (liste d'options d'Eryod Soft ; idem « Obligation de pisser atout » chez Belote Andr). « surcoupe obligatoire **ou non** » (lemagloisirs, dans sa liste de points à trancher avant de jouer). |

**Divergence maximale, et elle traverse la fédération elle-même.**

1. **La FFB se contredit à l'intérieur d'une même publication.** Les deux règlements du
   **27.01.2016** ont la même structure, la même numérotation, la même date de mise en
   application — et leur **paragraphe 5.2.2, identique jusqu'à sa dernière proposition**,
   se termine par « **il faut fournir un atout plus faible** » dans la belote classique et
   par « **on peut se défausser de n'importe quelle carte sans exception** » dans la
   contrée. Le même partage se retrouve sur le site : la page *belote* impose de pisser,
   les pages *contrée* et *coinche* l'excluent. Ce n'est donc pas une bévue isolée :
   **la FFB pose délibérément deux règles opposées selon le jeu** — on pisse à la belote,
   on ne pisse pas à la contrée/coinche.

2. **Pagat décrit exactement la même ligne de fracture, mais géographiquement** : le
   sous-coup obligatoire est la règle de base, « However, some players, **especially in the
   south of France**, do not require this. » Or la contrée *est* le jeu du Midi
   (`divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` : « La contrée se joue plus
   particulièrement dans le Midi de la France »). Les deux lectures — par jeu (FFB) et par
   région (Pagat) — désignent probablement le même clivage.

3. **Le corpus des tournois réels penche vers l'obligation de pisser** (ASCEE 2A, ASCEE/DDT
   79 + Pont-de-Claix, AIL Manissieux, ange.heureux), y compris pour des concours qui
   s'annoncent « Belote Contrée / Coinche » — l'ASCEE 2A l'écrit d'ailleurs en tête de son
   règlement comme un des « points particuliers à respecter », c'est-à-dire comme un point
   où il *sait* que les joueurs n'ont pas la même habitude.

4. **Les deux implémentations open source qui tranchent ont choisi B**, y compris celle qui
   tourne en production sur BoardGameArena : `coinche.game.php` lève « You must cut with a
   %s » dès que le partenaire n'est pas le plus fort, **sans exception pour l'absence
   d'atout supérieur**.

---

## 10. Le partenaire a coupé et tient le pli — et je n'ai plus que des atouts

Sous-cas de l'axe 7 que quatre rédactions FFB traitent séparément, et sur lequel elles ne
disent **pas la même chose**.

| Position | Sources | Extrait |
|---|---|---|
| **Non obligé de monter — c'est le seul cas où un atout inférieur est permis** | famille **FFB 2015** (contrée + coinche) · famille **FFB pages web** (contrée, coinche **et** belote) · `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` · `divers/carafons_fr_regles_de_la_coinche.txt` · `tournois/cdf_missegre11_com_....txt` · `tournois/villeconin_fr_....txt` | « Lorsque notre partenaire, maître, a coupé une carte adverse et que nous ne possédons plus que de l'atout, **il n'est pas obligé de fournir un atout supérieur. C'est le seul cas de figure, plutôt rare, où il est permis de jouer un atout inférieur.** » (FFB 2015, §5.2 règle 4). |
| **Obligé de monter** | **`federations/LOCAL_regles_officielles_belote_contree.txt`** | « Lorsque notre partenaire, maître, a coupé une carte adverse et que nous ne possédons plus que de l'atout, **il est obligé de fournir un atout supérieur.** » (§5.2 règle 4) |
| **Non obligé de couper, sous-coup du partenaire explicitement autorisé** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` · `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` · `divers/fr_wikipedia_org_wiki_Belote.txt` | « Si le joueur partenaire a déjà coupé et tient le pli, si on ne possède pas la couleur demandée, il n'est pas obligatoire de couper. On peut se défausser de n'importe quelle carte sans exception (**y compris un atout inférieur au sien**). » (FFB 2016, §5.2.3). « il peut également jouer un atout s'il le souhaite, **y compris inférieur à celui de son partenaire** » (Wikipédia Belote). |
| **Non obligé de couper, mais sous-couper le partenaire est INTERDIT si on a mieux** | `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` · `divers/pagat_com_jass_belote_html.txt` | « Il est en revanche **interdit de sous couper** (couper avec une carte inférieure) si le partenaire est maître, avec un atout, **alors qu'on a des atouts supérieurs dans sa main**. » (casimir, règle 4). « if you are able to overtrump, you may either do so or you may throw away […] **but you are not allowed to undertrump** » (Pagat belote, variante « toujours obligé de pisser »). |

**Divergence — et l'une d'elles est très probablement une coquille de réédition.**
`LOCAL_regles_officielles_belote_contree.txt` reprend le texte FFB 2015 **mot pour mot** sur
tout le paragraphe, puis inverse la conclusion en supprimant « n'est pas » **et** la phrase
d'explication qui suivait (« C'est le seul cas de figure, plutôt rare, où il est permis de
jouer un atout inférieur »). Comme cette phrase rendait la règle 2015 auto-cohérente, sa
disparition simultanée avec la négation ressemble à une coupe éditoriale maladroite plutôt
qu'à un choix de règle. **C'est pourtant cette édition-là qui a servi de base à Colver** —
voir la section « Où tombe Colver » en fin de document.

Au fond, trois options se partagent le champ : (i) obligation de monter, (ii) liberté totale
y compris sous-coup, (iii) liberté sauf sous-coup gratuit. Personne n'est majoritaire ; la
FFB 2016 et Wikipédia Belote portent la (ii), ce qui en fait la lecture la mieux étayée.

---

## 11. Ordre et valeur des cartes à l'atout et hors atout

| Position | Sources | Extrait |
|---|---|---|
| **Atout : V 20 · 9 14 · A 11 · 10 10 · R 4 · D 3 · 8 0 · 7 0 — Hors atout : A 11 · 10 10 · R 4 · D 3 · V 2 · 9 0 · 8 0 · 7 0** | **Toutes les sources du corpus qui donnent un barème.** Famille FFB (4 rédactions) · `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` · `divers/fr_wikipedia_org_wiki_Belote.txt` · `divers/pagat_com_jass_coinche_html.txt` · `divers/belotecontree_free_reglement.txt` · `tournois/cdf_missegre11_com_....txt` · `tournois/casimirdehauteclocque_fr_....txt` · `divers/adpoker_fr_belote_contree_html.txt` · `divers/jeux_regles_com_regles_coinche.txt` · `divers/regles_com_jeux_cartes_coinche_html.txt` · `divers/belotepoint_fr_regles_coinche.txt` · `divers/clubdejeux_com_....txt` · `apps-sites/gamerules_com_rules_coinche.txt` · `apps-sites/playjoy_com_en_coinche_rules.txt` · `divers/exoty_com_....txt` · `divers/lemagloisirs_fr_regle_coinche.txt` · `open-source/gyscos_libcoinche_master_src_points.rs` | « À l'atout : Valet 20 / Neuf 14 / As 11 / Dix 10 / Roi 4 / Dame 3 / Huit 0 / Sept 0. Hors atout : As 11 / Dix 10 / Roi 4 / Dame 3 / Valet 2 / Neuf 0 / Huit 0 / Sept 0. » (FFB contrée 2016, §6 — tableau identique dans les quatre rédactions). |
| **Variante d'inversion complète (« coinche inversée », version cavaillonnaise)** | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « Même principe que la coinche […] sauf que l'ordre des valeurs est **inversée**. Exemple pour l'atout : le 7 vaut 20 points ; le 8 vaut 14 points ; la dame vaut 11 points ; le roi vaut 10 points ; le dix vaut 4 points ; l'as vaut 3 points ; le neuf et le valet 0 point. » |

**Consensus total sur le barème à la couleur.** Aucune source ne diverge : 62 points à
l'atout, 30 par couleur ordinaire, 152 au total. La seule exception est une variante
locale nommément identifiée comme telle (Cavaillon), qui inverse la hiérarchie sans changer
la somme. Sur cet axe, la belote est un des jeux de cartes les mieux fixés du corpus.

Note de lecture : `divers/belotecontree_free_reglement.txt` présente les mêmes valeurs mais
liste les *forces* en ordre **croissant** (« 7, 8, Dame, Roi, 10, As, 9, Valet ») alors que
tout le reste du corpus les donne en ordre décroissant. Même règle, présentation inverse.

---

## 12. Sans Atout / Tout Atout : quelles valeurs, et sur quel total ?

Le Sans Atout / Tout Atout **n'existe pas à la contrée** (`divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt`
et `divers/fr_wikipedia_org_wiki_Coinche.txt` en font le critère qui sépare les deux jeux ;
`divers/belotecontree_free_reglement.txt` : « Dans le cadre de la contrée on ne joue pas au
SA/TA […] En tournoi officiel cela n'existe pas »). La FFB en fait néanmoins une **variante
d'organisateur**, et c'est là que les barèmes divergent.

| Position | Sources | Extrait |
|---|---|---|
| **Sans Atout : As à 19, le reste inchangé (total ramené à 162)** | famille FFB 2015 · famille FFB 27.01.2016 · famille FFB pages web · `divers/fr_wikipedia_org_wiki_Coinche.txt` · famille ange.heureux · `divers/jeux_regles_com_regles_coinche.txt` · `divers/pagat_com_jass_coinche_html.txt` | « Les As valent **19 points** afin de ramener les points du Paquet à 162. » (page FFB contrée). Somme : (19+10+4+3+2)×4 = 152, +10 de der = 162. **Aucune divergence.** |
| **Tout Atout — barème rééchelonné, total maintenu à 162** | famille **FFB 2015** (contrée + coinche) · `divers/fr_wikipedia_org_wiki_Coinche.txt` · `divers/jeux_regles_com_regles_coinche.txt` · `divers/pagat_com_jass_coinche_html.txt` | FFB 2015 : « Valet 13 / Neuf 9 / As 6 / Dix 5 / Roi 3 / Dame 2 / Huit 0 / Sept 0 », soit 38 par couleur → 152 + 10 = 162. Wikipédia Coinche et Pagat donnent la même idée avec **Dame à 1 et Valet à 14** : « le valet vaut 14 ; le neuf 9 ; l'as 6 ; le dix 5 ; le roi 3 ; la dame 1 » → également 38 par couleur. |
| **Tout Atout — barème d'atout conservé (258 points) et conversion ×162/258** | famille **FFB 27.01.2016** (contrée + belote) · famille **FFB pages web** · `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` · `divers/drasill_github_io_bga_coinche_rules_fr_html.txt` | « L'ordre et la valeur des cartes sont ceux de l'atout. Par conséquent le nombre total de points est de **258 points**. Pour vérifier si un contrat est réussi, il faut multiplier le nombre de points comptés (hors belote) par la fraction **162/258** […] L'organisateur du tournoi mettra à disposition des joueurs une **table de conversion**. » (FFB contrée 2016, §11.3) |
| **Tout Atout — conversion ×162/256** | famille ange.heureux | « la somme des points possibles (**256**) […] Il faut donc multiplier la valeur des points obtenus par **256/162 = 0,6328125**. Par commodité, on multipliera par **2/3**… » (le rapport est écrit à l'envers ; l'intention est ×162/256) |
| **Tout Atout — table de conversion non linéaire, plusieurs systèmes concurrents** | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « les points à sans atout et tout atout sont comptés normalement (sur 130 et 258 respectivement), puis ramenés sur 162, ce qui nécessite une règle de conversion » — suivie de **quatre** systèmes alternatifs, dont un qui supprime le dix de der et multiplie par 2/3. |
| **Obligation de monter à Tout Atout** | famille FFB (toutes rédactions) · `apps-sites/en_wikipedia_org_wiki_Coinche.txt` · `divers/drasill_github_io_bga_coinche_rules_fr_html.txt` · `apps-sites/gameduell_helpshift_com_..._faq_1054_coinche.txt` | « au jeu de la carte, **on est toujours obligé de monter sur la carte qui tient** si l'on peut. » (FFB). Mais Wikipédia EN y ajoute l'exception partenaire (cf. axe 8). |
| **Pas de belote à Sans Atout ; jusqu'à 4 belotes à Tout Atout** | famille FFB · `divers/fr_wikipedia_org_wiki_Coinche.txt` · `divers/exoty_com_....txt` | « Il n'y a pas de Belote/Rebelote possibles » (SA) ; « il peut y avoir jusqu'à 4 Belotes, pour un total de 80 points » (TA). Wikipédia ajoute la « **belote dorée** » (carré de rois + carré de dames) à 100 points au lieu de 80. |

**Divergence, et Pagat en donne la clé** : « Usually some adjustments are made to ensure that
the number of points in the pack remains 162 including the dix de der. There are **various
systems, none of them particularly elegant**. » Le clivage utile à retenir est
chronologique **au sein même de la FFB** : la rédaction **2015 rééchelonne les cartes** pour
que le total reste 162 nativement ; la rédaction **2016 garde les valeurs d'atout et convertit
après coup** (258 → 162). Les deux méthodes ne rendent pas les mêmes contrats réussis,
puisque l'arrondi de conversion ne tombe pas au même endroit que le rééchelonnement.

Le Sans Atout, lui, est un **consensus** : As à 19, tout le reste identique. Pagat le confirme
comme la pratique usuelle (« in sans atout the value of an ace may be increased to 19 »), et
personne ne propose autre chose dans le corpus.

---

## 13. Belote et rebelote

| Position | Sources | Extrait |
|---|---|---|
| **Roi + Dame d'atout dans la même main = 20 points, annoncés en jouant les deux cartes** | **Tout le corpus.** | « Lorsqu'un joueur détient ensemble le roi et la dame d'atout […] elle doit être annoncée en disant « Belote » lorsqu'il joue la première de ces deux cartes, et « Rebelote » lorsqu'il joue la seconde. » (FFB, toutes rédactions) |
| **L'ordre Roi/Dame est indifférent** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` · `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` · `divers/clubdejeux_com_....txt` · `tournois/casimirdehauteclocque_fr_....txt` · `apps-sites/gameduell_helpshift_com_..._faq_1056_contree.txt` | « en disant « belote » lorsqu'il joue la première de ces deux cartes (**indifféremment la Dame ou le Roi**) » (FFB contrée 2016, §7). « **quel que soit l'ordre** dans lequel il choisit de jouer ses cartes » (Belote.com). |
| **Le Roi doit obligatoirement être joué en premier** | famille **« coupe à droite »** (cdfcasson, jimcontent, Varennes, **rjcv.be**) · `tournois/geraudotloisirs_free_fr_....txt` · `divers/fr_wikipedia_org_wiki_Belote.txt` (comme règle belge / règle originelle) · `divers/pagat_com_jass_belote_html.txt` (comme variante) | « **Tout joueur est obligé de jouer Belote avec le Roi en premier** et d'annoncer « Belote » et « Rebelote » pour avoir les 20 points. » (cdfcasson art. 2 ; rjcv.be). « La belote s'annonce **par le roi** » (Géraudot art. 6). « notamment dans les concours **en Belgique** et dans les règles officielles de la Belote Classique, **le roi doit impérativement être joué avant la dame** […] S'il fait l'inverse, les 20 points ne peuvent dès lors pas être comptabilisés (ceci reste cependant une règle **optionnelle** qu'il convient de déterminer avant le début de la partie…) » (Wikipédia Belote). |
| **Oubli de « Rebelote » : tolérance, si le joueur s'en aperçoit avant le décompte** | famille **FFB 2015** · `federations/LOCAL_regles_officielles_belote_contree.txt` · famille **FFB pages web** · `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` | « En cas d'omission du terme « Rebelote », **une tolérance sera accordée si le joueur s'en aperçoit au plus tard lors de la comptabilisation des points.** Cette tolérance s'applique **une seule fois par équipe et par partie**. » (FFB 2015). **Mutation** : `LOCAL` reprend la phrase mais **supprime la limite « une seule fois par équipe et par partie »**. |
| **Oubli d'un des deux termes : la belote est perdue, sans tolérance** | **`federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt`** · `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` · `divers/carafons_fr_regles_de_la_coinche.txt` · `tournois/tcvb_bruche_free_fr_....txt` · `divers/fr_wikipedia_org_wiki_Belote.txt` · `divers/clubdejeux_com_....txt` · `tournois/casimirdehauteclocque_fr_....txt` | « puis **obligatoirement** « rebelote » lorsqu'il joue la seconde. **En cas d'omission d'un de ces deux termes, la belote n'est pas prise en compte.** » (FFB contrée 2016, §7). « LE JOUEUR QUI ANNONCE LA BELOTE DEVRA AUSSI ANNONCER LA REBELOTE, EN CAS D'OUBLI LES 20 POINTS NE SERONT PAS MARQUÉS. » (TCVB art. 11). |
| **« Rebelote » sans « Belote » préalable : pas de bonus** | famille FFB 2015 · `federations/LOCAL_...` · famille FFB pages web | « Si un joueur annonce rebelote **sans avoir préalablement annoncé la belote**, le bonus n'est pas accordé. » |
| **L'oubli volontaire est une tactique reconnue** | `divers/pagat_com_jass_coinche_html.txt` | « it can be in the bidding team's interest to **suppress the belote announcement** when their contract is likely to fail. They can do this by **not saying "rebelote"** when playing the second card, in which case the 20 points are not scored. » |

**Divergence sur la sanction de l'oubli, et elle est datée.** La FFB a **durci** entre 2015 et
2016 : la tolérance « une fois par équipe et par partie » disparaît au profit d'un « la belote
n'est pas prise en compte » sec. Le site web de la fédération, lui, est resté sur l'ancienne
rédaction — il diffuse donc aujourd'hui encore la tolérance que ses propres PDF de 2016 ont
supprimée. L'édition « Équipe Ludique » constitue une troisième position, plus permissive
encore que 2015 (tolérance sans plafond).

Sur **l'ordre Roi/Dame**, la ligne de fracture est nette et de nature régionale : les concours
de belote classique de l'Est/Nord et la Belgique imposent le Roi en premier ; toute la
littérature contrée/coinche dit que l'ordre est indifférent. Wikipédia et Pagat s'accordent
pour dire que la règle du Roi d'abord est **la règle originelle**, devenue optionnelle.

Deux points d'accord général qui restent : la belote **compte pour réaliser le contrat**
(FFB ; ASCEE 2A : « La belote (et re) permet de remplir le contrat : attention au moment de
contrer ! »), et elle **ne fait jamais partie des 152 points cartes** — c'est un bonus qui
s'ajoute au total du camp.

---

## 14. Dix de der et total de la donne

| Position | Sources | Extrait |
|---|---|---|
| **Dernier pli = +10 points ; total de la donne = 162 (152 cartes + 10)** | **Tout le corpus, sans exception.** Famille FFB (4 rédactions) · `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` · `divers/pagat_com_jass_coinche_html.txt` · `divers/belotecontree_free_reglement.txt` · tous les règlements de tournoi · toutes les apps | « L'équipe réalisant le dernier pli obtient un bonus de 10 points pour ce pli […] Le total des points du jeu est donc de **162**, en comptant le « dix de der ». » (FFB). « In each deal there is a total of 152 for the cards, plus 10 for the last trick » (Pagat). |
| **Capot : le dix de der vaut 100, total 252** | famille FFB (4 rédactions) · `divers/fr_wikipedia_org_wiki_Coinche.txt` · famille ASCEE/DDT 79 + Pont-de-Claix · `tournois/web_myassoc_org_..._pdf.txt` · `tournois/geraudotloisirs_free_fr_....txt` · `tournois/tcvb_bruche_free_fr_....txt` · famille « coupe à droite » · `divers/fr_wikipedia_org_wiki_Belote.txt` · `divers/pagat_com_jass_belote_html.txt` · `apps-sites/gameduell_helpshift_com_..._faq_1056_contree.txt` | « En cas de capot […] **le dix de der vaut 100 points**, portant ainsi le total à **252 points**. » (FFB). « Le capot compte alors pour **252 points**. » (ASCEE 79 art. 15 ; Lions Club art. 13 ; Géraudot art. 3 ; cdfcasson). |
| **Le dix de der ne compte pas en cas de capot** | famille ange.heureux | « Le "Dix de der" est marqué par l'équipe qui fait le dernier pli, soit 10 points supplémentaires, **sauf dans le cas de Tout-Atout ou du Capot où il ne compte pas.** » |
| **Le capot vaut 162 (pas 252)** | `tournois/aappmakoenigshoffen_e_monsite_com_..._pdf.txt` · `tournois/lagrandcombe_fr_..._Reglement_belote_2020_pdf.txt` | « Le capot est marqué **162 points**. » (AAPPMA Koenigshoffen art. 8). « Aucune annonce – Dedans : 162 points **- Capot : 162 points**. » (La Grand-Combe art. 1). |

**Consensus sur le dix de der, divergence sur le capot.** Le +10 du dernier pli et le total
de 162 sont l'un des rares points où le corpus entier — fédération, encyclopédies, tournois
de village, apps — dit exactement la même chose, y compris dans les mêmes termes. Le
sur-bonus du capot est presque aussi solide (252), avec **deux règlements alsacien et gardois
qui l'aplatissent à 162** pour éviter les écarts explosifs en concours à points cumulés, et
une source qui supprime le dix de der du capot.

Précision qu'une seule source ajoute, et qui est un contrôle utile : « Si la **somme des deux
totaux** obtenus est **différente de ce nombre**, il est procédé à un **nouveau décompte** des
points. » (famille FFB pages web) — l'invariant 162 comme test de recomptage.

---

## 15. Consultation du pli précédent

| Position | Sources | Extrait |
|---|---|---|
| **Le dernier pli est consultable tant que le pli suivant n'a pas été ramassé** | famille **FFB 2015** · `federations/LOCAL_...` · famille **FFB pages web** (+ copies Missègre, Villeconin, cartesetcie, carafons) · `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « Le dernier pli peut être consulté par n'importe quel joueur de la partie **tant que le pli suivant n'a pas été ramassé**. » (FFB 2015, §5.1). « **Seul le dernier pli peut être consulté.** » (Wikipédia contrée). |
| **Consultable seulement une fois le pli suivant complet, et jamais avant de jouer sa carte** | **`federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt`** · `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` · `divers/cartesetcie_fr_regles_officielles_de_la_belote.txt` | « Lorsqu'un pli est retourné, **il n'est consultable qu'à l'issue du pli suivant** (lorsque les quatre cartes sont jouées) avant que celui-ci ne soit retourné à son tour. **En aucun cas un joueur ne peut demander à consulter le pli précédent avant de jouer sa carte.** » (§5.1) |
| **Consultable seulement après avoir joué sa propre carte** | `tournois/data_over_blog_kiwi_com_..._reglement_table_coinche_pdf.txt` | « On ne peut, éventuellement, regarder le pli précédent **qu'après avoir joué sa carte**. » (art. 15) |
| **Consultable après la fin du coup, avant que le pli ne soit ramassé — jamais pendant** | `divers/belotecontree_free_reglement.txt` · `tournois/geraudotloisirs_free_fr_....txt` | « **Pendant le coup suivant le pli précédent ne peut être consulté.** Par contre après la fin du coup et avant de ramasser le pli, les quatre joueurs peuvent revoir le pli précédent uniquement (il n'est pas encore couvert). » (belotecontree). « On ne peut regarder que les cartes de la dernière levée et **AVANT** que le pli ne soit ramassé. » (Géraudot art. 7). |
| **Aucune consultation** | `tournois/maisondesessarts_fr_article116_html.txt` · `tournois/tcvb_bruche_free_fr_....txt` | « La levée ramassée et retournée **ne peut être remontrée à aucun joueur.** » (Maison des Essarts art. 6). « TOUS LES PLIS DOIVENT ÊTRE RAMASSÉS **ET COUVERTS** JUSQU'À LA FIN DU JEU. » (TCVB art. 10). |
| **Toléré, mais uniquement la dernière levée** | `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` | « **il est toléré** de regarder la dernière levée à avoir été jouée. Il est en revanche **interdit** de regarder les autres cartes retournées lors des levées précédentes. » |

**Divergence sur la fenêtre, consensus sur le fond.** Personne n'autorise à revoir plus que
**le dernier pli** ; le désaccord porte entièrement sur *quand* on peut le faire, et il
s'ordonne du plus permissif au plus strict : (a) tant que le pli suivant n'est pas ramassé
(FFB 2015 et le site), (b) seulement après avoir joué sa carte (AIL Manissieux), (c) seulement
une fois le pli suivant achevé (FFB 2016), (d) jamais (Maison des Essarts, TCVB). Là encore la
**FFB a durci sa propre règle entre 2015 et 2016**, et la formulation 2016 vise explicitement
l'abus : consulter *avant* de jouer, c'est-à-dire pour décider de sa carte.

---

## 16. Carte posée = carte jouée

| Position | Sources | Extrait |
|---|---|---|
| **Toute carte MONTRÉE doit être jouée, sauf accord des adversaires** | famille FFB 2015 · `federations/LOCAL_...` · famille FFB pages web (+ copies Missègre, Villeconin, carafons) | « **Toute carte montrée doit être jouée** sauf autorisation ou avis contraire de la part des adversaires. » |
| **Toute carte SORTIE DU JEU doit être jouée, sauf accord des adversaires** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` · `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` · `divers/cartesetcie_fr_regles_officielles_de_la_belote.txt` | « **Toute carte sortie du jeu** doit être jouée sauf autorisation ou avis contraire de la part des adversaires. » |
| **Toute carte posée / tombée / jetée est jouée, sans échappatoire** | `tournois/web_myassoc_org_..._pdf.txt` · `tournois/geraudotloisirs_free_fr_....txt` · `tournois/aappmakoenigshoffen_..._pdf.txt` · famille « coupe à droite » · `divers/adpoker_fr_belote_contree_html.txt` | « Toute carte posée est jouée **et ne peut être changée**. » (Lions Club art. 14 ; Géraudot art. 7). « Toute carte **jetée** est considérée jouée. » (AAPPMA art. 11). « Toute carte **tombée sur la table** est considérée comme jouée. » (cdfcasson art. 9). « Si, au cours du jeu, un des joueurs se trompe en fournissant une carte, **il ne peut la reprendre.** » (adpoker). |
| **Toute carte posée est jouée, SAUF si le joueur pouvait fournir — c'est alors une faute** | `tournois/tcvb_bruche_free_fr_....txt` | « TOUTE CARTE POSÉE EST CONSIDÉRÉE COMME JOUÉE, **SAUF SI LE JOUEUR PEUT FOURNIR. IL Y A ALORS FAUTE DE JEU.** TOUTE FAUTE DE JEU DONNE 162 POINTS À L'ADVERSAIRE. » |
| **Interdits connexes** | famille FFB pages web · `tournois/cdf_missegre11_com_....txt` | « Il est **interdit de préparer sa carte avant son tour**. » ; « La carte devra être **posée sur le tapis de la même manière** tout au long de la partie. » |

**Consensus sur la règle, divergence sur sa clémence.** Tout le corpus tient la carte pour
jouée dès qu'elle quitte la main. La fracture est de nature disciplinaire : la FFB et ses
dérivés laissent une **porte de sortie négociée** (« sauf autorisation des adversaires »),
alors que les règlements de concours l'excluent (« ne peut être changée ») — logique quand
la table est en compétition et qu'un arbitre existe. Un détail de rédaction FFB mérite
attention : le passage de « montrée » (2015) à « **sortie du jeu** » (2016) élargit la règle
aux cartes tombées face cachée.

---

## 17. Incidents : fausse donne / maldonne

| Position | Sources | Extrait |
|---|---|---|
| **Fausse donne → simple redonne, sans pénalité, par le même donneur** | famille FFB pages web (+ Missègre, cartesetcie, carafons) · famille ASCEE 2A · `divers/adpoker_fr_belote_contree_html.txt` · famille ange.heureux · `tournois/tcvb_bruche_free_fr_....txt` · famille « coupe à droite » · `divers/belotecontree_free_reglement.txt` | « Si le donneur commet une irrégularité en distribuant les cartes (carte retournée, erreur dans le nombre de cartes, etc.) **il doit simplement redistribuer les cartes.** » (page FFB). « **Fausse donne redonne.** » (ASCEE 2A). « **"Maldonne Redonne"** » (ange.heureux). En cas de mène annulée : « le **« tour ne passe pas »** c'est-à-dire que le même joueur redonne (mélange et coupe obligatoire). » (belotecontree). |
| **Escalade en trois temps : redonne → interdiction de prendre → 160/162 points à l'adversaire** | famille FFB pages web (contrée, coinche, belote) · `tournois/cdf_missegre11_com_....txt` · `divers/fr_wikipedia_org_wiki_Belote.txt` | « En cas de **seconde irrégularité consécutive**, l'équipe ayant commis la faute se verra pénalisée et **interdite de toute prise** sur cette seconde donne […] En cas de **troisième irrégularité consécutive, 160 points** [162 sur la page belote] sont donnés à l'équipe adverse. » Wikipédia résume la même échelle : « Dans les règles officielles, la première fausse donne conduit à la première règle ; une seconde […] l'équipe fautive est interdite de prise ; et une troisième […] l'équipe adverse reçoit 162 points. » |
| **Escalade en deux temps : redonne → 162 à l'adversaire dès la 2ᵉ** | famille « coupe à droite » (cdfcasson art. 8, jimcontent, Varennes) · `tournois/lagrandcombe_fr_....txt` · `tournois/aappmakoenigshoffen_..._pdf.txt` | « En cas de fausse donne, le coup est nul. Le même joueur recommence à redistribuer les cartes. **Une deuxième fausse donne sera pénalisée par 162 points** pour l'adversaire. » (cdfcasson). La Grand-Combe ajoute la privation de prise dès la 1ʳᵉ : « celle-ci refait la donne **et n'a pas le droit de prendre**. » |
| **Pénalité de 162 dès la 3ᵉ tentative** | `tournois/web_myassoc_org_..._pdf.txt` · `tournois/geraudotloisirs_free_fr_....txt` | « En cas de fausse donne, le coup est NUL. Le même joueur recommence à distribuer les cartes. S'il y a récidive, **au 3ème essai, l'équipe adverse marque 162 points** et cela compte pour un coup de cartes. » |
| **Pénalité selon le moment : gratuite avant que l'atout soit fixé, 162 après** | famille ASCEE/DDT 79 + Pont-de-Claix · `tournois/s1_static_footeo_com_..._pdf.txt` · `tournois/rjcv_be_belote_regles_pdf.txt` | « Il n'y a **pas de pénalisation pour maldonne avant mise à l'atout**. Si une maldonne se produit **après** la mise à l'atout, l'équipe distribuant sera pénalisée, **162 pts** seront attribués à l'équipe adverse. » (ASCEE 79 art. 10). « En cas de maldonne **pendant les 5 premières cartes**, le même joueur remèle. Pendant la retouche, c'est pénalisé de **16** plus les annonces et **la main passe**. » (rjcv.be). |
| **Redonne avec mélange et coupe obligatoires ; arbitre à la 2ᵉ récidive** | `divers/belotecontree_free_reglement.txt` | « il doit refaire la donne **avec obligation dans ce cas de mélanger le jeu et de refaire couper**. En cas de deuxième irrégularité, l'autre équipe pourra **exiger la présence d'un arbitre** à partir de la troisième donne. En cas d'irrégularités répétées, l'organisation pourra décider **d'exclure l'équipe**. » |

**Consensus sur le premier coup, divergence sur l'escalade.** Aucune source ne pénalise une
fausse donne isolée : partout, on redonne, et **le donneur ne change pas** — ce point est
unanime et vaut la peine d'être noté, car il implique que la fausse donne ne fait pas
tourner le tour. Au-delà, chaque organisateur a son barème (2 temps, 3 temps, ou seuil lié à
la fixation de l'atout), et la sanction terminale est presque toujours la même :
**162 points (ou 160) à l'adversaire**.

Cas particulier récurrent, en belote classique uniquement : « **Si personne ne prend à son
tour, le même joueur redonne** jusqu'à ce que l'atout soit pris. » (Lions Club art. 10 ;
Géraudot art. 5 ; ASCEE 79 art. 8) — à la contrée, quatre passes font au contraire tourner
le donneur.

---

## 18. Incidents : renonce, jeu irrégulier, sanctions

| Position | Sources | Extrait |
|---|---|---|
| **Faute vue immédiatement → le jeu continue, couleur interdite au tour suivant ; faute vue en fin de donne → donne perdue + 160/162 à l'adversaire** | famille **FFB pages web** (contrée, coinche, belote) · `tournois/cdf_missegre11_com_....txt` · `divers/carafons_fr_regles_de_la_coinche.txt` | « Si la faute est constatée immédiatement, le cours du jeu continue. Il sera alors **interdit à l'équipe ayant commis la faute de jouer cette couleur au tour suivant**. En cas d'erreur constatée en fin de donne telle que (**ne pas avoir fourni à l'atout** alors que le joueur en disposait, **ne pas avoir coupé**, **ne pas être monté**), la donne est considérée comme **perdue pour l'équipe fautive** et 160 points [162 sur la page belote] sont donnés à l'équipe adverse. » |
| **Coup annulé + points du contrat à l'adversaire** | `divers/adpoker_fr_belote_contree_html.txt` | « Si cette erreur contrarie le bon déroulement du jeu (**couleur non fournie, oubli de coupe ou de surcoupe, ne pas monter dans la couleur de l'atout**), le coup est annulé et **les points du contrat en cours sont marqués par l'équipe adverse**. De même, **si un joueur montre plusieurs cartes à la fois**, l'équipe adverse marque les points du contrat demandé. » |
| **162 points forfaitaires à l'adversaire pour toute faute de jeu** | famille « coupe à droite » (cdfcasson art. 13, jimcontent art. 10, Varennes) · `tournois/s1_static_footeo_com_..._pdf.txt` · famille ASCEE/DDT 79 + Pont-de-Claix · `tournois/web_myassoc_org_..._pdf.txt` · `tournois/tcvb_bruche_free_fr_....txt` · `tournois/aappmakoenigshoffen_..._pdf.txt` · `tournois/lagrandcombe_fr_....txt` | « Toute faute de jeu (**atout non fourni, coupe d'une couleur alors qu'on en possède, non-respect général du jeu de belote**) sera pénalisé par **162 points** pour l'adversaire. » (cdfcasson). Barème détaillé chez FC Plouay : « Omettre de couper : 162 points de pénalité » ; « **Ne pas monter à l'atout même sur son partenaire : 162 points de pénalité** ». « En cas d'**impasse à l'atout** ou à la couleur demandée, l'équipe adverse marque 162 points. » (La Grand-Combe art. 7). |
| **Choix laissé à l'équipe lésée : rectifier, ou arrêter la mène et prendre tous les plis restants** | `divers/belotecontree_free_reglement.txt` | « Le principe est que **l'erreur ne doit pas bénéficier à l'équipe qui l'a commise**. […] l'équipe adverse pourra **soit accepter de continuer la mène après rectification** du coup erroné, **soit de l'arrêter au coup erroné et de devenir alors bénéficiaire des coups restant à jouer**. Cependant […] l'équipe ayant commis l'erreur **conservera les plis normalement faits avant cette erreur**. » Avec une clause de bonne foi : « En cas d'erreur manifestement de bonne foi faite à l'avant dernier coup […] s'il ressort à l'évidence que cette erreur n'a aucune incidence sur les deux derniers coups elle sera considérée comme **sans conséquence**. » |
| **Renonce d'annonce (coinche) : le camp adverse marque ce que le fautif avait annoncé** | famille FFB pages web · famille FFB 2015 (coinche) · `tournois/rjcv_be_belote_regles_pdf.txt` | « Si un joueur se révèle **incapable de montrer les combinaisons qu'il a annoncées**, il y a **renonce** : le camp adverse marquera les points que le camp fautif avait annoncés. » (FFB coinche 2015, §8.4). « En cas d'erreur d'un joueur (**renon, coupe, fausse annonce**…), les points plus annonces reviendront à l'équipe adverse. » (rjcv.be) |
| **Signaux et communication irrégulière = mise hors concours / 162 points** | `divers/belotecontree_free_reglement.txt` · famille FFB pages web · `tournois/aappmakoenigshoffen_..._pdf.txt` · `tournois/tcvb_bruche_free_fr_....txt` · `tournois/lagrandcombe_fr_....txt` | « Tout système consistant à fournir […] à son partenaire des indications **autrement que par ces deux moyens** [annonces et cartes jouées], serait assimilé à un **jeu irrégulier pouvant entraîner la mise hors concours de l'équipe**. » (belotecontree). « Il est interdit de faire connaître son jeu par n'importe quel moyen, **notamment en tapant sur la table**, dans ce cas l'adversaire marque **162 points**. » (AAPPMA art. 9). « Tous signes ou commentaires pourront **annuler la donne** sauf accord tacite des 4 joueurs. » (La Grand-Combe art. 5). |
| **Jouer avant son tour** | `tournois/tcvb_bruche_free_fr_....txt` | « LORSQU'UN JOUEUR JOUE AVANT SON TOUR, L'ADVERSAIRE MARQUE **162 POINTS**. » |

**Consensus sur le principe, divergence sur le tarif.** Le principe est unanime et souvent
cité à l'identique : **« l'erreur ne doit pas bénéficier à l'équipe qui l'a commise »** — on
le trouve mot pour mot chez belotecontree.free.fr, sur les trois pages FFB, chez Missègre et
chez Carafons. Sur la sanction, trois écoles :

- **fédérale, graduée** : rectification si la faute est vue tout de suite, donne perdue si
  elle est vue trop tard — la faute a un coût *proportionné au moment où elle est découverte* ;
- **de concours, forfaitaire** : 162 points, quel que soit le contexte, ce qui est un choix
  de simplicité d'arbitrage sur des dizaines de tables ;
- **négociée** : belotecontree laisse l'équipe lésée choisir entre rectification et arrêt de
  la mène à son profit, avec une clause de bonne foi qui neutralise les erreurs sans effet.

Point commun à noter : **les trois fautes que tous les règlements nomment sont exactement les
trois obligations des axes 6 à 9** — ne pas fournir, ne pas couper, ne pas monter. C'est un
indice de plus que ce sont bien les seules obligations dures du jeu de la carte.

---

## Récapitulatif des lignes de fracture

| Axe | Verdict | Où passe la ligne |
|---|---|---|
| 6. Fournir la couleur | **Consensus total** | — |
| 11. Valeur des cartes (couleur) | **Consensus total** | Une variante nommée (Cavaillon, valeurs inversées) |
| 14. Dix de der / 162 | **Consensus** | Deux concours aplatissent le capot à 162 |
| 3. Coupe obligatoire | **Consensus sur le principe** | Seuil : 3 (FFB) vs 2 (Rhône-Alpes) ; côté gauche vs droit (suit le sens du jeu) |
| 7. Couper sauf si le partenaire est maître | **Consensus** | Personne n'est contre ; quelques sources sont muettes |
| 12. Sans Atout (As à 19) | **Consensus** | — |
| 8. Monter à l'atout sur son partenaire | **Consensus large** | Strict (FFB, Wikipédia, Pagat, concours, BGA) vs souple (4 sites/apps) |
| 16. Carte posée = jouée | **Consensus sur la règle** | Négociable (FFB) vs irrévocable (concours) |
| 17. Fausse donne → redonne | **Consensus sur le 1er coup** | Escalade à 2 temps, 3 temps, ou seuil lié à la fixation de l'atout |
| 1. Sens du jeu | **Divergence** | Contrée/coinche antihoraire vs concours de belote classique horaire |
| 2. Mélange | **Divergence** | Fédération (obligatoire, anti-triche) vs tradition de table (interdit, pour lire la donne précédente) |
| 4. Découpage de la donne | **Divergence mineure** | 3-2-3 & co. partout ; le 4-4 est interdit par la FFB, admis dans le Sud-Est et en Corse |
| 12. Tout Atout | **Divergence** | FFB 2015 rééchelonne les cartes (total 162) vs FFB 2016 convertit ×162/258 |
| 13. Belote : ordre & oubli | **Divergence** | Roi d'abord (Belgique, Est/Nord) vs indifférent (contrée) ; tolérance 2015 vs sanction sèche 2016 |
| 15. Consultation du pli | **Divergence** | Quatre fenêtres, de « tant que le suivant n'est pas ramassé » à « jamais » |
| 10. Partenaire a coupé, je n'ai que de l'atout | **Divergence** | Obligé de monter (LOCAL) vs libre y compris sous-coup (FFB 2016) vs libre sauf sous-coup gratuit (Pagat, casimir) |
| 9. **« Pisser »** | **Divergence maximale** | **La FFB impose de pisser à la belote et l'exclut à la contrée, dans deux textes du même jour** ; Pagat lit la même fracture comme un clivage Nord/Midi |

---

## Où tombe Colver

Constat de lecture du moteur, à la date du 2026-08-01. **Aucun fichier du moteur n'a été
modifié** : cette section décrit ce que le code fait, pas ce qu'il devrait faire.
Fichiers lus : `colver-core/src/engine/play.rs` (fonction `legal_plays()` →
`legal_plays_color()`), `colver-core/src/engine/state.rs`, `docs/RULES.md`, `CLAUDE.md`.

### Axe 9 — « pisser » : Colver est dans le camp **A (« on ne pisse pas »)**, celui de la FFB contrée

`play.rs`, branche « on ne peut pas fournir, le partenaire n'est pas maître, on a de
l'atout », l. 100-113 :

```rust
let best_trump_rank = best_trump_rank_on_trick(state, trump_suit);
if let Some(br) = best_trump_rank {
    let higher = overtrump_in_suit(in_trump, trump_suit, br);
    if higher != 0 {
        return higher;                       // surcoupe obligatoire si possible
    }
    // "Ne pisse pas": can't overtrump opponent's trump
    // → can discard (non-trump) instead of undertrumping
    let non_trump = hand & !SUIT_MASK[trump_suit as usize];
    if non_trump != 0 {
        return in_trump | non_trump;          // défausse OU sous-coup, au choix
    }
    // Only have trump → must undertrump
    return in_trump;
}
```

Deux remarques utiles :
1. Le moteur ne se contente pas d'**autoriser** la défausse, il rend `in_trump | non_trump`,
   c'est-à-dire qu'il laisse aussi le **sous-coup** disponible. C'est donc exactement la
   **position C** du tableau (au choix du joueur), qui n'a qu'un seul témoin explicite dans
   tout le corpus — le règlement de la **Coinche stéphanoise**. La FFB contrée, elle, écrit
   « on peut se défausser » sans dire si le sous-coup reste permis ; le moteur tranche cette
   ambiguïté dans le sens le plus permissif. C'est défendable (un moteur RL ne doit pas
   retirer d'option sans texte qui l'interdise), mais c'est un choix, pas une lecture.
2. Le cas « je n'ai que de l'atout » (l. 112-113) retombe sur le sous-coup **obligatoire** :
   là, Colver rejoint mécaniquement le camp B. Cohérent avec `docs/RULES.md` : « However, if
   you **only have trump cards** in your hand, you must undertrump. »

**Conséquence** : Colver est aligné sur la FFB contrée/coinche, Wikipédia contrée,
belotecontree.free.fr et la tradition du Midi — et désaligné sur Pagat, BoardGameArena,
ismo009, la FFB *belote*, et la moitié des règlements de concours du corpus.

### Axe 10 — partenaire a coupé, je n'ai que des atouts : **corrigé le 2026-08-01**

> **Le constat ci-dessous décrit l'état du moteur au moment de l'étude ; il a depuis été
> corrigé.** `legal_plays()` rend maintenant `hand` dès que le partenaire est maître, sans
> exception — le sous-coup du partenaire est redevenu légal. Le test
> `test_partner_cut_only_trump_is_free_choice` épingle les trois cas, et `docs/RULES.md` ainsi
> que `CLAUDE.md` portent la note de rupture : le correctif **élargit** l'ensemble des coups
> légaux, donc toute donnée DD antérieure est périmée.
>
> On garde le constat d'origine parce que c'est lui qui justifie le changement.

`play.rs`, l. 67-94, branche `partner_is_master(state)` — **avant correction** :

```rust
// Rule 2.1: Partner is winning → can discard (play anything)
// Rule 4 exception: if partner CUT with trump and we only have
// trump left, must overtrump the best trump on table.
let non_trump = hand & !SUIT_MASK[trump_suit as usize];
if non_trump != 0 {
    return hand;                              // défausse libre, sous-coup compris
}
// Only have trump. Did partner cut (play trump on non-trump lead)?
…
if partner_card != EMPTY && card_suit(partner_card) == trump_suit {
    // Partner cut with trump → must overtrump best trump on table
    …
    if higher != 0 { return higher; }         // MONTÉE OBLIGATOIRE sur le partenaire
    return in_trump;
}
```

`docs/RULES.md`, section « Partner Cut Exception », dit la même chose en clair : « But if you
**only have trump cards**, you must overtrump the highest trump on the table if possible. »

C'est **mot pour mot la règle 4 de `LOCAL_regles_officielles_belote_contree.txt`** — « il
**est** obligé de fournir un atout supérieur » — c'est-à-dire la seule source du corpus qui
dise cela, et celle dont l'axe 10 montre qu'elle a très probablement inversé une négation en
rééditant le texte FFB 2015. Les trois autres rédactions FFB, plus Wikipédia Belote, disent
l'inverse : c'est précisément le cas où un atout inférieur est permis (2015), voire
explicitement « y compris un atout inférieur au sien » (2016 §5.2.3).

Colver est donc, sur ce point précis, **seul avec une source unique et probablement fautive**.
Le commentaire du code (« Rule 4 exception ») confirme que la règle a été portée depuis ce PDF.
Le reste de la branche est en revanche conforme à la FFB 2016 : quand on a des cartes hors
atout, `return hand` autorise bien la défausse **et** le sous-coup du partenaire.

Portée réelle du désaccord : le cas exige (a) ne pas avoir la couleur demandée, (b) n'avoir
plus **que** des atouts, (c) que le partenaire ait coupé et tienne le pli. C'est rare — la
FFB écrit elle-même « plutôt rare » — mais quand il se produit, Colver **retire au joueur une
option légale** selon la FFB contrée 2016, ce qui biaise aussi bien les rollouts que les
solveurs DD qui s'appuient sur `legal_plays()`.

### Axe 1 — sens du jeu : Colver applique **la règle antihoraire** (camp contrée/coinche)

`state.rs` l. 131-132 :

```rust
// First bidder is to the right of the dealer (dealer+1 mod 4).
let first_bidder = (dealer + 1) % 4;
```

`bidding.rs` l. 175 et `cfn.rs` l. 178 posent `trick_lead = (state.dealer + 1) % 4` : **le
premier parleur est aussi l'entameur**, ce qui est le consensus de l'axe 5. La rotation est
la même pour les enchères (`bidding.rs` l. 162) et pour le jeu (`play.rs` l. 256) :
`(current_player + 1) % 4`.

Le moteur implémente donc bien « donne, parole et jeu tournent dans le même sens, en partant
du voisin **de droite** du donneur ». Réserve de vocabulaire, sans effet sur les règles :
`CLAUDE.md` étiquette les sièges « 0=N, 1=E, 2=S, 3=W », or le siège 1 est celui à la
*droite* du siège 0 — à une vraie table, c'est l'Ouest. Les lettres sont des étiquettes de
sérialisation (notamment CFN), pas une géométrie de table ; l'incrément `+1` est
antihoraire au sens des règlements.

### Axe 2 — mélange : **hors périmètre du moteur**, et c'est un choix cohérent

Il n'existe dans `colver-core/src/engine/` **ni mélange ni coupe modélisés**. La seule
occurrence de `shuffle` est `state.rs` l. 163, dans `deal_random` :

```rust
let mut cards: [u8; 32] = core::array::from_fn(|i| i as u8);
cards.shuffle(rng);
```

C'est un tirage uniforme des 4 mains, pas la simulation d'un mélange physique. Autrement dit
Colver se place implicitement du côté **« mélange obligatoire et parfait »** (celui de la
FFB), et ne peut pas représenter la tradition « on ne mélange pas » — laquelle a pourtant un
effet de jeu réel et documenté (Pagat : les joueurs déduisent la donne suivante de l'ordre
des plis de la précédente). Ce n'est pas un bug : modéliser un mélange imparfait rendrait les
donnes corrélées entre elles, ce qui casserait l'hypothèse i.i.d. sur laquelle reposent le
pool DD et l'entraînement. À signaler seulement parce que cela **écarte du modèle une
information que de vrais joueurs exploitent**.

### Synthèse

| Divergence | Camp appliqué par Colver | Alignement dans le corpus |
|---|---|---|
| 9. « Pisser » | **On ne pisse pas** — défausse *et* sous-coup autorisés | FFB contrée/coinche + Midi. La liberté de sous-couper en plus n'est explicitement écrite que par la Coinche stéphanoise |
| 10. Partenaire a coupé, je n'ai que de l'atout | ~~Montée obligatoire~~ → **choix libre**, corrigé le 2026-08-01 | Suivait une **source unique** (`LOCAL_…`, édition « Équipe Ludique ») contre FFB 2015, FFB 2016 et Wikipédia Belote. Désormais aligné sur la FFB |
| 1. Sens du jeu | **Antihoraire**, parole = entame | Consensus contrée/coinche |
| 2. Mélange | **Non modélisé** (tirage uniforme) | Équivaut au camp « mélange obligatoire » de la FFB |
