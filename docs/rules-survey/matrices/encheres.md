# Les enchères — qui dit quoi

Sources : le corpus de [../README.md](../README.md). Collecte du 2026-08-01.
Pendant de [arrondi.md](arrondi.md), sur l'autre moitié du règlement.

---

## Avant de lire : ce qui compte comme un témoignage

**Le corpus est plus gros que ce que le README annonce** : **202 fichiers `.txt`**, pas 77. Il
contient aussi un dossier `clubs/` (jass suisse) et une vingtaine de pages Pagat sur d'autres
jeux de la famille (pilotta, baloot, klaverjas, sidi barrani, pandoer…). **Cette matrice ne
porte que sur la contrée et la coinche françaises.** Les autres jeux sont écartés, et pas
seulement par commodité : la « belote bulgare » (`gambiter_com_cards_Belote`, `licitum`,
`officialgamerules`, `belot_bg_*`) a bien des enchères, mais **ordinales et sans valeur de
points** (trèfle < carreau < cœur < pique < sans atout < tout atout). Les compter comme des
voix sur « l'enchère minimale » n'aurait aucun sens.

**Écartés aussi** : la belote *classique* (prise sur retourne), qui n'a pas d'enchère chiffrée
— `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016`,
`ffbelote_org_regles_officielle_belote`, `ffbelote_org_reglements_de_la_belote_avec_ou_sans_annonce`,
`pagat_com_jass_belote`, `ibelote`, `playjoy_com_en_belote_rules`, et la majorité des
règlements de concours de `tournois/` (fnasce `Belote_Reglement_cle19a7cf` et
`reglement_belote`, pontdeclaix, lagrandcombe, cdfcasson, lesamisdutemps, footeo/plouay,
aappma, tcvb, web_myassoc, geraudot, rjcv), qui précisent tous « **sans annonce** » et se
jouent à la retourne. Muets sur les enchères, ils ne sont pas d'accord avec qui que ce soit.
Deux d'entre eux méritent d'être cités quand même, parce qu'ils suppriment ce que d'autres
tiennent pour acquis : `aappmakoenigshoffen` (« **Il n'y a pas d'annonce (ni belote)** ») et
`villeconin` (fiche FFB), dont l'annexe Sans Atout / Tout Atout est reprise mot pour mot par
la page contrée du site FFB.

### Copies verbatim — à ne compter qu'une fois

Mesuré par recouvrement lexical sur le corpus entier, puis vérifié à la main :

| Groupe | Fichiers | Ce qu'il faut en faire |
|---|---|---|
| **Texte FFB « contrée/coinche » du site** | `federations/ffbelote_org_belote_contree` ≈ `ffbelote_org_regles_coinche` (0,97) ; `tournois/cdf_missegre11_…` (0,85) ; `divers/carafons_fr_…` (0,88) ; `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee` ; `apps-sites/re_belote_fr_belote_sans_atout_tout_atout` (annexe SA/TA) | **Une seule voix**, celle de la FFB — mais voir ci-dessous : les copies ont *muté*. |
| **Texte FFB PDF** | `ffbelote_REGLES-DE-LA-BELOTE-CONTREE` ≈ `ffbelote_org_…REGLES_DE_LA_BELOTE_COINCHEE_pdf` (0,93) ≈ `LOCAL_regles_officielles_belote_contree` (0,85) ≈ `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016` (0,82) | Quatre rédactions d'un même texte. Sur les enchères elles divergent peu (sauf §capot et §surcontre). |
| **« Tournoi international »** | `divers/belotecontree_free_reglement` = `apps-sites/belotecontree_free_fr_article_php3_id_article_22` (même page, deux extractions) ≈ `tournois/ainesruraux_saintsever_…` (0,998) | **Une seule voix.** |
| **Pagat** | `divers/pagat_com_jass_coinche` ≈ `apps-sites/gambiter_com_cards_jass_coinche` (0,97) ≈ `divers/reglesdejeux_github_io_…` (**traduction automatique** du même texte : « coinche » y devient « pièce de monnaie », « valet » « cric ») | **Une seule voix**, celle de John McLeod. C'est la plus riche du corpus, mais c'est **une** source, pas trois. |
| **ASCEE 2A** | `tournois/fnasce_org_IMG_pdf_reglement` = `tournois/fnasce_org_IMG_pdf_belote_reglement_cle1a43c7` — section « Enchère » **strictement identique mot pour mot** (seule la fin de partie diffère : 1 010 vs 1 000 points) | **Une seule voix.** |
| **ange.heureux** | `divers/ange_heureux_free_fr_JeuxDeCartes_La_Coinche` ≈ `apps-sites/ange_heureux_free_fr_Jeux_LaCoinche` (0,93) | Une seule voix (variante Rhône-Alpes / Dauphiné). |
| **drasill (BGA)** | `divers/drasill_github_io_bga_coinche_rules_fr` = `…rules_en` (traduction stricte) ; `open-source/drasill_bga-coinche_master_coinche.game.php` en est **l'implémentation** | Deux voix à ne pas confondre : la doc et le code **ne disent pas la même chose** (cf. axe 11). |
| **IsCool** | `divers/iscool_…faq_157…` = `apps-sites/iscool_…faq_157…` (doublon de fichier) ≈ `apps-sites/iscool_…faq_497…` (0,96) | Une seule voix — mais `apps-sites/iscool_…faq_701…` est un **mode de jeu différent** et compte à part. |
| **Doublons de fichiers purs** | `apps-sites/jeux_regles_com_…` = `divers/jeux_regles_com_…` ; `apps-sites/exoty_com_regles_coinche_belote` = `divers/…` | Une seule voix chacun. |

**Le cas intéressant : carafons ↔ cartesetcie.** Ce sont deux copies du même texte FFB,
phrase pour phrase — et pourtant elles **se contredisent sur deux axes d'enchères** : plafond
**180** vs **650**, coinche **« à tout moment »** vs **« on ne peut pas Coincher à la
volée »**. La copie n'est donc pas une garantie d'accord : les recopieurs éditent. Même chose
entre la page FFB et sa copie de Missègre, qui autorise « **Je passe, Passe ou Allez** » là où
l'original interdit tout sauf « Je passe ». Chaque fois que c'est le cas, c'est signalé plus bas.

### Sources sans ligne `SOURCE:`

Quatre fichiers n'ont pas d'URL en tête (PDF déposés localement, ou dépôts de code) :
`federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt`,
`federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt`,
`federations/LOCAL_regles_officielles_belote_contree.txt`, et
`divers/belotecontree_free_reglement.txt` (dont le jumeau `apps-sites/…article_php3…` porte
`http://belotecontree.free.fr/article.php3?id_article=22`). Tous les fichiers d'`open-source/`
sont dans ce cas. Ils sont cités par nom de fichier seul, sans URL inventée.

---

## 1. Enchère minimale : 80 ou 82 ?

### Position A — « 80 », sans commentaire

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| min = 80 | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` §4.2.2 (pas d'URL, PDF FFB) | « L'enchère doit être **comprise entre 80 et 160**, doit être un multiple de 10 et doit être supérieure à l'enchère précédente. » |
| min = 80 | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` (pas d'URL) | idem, mot pour mot |
| min = 80 | `federations/LOCAL_regles_officielles_belote_contree.txt` §4.1b (pas d'URL) | « L'enchère doit être **de 80 au minimum** » — *sans plafond* |
| min = 80 | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` · [url](https://www.ffbelote.org/wp-content/uploads/2015/11/REGLES-DE-LA-BELOTE-COINCHEE.pdf) | « L'enchère doit être de 80 au minimum » |
| min = 80 | `federations/ffbelote_org_belote_contree.txt` · [url](https://www.ffbelote.org/belote-contree/) | « annoncer un contrat **de 80 à 160**, ou bien annoncer un capot » |
| min = 80 | `federations/ffbelote_org_regles_coinche.txt` · [url](https://www.ffbelote.org/regles-coinche/) | « annoncer un contrat **de 80 à 650** » |
| min = 80 | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` · [url](https://fr.wikipedia.org/wiki/Belote_contr%C3%A9e) | « L'enchère doit être comprise entre 80 et 160 » ; « La première enchère doit être **au minimum de 80 points** » |
| min = 80 | `divers/fr_wikipedia_org_wiki_Coinche.txt` · [url](https://fr.wikipedia.org/wiki/Coinche) | « Les enchères débutent au minimum à 80, ce qui correspond presque à la moitié du total des points (162) » |
| min = 80 | `divers/belotecontree_free_reglement.txt` (« tournoi international ») | « Les annonces commencent à un **minimum de 80** et s'élèvent par tranche de 10 points minimum jusqu'à 160. » |
| min = 80 | `tournois/cdf_missegre11_com_medias_files_belote_contre_e_pdf.txt` · [url](http://www.cdf-missegre11.com/medias/files/belote-contre-e.pdf) | « annoncer un contrat de 80 à 160, ou bien annoncer un capot » |
| min = 80 | `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` (ASCEE 2A) · [url](https://www.fnasce.org/IMG/pdf/reglement.pdf) | « Les enchères doivent être au minimum de **80 points** et doivent être des multiples de 10 » |
| min = 80 | `divers/ange_heureux_free_fr_JeuxDeCartes_La_Coinche_html.txt` · [url](http://www.ange.heureux.free.fr/JeuxDeCartes/La_Coinche.html) | « les valeurs des enchères possibles sont : 80 - 90 - 100 - 110 - 120 - 130 - 140 - 150 - 160 » |
| min = 80 | `divers/alhoa_free_fr_ALH_belote_rules_htm.txt` · [url](http://alhoa.free.fr/ALH/belote_rules.htm) | « Les enchères débutent au minimum à 80. » |
| min = 80 | `divers/clubdejeux_com_belote_coinchee_online_regles.txt` · [url](https://www.clubdejeux.com/belote-coinchee-online/regles) | « Les enchères possibles vont de 80 à 160, par pas de 10. » |
| min = 80 | `divers/adpoker_fr_belote_contree_html.txt` · [url](https://www.adpoker.fr/belote-contree.html) | « La hiérarchie des annonces est la suivante: 80 points dans une couleur. 90 points… 160 points… capot. » |
| min = 80 | `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` · [url](https://www.jeu-belote.fr/regles.php?part=regles-jeu-coinche) | « un contrat allant de 80 à 160 (de 10 en 10) » |
| min = 80 | `divers/regles_com_jeux_cartes_coinche_html.txt` · [url](https://www.regles.com/jeux-cartes/coinche.html) | « Le minimum est 80, puis on monte de 10 en 10. » |
| min = 80 | `divers/exoty_com_regles_coinche_belote.txt` · [url](https://exoty.com/regles-coinche-belote) | « Les enchères commencent à 80 points minimum et augmentent de 10 en 10. » |
| min = 80 | `divers/belotepoint_fr_regles_coinche.txt` · [url](https://www.belotepoint.fr/regles-coinche) | « minimum 80, par paliers de 10 » |
| min = 80 | `divers/jeux_regles_com_regles_coinche.txt` · [url](https://jeux-regles.com/regles-coinche/) | « Les contrats démarrent à 80 points » |
| min = 80 | `divers/bk_jeux_ducale_…_belote_coinchee_pour_joueur_expert_pdf.txt` · [url](https://bk.jeux-ducale.fr/app/uploads/2022/06/cartes-a-jouer-Ducale-regle-jeu-belote-coinchee-pour-joueur-expert.pdf) | « L'annonce minimum est de 80 jusqu'à 180 (160 + belote) puis capot » |
| min = 80 | `divers/iscool_…faq_157…txt` · [url](https://iscool.helpshift.com/hc/fr/10-belote-mobile/faq/157-how-to-play-coinche-coinche-rules/) | « en faire une d'au moins 80 points à une couleur donnée » |
| min = 80 | `apps-sites/playjoy_com_en_coinche_rules.txt` · [url](https://playjoy.com/en/coinche/rules/) | « Choose a contrat value from **80 to 160** points. » |
| min = 80 | `apps-sites/gameduell_…faq_1056_contree` · [url](https://gameduell.helpshift.com/hc/fr/16-belote-com---belote-coinche/faq/1056-contree/) | « Les enchères possibles sont les suivantes : 80, 90, …, 160, 260 (capot), 500 (générale). » |
| min = 80 | `apps-sites/exoty_com_regles_contree_belote.txt` · [url](https://exoty.com/regles-contree-belote) | « Les enchères montent par paliers de 10 points, de 80 à 160. » |
| min = 80 | `tournois/data_over_blog_kiwi_…reglement-table-coinche.pdf.txt` · [url](http://data.over-blog-kiwi.com/1/05/17/17/20150128/ob_1f68a4_2015-01-27-reglement-table-coinche.pdf) | « les enchères commencent à 80 points minimum » |
| min = 80 | `tournois/maisondesessarts_fr_article116_html.txt` · [url](https://www.maisondesessarts.fr/article116.html) | « le contrat (un nombre multiple de 10) **à partir de 80** et une couleur » |
| min = 80 | `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` · [url](https://casimirdehauteclocque.fr/jeux/coinche.pdf) | « Il y a dix valeurs de contrats possible : 80, 90, …, 160, capot. » |
| min = 80 | `apps-sites/ludicash_com_help_rules_coinche.txt` · [url](https://www.ludicash.com/help/rules-coinche) | « The bid must be at least 80 points or higher than the previous bid by a multiple of 10. » |
| min = 80 | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « Minimum opening bid: **80**. » |
| min = 80 | `open-source/ilyesbrh_twistedFate-belote_main_docs_games_coinche_GAME_RULES.md` | « **Minimum bid**: 80 » |
| min = 80 (code) | `open-source/theosaulus_coinche_main_coinche_utils.py` | `bid_value = 80 + ((action - 1) // 4) * 10` |
| min = 80 (code) | `open-source/ismo009_Coinche_main_game.js` | `const validPoints = [80, 90, 100, 110, 120, 130, 140, 150, 160, 250, 270, 500];` |

### Position B — « on dit 80, il faut faire 82 »

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| enchère = 82, prononcée « 80 » | `divers/pagat_com_jass_coinche_html.txt` · [url](https://www.pagat.com/jass/coinche.html) | « The bid must be for **at least 82 points (by convention, 82 is bid by saying “80”)**, must be a multiple of 10 » — et au score : « at least 82 if they bid 80 » |
| idem (copies) | `apps-sites/gambiter_com_cards_jass_coinche_html.txt` · [url](https://gambiter.com/cards/jass/coinche.html) ; `divers/reglesdejeux_github_io_…` · [url](https://reglesdejeux.github.io/regles-du-jeu-la-coinche/index.html) | même phrase — **ne comptent pas comme deux témoins de plus** |
| min = 82 | `divers/drasill_github_io_bga_coinche_rules_fr_html.txt` · [url](https://drasill.github.io/bga-coinche/rules-fr.html) | « Les enchères commencent à **82 points, soit la moitié du jeu** » |
| min = 82, dans le code | `open-source/drasill_bga-coinche_master_coinche.game.php` | `// Current bid value (from 82 to 170)` — et au décompte : `if ($bid == 82) { $bid = 80; }` (deux fois) |
| min = 82 | `apps-sites/en_boardgamearena_com_gamepanel_game_coinche.txt` · [url](https://en.boardgamearena.com/gamepanel?game=coinche) | « **Bidding starts with 82 points**, and goes up by units of 10. » |
| min = 82 | `apps-sites/en_wikipedia_org_wiki_Coinche.txt` · [url](https://en.wikipedia.org/wiki/Coinche) | « Bidding's start at least at **82 points (shortened to a bid/call of 80)** » |
| min = 82, échelle décalée | `apps-sites/contree_org_4_joueurs.txt` · [url](http://contree.org/4-joueurs/) | « L'enchère minimum … est donc de **81+1 = 82 points** … Un palier est passé tous les 10 points : **82-92-102-112-122-132-142-152-162-182-222** » |
| min = 82 | `tournois/web_archive_org_…coinche_stephanoise…reglement_coinche.pdf.txt` · [url](https://web.archive.org/web/2020/http://coinche-stephanoise.com/mesdocuments/reglement_coinche.pdf) | « minimum d'un contrat est de **82**… » |
| min = 82 | `apps-sites/regles_de_jeux_com_regle_coinche.txt` · [url](https://www.regles-de-jeux.com/regle-coinche/) | « Elle doit aussi atteindre un minimum de **82 points** de plus hors annonces et belote-rebelote » |
| min = 82 | `divers/jeubelote_com_regle_de_la_belote_html.txt` · [url](https://www.jeubelote.com/regle-de-la-belote.html) | « L'enchère minimale étant **82 points** puis 90,100 etc.. jusqu'à 162 ou 250 » |

### Position C — « 80 s'annonce et se marque, mais 82 pour l'honorer »

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| 80 annoncé, 82 exigé | `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` §2.4.5 | « Le contrat « 80 » est particulier. **Il faut en réalité marquer 82 points pour honorer ce contrat** (soit strictement plus de la moitié des points en jeu). » |
| idem | `apps-sites/gamerules_com_rules_coinche.txt` · [url](https://gamerules.com/rules/coinche/) | « The lowest bid allowed is 80, **although if a bid of 80 is made the team must earn 82 points** » |
| idem | `tournois/data_over_blog_kiwi_…` | « Seule la demande de 80 implique une [exigence de 82] » |
| idem | `divers/carafons_fr_regles_de_la_coinche.txt` · [url](https://carafons.fr/regles-de-la-coinche/) | contrat réussi = « supérieur ou égal au contrat annoncé, **et de minimum 82 points** » |
| **contrat + 2, toujours** | `tournois/maisondesessarts_fr_article116_html.txt` | « le preneur doit réaliser **son contrat + 2 points** et réaliser plus de points que l'adversaire » |
| variante « +2 partout » | `divers/pagat_com_jass_coinche_html.txt` | « A few groups require an extra 2 points for every bid - for example **at least 112 points to win a bid of 110**. » |

### Position D — « non, 80 c'est 80 »

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| pas de seuil 82 | `apps-sites/gameduell_…faq_1056_contree` | « il **n'est pas obligatoire** pour l'équipe preneuse d'atteindre un minimum de 82 pts de plis … pour un contrat de 80 Pique : il suffit à l'équipe de réaliser **exactement 80** points » |
| pas de seuil 82 | `federations/*` (les 6 fichiers FFB) | 82 **n'apparaît nulle part**. « Le contrat est réussi si les preneurs obtiennent un total **supérieur ou égal à l'enchère demandée**. Ceci est valable **même si les défenseurs ont réalisé plus de points** que les preneurs. » |
| pas de seuil 82 | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « si une équipe annonce 80, qu'elle **réalise exactement 80 points** et que l'équipe adverse fait 82 points …, le contrat est considéré comme réussi » |

**Divergence.** La ligne de fracture n'est pas géographique, elle est **logique** : 82 est le
vestige de la règle « le preneur doit faire plus de points que la défense » (162/2 = 81, donc
82). Pagat le dit noir sur blanc : « *It is a vestige of this rule that requires a score of at
least 82 to win a bid of 80* ». Les sources qui ont abandonné cette seconde condition — dont
**toute la FFB** — n'ont plus aucune raison d'exiger 82, et ne l'exigent pas. Les sources
anglophones et les implémentations (BGA, drasill) ont gardé les deux. **Le corpus contient
donc deux jeux cohérents, et une zone floue au milieu** (carafons, casimir, gamerules) qui
exige 82 sans exiger de battre la défense — un état intermédiaire qui ne se justifie que par
la tradition.

---

## 2. Pas d'enchère et enchère maximale

### Pas : consensus sur 10

Toutes les sources qui se prononcent disent « multiple de 10 » ou « par tranche de 10 points
minimum ». Trois exceptions, toutes annoncées comme telles :

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| pas de 5 | `divers/fr_wikipedia_org_wiki_Coinche.txt` (liste de variantes) | « les enchères tout comme le décompte des points **se font de 5 en 5** au lieu de 10 en 10 » |
| pas de 10 **ou plus** | `divers/drasill_github_io_bga_coinche_rules_fr_html.txt` | « montent de 10 points en 10 points (**ou plus**) » |
| pas de 10 sur base 82 | `apps-sites/contree_org_4_joueurs.txt` | 82-92-102-…-162, puis **182**, puis **222** |

**Consensus.** La seule vraie question est le point de départ de l'échelle (axe 1), pas son
incrément.

### Plafond : c'est là que ça casse

| Plafond | Source | Ce qu'elle dit |
|---|---|---|
| **160** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016` ; `ffbelote_REGLES-DE-LA-BELOTE-CONTREE` ; `ffbelote_org_belote_contree` ; `divers/fr_wikipedia_…Belote_contrée` ; `divers/belotecontree_free_reglement` ; `tournois/cdf_missegre11_…` ; `ange_heureux` ; `clubdejeux` ; `adpoker` ; `jeu_belote_fr` ; `iscool` ; `exoty (contrée)` ; `playjoy` ; `jeux_regles` ; `casimir` ; `maviedesenior` ; `apps-sites/gameduell_…contree` | « comprise **entre 80 et 160** » (FFB) ; « l'enchère la plus grande (hormis le capot) est **160** » (Wikipédia) |
| **160** (code) | `open-source/theosaulus_coinche_main_coinche_utils.py` ; `open-source/ismo009_Coinche_main_game.js` | 36 actions ÷ 4 couleurs = 9 paliers de 80 à 160 ; `validPoints = [80, …, 160, 250, 270, 500]` |
| **160**, dit comme variante | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « la valeur maximale de l'enchère est de 160 points (**à la contrée en particulier**) » — c'est-à-dire que ce n'est *pas* la règle par défaut de la coinche |
| **162** | `divers/belotepoint_fr_regles_coinche.txt` | « Les enchères montent ainsi **de 80 à 162 points** (le total maximum de points de pli dans une manche) » |
| **162 ou 250** | `divers/jeubelote_com_regle_de_la_belote_html.txt` | « jusqu'à **162 ou 250** (capot) » |
| **170** | `open-source/drasill_bga-coinche_master_coinche.game.php` | `// Current bid value (from 82 to 170)` |
| **180** (= 160 + belote) | `divers/bk_jeux_ducale_…pdf.txt` | « L'annonce minimum est de 80 **jusqu'à 180 (160 + belote)** puis capot » |
| **180** | `divers/carafons_fr_regles_de_la_coinche.txt` | « annoncer un contrat **de 80 à 180** » |
| **180** | `divers/alhoa_free_fr_ALH_belote_rules_htm.txt` | « Elles peuvent aller à 160, **voire 180 avec la belote (rare)** » |
| **180**, avec la raison | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « Maximum numeric bid: **180**. […] The 170 and 180 steps are **only feasible with Belote** in hand … The auction does **not** enforce that constraint at bid time — announcing 170 / 180 without Belote is legal but commits the bidder to a contract they cannot make on cards alone » |
| **222** | `apps-sites/contree_org_4_joueurs.txt` | « 82-92-…-162-**182-222** » |
| **650** | `federations/ffbelote_org_regles_coinche.txt` | « annoncer un contrat **de 80 à 650** » |
| **650** | `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` · [url](https://cartesetcie.fr/regle-du-jeu-la-belote-coinchee/) | « un contrat de **80 à 650 points**, ou annoncer un capot » (copie de la page FFB coinche) |
| **650** | `open-source/ilyesbrh_twistedFate-belote_main_docs_games_coinche_GAME_RULES.md` | « up to 160 (**and beyond by mutual escalation**) … **Maximum: 650, or capot** » |
| **aucun** | `tournois/maisondesessarts_fr_article116_html.txt` ; `tournois/fnasce_org_IMG_pdf_reglement` | aucun plafond énoncé ; Les Essarts donne « (ex : 90 pique, 120 trèfle, **250 cœur**…) » |
| **muet** | `pagat` (+ copies), `regles_com`, `lemagloisirs`, `gamerules`, `en_wikipedia`, `gameduell (en)`, `exoty` | pas de plafond chiffré ; le seul plafond est le capot |

**Divergence, et elle est arithmétique.** Trois familles :

1. **160** — plafond posé sur les *plis seuls* (162 arrondi vers le bas). C'est la contrée.
2. **170–222** — plafond qui **intègre la belote** (+20 → 180) ; ContrAI est le seul à expliquer
   pourquoi et à noter que le moteur ne le vérifie pas.
3. **650** — plafond qui intègre **les annonces** (162 + carré de valets 200 + cent 100 + …).
   C'est la coinche : quand les annonces comptent pour le contrat, le plafond des plis ne veut
   plus rien dire.

La ligne de fracture est donc **« les annonces comptent-elles ? »** (axe 12), pas une
préférence de plafond. Et **c'est la FFB qui l'illustre le mieux, contre elle-même** : sa page
contrée dit 160, sa page coinche dit 650 — même site, même semaine, deux jeux.

---

## 3. Surenchérir : en valeur seulement, ou aussi en couleur ?

### Position A — la valeur seule ; les couleurs ne sont pas ordonnées

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| valeur seule, explicite | `divers/ange_heureux_free_fr_JeuxDeCartes_La_Coinche_html.txt` | « Pour pouvoir parler, il faut monter au niveau de la valeur. **Par contre, il n'y a pas d'ordre au niveau de la couleur.** » |
| valeur seule, explicite | `divers/clubdejeux_com_belote_coinchee_online_regles.txt` | « **Les couleurs n'étant pas ordonnées**, le nombre de points doit donc [être] plus élevé. » |
| valeur seule, explicite | `divers/alhoa_free_fr_ALH_belote_rules_htm.txt` | « **La valeur des couleurs est, elle, identique, à la différence du bridge.** » |
| valeur seule, explicite | `divers/iscool_…faq_157…txt` | « surenchérir d'au moins 10 points sur la dernière annonce **dans n'importe quelle couleur** » |
| valeur seule | `federations/*` (FFB, les 6 fichiers) | « doit être **supérieure à l'enchère précédente** » — aucune hiérarchie de couleurs nulle part |
| valeur seule | `divers/adpoker_fr_belote_contree_html.txt` | « à condition d'annoncer un contrat dont **la valeur en points est supérieure** (par exemple: après 80 ♥, 100 ♠) » |
| valeur seule | `divers/bk_jeux_ducale_…pdf.txt` | « surenchérir avec un nombre de points plus grand et **la couleur de son choix** … Les joueurs ont le droit d'annoncer à une couleur choisie par quelqu'un d'autre. » |
| valeur seule | `divers/pagat_com_jass_coinche_html.txt` (règle de base) | « must be higher than the previous bid » |
| valeur seule | `apps-sites/contree_org_4_joueurs.txt` | « sur « 82 à cœur » il peut demander « 92 à Pique » **ou tout autre montant supérieur avec la couleur d'Atout de son choix** » |
| valeur seule | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « Each new bid must be **strictly higher** than the current one » (aucune contrainte de couleur) |
| valeur seule (code) | `open-source/ismo009_Coinche_main_game.js` | `if (this.contract && bid.points <= this.contract.points) { … }` — seul test ; **on peut donc réenchérir dans la même couleur** |

### Position B — il existe un ordre des couleurs

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| ordre local des couleurs | `divers/belotepoint_fr_regles_coinche.txt` | « Chaque enchère doit être supérieure à la précédente, soit par un montant plus élevé, **soit par une couleur considérée comme prioritaire selon les conventions locales**. » |
| ordre ♣<♦<♥<♠<SA<TA, **en variante** | `divers/pagat_com_jass_coinche_html.txt` | « Some introduce a ranking order of suits … from lowest to highest: **clubs, diamonds, hearts, spades, sans atout, tout atout**. A bid can be overcalled by an equal bid in a higher denomination … 100 clubs is higher than 90 sans atout » ; « This variant leads to **lengthier auctions** » |
| SA/TA sur une autre échelle | `divers/fr_wikipedia_org_wiki_Coinche.txt` | tableau d'équivalence Sans atout 70 ↔ Atout 80 ↔ Tout atout 130 : une hiérarchie de dénominations déguisée en conversion |

### Position C — obligation de changer de couleur, mais seulement pour se relancer soi-même

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| couleur obligatoire sur soi-même | `open-source/drasill_bga-coinche_master_coinche.game.php` | `if ($previousColor == $color && $playerId == $previousBidPlayer) { throw … 'You must change color to bid higher on yourself'; }` |
| idem | `divers/ange_heureux_free_fr_…` ; `tournois/data_over_blog_kiwi_…` ; `apps-sites/en_wikipedia_org_wiki_Coinche.txt` | voir axe 4b |

**Consensus, avec une exception nommée.** L'écrasante majorité du corpus dit : **la valeur
seule**, et quatre sources prennent la peine de le dire *contre le bridge*. La hiérarchie des
couleurs n'est défendue en règle principale que par une source (belotepoint, en renvoyant aux
« conventions locales ») ; Pagat la donne comme variante. Là où elle réapparaît vraiment,
c'est **entre couleur, SA et TA** (axe 8) — et, sous une tout autre forme, comme *condition de
relance sur soi-même* (axe 4b).

---

## 4. Parler sur soi-même / surenchérir sur son partenaire

Trois questions que les textes mélangent constamment.

### 4a. Se relancer soi-même quand les trois autres ont passé — non

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| interdit | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` | « **Il est impossible de surenchérir sur soi-même.** Le tour d'enchères prend fin lorsqu'il revient à la dernière personne ayant fait une annonce, autre que « je passe ». » |
| interdit | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE`, `…COINCHEE_pdf`, `LOCAL_…` | « Il est impossible de surenchérir sur soi-même : le tour d'enchère s'achève lorsqu'**après une enchère ou un contre, les trois joueurs suivants passent**. » |
| interdit | `federations/ffbelote_org_belote_contree` + `…regles_coinche` | « Les enchères se terminent après 3 « Passe » consécutifs. **Un joueur ne peut donc surenchérir sur lui-même** si tous les autres joueurs ont passé. » |
| interdit | `tournois/cdf_missegre11_…` | même phrase (copie FFB) |
| interdit | `divers/belotecontree_free_reglement.txt` | « Un joueur ne peut « **parler sur lui-même** » c'est-à-dire que si trois joueurs passent, aucune annonce nouvelle n'est plus possible et c'est le contrat annoncé en dernier qui est joué. » |
| interdit | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « Il est impossible de surenchérir sur soi-même. » |
| interdit | `divers/fr_wikipedia_org_wiki_Coinche.txt` | fin des enchères « lorsque trois joueurs d'affilée n'ont rien annoncé (**impossibilité de monter sur sa propre enchère**) » |
| interdit | `divers/carafons_fr_…` / `divers/cartesetcie_fr_…` | « Un joueur ne peut surenchérir sur lui-même si les autres joueurs ont "passé". » |
| interdit | `divers/alhoa_free_fr_…` | « Un joueur ne peut donc pas surenchérir sur sa propre annonce si tout le monde a passé entre temps. » |
| interdit | `apps-sites/gameduell_…faq_1054_coinche` · [url](https://gameduell.helpshift.com/hc/en/16-belote-com---belote-coinche/faq/1054-coinche/) | « (the player is **not allowed to overbid if they made the last offer**) » |
| interdit | `apps-sites/ludicash_com_help_rules_coinche.txt` | « A player cannot therefore outbid himself if all the other players have passed. » |

### 4b. …sauf en changeant de couleur

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| autorisé si changement de couleur | `divers/ange_heureux_free_fr_JeuxDeCartes_La_Coinche_html.txt` | « quand trois joueurs passent, **le dernier qui a parlé a le droit de reparler mais il est obligé de changer la couleur demandée. Les autres joueurs ont alors le droit de reparler.** » |
| autorisé si changement de couleur | `tournois/data_over_blog_kiwi_…reglement-table-coinche.pdf.txt` | « **on ne peut pas remonter sur soit même sauf si on change de couleur** » |
| autorisé si changement de couleur | `apps-sites/en_wikipedia_org_wiki_Coinche.txt` | « one player can not bid higher than himself **except if he changes the suit** » |
| autorisé si changement de couleur (code) | `open-source/drasill_bga-coinche_master_coinche.game.php` | `'You must change color to bid higher on yourself'` |
| en variante, **les deux** | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « on peut surenchérir sur sa propre enchère » **et** « on peut surenchérir sur sa propre enchère **uniquement si on propose une autre couleur** » — listées comme deux variantes distinctes |

### 4c. Surenchérir sur son partenaire — autorisé, et c'est le cœur du jeu

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| autorisé et recommandé | `divers/adpoker_fr_belote_contree_html.txt` | « Pour annoncer vos As à votre partenaire qui a déclaré un contrat (par exemple, 90 ♥), demandez à votre tour un contrat avec **le même atout** en l'augmentant d'autant de fois dix que vous avez d'As » |
| autorisé | `divers/bk_jeux_ducale_…pdf.txt` | « si un joueur parle à 80 et que **son partenaire** a le Valet de Trèfle, ce partenaire pourra annoncer 100 à son tour » |
| autorisé | `tournois/maisondesessarts_fr_article116_html.txt` | « Les enchères ne sont pas closes pour autant et l'adversaire **ou le partenaire** peut surenchérir. » |
| autorisé | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « **relancer 10 points son partenaire** revient à lui annoncer un pli supplémentaire (1 as hors atout ou 2 atouts dont le neuf) » |
| autorisé | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` (conventions) | tout le système « aux as » repose là-dessus : « le partenaire répondra 10 pour indiquer qu'il a le valet ou le neuf » |
| recommandé | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « If your partner has already bid and you can add value, **raise their contract** rather than start a new one in another suit » |
| autorisé (code) | `open-source/ismo009_Coinche_main_game.js`, `…coinche.game.php` | aucun contrôle d'équipe sur l'enchère — seul le **contre** est bridé par équipe |
| **aucune source du corpus ne l'interdit** | — | — |

**Consensus sur 4a et 4c ; divergence sur 4b.** Personne ne conteste qu'on soutient son
partenaire en montant ; personne ne conteste qu'on ne se relance pas soi-même dans le silence
général. La fracture est étroite et bien délimitée : **quatre sources indépendantes**
(ange.heureux, un règlement de table over-blog, Wikipédia EN, et l'implémentation BGA)
**rouvrent l'enchère au dernier parleur à condition qu'il change de couleur**, ce que la FFB
ferme sans réserve. Conséquence directe sur l'axe 11 : chez eux, trois passes **ne terminent
pas** les enchères.

---

## 5. Un joueur qui a passé peut-il reparler ?

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| oui | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` §4.2.1 | « **Un joueur qui a passé une fois pourra enchérir par la suite s'il en a l'opportunité.** » |
| oui | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE`, `…COINCHEE_pdf`, `LOCAL_…` | même phrase, mot pour mot (une seule voix FFB) |
| oui | `divers/pagat_com_jass_coinche_html.txt` | « Pass, **which does not prevent the player from bidding in future** if some other player has bid meanwhile. » |
| oui | `apps-sites/gamerules_com_rules_coinche.txt` | « When a player passes this **does not take them out** of a round of bidding. They can choose another option if their turn in the order comes up again. » |
| oui | `apps-sites/en_wikipedia_org_wiki_Coinche.txt` | « Pass (it does not prevent him from bidding later **with the exception if everyone passes**) » |
| oui | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « soit il annonce qu'il passe, **ce qui ne l'empêche pas de surenchérir aux tours suivants**. Une annonce " en passant " peut vouloir signifier à son partenaire de laisser partir l'équipe adverse » |
| oui | `divers/ange_heureux_free_fr_…` | « **On peut passer et reparler au tour suivant.** » |
| oui | `divers/bk_jeux_ducale_…pdf.txt` | « **Un joueur qui a déjà passé peut annoncer si le tour revient à lui.** » |
| oui | `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` | « Il est possible de proposer un contrat **même si on a passé auparavant** dans le tour d'enchère. » |
| oui | `apps-sites/ludicash_com_help_rules_coinche.txt` | « pass their turn, **while being able to speak in the next round** if a bid has been made in the meantime » |
| oui (implicite, stratégie) | `divers/exoty_com_regles_coinche_belote.txt` | « **Passer au premier tour pour monter au second** montre souvent à votre partenaire que vous avez du jeu » |
| oui, y compris pour contrer | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « A player who passes **may re-enter the bidding later** » ; « **Intervening passes do not close the Coinche / Surcoinche window** … players who passed earlier may re-enter and call *contre* or *surcontre* » |
| oui (code) | `open-source/ismo009_Coinche_main_game.js`, `…coinche.game.php` | passer n'inscrit aucun verrou ; toute enchère remet `passCount` à zéro |
| **muet** | carafons, cartesetcie, regles.com, jeux-regles, lemagloisirs, clubdejeux, belotepoint, adpoker, jeu-belote.fr, iscool, playjoy, missègre, ainesruraux, ASCEE, ilyesbrh | ne se prononcent pas |

**Consensus, le plus net de toute la matrice.** Treize sources indépendantes disent oui ;
**aucune source du corpus ne dit non.** La seule restriction, chez Wikipédia EN, n'en est pas
une : « sauf si tout le monde passe » décrit la fin du tour. Attention toutefois : « muet »
domine numériquement — beaucoup de vulgarisations n'y pensent tout simplement pas, et le
lecteur peut en déduire à tort qu'un passe est définitif.

---

## 6. Le capot : enchère ou bonus ? et qu'est-ce qui passe par-dessus ?

### 6a. Le capot est une enchère, et il ferme l'enchère chiffrée

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| enchère insurpassable | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` | « En lieu et place d'un nombre de points, il est possible **à tout moment** de demander un capot … **Il n'existe pas d'enchère supérieure au capot ; dès lors l'équipe adverse ne peut plus que passer ou contrer.** » |
| idem | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE`, `…COINCHEE_pdf` (« passer ou **coincher** »), `LOCAL_…` | même phrase |
| enchère insurpassable | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « Il n'existe pas d'enchère supérieure au capot ; dès lors l'équipe adverse ne peut plus que passer ou contrer. » |
| enchère bloquante | `divers/belotecontree_free_reglement.txt` / `tournois/ainesruraux_…` | « Ces annonces **bloquent le jeu**, c'est à dire qu'il n'est ensuite plus possible de parler à une autre couleur : **un capot demandé ne peut être que contré et éventuellement surcontré** » |
| enchère bloquante | `divers/adpoker_fr_belote_contree_html.txt` | « Autre enchère **stoppant toute possibilité de surenchérir**: le capot … Après une telle annonce, les adversaires **ne peuvent que contrer** » |
| enchère bloquante | `divers/iscool_…faq_157…txt` | « **Si la dernière annonce n'est pas un capot**, surenchérir d'au moins 10 points » |
| enchère bloquante | `tournois/web_archive_org_…coinche_stephanoise…` | les enchères s'arrêtent quand « Un joueur demande **« LE CAPOT »** » |
| enchère bloquante | `open-source/ilyesbrh_…coinche_GAME_RULES.md` | « **ends bidding but can be coinched** » |
| enchère bloquante, asymétrique | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « Slam outranks any numeric bid: **once declared, no further contract bid is legal** (numeric, Slam, or Solo Slam). *Contre* and *surcontre* remain available against a Slam. » |

### 6b. …et il n'est même pas contrable

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| capot non coinchable | `divers/pagat_com_jass_coinche_html.txt` | « **A capot bid ends the bidding and cannot be doubled.** » (+ variante : « Some allow a capot bid to be doubled and redoubled ») |
| capot non coinchable | `apps-sites/gamerules_com_rules_coinche.txt` | « This ends the round of bidding and **cannot be doubled in any way**. » |
| capot non coinchable, en variante | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « on ne peut coincher un adversaire qui **annonce un capot ou une générale** » |

**Divergence tranchée.** La FFB et toute la tradition française laissent le contre ouvert sur
un capot — et chiffrent même « capot contré 1 000 / surcontré 2 000 » ; **Pagat et gamerules le
ferment complètement.**

### 6c. …et l'exception qui montre le mécanisme

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| **les enchères continuent** | `tournois/maisondesessarts_fr_article116_html.txt` | « Les autres expressions comme … **capote, générale… ne sont pas autorisées** … Si on a un jeu pour mettre l'adversaire capot, **on peut demander par exemple 250 carreau.** … **Les enchères ne sont pas closes pour autant** et l'adversaire ou le partenaire peut surenchérir. » |

Ce règlement de concours est le plus instructif du lot : il **supprime le mot « capot » du
vocabulaire d'enchère** et le remplace par un nombre. Résultat, le capot cesse d'être un palier
spécial et redevient une enchère comme une autre — donc surpassable. Ce n'est pas une lubie
locale, c'est la démonstration que « le capot ferme l'enchère » est une convention de
*vocabulaire*, pas une nécessité du jeu.

### 6d. Valeur d'enchère du capot

| Valeur | Source | Ce qu'elle dit |
|---|---|---|
| **250** (contrat) | `federations/LOCAL_regles_officielles_belote_contree.txt` | « le montant du contrat demandé (**250 points pour un capot demandé**) » |
| **250** | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « Une annonce de capot équivaut à **250 points**. Un capot réussi rapporte donc 500 points » |
| **250** | `tournois/fnasce_org_IMG_pdf_reglement` (ASCEE 2A) | « **Capot annoncé = 250 points.** » (dans la section « Enchère ») |
| **250** | `ange_heureux`, `iscool`, `jeux_regles`, `playjoy`, `clubdejeux`, `exoty`, `belotepoint`, `ludicash`, `alhoa`, `casimir`, `maviedesenior` | « Le Capot vaut 250 points » |
| **250** (code) | `open-source/theosaulus_…utils.py` ; `open-source/valmathieu_ContrAI…md` ; `open-source/ismo009_…game.js` | `return (250, atout_suit) # capot is encoded as 250` ; « Contract base value **250** points » ; `validPoints = [… 160, 250, 270, 500]` |
| **250 + 270 « capot beloté »** | `open-source/ismo009_Coinche_main_game.js` | `// Capot beloté (270): tous les plis + belote/rebelote de l'attaque` — corrobore la variante Wikipédia « les annonces capot beloté et générale belotée sont supérieures respectivement aux capot et générale » |
| **260** | `apps-sites/gameduell_…faq_1056_contree` | échelle « …, 150, 160, **260 (capot)**, 500 (générale) » |
| **350** (aux points faits) | `divers/bk_jeux_ducale_…pdf.txt` | « Le capot non annoncé rapporte 250 points et le **capot annoncé rapporte 350 points** » |
| **500** (forfait, pas une enchère) | `federations/ffbelote_org_belote_contree.txt` ; `divers/belotecontree_free_reglement.txt` ; `tournois/cdf_missegre11_…` | « Le capot demandé et réalisé, ou chuté, vaut **500 points**. Le capot contré vaut 1.000. Le capot surcontré vaut 2.000. » |
| **pas une enchère du tout** | `apps-sites/contree_org_4_joueurs.txt` | le capot y est une **prime**, et elle s'appelle « générale » : « « **Générale** » est le nom de la prime de capot … forfait de 100 + 152 = 252 points » |
| **absent du code** | `open-source/drasill_bga-coinche_master_coinche.game.php` | le capot n'existe **qu'au décompte** : `// If a team scored zero points, it's a "capot", so 250pts` — jamais comme enchère |

**Consensus sur 250 comme valeur de contrat** (14 sources indépendantes) — mais **divergence
sur ce qu'est le capot** : une enchère (majorité), un forfait de score (FFB ancienne, tournoi
international), une prime non annonçable (contree.org, BGA), ou un simple contrat à 250 qu'il
est interdit d'appeler « capot » (Les Essarts).

---

## 7. La générale

### Position A — elle existe, un seul joueur fait les huit plis

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| existe, 500, **partenaire écarté** | `divers/belotepoint_fr_regles_coinche.txt` | « remporter tous les plis **à lui seul**, sans l'aide de son partenaire … **Ce dernier ne joue pas de la manche.** En cas de réussite, le bonus est de 500 points. » |
| existe, 500, **il entame** | `divers/ange_heureux_free_fr_…` | « Le joueur annonce qu'il va **lui-même** faire tous les plis, **qu'il entamera la partie** … La Générale vaut 500 points » |
| existe, 500 | `apps-sites/playjoy_com_en_coinche_rules.txt` | « If a **single player** succeeds in winning all 8 tricks …, adding 490 extra points to the “last ten” (**500 in total**). » |
| existe, 500 | `divers/maviedesenior_com_…` · [url](https://maviedesenior.com/loisirs/comment-jouer-a-la-belote-coinchee) | « ce dernier est dans l'obligation de **remporter seul** tous les plis, soit un total de 500 points » |
| existe, 500 | `apps-sites/gameduell_…faq_1056_contree` | échelle « …, 260 (capot), **500 (générale)** » ; « pour une générale …, le dernier pli rapporte **350** points » |
| existe, 500 | `divers/alhoa_free_fr_…` | « Générale (tous les plis pour **une personne**) … Remplacer ce nombre par 500 en cas de générale » |
| existe, valeur libre | `divers/lemagloisirs_fr_regle_coinche.txt` · [url](https://www.lemagloisirs.fr/regle-coinche/) | « un seul joueur annonce qu'il fera tous les plis **sans l'aide de son partenaire** … **Les valeurs exactes de capot et générale peuvent varier selon les cercles.** » |
| existe | `divers/jeux_regles_com_regles_coinche.txt` | « Générale : **le joueur qui prend doit gagner seul tous les plis.** » (valeur MUETTE) |
| existe, 500, **partenaire bridé** | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « the **bidder personally** will win every one of the 8 tricks — their **partner may play normally but is forbidden from winning any trick**. Contract base value **500**. … it **cannot be announced after a Slam** (asymmetric block) » |
| existe, 500 (code) | `open-source/slim0_contree_main_backend_game_scoring.py` | `if r.contract.bid.is_generale: contract_made = all(t.winner == r.contract.bid.position for t in r.tricks)` |
| existe, 500 (code) | `open-source/ismo009_Coinche_main_game.js` | `// Générale: 500 points` + vérification que **chaque** pli est gagné par le preneur lui-même |
| existe, en **variante** | `divers/pagat_com_jass_coinche_html.txt` | « **Some allow** an additional (highest) bid of générale, in which the bidder has to win all eight tricks alone, without help from partner. The bidder of a générale **may have the right to lead** to the first trick. » — valeur « for example **1000** » |
| **différée** | `open-source/ilyesbrh_…coinche_GAME_RULES.md` | « **Deferred V1** — trigger: ≥5% of online matches reach a position where it would matter » |
| existe, valeur **ouverte** | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « **la générale n'existe pas** ; la générale rapporte **350** ; …**500** ; …**700** ; …**1 000** ; la générale **n'est comptée que si elle est annoncée** ; la générale **ne donne pas la main** à celui qui l'annonce ; **la générale ne se coinche pas** » (sept variantes d'affilée) |

### Position B — elle n'existe pas / ce n'est pas ça

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| absente | toute la FFB (`…Contree-27-01-2016`, `…BELOTE-CONTREE`, `LOCAL_…`, `…COINCHEE_pdf`, `ffbelote_org_belote_contree`, `ffbelote_org_regles_coinche`) | le mot **n'apparaît nulle part**. Et le règlement 2016 ferme la porte aux ajouts : « Toute variante à l'instar de « on ne finit pas sur un capot », « le carré de 7 annule la donne » etc, sont **nulles et non avenues** dans les tournois homologués » |
| absente | `fr_wikipedia_Belote_contrée`, `belotecontree_free`, `ainesruraux`, `missègre`, `adpoker`, `carafons`, `cartesetcie`, `clubdejeux`, `regles_com`, `drasill` (doc **et** code), `jeu_belote_fr`, `ducale`, `casimir`, `gamerules`, `ASCEE`, `theosaulus` | MUETS — le mot ne figure pas |
| **interdite en concours** | `tournois/maisondesessarts_fr_article116_html.txt` | « Les autres expressions comme … capote, **générale… ne sont pas autorisées** » |
| **c'est le nom du capot** | `apps-sites/contree_org_4_joueurs.txt` | « « **Générale** » est le nom de la prime de capot attribuée à l'équipe ayant fait toutes les levées. » |
| **c'est le nom du capot** | `divers/iscool_…faq_157…txt` | « Seule exception : « **la générale** » qui est la plus grande annonce possible (« **mettre les adversaires capot** » : faire tous les plis) » |

**Divergence, et la ligne de fracture est le nom du jeu.** La générale est **une enchère de
coinche, pas de contrée** : présente chez les sites de coinche et les apps, absente de *tous*
les textes qui se disent « contrée » et de l'intégralité du corpus FFB. Wikipédia FR le
confirme par construction, en la rangeant dans sa liste de variantes de la coinche. **Sa valeur
n'a aucun consensus** (350 / 500 / 700 / 1 000, ou « à convenir »), et **deux sources ont gardé
le mot en lui donnant le sens de « capot »** — ce qui le rend dangereux à table. Le seul point
d'accord réel : le preneur y joue seul. Sur ce que devient le partenaire, deux sources seulement
tranchent, et **elles ne disent pas la même chose** : il « ne joue pas de la manche »
(belotepoint) vs il « joue normalement mais n'a pas le droit de gagner un pli » (ContrAI).

---

## 8. Sans Atout / Tout Atout

### 8a. Sont-ils proposés ?

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| **définitoire de la coinche** | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « La coinche … se distingue de la belote contrée **par la présence des enchères « tout atout » et « sans atout »**. » |
| **définitoire, en creux** | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « Elle se distingue de la coinche **par l'absence des enchères « tout atout » et « sans atout »**. » |
| **définitoire** | `divers/en_wikipedia_org_wiki_Belote.txt` · [url](https://en.wikipedia.org/wiki/Belote) | « Belote contrée … **differs in that there are no no trump and all trumps contracts** » |
| **option de tournoi** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` §11 | « **L'organisateur d'un tournoi peut décider** de le jouer avec la variante du Sans Atout / Tout Atout. » |
| **option**, dite comme telle | `federations/ffbelote_org_belote_contree.txt` (annexe) ; `apps-sites/re_belote_fr_belote_sans_atout_tout_atout.txt` · [url](https://www.re-belote.fr/belote-sans-atout-tout-atout/) ; `tournois/villeconin_…` (**mêmes mots**) | « **Important : le Sans Atout / Tout Atout est une option. Il n'est pas appliqué dans tous les tournois.** Il est important de demander aux organisateurs » |
| variante | `divers/pagat_com_jass_coinche_html.txt` | « Another Bridge-like modification is the no trump (sans atout) bid » |
| **option de code** | `open-source/drasill_bga-coinche_master_coinche.game.php` | `if ($color >= 5 && $this->getGameStateValue('hasAllNoTrumps') == 2) { throw … 'All Trumps / No Trumps disabled for this game'; }` |
| proposés | `carafons`, `cartesetcie`, `jeux_regles`, `exoty`, `ange_heureux`, `alhoa`, `drasill (doc)`, `belotepoint`, `iscool (mode dédié)`, `eryodsoft`, `playjoy`, `gameduell (en)`, `en_wikipedia_Coinche`, `coinchegratuit`, `regles_de_jeux`, `ilyesbrh`, `ismo009`, `slim0` | oui |
| **explicitement interdits** | `tournois/data_over_blog_kiwi_…reglement-table-coinche.pdf.txt` | « **Pas d'annonce « sans atout ou tout atout »** » |
| **hors périmètre assumé** | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « The base ContrAI engine does **not** implement them » |
| absents | `ffbelote_org_regles_coinche` (corps), `adpoker`, `clubdejeux`, `jeu_belote_fr`, `maviedesenior`, `gamerules`, `casimir`, `belotecontree_free`, `ainesruraux`, `missègre`, `ducale`, `ASCEE`, `theosaulus` | MUETS |

### 8b. Comment on rétablit les 162 points

| Méthode | Source | Ce qu'elle dit |
|---|---|---|
| **SA : As = 19** | FFB (les 4 PDF + les 2 pages web + villeconin), `re_belote_fr`, `carafons`, `cartesetcie`, `jeux_regles`, `exoty`, `ange_heureux`, `alhoa`, `pagat`, `en_wikipedia_Coinche`, `playjoy`, `ilyesbrh`, `ismo009` | « **Les As valent 19 points** afin de ramener les points du Paquet à 162 » — **le seul chiffre du corpus sur lequel tout le monde est d'accord** |
| SA : Valet = 2 vs Valet = 0 | `open-source/ismo009_…game.js` (`'valet': 2`) vs `open-source/ilyesbrh_…md` (« **J = 0** ») | désaccord sur une seule carte, dans deux implémentations |
| SA : As = 11 (!) | `divers/lemagloisirs_fr_regle_coinche.txt` | « sans atout : **as = 11**, 10 = 10, roi = 4, dame = 3, valet = 2 » — isolée, et arithmétiquement fausse (total 120, pas 152) |
| **TA : ratio 162/258** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` §11.3 | « il faut **multiplier le nombre de points comptés (hors belote) par la fraction 162/258** puis rajouter les éventuelles belotes. L'organisateur … mettra à disposition une **table de conversion** » |
| TA : ratio 162/258 | `ffbelote_org_belote_contree`, `villeconin`, `re_belote_fr`, `carafons`, `cartesetcie` | même formule |
| TA : ratio 152/248 (code) | `open-source/drasill_bga-coinche_master_coinche.game.php` | `if ($trumpColor == 5) { $arrangeMultiplier = 152 / 248; } elseif ($trumpColor == 6) { $arrangeMultiplier = 152 / 120; }` — **il rééchelonne aussi le Sans Atout**, ce que personne d'autre ne fait |
| TA : ratio 256/162 puis 2/3 | `divers/ange_heureux_free_fr_…` | « multiplier … par 256/162 = 0.6328125. Par commodité, on multipliera par **2/3** … (certains utilisent la « **Qonstante de Qoinche** » = 27/43 = 0.6279069767 !) » — avec table 80→123, 160→245 |
| **TA : V14 / 9-9 / A6 / 10-5 / R3 / D1** | `pagat`, `jeux_regles`, `fr_wikipedia_Coinche`, `reglesdejeux`, `ilyesbrh`, `ismo009` | « in tout atout the card values are reduced to **J=14, 9=9, A=6, 10=5, K=3, Q=1** » |
| TA : V**13** / 9-9 / A**7** / 10-5 / R3 / D1 | `apps-sites/playjoy_com_en_coinche_rules.txt` | valeurs **différentes** de la ligne précédente |
| TA : V**13** / 9 / A6 / 10 / R / D (barème « atout » réduit) | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` | « Valet 13 points, Neuf 9, As 6, Dix 5, Roi 3, Dame 2 » — la FFB elle-même a deux barèmes selon la rédaction |
| TA : V14 / 9-9 / A**7** / 10-5 / R3 / **D2** | `divers/exoty_com_compter_points_coinche.txt` · [url](https://exoty.com/compter-points-coinche) | encore d'autres |
| TA : V14 / 9-9 / A6 / **10-4** / R3 / D2 | `divers/alhoa_free_fr_…` | encore d'autres |
| **SA : contrat doublé** | `tournois/web_archive_org_…coinche_stephanoise…` | « **Qu'un contrat à sans atout double le montant de celui-ci** : par exemple vous demandez 82 sans atout si votre contrat est réalisé vous marquerez 160 points. » |
| **table d'équivalence non linéaire** | `divers/fr_wikipedia_org_wiki_Coinche.txt` | Sans atout 70 ↔ Atout 80 ↔ Tout atout 130 ; « une enchère faite sur tout atout est faite sur 160 points, **puis on rajoute 50** pour avoir l'enchère sur 250 » |

### 8c. Conséquences sur l'enchère elle-même

- **Même échelle 80–160** : FFB (« Un joueur peut enchérir à la couleur **ainsi qu'à** Sans Atout
  et Tout Atout »), ange.heureux, alhoa, ilyesbrh, ismo009, drasill (82–170), et surtout IsCool,
  qui l'explicite : « TA et SA sont à considérer comme des « couleurs » spéciales ; cela signifie
  que c'est bien **la valeur du contrat la plus forte** qui déterminera quelle équipe remporte
  l'enchère » (`apps-sites/iscool_…faq_701…` · [url](https://iscool.helpshift.com/hc/fr/17-belote-facebook/faq/701-coinche-with-announces-at-nt/)).
- **SA/TA comme dénominations supérieures** : Pagat en variante (♣<♦<♥<♠<SA<TA), Wikipédia FR
  par sa table d'équivalence.
- **Pas de belote à SA** : consensus total (FFB, villeconin, carafons, cartesetcie, exoty,
  ange.heureux, drasill, pagat, en_wikipedia, ilyesbrh). **Jusqu'à 4 belotes à TA** : consensus
  aussi — sauf carafons (« il peut y avoir qu'une seul Belote ») et ange.heureux (aucune belote
  à TA non plus).

**Divergence structurée.** Le *principe* fait consensus : SA/TA existent, il faut ramener le
total à 162, l'As vaut 19 à SA. **La méthode pour TA n'a, elle, aucun consensus** — six barèmes
de cartes incompatibles (dont deux à l'intérieur de la seule FFB), trois ratios, une table
d'équivalence non linéaire, et un règlement qui préfère doubler le contrat plutôt que toucher
aux cartes. Pagat résume mieux que quiconque : « *There are various systems, **none of them
particularly elegant**.* » **Et le vrai clivage est en amont** : SA/TA sont ce qui *sépare* la
coinche de la contrée (les deux Wikipédia le disent, en miroir), quand la FFB en fait une simple
case à cocher d'organisateur.

---

## 9. Coincher / contrer : à son tour, ou à la volée ?

### Position A — à son tour de parole seulement

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| à son tour | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` §4.2.3 | « On ne peut contrer que lorsque c'est **à son tour de parler** (on ne contre pas « à la volée »). » |
| à son tour | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE`, `LOCAL_…`, `…COINCHEE_pdf` | même phrase (« on ne **coinche** pas à la volée » dans la version coinche) |
| à son tour | `federations/ffbelote_org_belote_contree.txt` + `…regles_coinche.txt` | « Un contre doit être annoncé à son tour. **On ne peut pas contrer « à la volée ».** » |
| à son tour | `tournois/cdf_missegre11_…` | même phrase (copie FFB) |
| à son tour | `tournois/ainesruraux_saintsever_…` / `divers/belotecontree_free_reglement.txt` | « (Là encore bien entendu le joueur **ne pouvant faire de telles annonces qu'à son tour de parler**.) » |
| à son tour | `tournois/maisondesessarts_fr_article116_html.txt` | « **La coinche se fait à son tour de parole.** » |
| à son tour | `divers/cartesetcie_fr_regle_du_jeu_la_belote_coinchee.txt` | « Une Coinche doit être annoncé **chacun son tour** lors de son tour de jeu. On ne peut pas Coincher « à la volée ». » |
| à son tour, deux fois | `divers/regles_com_jeux_cartes_coinche_html.txt` | « Coincher ne se fait pas à la volée. Il faut le faire à son tour de parole, **pas en coupant la parole de la table.** » ; FAQ : « Peut-on coincher à la volée ? **Non.** » |
| à son tour | `divers/jeu_belote_fr_regles_php_part_regles_jeu_coinche.txt` | « Les joueurs doivent parler **chacun leur tour**, ils ont la possibilité de passer, ou … de coincher. » |
| à son tour | `divers/iscool_…faq_157…txt` | la coinche est listée comme une option **du joueur dont c'est le tour** |
| à son tour | `tournois/data_over_blog_kiwi_…` | « **On ne coinche pas à la volée** » |
| à son tour, formellement | `tournois/web_archive_org_…coinche_stephanoise…` | « Il est **formellement interdit** de surenchérir, coincher ou surcoincher « **A LA VOLEE** ». On doit s'exprimer l'un après l'autre. **Il faut que le précédent joueur ait dit « Je Passe »** pour annoncer son enchère. » |
| à son tour, en tournoi | `divers/belotecontree_free_reglement.txt` (fil de discussion de la page) | « Le contre, dans le règlement officiel, se fait au tour, **jamais à la volée**. … Nous avons nous-même débuté en contrant à la volée, ce qui est indéniablement **plus fun… mais interdit en tournoi.** » |
| à son tour (code) | `open-source/ismo009_Coinche_main_game.js` | contrôle `playerPosition !== this.currentPlayer` en tête de `placeBid()` — la coinche y passe comme les autres actions |

### Position B — à la volée

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| à la volée (règle de base) | `divers/pagat_com_jass_coinche_html.txt` | « **It is not necessary to wait for your turn to say coinche**, but you can only double if the most recent bid was by an opponent. » |
| à la volée | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « il peut coincher **pendant tout le temps où cette enchère dure** (dénommé « **coincher à la volée** ») … Parfois, le fait de frapper un petit coup sec sur la table accompagne (voire remplace) l'annonce » |
| à la volée | `divers/ange_heureux_free_fr_…` | « peut être fait **à la volée** (c'est-à-dire que l'on peut coincher immédiatement sans attendre son tour) » |
| **à la volée, en tournoi** | `tournois/fnasce_org_IMG_pdf_reglement_pdf.txt` (= `…belote_reglement_cle1a43c7`, ASCEE 2A) | « **On contre (ou coinche) à la volée : pas d'obligation d'attendre son tour.** » |
| à la volée | `divers/clubdejeux_com_…` | « **À tout moment**, un joueur peut coincher ; les enchères s'arrêtent alors » |
| à la volée | `divers/adpoker_fr_belote_contree_html.txt` | « **À tout moment des enchères**, les adversaires du preneur peuvent contrer. » |
| à la volée | `divers/carafons_fr_regles_de_la_coinche.txt` | « **Une Coinche peut être annoncé à tout moment.** » (alors que sa jumelle cartesetcie dit le contraire) |
| à la volée | `divers/drasill_github_io_bga_coinche_rules_fr_html.txt` | « Lors de la phase d'enchères, **à tout moment**, un joueur peut Coincher » |
| à la volée | `divers/bk_jeux_ducale_…pdf.txt` | « Les joueurs peuvent coincher leurs adversaires **à tout moment** pendant les annonces … taper sur la table en disant « je coinche » » |
| à la volée | `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` | « **À tout moment**, n'importe quel joueur autour de la table peut, lorsqu'un adversaire vient de proposer un contrat, coincher » |
| à la volée | `apps-sites/gamerules_com_rules_coinche.txt` | « this can be done **on any player's turn** but the last bid made must be from the opposing team » |
| à la volée | `apps-sites/en_wikipedia_org_wiki_Coinche.txt` | « this can be called **at any time**, and by **either player** in the team » |
| à la volée | `open-source/ilyesbrh_…coinche_GAME_RULES.md` | « **Either opponent may say "coinche" without waiting for their turn.** Available only on the most recent bid » |
| à la volée, y compris après des passes | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « **Intervening passes do not close the Coinche / Surcoinche window.** Both *contre* … and *surcontre* … remain legal up until the auction terminates on three consecutive passes » |
| à la volée (par construction) | `open-source/drasill_bga-coinche_master_coinche.game.php` | `coinche()` utilise `checkPossibleAction()` + `getCurrentPlayerId()`, là où `bid()`/`pass()` font `checkAction()` + `getActivePlayerId()` → **la coinche n'est pas restreinte au joueur actif** |
| à la volée (option) | `apps-sites/eryodsoft_com_fr_jeux_coinche.txt` · [url](http://www.eryodsoft.com/fr/jeux/coinche) | « Possibilité de **Coincher à la volée**. » |
| **paramétrable** | `apps-sites/play_google_…com.aandrill.belote` · [url](https://play.google.com/store/apps/details?id=com.aandrill.belote&hl=fr) | « ✓ **Coincher à tout moment ou sur un 80** » |

### Position C — c'est une convention à fixer avant la partie

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| au choix de la table | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « **Selon les conventions variables, à définir avant l'entame de la partie**, on peut contrer à la volée ou lorsque c'est à son tour de parler (**dans les concours le contre à la volée est souvent banni**). » |
| variante | `divers/pagat_com_jass_coinche_html.txt` | « Some play that a player may **only say "coinche" in turn**. In this version … West is not allowed to say anything **or give any indication** that he or she wishes to coincher until East and North have had their turns to speak. » |

**Divergence maximale.** Le clivage suit **la FFB et ses copies contre à peu près tout le
reste** : les six documents FFB, Missègre, le tournoi international, Les Essarts, la Coinche
Stéphanoise et le règlement de table over-blog interdisent la volée ; Pagat, Wikipédia FR,
ange.heureux, clubdejeux, adpoker, Ducale, drasill, casimir, gamerules, ContrAI et ilyesbrh
l'autorisent. **La lecture « tournoi = à son tour, bistrot = à la volée » est tentante mais
fausse** : le règlement de l'ASCEE 2A est un règlement de tournoi et il écrit noir sur blanc
« On contre (ou coinche) **à la volée** : pas d'obligation d'attendre son tour ». Wikipédia FR
formule la bonne nuance — « dans les concours le contre à la volée est **souvent** banni » —
et le forum de belotecontree.free.fr en donne la raison sociale : « indéniablement plus fun…
mais interdit en tournoi ». **Et le désaccord traverse une copie verbatim** : carafons et
cartesetcie reprennent la même page FFB et se contredisent sur ce point précis.

### 9b. Le contre gèle-t-il les enchères ?

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| **gèle**, et fait taire le camp du contreur | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` | « Le fait de contrer **fige le contrat**, empêchant ainsi toute possibilité de surenchère à l'exception du surcontre. Dans le cas d'un contre, **seuls les joueurs de l'équipe contrée sont amenés à parler**, en passant ou surcontrant. Le partenaire du joueur ayant contré **ne doit pas parler** (ne pas dire je passe) » |
| gèle | FFB (autres versions), `fr_wikipedia_Belote_contrée`, `belotecontree_free`, `ainesruraux`, `adpoker`, `clubdejeux`, `ange_heureux`, `drasill (doc)`, `ducale`, `alhoa`, `regles_com`, `casimir`, `gamerules`, `pagat` (base), `maisondesessarts` | « les enchères sont **immédiatement stoppées** » / « A coinche **ends the bidding** » / « Une "coinche" **stoppe les enchères** » |
| gèle (code) | `open-source/ismo009_…game.js` ; `…coinche.game.php` | `'Enchère coinchée, vous pouvez seulement passer ou surcoincher'` ; `if ($countered > 0) { … nextState('waitForRedouble'); }` |
| gèle | `open-source/valmathieu_ContrAI…md` | « This **freezes** the auction at the current contract » |
| **ne gèle pas** (variante) | `divers/pagat_com_jass_coinche_html.txt` | « Some play that a coinche can only be said in turn and **does not end the bidding**. So if South bids 100 Hearts and East says "coinche", **North can escape by bidding a different suit** … The bidding will only end after three consecutive passes. » — et Pagat rattache **explicitement ce cas au nom « contrée »** |
| **ne gèle pas : on peut « décoincher »** | `tournois/web_archive_org_…coinche_stephanoise…` | « **Si un joueur est coinché les deux autres joueurs peuvent décoincher en faisant une enchère supérieure.** » |
| **ne gèle pas** (variante) | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « on peut **décoincher** en annonçant une annonce supérieure à celle qui a été coinchée » |
| **ne gèle pas** | `tournois/maisondesessarts_fr_article116_html.txt` | (sur le capot) « Les enchères **ne sont pas closes pour autant** et l'adversaire ou le partenaire peut surenchérir. » |
| non contrable à 80 | `divers/alhoa_free_fr_…` ; `divers/fr_wikipedia_org_wiki_Coinche.txt` | « Par convention, facultative, **on ne coinche pas un 80.** » / « on ne peut pas coincher les contrats de 80 points » |
| non contrable sur son partenaire | `open-source/ismo009_…game.js` ; `…coinche.game.php` | `'Vous ne pouvez pas coincher votre partenaire'` ; `'Cannot double on you partner\'s bid'` |
| non surcoinchable sur une « voiture » | `tournois/clublafontainedejouvence_fr_règlement_coinchée.txt` · [url](https://www.clublafontainedejouvence.fr/r%C3%A8glement/coinch%C3%A9e) | « Une **voiture** peut être coinchée **mais pas surcoinchée**. » |

**Divergence.** Le gel est très majoritaire, mais **le décoinchage existe et il est porté par
un règlement de compétition** (la Coinche Stéphanoise, qui se réclame du « Championnat de
France, National de Coinche »), pas seulement par une liste de variantes. Pagat fait de la
combinaison *contre à son tour + contre non bloquant* **la définition même de « contrée »** par
opposition à « coinche » — exactement l'inverse de l'usage français, où la FFB appelle
« contrée » un jeu où le contre gèle et se fait à son tour.

---

## 10. Surcoincher / surcontrer

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| **par l'équipe contrée, arrête tout** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` §4.2.4 | « Lorsqu'une équipe a contré, l'équipe adverse a la possibilité de surcontrer … **Le tour d'enchères s'arrête alors immédiatement.** » |
| idem | FFB (toutes versions), `missègre`, `carafons`, `cartesetcie` | même phrase |
| par le preneur **ou** son partenaire | `divers/adpoker_fr_belote_contree_html.txt` | « **Le preneur ou son partenaire** peuvent alors, à leur gré, surcontrer. » |
| par le preneur ou son partenaire | `divers/pagat_com_jass_coinche_html.txt` | « either **the bidder or his partner** can surcoincher » |
| par l'un des membres, clôt tout | `divers/belotepoint_fr_regles_coinche.txt` | « **l'un de ses membres** peut surcoincher … **Après une surcoinche, les enchères sont définitivement closes** et le jeu commence. » |
| clôt tout | `apps-sites/en_wikipedia_org_wiki_Coinche.txt` | fin des enchères : « one player has "surcoinchéd" » |
| clôt tout (code) | `open-source/ismo009_…game.js` | `// La sur-coinche lance directement le jeu` → `return this.startPlaying();` |
| par le camp du preneur, clôt tout (code) | `open-source/drasill_bga-coinche_master_coinche.game.php` | `'Cannot redouble on yourself'` / `'Cannot redouble on your partner'` puis `nextState('endBidding')` ; un joueur peut décliner (`nosurcoinche()` → « ${player_name} does not redouble. ») |
| **on ne peut pas surcontrer son propre camp** | `federations/ffbelote_…Contree-27-01-2016` §4.2.3 | « Le partenaire du joueur ayant contré **ne doit pas parler** » — le surcontre ne peut venir que du camp contré |
| **absente du texte** | `divers/clubdejeux_com_…`, `divers/alhoa_free_fr_…`, `divers/drasill_github_io_…` (fr et en), `divers/bk_jeux_ducale_…` | MUETS : la surcoinche n'est **jamais mentionnée** |
| **optionnelle** | `apps-sites/eryodsoft_com_fr_jeux_coinche.txt` | « **Autoriser ou non** la Surcoinche, la Générale. » |
| **interdite sur une « voiture »** | `tournois/clublafontainedejouvence_fr_règlement_coinchée.txt` | « Une voiture peut être coinchée **mais pas surcoinchée**. » |
| multiplicateur ×4 | `ffbelote_REGLES-DE-LA-BELOTE-CONTREE`, `ffbelote_org_belote_contree` (640), `missègre` (640), `belotecontree_free` (640), `pagat` (base), `ange_heureux`, `regles_com`, `casimir`, `exoty`, `maviedesenior`, `ASCEE`, `ilyesbrh`, `slim0`, `theosaulus` | « Le score de la donne sera multiplié par 4 » |
| multiplicateur ×3 | `federations/LOCAL_regles_officielles_belote_contree.txt` | « Le score de l'annonce sera **multiplié par 3**. » |
| multiplicateur ×3 | `divers/iscool_…faq_157…`, `apps-sites/iscool_…faq_701…`, `divers/jeu_belote_fr_…`, `open-source/ismo009_…game.js` | « Les points seront **multipliés par 3** » / « Le coefficient multiplicateur passe à **trois** » / `if (this.contract.surcoinched) multiplier = 3;` |
| ×3 en variante | `divers/pagat_com_jass_coinche_html.txt` | « Some play that a surcoinche does not double the score again, but **only increases the multiplier from 2× to 3×**. » |
| **×3 mais 480, pas 640** | `tournois/maisondesessarts_fr_article116_html.txt` | « La coinche vaut **160 × 2 = 320** … La surcoinche vaut **160 × 3 = 480** » |

**Consensus sur la mécanique, divergence sur le multiplicateur.** Tout le monde s'accorde sur
deux points : *seul le camp contré peut surcontrer*, et *ça termine les enchères*. En revanche
le multiplicateur se partage nettement **×4** (tradition, FFB ancienne, Pagat) contre **×3**
(**FFB récente** « Équipe Ludique », IsCool, jeu-belote.fr, ismo009, Les Essarts) — c'est
exactement le même clivage entre rédactions FFB que celui relevé dans [arrondi.md](arrondi.md).
À signaler : **quatre sources ne connaissent pas la surcoinche du tout**, dont la documentation
drasill/BGA — alors que son propre code l'implémente.

---

## 11. Fin des enchères

### 11a. Après une enchère : trois passes

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| 3 passes | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` | « le tour d'enchère s'achève lorsqu'**après une enchère ou un contre, les trois joueurs suivants passent** » |
| 3 passes | `federations/ffbelote_org_belote_contree` + `…regles_coinche` + `tournois/cdf_missegre11_…` | « Les enchères se terminent **après 3 « Passe » consécutifs.** » |
| 3 passes | `fr_wikipedia_Belote_contrée`, `belotecontree_free`, `ainesruraux`, `carafons`, `cartesetcie`, `jeux_regles`, `lemagloisirs`, `clubdejeux`, `belotepoint`, `jeu_belote_fr`, `ducale`, `casimir`, `gamerules`, `pagat`, `ludicash`, `coinche_stephanoise`, `adpoker`, `ilyesbrh` | « lorsque 3 joueurs passent après l'enchère de l'un des joueurs » |
| 3 passes | `tournois/fnasce_org_IMG_pdf_reglement` (ASCEE 2A) | « **Si 3 personnes passent après une enchère, le jeu commence.** » |
| 3 passes | `apps-sites/gameduell_…faq_1054_coinche` | « as soon as **3 people have passed since the last offer** » |
| 3 passes | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « The auction ends when **three consecutive players pass after the last bid** » |
| 3 passes (code) | `open-source/ismo009_Coinche_main_game.js` | `if (this.lastBidder && this.passCount >= 3) { return this.startPlaying(); }` |

### 11b. …sauf pour ceux qui comptent quatre passes

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| **4 passes** | `divers/drasill_github_io_bga_coinche_rules_fr_html.txt` (+ version EN) | « **Dès que 4 joueurs d'affilée passent**, les enchères prennent fin : si aucune enchère n'a été faite, une nouvelle main est distribuée, sinon, la dernière enchère est prise en compte » |
| **4 passes** (code) | `open-source/drasill_bga-coinche_master_coinche.game.php` | `if ($passCount >= 4) { … }` — le preneur lui-même doit repasser, ce qui est cohérent avec sa règle « relance sur soi-même autorisée en changeant de couleur » |
| **tour complet de passe** | `divers/alhoa_free_fr_…` | « jusqu'à ce qu'un **tour complet de "passe"** soit effectué » |
| **le dernier parleur peut rouvrir** | `divers/ange_heureux_free_fr_…` | « quand trois joueurs passent, le dernier qui a parlé **a le droit de reparler** … Les autres joueurs ont alors le droit de reparler. » |

### 11c. Quatre passes d'entrée : donne annulée, redonne, donneur suivant

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| donne terminée, **sans points**, donneur suivant | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` §4.2.1 | « Si les 4 joueurs passent sans enchérir, la donne est terminée **sans points marqués**. **Le joueur à la droite du donneur devient le nouveau donneur.** » |
| idem | FFB (toutes versions) | « Le donneur rassemble les cartes et les fait passer au joueur à sa droite qui sera le nouveau donneur. » |
| **la donne ne compte pas dans le quota** | `federations/ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt` §10.4.1 (belote classique, mais règle de tournoi) | « Si les quatre joueurs passent sur les deux tours d'enchères, **on ne décompte pas cette donne.** » |
| mène non jouée, donneur suivant | `divers/belotecontree_free_reglement.txt` / `tournois/ainesruraux_…` | « Si les quatre joueurs passent après la donne, **la mène n'est pas jouée** et le jeu passe … au joueur suivant. » |
| redonne | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « les cartes sont ramassées, une nouvelle donne est réalisée par le joueur placé à la droite du précédent donneur » |
| redonne | `jeux_regles`, `clubdejeux`, `belotepoint`, `jeu_belote_fr`, `alhoa`, `casimir`, `drasill`, `pagat`, `gamerules`, `coinche_stephanoise`, `gameduell (fr et en)`, `en_boardgamearena` | « If all four players pass, the cards are thrown in and the next dealer deals a new hand. » |
| redonne **même donneur** | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « the round is annulled, cards are collected and **redealt (with the same dealer)** » |
| redonne + rotation (code) | `open-source/ismo009_…game.js` ; `…coinche.game.php` | `if (this.passCount >= 4 && !this.lastBidder) { this.dealer = getNextPlayer(this.dealer); … 'redistribute' }` ; `clienttranslate('Everybody passes, no bid')` + `setNextFirstPlayer()` |
| **muets** | carafons, cartesetcie, missègre, ASCEE, regles.com, lemagloisirs, adpoker, ange_heureux, exoty, ducale, iscool, maviedesenior, maisondesessarts, ilyesbrh | ne disent rien du cas des 4 passes |

**Consensus fort sur 11a et 11c** — trois passes après une enchère, redonne après quatre.
**Divergence sur 11b** (drasill/BGA compte quatre passes, ange.heureux et alhoa rouvrent
l'enchère) — mais ce n'est **pas un axe indépendant** : c'est l'axe 4b vu de l'autre bout. La
formule FFB « le tour d'enchères prend fin lorsqu'il revient à la dernière personne ayant fait
une annonce » et la formule « trois passes » ne sont équivalentes *que si* on interdit de
parler sur soi-même. **Divergence secondaire sur qui redonne** : la FFB fait tourner le
donneur, ContrAI le garde.

---

## 12. Les annonces (tierce, cinquante, cent, carré)

### 12a. Jouées ou non — c'est la définition du jeu

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| **jamais** (contrée) | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` §8 | « **La belote contrée se joue toujours sans annonces.** » |
| jamais | `federations/ffbelote_REGLES-DE-LA-BELOTE-CONTREE.txt` §8 | même phrase |
| **toujours** (coinche) | `federations/ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.txt` §8 | « **La coinche se joue toujours avec les annonces.** » |
| jamais | `divers/adpoker_fr_belote_contree_html.txt` | « À la belote contrée, **il n'y a pas d'annonce** : seuls le dix de der (10 points) et la belote-rebelote (20 points) sont comptabilisés. » |
| jamais | `divers/ange_heureux_free_fr_…` | « On joue **sans annonce, sauf la belote** qui sert à l'attaque. » |
| jamais | `tournois/fnasce_org_IMG_pdf_reglement` (ASCEE 2A) | « **On joue sans annonce (tierce, carré, etc.) hormis la belote (et re)** … La belote (et re) **permet de remplir le contrat : attention au moment de contrer !** » |
| jamais, et c'est la définition | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « Étant donné qu'à la contrée **les annonces ne comptent pas** (à part la belote), il a été possible de mettre au point un système d'enchères codifiées standard permettant la recherche du capot » |
| jamais, frontière assumée | `open-source/valmathieu_ContrAI_main_contree-domain.md` | « **Out of scope for ContrAI** — this is what distinguishes contrée (without annonces) from coinche. … This is the **explicit boundary of the project**: Contrée *without* annonces. » |
| jamais | `carafons`, `alhoa`, `drasill` (doc **et** code), `ducale`, `jeu_belote_fr`, `iscool (mode standard)`, `lemagloisirs`, `maviedesenior`, `belotecontree_free`, `ainesruraux`, `missègre`, `casimir`, `theosaulus`, `ismo009` | MUETS ou explicitement « sans annonce » |
| **paramétrable** | `apps-sites/eryodsoft_com_fr_jeux_coinche.txt` | « **Jeu avec ou sans annonces.** » ; « **Annonces comptent ou non pour réussir le contrat.** » ; « Annonces perdues ou non en cas de chute ou de capot. » |
| paramétrable | `apps-sites/play_google_…com.aandrill.belote` | « ✓ Avec ou sans annonce » ; « ✓ **Les annonces comptent ou non pour passer** » ; « ✓ Annonces prenables sur capot » |
| toujours | `cartesetcie`, `regles_com`, `jeux_regles`, `clubdejeux`, `belotepoint`, `exoty`, `pagat`, `playjoy`, `ludicash`, `contree_org`, `coinche_stephanoise`, `data_over_blog`, `maisondesessarts`, `iscool (mode dédié)`, `ilyesbrh`, `villeconin` | oui |
| **jamais**, note historique | `divers/pagat_com_jass_coinche_html.txt` | « **Some reserve the name Belote Contrée for variants with few or no announcements** … Since the late 20th century, versions without announcements have become more popular, as this is thought to **reduce the element of luck**. » |

### 12b. Comptent-elles pour réaliser le contrat ?

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| **oui** | `federations/ffbelote_org_regles_coinche.txt` | « À la coinche, les annonces **AIDENT** à faire le contrat. Si un joueur demande 100, qu'il réalise 83 points mais a une tierce, il a donc 103 points. **Son contrat est donc réalisé.** » |
| oui | `federations/ffbelote_org_…COINCHEE_pdf` §4.2 | « Le nombre de points comprend les points remportés lors des plis, les points de la belote éventuelle **ainsi que les points des annonces éventuelles.** » |
| oui | `tournois/villeconin_…` (fiche FFB) | « Le nombre de points comprend non seulement les points remportés lors des plis, **mais aussi les annonces éventuelles** dont il faudra donc tenir compte. » |
| oui | `divers/cartesetcie_fr_…` | « les annonces **participent à remplir le contrat** … il aura après addition des deux sommes 102 points » |
| oui | `divers/regles_com_…` | « Les annonces comptent-elles pour faire le contrat ? **Oui dans la coinche officielle.** » |
| oui | `divers/exoty_com_compter_points_coinche.txt` | « Oui, à la coinche, les points des annonces s'ajoutent aux points de vos plis pour atteindre l'objectif du contrat. **La seule exception est la Belote-Rebelote.** » |
| oui, avec le calcul | `divers/clubdejeux_com_…` | « si un joueur prend avec un bon jeu, mais qu'un adversaire annonce un cinquante, il lui sera difficile de remplir son contrat car il lui faudra obtenir, **non plus 82 points, mais 107** (plus de la moitié de 212) » |
| oui | `apps-sites/gameduell_…faq_1054_coinche` | « The melds play a **major role in the fulfilment of the contract** » |
| oui, et ça change l'enchère | `divers/pagat_com_jass_coinche_html.txt` | « Playing with announcements makes it possible to **bid much higher** … a player who has four jacks can clearly make **at least 220** with any suit as trumps » |
| oui | `tournois/data_over_blog_kiwi_…` ; `tournois/web_archive_org_…coinche_stephanoise…` | « Les annonces servent à « **PRENDRE ou FAIRE SON COUP** ». » (formule identique dans les deux) |
| **non** | `divers/belotepoint_fr_regles_coinche.txt` | « Les points d'annonce s'ajoutent au score de l'équipe **indépendamment du contrat**. » |
| **non** | `apps-sites/iscool_…faq_701_coinche_with_announces_at_nt` | « L'équipe qui prend doit réaliser le nombre de points du contrat, **sans compter sur les 20 points de Belote-Rebelote ni sur les points des annonces**. » |
| **non** | `open-source/ilyesbrh_…coinche_GAME_RULES.md` | « Bidding team must reach at least their bid value in card points (**excluding belote/announcements**) » |
| **les deux, dans la même page** | `apps-sites/gameduell_…faq_1054_coinche` | « scoring a minimum of 82 points **through tricks, without melds and Belote-Rebelote** » — juste après avoir dit que les melds jouent « a major role in the fulfilment of the contract » |
| **les deux, dans la même phrase** | `divers/jeux_regles_com_regles_coinche.txt` | « **elles aident à la réalisation du contrat**. Cependant, **leurs points ne permettent pas la réussite du contrat**, ils sont ajoutés à la fin de la partie » |
| variante | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « les annonces sont comptabilisées pour déterminer si le contrat est réussi » — listé comme *une* variante parmi d'autres |

### 12c. Quand les annoncer, et quelques règles orphelines

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| au 1er pli, hauteur seule ; révélées au 2e | `federations/ffbelote_org_…COINCHEE_pdf` §8.2-8.3 | « Lors du premier pli, chacun des joueurs annonce **simplement la nature** de son ou ses annonce(s) au moment où il joue sa carte. … Lorsque la **première carte du second pli** vient d'être jouée, on procède à la résolution des annonces. » |
| idem | `villeconin`, `cartesetcie`, `regles_com`, `clubdejeux`, `iscool (mode dédié)`, `ludicash`, `data_over_blog`, `maisondesessarts`, `rjcv` | même procédure en deux temps, souvent mot pour mot |
| juste avant la 1re carte | `divers/pagat_com_jass_coinche_html.txt` ; `open-source/ilyesbrh_…md` | « may announce them **just before playing to the first trick** » ; « single-moment » (la procédure FFB en deux temps y est différée) |
| **à la 5e carte** | `tournois/web_archive_org_…coinche_stephanoise…` | « **A la cinquième carte on « arrête le jeu »** et on précise la hauteur de l'annonce. … A SANS ATOUT, les joueurs montrent leur(s) annonces **à la 9ème carte jouée**. » |
| avant que la 1re carte ne tombe | `apps-sites/contree_org_4_joueurs.txt` | « Elle doit être révélée **avant que la première carte ne soit dévoilée**, si elle est reconnue comme étant bonne, dès que la carte du second tour est tombée. » |
| **une seule équipe marque** | FFB coinche, villeconin, pagat, cartesetcie, clubdejeux, playjoy, jeux_regles, regles_com, iscool, ilyesbrh | « les annonces **d'un seul camp** peuvent être prises en compte : celui qui montre l'annonce la plus haute » |
| **une carte = une annonce** | FFB coinche, villeconin, cartesetcie, ludicash, rjcv | « une carte ne peut compter que pour **une seule annonce** à la fois » |
| **cartes cumulables** | `apps-sites/contree_org_4_joueurs.txt` ; `tournois/web_archive_org_…coinche_stephanoise…` | « Elles sont **cumulables dans une même main** : le joueur ayant la tierce d'Atout V-D-R et les 3 autres valets comptera donc un Carré de Valets (200), une Tierce (20), et la Belote (20) soit **240 points**. » ; « Une carte peut servir pour présenter **plusieurs annonces** y compris pour la belote » — **contredit frontalement la FFB** |
| **carré de 7/8 annule les annonces** | `apps-sites/iscool_…faq_701…` | « Si un joueur annonce un Carré de 7 ou un Carré de 8, **toutes les annonces de la manche sont tout simplement annulées.** » |
| **carré de 7 annule la donne** | `divers/fr_wikipedia_org_wiki_Coinche.txt` | « un carré de 7 permet d'**annuler la partie**, à la fin des enchères » |
| **explicitement nul et non avenu** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` | « Toute variante à l'instar de « on ne finit pas sur un capot », « **le carré de 7 annule la donne** » etc, sont **nulles et non avenues** dans les tournois homologués » |
| **renonce** | FFB coinche, villeconin, cartesetcie | « Si un joueur se révèle incapable de montrer les combinaisons qu'il a annoncées, il y a renonce : **le camp adverse marquera les points que le camp fautif avait annoncés.** » |
| valeurs | FFB coinche, villeconin, pagat, cartesetcie, clubdejeux, playjoy, jeux_regles, regles_com, exoty, ludicash, iscool, ilyesbrh, rjcv (partiel) | carré valets 200, neufs 150, As/10/R/D 100 ; cent 100, cinquante 50, tierce 20 ; **carrés de 7 et 8 nuls** ; « un carré de 100 points est plus fort qu'un cent » |
| valeurs **à Sans Atout** | FFB (contrée + coinche), villeconin, re_belote_fr, cartesetcie, ludicash, fr_wikipedia_Coinche | « 4 As **200**, 4 Dix **150**, 4 Rois/Dames/Valets/Neufs 100 » — l'échelle des carrés bascule avec la valeur des cartes |

**Ce n'est pas un désaccord, c'est une définition.** La FFB tranche par décret : « la belote
contrée se joue **toujours sans** annonces » / « la coinche se joue **toujours avec** ». Ce sont
deux jeux, pas deux opinions — et les deux Wikipédia, Pagat, l'anglais et ContrAI en tirent la
même conséquence : **contrée = pas d'annonces + pas de SA/TA ; coinche = annonces + SA/TA**. Les
valeurs, elles, font consensus quasi total. **La vraie divergence résiduelle est 12b** : les
annonces comptent-elles pour *faire* le contrat ? La FFB dit oui sans ambiguïté, IsCool,
belotepoint et ilyesbrh disent non, gameduell dit les deux dans la même page, et
jeux-regles.com se contredit **dans la même phrase**. C'est aussi ce qui explique l'axe 2 : un
plafond de 650 n'a de sens que si la réponse est « oui ». **Divergence secondaire mais nette** :
une carte peut-elle servir à deux annonces ? La FFB dit non, la Coinche Stéphanoise et
contree.org disent oui.

---

## 13. Axes que le corpus ajoute de lui-même

Trois points qui divergent et qu'on n'attendait pas.

### 13a. Dans quel sens tourne la parole ?

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| **à droite du donneur**, sens antihoraire | FFB (les 6), `fr_wikipedia_Belote_contrée`, `belotecontree_free`, `ainesruraux`, `missègre`, `pagat`, `adpoker`, `ange_heureux`, `contree_org`, `coinche_stephanoise`, `ducale`, `ludicash` | « Les joueurs s'expriment chacun à leur tour **en commençant par le joueur placé à droite du donneur**. » |
| **à gauche du donneur** | `apps-sites/gameduell_…faq_1054_coinche` | « The player **to the left of the dealer** starts the bidding » |
| à gauche | `divers/iscool_…faq_157…`, `apps-sites/exoty_com_regles_contree_belote`, `apps-sites/coinchegratuit_fr_…` · [url](https://www.coinchegratuit.fr/quelles-sont-les-regles-du-jeu-de-coinche/), `apps-sites/regles_de_jeux_com_…` | « Le joueur se situant **à gauche du donneur** commence à parler. » |
| **sens horaire** | `tournois/casimirdehauteclocque_fr_jeux_coinche_pdf.txt` ; `apps-sites/gameduell_…faq_1056_contree` ; `tournois/clublafontainedejouvence_…` | « On distribue et change de premier joueur **dans le sens des aiguilles d'une montre** » ; « **Les cartes sont distribuées dans le sens des aiguilles d'une montre : 3 puis 2 puis 3** » |

**Divergence** — mais probablement moins une variante régionale qu'une **erreur importée chez
les éditeurs de jeux en ligne**, qui appliquent le sens du bridge. Le corpus français
traditionnel est unanime : à droite, sens antihoraire. À noter que `casimir` est cohérent avec
lui-même (il fait tourner *tout* dans le sens horaire), ce qui est une variante réelle et non
une coquille.

### 13b. Formulation obligatoire de l'enchère

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| valeur puis couleur, « X de mieux » **interdit** | `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` | « On annonce d'abord **la valeur** du contrat (80, 90, 100…), **puis la couleur** … Les locutions du type « **20 de mieux** », « **10 de plus** », etc… **sont interdites.** » |
| idem | FFB (toutes versions), `carafons`, `cartesetcie`, `regles_com`, `missègre` | « Ce qui donnera 80 pique, et **non pique 80** » — Missègre ajoute « ou encore 80 à pique » |
| « à éviter » (et non interdit) | `divers/fr_wikipedia_org_wiki_Belote_contr_C3_A9e.txt` | « Les locutions du type « 20 de mieux », « 10 de plus », **sont à éviter.** » |
| **« dix et vingt de mieux » AUTORISÉ** | `tournois/data_over_blog_kiwi_…reglement-table-coinche.pdf.txt` | « Les annonces acceptées sont : 80…90…100, etc…, je passe, c'est bon, **dix et vingt de mieux**. » |
| « je passe » obligatoire, « passe »/« allez » **interdits** | `federations/ffbelote_org_belote_contree.txt` | « Il devra dire directement « Je passe ». **Toute autre expression telle que « Passe » ou « Allez » est interdite.** » |
| **« Je passe », « Passe » ou « Allez » AUTORISÉS** | `tournois/cdf_missegre11_…` (pourtant copie du texte FFB) | « « Je passe », « Passe » ou « Allez » et **toujours utiliser la même expression** au long de la partie. Les autres expressions sont interdites. » |
| « je passe » **ou** « allez », mais constant | `divers/belotecontree_free_reglement.txt` / `tournois/ainesruraux_…` | « Celui qui n'a pas d'annonce particulière à formuler devra dire « je passe » **ou « allez »**. … Quelle que soit l'expression employée, elle devra **rester identique pendant toute la partie**. » |
| « parole » | `apps-sites/contree_org_4_joueurs.txt` | « soit passer en prononçant « **parole** » » |
| **pas de délai de réflexion** | `divers/belotecontree_free_reglement.txt` | « Si un joueur passe, il doit le dire **immédiatement, sans délai de réflexion**. De la même manière toute annonce doit être formulée sans délai de réflexion qui apparaîtrait anormalement long. » |
| idem | `federations/ffbelote_org_belote_contree.txt` | « lors de la prise, celle-ci doit être annoncée **distinctement et sans temps d'hésitation** … il est strictement interdit de faire transparaître un soupçon d'hésitation ou bien une **intonation d'interrogation** dans la voix » |

**Divergence directe, et instructive** : la FFB **interdit** exactement les locutions qu'un
règlement de table **autorise nommément** (« dix et vingt de mieux »), et sa propre copie de
Missègre rouvre « Passe » et « Allez » qu'elle venait de bannir. L'enjeu n'est pas cosmétique :
« 20 de mieux » est une information supplémentaire (un delta explicite, indépendant du niveau
absolu), et la vraie règle anti-signaux n'est pas le vocabulaire mais le **tempo** — que seules
deux sources codifient.

### 13c. Enchère annulée pour irrégularité

| Règle | Source | Ce qu'elle dit |
|---|---|---|
| annonce nulle, l'équipe fautive ne peut plus que passer | `divers/belotecontree_free_reglement.txt` / `tournois/ainesruraux_…` art. 9B | « Si un joueur se trompe en annonçant (valeur d'annonce insuffisante, intempestive…) **son annonce est nulle et son équipe ne peut plus que passer**, l'autre équipe étant libre de poursuivre les enchères … ou de jouer le contrat fixé avant l'annonce annulée. » |
| l'adversaire peut annuler la mène | idem | « Si le contrat doit être joué par l'équipe ayant commis l'erreur d'annonce, l'autre équipe pourra décider l'annulation de la mène **mais à condition qu'elle prenne cette décision immédiatement** » |
| interdiction de prise à la 2e maldonne | `federations/ffbelote_org_belote_contree.txt` ; `tournois/cdf_missegre11_…` | « l'équipe ayant commis la faute se verra pénalisée et **interdite de toute prise sur cette seconde donne** » |
| le donneur fautif ne peut plus enchérir | `tournois/web_archive_org_…coinche_stephanoise…` | « En cas de maldonne, le donneur redistribue, **mais il ne pourra pas faire d'enchère.** » |
| contrat irrévocable | `federations/ffbelote_org_belote_contree.txt`, `carafons`, `cartesetcie`, `missègre`, `ilyesbrh` | « **Tout contrat annoncé ne peut être annulé.** » / « A bid, once announced, **cannot be cancelled** » |

**Consensus de principe** (« l'erreur ne doit pas bénéficier à l'équipe qui l'a commise »),
**divergence sur la sanction** : privation de prise pour la donne, annonce nulle avec équipe
réduite au silence, ou annulation de la mène au choix de l'adversaire. Seuls cinq documents du
corpus abordent la question — **tous des règlements de compétition ou des specs de projet**.
Les sites de règles sont muets.

---

## 14. Récapitulatif

| Axe | Verdict | Ligne de fracture |
|---|---|---|
| 1. Enchère minimale | **Divergence** | 80 (FFB + francophones) vs 82 (Pagat, anglophones, implémentations) — vestige de la règle « faire plus que la défense », que la FFB a supprimée |
| 2. Pas | **Consensus** (10) | seules variantes annoncées : 5 en 5 (Wikipédia), « 10 ou plus » (drasill) |
| 2b. Plafond | **Divergence** | 160 si les annonces ne comptent pas ; 170/180 si la belote compte ; 650 si les annonces comptent. **La FFB dit 160 sur sa page contrée et 650 sur sa page coinche** |
| 3. Surenchère en couleur | **Consensus** (valeur seule) | 4 sources le disent *contre le bridge* ; hiérarchie de couleurs = variante Pagat + « conventions locales » (belotepoint) |
| 4a. Parler sur soi-même | **Consensus** (interdit) | 11 sources, aucune opposition frontale |
| 4b. …sauf en changeant de couleur | **Divergence** | 4 sources indépendantes rouvrent l'enchère au dernier parleur ; conséquence directe sur l'axe 11 |
| 4c. Surenchérir sur son partenaire | **Consensus** (autorisé) | c'est le mécanisme de communication ; aucune source ne l'interdit |
| 5. Reparler après avoir passé | **Consensus** (oui) | 13 sources pour, **zéro contre** — le plus net de la matrice |
| 6a. Capot ferme l'enchère chiffrée | **Consensus** | « il n'existe pas d'enchère supérieure au capot » |
| 6b. Capot contrable | **Divergence** | FFB et tradition FR : oui (capot contré 1 000) ; Pagat + gamerules : **« cannot be doubled »** |
| 6c. Valeur du capot | **Consensus** (250) | sauf 260 (gameduell), 270 « capot beloté » (ismo009), 350 (Ducale), 500 forfait (FFB ancienne), et Les Essarts qui interdit de dire « capot » — et en tire la conséquence que **les enchères continuent** |
| 7. Générale | **Divergence** | enchère de **coinche**, absente de tous les textes « contrée » et de toute la FFB ; valeur sans consensus (350/500/700/1000) ; deux sources l'emploient au sens de « capot » ; désaccord sur ce que fait le partenaire |
| 8. SA/TA | **Consensus** sur le principe et sur As=19 à SA ; **divergence totale** sur TA | six barèmes incompatibles (deux à l'intérieur de la seule FFB), trois ratios, une table non linéaire, un règlement qui double le contrat. « None of them particularly elegant » (Pagat) |
| 9. Coinche à la volée | **Divergence maximale** | FFB + ses copies + 3 règlements de concours contre presque tout le reste. **Ce n'est pas « tournoi vs bistrot »** : l'ASCEE 2A est un règlement de tournoi et autorise la volée. Le désaccord traverse même une copie verbatim (carafons vs cartesetcie) |
| 9b. Le contre gèle | **Consensus majoritaire**, mais | le « décoinchage » est porté par un règlement de championnat (Stéphanoise) ; Pagat fait du contre *non bloquant* la définition de « contrée », à l'inverse de l'usage français |
| 10. Surcoinche | **Consensus** (camp contré, fin immédiate) ; **divergence** sur ×3 vs ×4 | même clivage FFB ancienne/récente que dans arrondi.md ; 4 sources ne la connaissent pas, dont la doc drasill alors que son code l'implémente |
| 11. Fin des enchères | **Consensus** (3 passes ; 4 passes = redonne) | drasill compte 4 passes, ange.heureux/alhoa rouvrent au dernier parleur — c'est l'axe 4b vu de l'autre bout. Divergence secondaire : le donneur tourne (FFB) ou pas (ContrAI) |
| 12. Annonces | **Pas un désaccord : une définition** | contrée = sans annonces ni SA/TA ; coinche = avec. Divergences réelles : comptent-elles pour *faire* le contrat (FFB oui, IsCool/belotepoint/ilyesbrh non, gameduell les deux, jeux-regles.com les deux dans la même phrase) ; une carte peut-elle servir à deux annonces (FFB non, Stéphanoise et contree.org oui) |
| 13a. Sens de la parole | **Consensus** (à droite du donneur) | les éditeurs en ligne disent « à gauche » — vraisemblablement une erreur importée du bridge |
| 13b. Formulation | **Divergence** | la FFB **interdit** « 20 de mieux » ; un règlement de table l'**autorise nommément** ; la copie FFB de Missègre rouvre « Passe » et « Allez » |
| 13c. Enchère irrégulière | **Consensus de principe, divergence de sanction** | cinq documents seulement, tous des règlements de compétition |

---

## 15. Ce que personne ne dit

Utile à savoir avant d'implémenter quoi que ce soit : sur ces points, le corpus est **muet**,
pas divisé.

- **Combien de tours d'enchères au maximum ?** Une seule source pose une borne : Wikipédia FR,
  en variante — « **on ne peut enchérir plus de deux fois** ». Aucune autre. Une enchère peut
  donc formellement tourner indéfiniment tant que quelqu'un monte de 10.
- **Que se passe-t-il si quelqu'un veut surenchérir sur un capot ?** Personne ne prévoit le cas
  « capot après capot » — sauf `open-source/valmathieu_ContrAI_main_contree-domain.md`, qui en
  fait une règle explicite et **asymétrique** : « **Solo Slam … cannot be announced after a
  Slam** — once a Slam is on the table, the auction is closed to further contract bids ».
- **Le partenaire du contreur doit-il parler ?** Une seule source tranche, la FFB : « Le
  partenaire du joueur ayant contré **ne doit pas parler (ne pas dire je passe)** ». Toutes les
  autres laissent la question ouverte, ce qui rend le « 3 passes » ambigu après un contre — et
  c'est exactement là que le code de drasill diverge en exigeant 4 passes.
- **Peut-on contrer après avoir soi-même passé ?** Deux sources seulement, et elles disent oui :
  ContrAI (« Intervening passes do not close the Coinche / Surcoinche window ») et, de fait, le
  code de BGA. Toutes les autres sont muettes, alors que la question se pose à chaque donne.
- **Le temps de parole.** Deux sources, et ce sont celles qui interdisent la volée : « sans
  délai de réflexion qui apparaîtrait anormalement long » (tournoi international), « annoncée
  **distinctement et sans temps d'hésitation** » (FFB). C'est pourtant le seul vrai garde-fou
  contre le signal par le tempo — plus efficace, sans doute, que d'interdire « 20 de mieux ».
