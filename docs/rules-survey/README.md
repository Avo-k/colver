# Corpus de règles — belote / coinche / contrée

Collecte brute des règlements que l'on peut trouver en ligne : fédérations, tournois et
concours d'associations, sites de règles, applications, implémentations open source.
But : répondre à **qui arrondit quoi, quand, et dans quel sens**, et plus largement mesurer
sur quoi les règlements sont d'accord et sur quoi ils ne le sont pas.

**La synthèse est dans [SYNTHESE.md](SYNTHESE.md).** Ce fichier-ci n'est que l'index des sources.

Le détail axe par axe est dans `matrices/` :
[arrondi](matrices/arrondi.md) ·
[enchères](matrices/encheres.md) ·
[barème](matrices/bareme.md) ·
[jeu de la carte](matrices/jeu-de-la-carte.md) ·
[fin de partie](matrices/fin-de-partie.md).
L'inventaire de la seconde vague de collecte est dans [COLLECTE-2.md](COLLECTE-2.md).

**Le corpus brut n'est pas versionné** (~92 Mo de documents de tiers). Pour retrouver la source
d'une citation, cherchez son nom de fichier dans [SOURCES.md](SOURCES.md), qui donne l'URL de
chacune des 594 sources. Pour tout retélécharger : `python _refetch.py`. Comment le corpus a été
constitué, avec quels outils et quels biais : [METHODE.md](METHODE.md).

Collecte du 2026-08-01. La première passe (décrite ci-dessous) a réuni ~50 sources ; une seconde
passe a porté le corpus à ~1 100 fichiers, dont le dossier `clubs/` et des sources belges,
suisses, bulgares et néerlandaises.

## Comment c'est rangé

Les sources brutes vivent dans **`data/rules-corpus/`**, hors du dépôt (`data/` est gitignoré) —
tous les chemins ci-dessous et dans les matrices sont relatifs à ce dossier.

| Dossier | Contenu |
|---|---|
| `federations/` | FFB (Fédération Française de Belote) — **quatre versions différentes** du même règlement, plus les pages HTML du site |
| `tournois/` | Règlements de concours et de tournois réels (associations, comités, clubs) |
| `divers/` | Sites de règles de référence (Pagat, Wikipédia, éditeurs de cartes, blogs) |
| `apps-sites/` | Applications et plateformes en ligne, y compris hors de France |
| `open-source/` | Code de calcul de score et documents de règles de dépôts GitHub |

Chaque source existe en deux exemplaires : l'original (`.pdf` / `.html`) et un `.txt` dont la
première ligne est `SOURCE: <url>`. Outillage : `_fetch.py` (téléchargement + extraction),
`_manifest.py` (régénère `SOURCES.md`), `_refetch.py` (reconstitue le corpus). Détail dans
[METHODE.md](METHODE.md).

## Fédérations — `federations/`

Il y a **deux** fédérations françaises, et elles ne se reconnaissent pas.

- **FFB** — Fédération Française de Belote. Elle a publié **au moins quatre rédactions
  incompatibles** de son règlement de contrée, toutes encore en circulation (tableau ci-dessous).
- **FF Coinche** — Fédération Française de Coinche, fondée en 1997 à Saint-Étienne par Emmanuel
  Marquez, sur un règlement de coinche **déposé à l'INPI en 1996**. Son texte est dans
  `tournois/web_archive_org_…coinche_stephanoise…reglement_coinche_pdf.*` (récupéré via Wayback,
  le site d'origine ne le sert plus) ; contexte institutionnel dans
  `federations/coinche_en_ligne_com_federation_francaise_de_coinche.*` et
  `federations/net1901_org_association_FEDERATION_FRANCAISE_DE_COINCHE_804219.*`. Il diverge de la
  FFB sur l'enchère minimale (82), la distribution (6 puis 2), le décoinchage, le sous-coup, la
  montée sur le partenaire et le sans-atout.

Et un troisième label, indépendant des deux : le **Championnat de France de Belote Contrée** de
Cannes, organisé par BELOTE CONTREE MARALPINE au Festival International des Jeux
(`tournois/festivaldesjeux_cannes_com_*`) — parties en **2001 points**, finale 4 × 2001 + 1 × 2501,
148 équipes maximum. **Son règlement de jeu a été retrouvé** (édition 2016, signée du responsable
des tournois du FIJ) : [reglement-cannes.md](reglement-cannes.md) pour l'enquête et les réserves,
[matrices/jeu-de-la-carte.md](matrices/jeu-de-la-carte.md) pour ce qu'il dit axe par axe. C'est la
seule source du corpus qui tranche à la fois le barème et le jeu de la carte au nom d'une
compétition nationale.

Les quatre rédactions FFB :

| Fichier | Version | Ce qui la distingue |
|---|---|---|
| `ffbelote_REGLES-DE-LA-BELOTE-CONTREE.*` | ~2015 | Arrondi dizaine la plus proche (85→90). Chute = 160 + contrat. Contre = 320 + contrat×2, surcontre ×4 |
| `ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.*` | 27/01/2016 | Idem, plus le tableau complet des 16 cas de score (p. 7) et la règle de fin de partie |
| `LOCAL_regles_officielles_belote_contree.*` | plus récente, éditée par « Équipe Ludique » (Paris 9e) | **Arrondi inversé : 85→80, 86→90.** Surcontre **×3** et non ×4. Capot = contrat à 250. Pas d'arrondi sur la dernière partie. C'est la version déjà présente dans `docs/` du dépôt et celle dont Colver s'est inspiré |
| `ffbelote_regles-officielles-de-la-Belote-27-01-2016.*` | 27/01/2016 | Belote **classique** : l'arrondi y est **optionnel**, à annoncer avant le tournoi |
| `ffbelote_org_wp_content_uploads_2015_11_REGLES_DE_LA_BELOTE_COINCHEE_pdf.*` | ~2015 | Coinche (= contrée + annonces) |
| `ffbelote_org_*.html/.txt` | pages web du site | Reprises HTML, parfois divergentes des PDF qu'elles prétendent refléter |

À noter : la page `ffbelote_org_belote_contree` donne **capot contré = 500 / surcontré = 1000**
dans sa section « points annoncés » et **1000 / 2000** dans ses sections « points faits » et
« points faits + annoncés ». La contradiction est donc **interne à une seule page**, pas entre le
site et les PDF. C'est l'une des huit contradictions FFB recensées dans
[matrices/bareme.md](matrices/bareme.md) §10.

## Tournois et concours — `tournois/`

Les règlements réellement appliqués à des tables, par ordre décroissant d'intérêt :

| Source | Ce qu'on y apprend |
|---|---|
| `cdf_missegre11_com_*` (Missègre 11, 2015) | Reprend le texte FFB mot pour mot, arrondi 5→haut, chute = **160 tout court** |
| `fnasce_org_IMG_pdf_reglement_pdf` + `belote_reglement_cle1a43c7` (ASCEE 2A) | **Décompte aux points annoncés seuls** ; contre ×2 / surcontre ×4 ; partie en **1 010 points** ; « obligation de pisser » (l'inverse de la FFB) |
| `fnasce_org_IMG_pdf_Belote_Reglement_*` + `reglement_belote` | Concours de belote classique : « la mise dedans compte pour 162 points » |
| `web_myassoc_org_*` (Lions Club) | « Les parties se comptent **au point, sans arrondir**. Le capot compte 252 » |
| `geraudotloisirs_free_fr_*` | Idem : au point, sans arrondir, capot 252 |
| `rjcv_be_*` (Belgique) | Manche en **101** points ; arrondi supérieur **à partir de 6** ; litige à 81-81 |
| `maisondesessarts_fr_*` | Coinche = 160×2 = 320, **surcoinche = 160×3 = 480** |
| `ainesruraux_saintsever_com_*` | Même texte que `belotecontree.free.fr` : le vieux règlement « tournoi international » |
| `aappmakoenigshoffen_*`, `tcvb_bruche_*` | Concours de belote alsaciens : le capot vaut **162**, pénalités en 162 points |
| `villeconin_fr_*` | Fiche FFB « informations sur le jeu » : au point près, arrondi toléré pour les jetons |
| `jeu_belote_fr_*` | Objectifs de score par format : 600 en libre, 1000 en tournoi, 1500 en compétition |

## Sites de référence — `divers/`

- `pagat_com_jass_coinche` — Pagat (John McLeod), la référence encyclopédique en anglais
- `fr_wikipedia_org_wiki_Belote` — **la source la plus complète sur l'arrondi**, quatre conventions décrites, dont deux qui préservent le total à 160
- `fr_wikipedia_org_wiki_Belote_contr_C3_A9e`, `fr_wikipedia_org_wiki_Coinche`
- `belotecontree_free_reglement` — « règlement officiel récupéré lors du tournoi international », barème historique en forfaits (contre 320, surcontre 640, capot 500/1000/2000)
- `bk_jeux_ducale_*` — règle imprimée par le cartier **Ducale**
- `cartesetcie`, `carafons`, `jeux_regles`, `regles_com`, `lemagloisirs`, `clubdejeux`, `adpoker`, `exoty`, `belotepoint`, `maviedesenior`, `alhoa_free_fr`, `ange_heureux_free_fr`, `jeubelote_com`, `reglesdejeux_github_io`, `drasill_github_io` (règles BoardGameArena)

Attention aux faux témoins :
- `cartesetcie`, `carafons`, `missegre` et `villeconin` recopient les pages `ffbelote.org` — cinq
  pages, **une seule voix**.
- `gambiter` est une copie **verbatim** de Pagat (vérifiée par `diff`) et `reglesdejeux.github.io`
  en est une traduction automatique — trois pages, **une seule voix**.
- `ainesruraux-saintsever` reprend intégralement `belotecontree.free.fr`.
- En revanche `jeux-regles.com` et `regles.com`, souvent pris pour des copies FFB, sont des
  témoins **indépendants** sur l'axe barème : ils n'ont pas la signature « 89 points pour un
  contrat à 90 » et donnent une règle de coinche absente de tout texte FFB.

## Applications et plateformes — `apps-sites/`

`gambiter` (coinche + belote bulgare), `playjoy`, `gamerules`, `officialgamerules`, `ibelote`,
`licitum` (belote bulgare), `en_wikipedia_org_wiki_Coinche`, `gameduell` (Belote.com),
`iscool` (Belote Mobile), `eryodsoft` (La Coinche / Belote Contrée — l'app la plus complète en
options de règles), `play_google_com_*` (Eryod Soft, Belote Andr), `vipgames`, `ludigames`.

La belote bulgare (`gambiter_com_cards_Belote`, `officialgamerules`, `licitum`) est incluse
exprès : c'est le seul corpus qui traite l'arrondi comme un **problème d'invariant** et pas comme
une commodité de marquage.

## Open source — `open-source/`

| Fichier | Intérêt |
|---|---|
| `CephaloSophie_kydos_*_donneScoring.ts` | Le mieux documenté : base 162, der ajouté **avant** arrondi, arrondi par équipe, et l'auteur nomme le phénomène de la « casse » (total 170) |
| `ismo009_Coinche_main_game.js` | `Math.round(v/10)*10` appliqué aux deux scores finaux |
| `slim0_contree_main_backend_game_scoring.py` | Contrée, pas d'arrondi |
| `drasill_bga_coinche_main_coinche.game.php` | L'implémentation BoardGameArena |
| `ilyesbrh_twistedFate-belote_*_SOURCES.md` | Une matrice de variantes déjà construite par quelqu'un d'autre (2026-05), utile en recoupement — mais c'est une source **secondaire** produite par un agent, à ne pas citer comme un règlement |
| `gyscos_libcoinche_master_src_points_rs` | Barème en Rust |
| `valmathieu_ContrAI_main_contree-domain.md` | Modélisation du domaine par un autre projet d'IA contrée |
| `theosaulus_coinche_*`, `39Olivier_*` | Divers |

## Ce qui manque

- **La liste de tournois de l'artefact** `claude.ai/public/artifacts/2fa3291c-…` n'a pas pu être
  lue : Cloudflare renvoie la coquille SPA à `WebFetch` comme à `curl`. À coller à la main.
- Aucune fédération **belge, suisse ou québécoise** ne publie de règlement de contrée ; seul un
  club belge (`rjcv.be`) a été trouvé, et pour de la belote classique.
- Les règlements de concours de village sont majoritairement **hors ligne** (feuille A4 sur la
  table). Ce qui est indexé sur le web sur-représente les gros organisateurs et les copies FFB.
- `coinche-stephanoise.com` et `pcsgc.fr` référencent des règlements que leurs serveurs ne
  servent plus (404 / connexion refusée).
