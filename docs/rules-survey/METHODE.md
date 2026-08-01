# Méthode — comment ce corpus a été réuni

Ce dossier documente les règles réellement appliquées à la belote, la coinche et la contrée :
fédérations, tournois, clubs, sites de règles, applications, implémentations open source. Le but
est de savoir **sur quoi les règlements sont d'accord et sur quoi ils ne le sont pas** — d'abord
l'arrondi, puis tout le reste.

Collecte du **1ᵉʳ août 2026**.

---

## 1. Ce qui est versionné, ce qui ne l'est pas

L'analyse et le corpus vivent à deux endroits différents, et c'est délibéré.

**Versionné** — `docs/rules-survey/`, 15 fichiers, ~700 Ko : nos analyses, les scripts, le
manifeste.

```
docs/rules-survey/
  SYNTHESE.md      la lecture de tête : consensus, fractures, où tombe Colver
  README.md        index raisonné des sources + les faux témoins
  SOURCES.md       manifeste nom de fichier → URL (généré)
  COLLECTE-2.md    inventaire de la seconde vague de collecte
  METHODE.md       ce fichier
  _sources.tsv     le manifeste, version machine (généré)
  _fetch.py        téléchargement + extraction de texte
  _manifest.py     régénère _sources.tsv et SOURCES.md depuis le disque
  _refetch.py      reconstitue le corpus brut depuis _sources.tsv
  matrices/        les 5 matrices « qui dit quoi » + la transcription du tableau FFB 2016
```

**Non versionné** — `data/rules-corpus/`, ~92 Mo, 594 sources, réparties en
`federations/ tournois/ clubs/ divers/ apps-sites/ open-source/`. Ce sont des documents de tiers,
souvent sous droit d'auteur, et leur volume n'a rien à faire dans l'historique d'un moteur de jeu.

**Pourquoi `data/` et pas un dossier à part.** `data/` est déjà gitignoré en entier et c'est déjà
là que ce dépôt met ce qui est volumineux et reconstituable (pools de donnes, corpus de belief,
replays). Le corpus de règles y tombe donc **sans une seule ligne de `.gitignore` en plus** —
alors qu'un dossier dédié aurait demandé une liste blanche (`dossier/*` puis six `!`), c'est-à-dire
exactement le genre de règle qu'un ajout futur casse en silence.

**Conséquence pratique.** Une citation de matrice comme
`federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt` est un chemin
**relatif à `data/rules-corpus/`**, et ce fichier n'est pas dans le dépôt. Pour le retrouver :
cherchez son nom dans [SOURCES.md](SOURCES.md), qui donne son URL.

---

## 2. Reconstituer le corpus

```bash
cd docs/rules-survey
uv run --with pypdf --no-project python _refetch.py                 # les 594 sources
uv run --with pypdf --no-project python _refetch.py federations     # un dossier seulement
```

**468 sources sur 594 se retéléchargent seules.** Les 126 autres sont les fichiers de
`open-source/`, récupérés en direct sur `raw.githubusercontent.com` : leur nom a été fabriqué en
remplaçant les `/` du chemin par des `_`, et cette conversion **n'est pas réversible** (des noms
de fichiers contiennent déjà des `_`). Le manifeste ne garde donc que le dépôt GitHub, ce qui
suffit largement à retrouver le fichier à la main. `_refetch.py` les liste en fin d'exécution
au lieu de les traiter.

**Ce qui ne reviendra pas identique**, et il faut le savoir avant de conclure qu'une citation est
fausse :

- Les pages mortes depuis la collecte. Le web pourrit vite sur ce sujet : sur la seule première
  passe, `coinche-stephanoise.com` et `pcsgc.fr` étaient déjà tombés et n'ont été récupérés que
  par la Wayback Machine ; `aubignanfaitses24h.fr` renvoie 520 sans aucun instantané archivé.
- Le contenu d'une page dynamique, qui aura changé — typiquement les pages d'événement du
  Festival de Cannes, dont le programme bascule d'une édition à l'autre.
- Les extractions PDF, qui dépendent de la version de `pypdf`.

Après une reconstitution ou un ajout, régénérez le manifeste :

```bash
python3 _manifest.py
```

---

## 3. Le pipeline

### `_fetch.py` — un document, deux fichiers

```bash
uv run --with pypdf --no-project python _fetch.py <dossier> <url> [<url> ...]
```

Pour chaque URL, il écrit **l'original** (`.pdf` ou `.html`) et **le texte extrait** (`.txt`), le
second commençant toujours par `SOURCE: <url>`. C'est cette ligne qui fait office de provenance et
qui alimente le manifeste. Le nom de fichier est dérivé de l'URL (hôte + chemin, non alphanumérique
remplacé par `_`), ce qui rend la collecte idempotente : relancer sur la même URL réécrit le même
fichier plutôt que d'en créer un doublon.

Détails qui ont compté :
- l'encodage est lu dans le `<meta charset>` et non deviné — plusieurs sites français des années
  2000 sont en `cp1252`, et l'ignorer transforme « arrondi à la dizaine » en bouillie ;
- un `User-Agent` de navigateur est obligatoire, plusieurs hébergeurs associatifs répondent 403
  sans ;
- les PDF sont détectés sur la signature `%PDF` et pas sur le `Content-Type`, qui ment souvent.

### Extraire ce que `pypdf` ne voit pas

Le tableau des 16 cas de score du règlement FFB contrée 2016 est une **image** dans le PDF : il
n'apparaît dans aucune extraction texte, et c'est précisément le document le plus important du
corpus. Il a fallu rendre la page en PNG (`pymupdf`, 170 dpi) et la transcrire à la main →
[matrices/tableau-ffb-2016.md](matrices/tableau-ffb-2016.md), versionnée pour cette raison.

Même problème en pire pour le règlement de la Fédération Française de Coinche hébergé par un
club : un `.doc` binaire OLE, dont le texte a été récupéré par extraction des fragments `cp1252`.

### Trouver les sources

- **La recherche web intégrée** a fait tout le travail. Elle sature vite : au bout de trois ou
  quatre requêtes tournant autour de la même formulation, elle ressort les mêmes huit résultats.
  Il faut changer de vocabulaire, pas de mot-clé — chercher les phrases que les documents
  *contiennent* (« la mise dedans compte pour », « le concours se déroule en X parties de Y
  donnes », « les parties se comptent au point ») plutôt que ce qu'on veut savoir.
- **DuckDuckGo et Mojeek ont été essayés et abandonnés** : le premier renvoie une page
  anti-robot (HTTP 202), le second a un index trop petit (zéro résultat sur « belote contree
  reglement arrondi »). Le script `_search.py` écrit pour ça a été supprimé, il ne marchait pas.
- **GitHub via `gh api`** — le `gh` installé ici est ancien, `gh search` n'existe pas :

  ```bash
  gh api "search/repositories?q=belote+contree&sort=stars&per_page=25" --jq '.items[].full_name'
  gh api "search/code?q=surcontre+coinche+in:file&per_page=15" --jq '.items[] | "\(.repository.full_name) :: \(.path)"'
  ```

  La recherche de code est le meilleur filon : elle trouve les fonctions de calcul de score, qui
  disent ce qui est *effectivement* appliqué plutôt que ce qui est écrit.
- **La Wayback Machine** pour tout ce qui est mort : `https://web.archive.org/web/2020/<url>`.

### La seconde vague

La première passe (~50 sources) a été faite à la main. La seconde a été confiée à des sous-agents
travaillant en parallèle : un pour élargir la collecte, quatre pour construire une matrice chacun
(enchères, jeu de la carte, barème, fin de partie) à partir du corpus local, avec interdiction
d'aller sur le web pour que les matrices ne citent que des documents présents sur le disque.
Inventaire de cette vague : [COLLECTE-2.md](COLLECTE-2.md).

---

## 4. Les règles d'analyse

Elles ne sont pas cosmétiques : sans elles, le corpus donne des conclusions fausses.

1. **Les copies ne votent pas plusieurs fois.** Le web sur ce sujet est un jeu de miroirs.
   `cartesetcie`, `carafons`, `missegre` et `villeconin` recopient les pages `ffbelote.org` :
   cinq pages, une voix. `gambiter` est une copie verbatim de Pagat (vérifiée par `diff`) et
   `reglesdejeux.github.io` en est une traduction automatique : trois pages, une voix. Chaque
   matrice ouvre sur le recensement de ces familles avant de compter quoi que ce soit.
2. **Muet ≠ d'accord.** Une source qui ne traite pas une question ne conforte aucune position.
   Les matrices distinguent systématiquement les deux, et signalent les axes où la majorité des
   sources est silencieuse — ce sont ceux qui se règlent à la table.
3. **Une fédération n'est pas une voix, c'est un document daté.** La FFB a publié au moins quatre
   rédactions incompatibles de son règlement de contrée, et son site HTML contredit ses propres
   PDF. Chaque citation nomme donc le document, jamais « la FFB ».
4. **Le code est une source primaire.** Une implémentation dit ce qui est appliqué, pas ce qu'on
   souhaite. Deux moteurs du corpus tranchent en sens opposé sur « atteindre ou dépasser », avec
   un commentaire explicite de part et d'autre.
5. **Citer littéralement et court.** Les matrices donnent l'extrait, pas une paraphrase : sur ce
   sujet la formulation exacte est le fait (« arrondi à la dizaine inférieure jusqu'à 5 » et
   « jusqu'à 4 » ne diffèrent que d'un caractère et désignent deux règles distinctes).

---

## 5. Biais connus du corpus

À garder en tête avant de lire un décompte de sources comme une mesure de popularité.

- **Le web sur-représente les gros organisateurs et les copies FFB.** La majorité des règlements
  de concours de village est une feuille A4 posée sur la table, qui n'existe nulle part en ligne.
- **Aucune fédération belge, suisse ou québécoise ne publie de règlement de contrée.** Le corpus
  étranger est donc fait de clubs et de sites, pas d'autorités.
- **Les applications documentent mal leurs règles.** Ce qu'elles offrent de plus utile est la
  liste de leurs *options paramétrables*, qui est le meilleur inventaire de variantes existant —
  mais elle est rarement publiée hors de l'application.
- **La collecte est datée.** Elle décrit l'état du web au 1ᵉʳ août 2026, pas un état stable.

---

## 6. Ajouter une source

```bash
cd docs/rules-survey
uv run --with pypdf --no-project python _fetch.py tournois "https://exemple.fr/reglement.pdf"
python3 _manifest.py                       # met à jour _sources.tsv et SOURCES.md
```

Puis, si elle apporte quelque chose : la citer dans la matrice concernée (`matrices/`) avec son
nom de fichier **et** un extrait littéral, en vérifiant d'abord qu'elle n'est pas une énième copie
d'un texte déjà présent. Si elle change une conclusion, remonter dans [SYNTHESE.md](SYNTHESE.md).
