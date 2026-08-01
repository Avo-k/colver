# Backlog web (non implémenté)

Idées notées, rien d'engagé. (Dernière mise à jour : 2026-08-01)

Rangé par **gain / effort**, pas par thème : les premiers blocs sont ce
qu'il faut faire pour passer d'un site personnel à un site à quelques dizaines
de joueurs, dans l'ordre où ça se fait. L'effort est en journées de travail,
très grossièrement.

L'ancien bloc 1 (« gain fort, effort faible » : fuite des mains par
`/api/games/{id}`, limites de débit, SEO, backup de la base, logging + `/health`,
rename `dmc_50.bin`) a été **implémenté le 2026-08-01** et retiré d'ici — voir
les commits de ce jour-là. La numérotation des blocs restants est conservée
pour que les renvois (§2.2, §3.1…) restent stables.

Un fil relie plusieurs entrées : **l'état d'une partie solo ne vit que dans les
locales de `_websocket_session`** (le corps du gestionnaire WebSocket,
extrait de `websocket_endpoint` le 2026-08-01 sans changer sa structure).
C'est ce qui empêche à la fois de drainer un
déploiement (§2.6), de découper le gestionnaire WebSocket (§3.2) et de faire
tourner plus d'un worker (§3.3). Un registre des sessions solo est le préalable
commun. (Le plafond de recherches du §2.2, lui, n'en a **pas** besoin : tout
coup de bot, solo comme salon, passe par `AgentTable.decide` — un sémaphore de
module suffit.)

---

## 2. Gain fort, effort moyen

### 2.1 Cycle de vie du compte (2-3 j, indispensable au-delà du cercle proche)

Il n'y a **ni changement de mot de passe, ni réinitialisation, ni adresse
e-mail, ni suppression de compte** — `auth.py` ne fait que register / login /
logout. À quelques dizaines de joueurs, le premier « j'ai perdu mon mot de
passe » n'a aucune réponse possible.

La suppression n'est pas qu'un confort : on stocke un pseudo, un bcrypt et
l'historique de jeu de personnes réelles, sans mentions légales ni politique de
confidentialité. Décider aussi ce que devient une donne quand son joueur part
(anonymiser `games.user_id` plutôt que casser les parties de salon).

### 2.2 Dédé : inverser le budget — qualité fixe, latence bornée (1-2 j, qualité de jeu)

Le plafond du serveur, c'est le CPU de la recherche, pas l'I/O. Dédé cherche
1000-1200 ms par coup avec `parallel = true` par défaut (`agent/spec.rs:127`),
donc **un seul coup occupe tout le pool rayon**, et une donne fait ~24 coups de
bot.

Le piège est là : en mode budget-temps, la contention ne rend pas Dédé plus
lent, elle le rend **plus faible** — il résout moins de mondes dans les mêmes
1,2 s. La latence reste parfaite sur le tableau de bord pendant que la force de
jeu s'effondre. Même mécanique côté sidecar playgen : une seule machine, un
aller-retour par 256 mondes et par coup, avec `fallback = "uniform"`
(`agents.py:60`) — saturé, tout le monde retombe silencieusement sur des mondes
uniformes.

Le correctif n'est pas d'abord de surveiller, c'est d'**inverser le mode de
budget** : N mondes fixes (mode compte) sous un plafond de temps — « N mondes
ou T ms, premier atteint ». La qualité devient la constante et la dégradation
change de canal : elle devient de la latence, le seul canal honnête — un humain
voit un bot réfléchir plus longtemps, il ne verra jamais qu'il réfléchit plus
mal. Trois raisons produit :

- Dédé doit être **le même adversaire** à vide et en charge : on étudie son jeu
  contre lui, et son Elo est une identité (`("bot", "dede")`) — une force qui
  flotte avec la charge bruite aussi les classements humains gagnés contre lui.
- **La dégradation devient la métrique** : `determinizations < N` (déjà rapporté
  à chaque décision, `agents.py:176`) dit exactement ce que le plafond a coûté.
  « Mesurer d'abord » se réduit à brancher des chiffres déjà calculés sur le
  logging en place depuis le 2026-08-01.
- Le surcoût de latence est en partie **absorbé par construction** : le bot
  réfléchit dans la pause d'affichage (`pacing.hold`) ; seuls les vrais pics
  émergent, et ce sont eux qu'on veut voir.

Ce que le code impose (relevé 2026-08-01) :

- **Le mode combiné n'existe pas** : temps et compte sont exclusifs
  (`is_dd.rs:853-860`, `spec.rs:362`) ; sortir de la boucle sur l'une **ou**
  l'autre borne est le petit changement Rust.
- **Nouvelle clé TOML obligatoire** (`time_cap_ms`, active en mode compte
  seulement). Ne pas réinterpréter « les deux clés posées = les deux bornent » :
  `determinizations` vaut 20 par défaut et les TOML d'arène posent déjà les
  deux avec temps-qui-gagne (`v6_isdd_75M_isdd.toml` : `time_ms = 50` *et*
  `determinizations = 20`) — chaque bot temps existant deviendrait
  silencieusement un bot à 20 mondes.
- **Garder l'échelle cards-left** pour le plafond (le budget temps actuel est
  × cards_left/8, `is_dd.rs:830`) : un solve coûte ~6× moins cher en milieu de
  donne, un plafond plat serait trop serré tôt et inutile tard.
- Déjà en place, gratuit : en mode compte le sidecar fait **un aller-retour par
  coup** (tout le budget demandé d'un bloc, `is_dd.rs:876`) au lieu d'un par
  batch de 256 — moins de fenêtres de repli ; et `set_time_ms` (retune de
  Regarder) est déjà no-op en mode compte (`colver-py/src/lib.rs:1333`), rien
  ne re-bascule un agent en mode temps par accident.

Ordre : (1) Rust `time_cap_ms`, + test que le plafond borne et que
`determinizations` avoue le manque ; (2) journaliser par coup de bot — elapsed,
determinizations, worlds_source (le repli sidecar est la deuxième dégradation
silencieuse, la même passe le couvre) ; (3) calibrer N = p50 d'un serveur **à
vide** à 1200 ms, par phase de donne si possible — le basculement ne change
alors rien au cas non chargé, il le fige ; plafond ≈ 2-3× le temps nominal.
L'avertissement « l'entame peut prendre plusieurs secondes » (`agents.py:27`)
date d'avant le solve fenêtré du 2026-07-26, à re-mesurer ; (4) basculer le
défaut web — `ISDD_DETS` (`agents.py:29`) existe déjà comme interrupteur ;
(5) plafonner les recherches simultanées.

Le sémaphore (5) reste nécessaire — le mode compte le rend même plus rentable :
avec `parallel = true`, **une** recherche sature déjà le pool rayon, K
recherches simultanées se ralentissent K× sans débit en plus. Plafond à 1
(2 max) = file FIFO de recherches pleine vitesse : l'attente devient de la
latence visible, et la qualité ne s'entame qu'au plafond, en l'avouant. Pas
besoin du registre §3.1 pour ça (voir l'entête du fichier).

Corollaire d'affichage : **§4.4**. Tout ce bloc repose sur « la dégradation
devient de la latence, le seul canal honnête » — encore faut-il que le joueur
lise cette latence comme un bot qui réfléchit, et pas comme un serveur bloqué.
Sans le signal, inverser le budget échange une dégradation invisible contre une
autre.

### 2.3 File d'attente visible pour `agent_review` (1 j, UX sous charge)

`_gate = asyncio.Semaphore(1)` (`agent_review.py`) sérialise la revue à
l'échelle du processus, à 7-10 s la donne. C'est le bon choix aujourd'hui — il
protège le sidecar — mais à plusieurs joueurs sur Rejouer la file grandit sans
borne et **le client ne voit rien**, ni sa position, ni qu'il attend. Le flux
`agent_review_start` peut porter un rang d'attente ; refuser au-delà d'une
profondeur donnée plutôt que promettre.

Accessoire : `_locks` (`analysis.py:35`, `agent_review.py:49`) ne se vide que
sur le chemin nominal.

### 2.4 Valider les messages WebSocket (2 j, robustesse)

Tout est en `data.get(...)` brut, avec des `int()` non vérifiés
(`int(data.get("human_seat", 2))` et compagnie). Une exception dans la boucle
**tue le socket**, donc tue la donne en cours (qui n'est pas reprenable). Un
modèle Pydantic par type de message supprime toute une classe de plantages et
documente le protocole au passage — c'est le même travail.

### 2.5 Premiers tests Python et CI qui teste (2-3 j, dette)

Le noyau Rust est bien testé ; la couche web (6 600 lignes, et la logique la
plus retorse du projet : reprise de partie, pilote de salon, course du dernier
pli dans `_wait_human_card`) n'a **aucun test**. Et la CI se limite à
`publish.yml` : ni test, ni lint, ni typage au push, aucune configuration
ruff/mypy/pytest dans `pyproject.toml`.

Par où commencer, dans l'ordre du risque : `match_state.Match` (record
idempotent, `finished` à égalité), la reprise (`load_open_match` → `restore`,
et le fait que le score marqué ne se reconstitue pas en sommant les donnes), le
passe forcé, `pacing.resolve` dégradé.

### 2.6 Déployer en fin de donne, pas au milieu (drain)

Aujourd'hui un déploiement, c'est `docker compose up -d --build` : le conteneur
est tué, toutes les WebSocket tombent d'un coup. Le coût n'est pas symétrique
entre les deux échelles de jeu :

- La **partie** survit — le score cumulé est en base (`matches.points_ns/ew`) et
  un joueur connecté la reprend via `_resume_match`.
- La **donne en cours est perdue pour de bon** : les bots n'ont aucun état
  persistant (IS-DD, sources de mondes), rejouer les actions ne le leur rendrait
  pas. C'est exactement pour ça que « Reprendre » concède la donne courante. Un
  déploiement pendant qu'on joue inflige ça sans prévenir, et un joueur anonyme
  (pas de `matches.user_id`) perd tout, pas seulement la donne.

Donc : attendre la fin des donnes actives avant de couper. Fin de **donne**
suffit, pas fin de partie — c'est la seule granularité qui soit à la fois
nécessaire et bornée (~42 s en standard, ~16 s en rapide ; une partie en 2000
points, elle, n'a pas de durée maximale).

Forme envisagée :

1. Signal d'arrêt (SIGTERM, ou endpoint admin) → le serveur passe en état
   *drain* : refuse les nouvelles donnes et parties (`play_start`,
   `room_create`, `room_join`, `room_next_deal`) avec un message de maintenance,
   et le dit aux clients déjà connectés.
2. Attendre que les donnes actives se terminent : les salons sont énumérables
   (`rooms.ROOMS`), le solo **ne l'est pas** — l'état d'une partie solo ne vit
   que dans les locales de `_websocket_session`, il faudrait un registre.
3. Plus rien d'actif (ou timeout) → sortir, `restart: unless-stopped` relance.

Pièges repérés :

- **Il n'y a pas de hook d'arrêt** : `server.py` n'a que des
  `@app.on_event("startup")` (backfill Elo, backup de la base — l'idiome est
  lui-même déprécié, passer à `lifespan` en même temps).
- **`stop_grace_period` vaut 10 s par défaut** dans Compose : sans l'allonger,
  Docker `SIGKILL` le conteneur avant la fin de la donne et le drain ne sert à
  rien.
- **Timeout obligatoire**, et compter l'*activité*, pas la connexion : un onglet
  laissé ouvert sur une donne jamais jouée bloquerait le déploiement.
- **Deux processus à coordonner** : le web et le sidecar playgen
  (`playgen-gpu.service` sur moxxi). Redéployer le sidecar coupe IS-DD en pleine
  recherche — le drain du web ne protège rien si le sidecar tombe pendant.
- Les tâches d'analyse (`agent_review` qui streame, `annonces_sim`,
  `card_analysis`) sont à **annuler**, pas à drainer : elles sont recalculables
  et déjà annulables.

### 2.7 Recharger la page efface une donne mal engagée (1-3 j, équité)

Signalé par un joueur : en pleine donne, quitter la page (retour navigateur,
rechargement) puis revenir suffit à faire disparaître la donne en cours et à en
obtenir une neuve. Vérifié — c'est même le comportement *documenté* de la
reprise (`_resume_match`, `server.py:1348`) : « une donne abandonnée n'a pas eu
lieu », le même donneur redistribue, au score acquis. Or elle ne coûte rien
nulle part : la ligne `games` reste `is_complete = 0`, donc ni les points de la
partie (`Match.record` n'est jamais atteint), ni l'Elo (`rate_game` exige
`is_complete = 1`, `elo.py:89`) ne la voient.

L'exploit : contrat coincé ou mal engagé → F5 → « Reprendre » → redonne
gratuite, répétable à volonté. En partie de 1000, on ne perd que les donnes
qu'on choisit de finir. Même mécanique hors partie pour l'Elo : recharger avant
l'état terminal et la donne n'est jamais notée. Le salon n'est pas touché (le
driver continue, la reconnexion rebranche le siège).

La vraie sortie : **reprendre la donne en cours**, pas seulement la partie.
§2.6 affirme que « rejouer les actions ne rendrait pas leur état aux bots »,
mais c'est plus vrai en principe qu'en pratique avec les bots actuels :
DouDou50 est sans état, et Dédé reconstruit ses mondes à chaque coup depuis le
préfixe visible, que ses sources suivent via `init_deal`/`observe` — préfixe
entièrement rejouable, puisque `games.hands` est écrit à la création et
`games.actions` à chaque coup (`database.py::append_action`). Reconstruire une
`PlaySession` depuis la base et rejouer les actions (env + `observe` des
agents) semble donc faisable. Bénéfice collatéral : ça rendrait indolores le
déploiement en pleine donne (§2.6) et le socket tué par un message malformé
(§2.4).

En garde-fou résiduel (ou en attendant) : une donne abandonnée en partie ne
doit pas être gratuite — à trancher entre la marquer chutée pour l'abandonneur
et n'en compter que les points cartes déjà pris par l'adversaire. À couvrir par
§2.5 : la reprise est déjà en tête de la liste des tests à écrire.

---

## 3. Effort important, gain structurel

### 3.1 Registre des sessions solo (2-3 j, débloque le reste)

Préalable de §2.6 (drain) et §3.2 — plus du plafond de recherches de §2.2,
qui passe par `AgentTable.decide`. Aujourd'hui une
partie solo n'existe nulle part ailleurs que dans la pile de
`_websocket_session` : rien ne peut l'énumérer, la compter ni l'attendre.

### 3.2 Découper `_websocket_session` (3-5 j, maintenabilité)

`server.py` fait ~2 650 lignes et le gestionnaire WebSocket est **une fonction
d'environ 1 400 lignes**, 29 branches `elif msg_type ==` et une douzaine de
variables `nonlocal`. Les commentaires sont excellents et expliquent le
*pourquoi* mieux que la moyenne — c'est la forme qui pose problème : chaque
nouveau message élargit la même closure.

Cible : un objet session portant l'état aujourd'hui `nonlocal` (= §3.1), une
table de dispatch message → handler, et les helpers métier qui traînent en fin
de `server.py` (la simulation `doudou`, par exemple) remontés dans leur module.
À faire **après** §2.4 : les modèles de messages donnent les frontières.

### 3.3 Sortir l'état du processus (gros, à ne pas faire tout de suite)

Le serveur est mono-processus et à état : salons en mémoire (`rooms.ROOMS`,
`MAX_ROOMS = 20`), sessions solo en locales, SQLite en connexion unique. On ne
peut donc **pas** lancer `--workers 2`, ni un second conteneur, et chaque
déploiement est une coupure franche (d'où §2.6).

À quelques dizaines de joueurs c'est le bon choix, et « scaler » veut dire
acheter une machine plus grosse — ce qui est honnête tant qu'on connaît son
plafond CPU (§2.2). À noter comme dette assumée, pas comme travail à faire :
distribuer avant d'avoir mesuré coûterait cher pour rien.

Manque aussi, plus petit : aucun délai d'inactivité sur les WebSocket (le
plafond — par IP et global, `COLVER_WS_PER_IP` / `COLVER_WS_TOTAL` — existe
depuis le 2026-08-01, mais un socket ouvert et muet est gardé pour toujours).

---

## 4. Confort et cohérence

### 4.1 Le score de partie dans les pages d'analyse

Les pages d'analyse raisonnent toutes à **0-0**. `AgentTable.set_scores` n'est
appelé que depuis `game_manager.py` (donc en jeu, solo et salon) ; ni
`/analyse/annonces` ni `/analyse/jeu` ne le font. Or bid v6 lit une observation
*score-aware* : la même main s'annonce autrement à 900-200 qu'à 0-0. Analyser
une annonce hors de son score, c'est donc poser une autre question que celle
que le joueur s'est posée à la table.

Ce qu'il faudrait :

- **Annonces** : un score de partie saisissable (et pré-rempli depuis la partie
  d'origine quand on arrive par « Analyser cette annonce », via `from=<gameId>`
  — le score cumulé vit dans `matches.points_ns/ew`, pas dans `games`). Il
  appartient à la question, donc à la clé de déduplication des mains
  enregistrées, au même titre que les enchères précédentes.
- **Analyse du jeu** : même chose, mais l'effet est indirect — c'est le bidder
  qui est score-aware, pas DouDou50 ni le solveur. Ça ne change les chiffres que
  pour les mondes échantillonnés (playgen tokenise l'auction) ; à vérifier avant
  d'y mettre du travail.
- Le CFN complet ne porte pas le score de partie : il faudra un paramètre d'URL
  de plus (`?ns=&ew=`) pour que la situation reste partageable — c'est le
  principe déjà en place sur ces deux pages.

### 4.2 Le « vrai monde » sur la page annonces

`/analyse/jeu` sépare deux questions que la page annonces confond en n'en posant
qu'une : *le vrai monde* (un solve sur la donne réelle, information parfaite) et
*les mondes de l'information set* (échantillonnés depuis ce que le siège pouvait
savoir). Le raisonnement est dans [web_analyse_jeu.md](web_analyse_jeu.md) §2 et
l'implémentation dans `card_analysis.true_world()` — une fonction à part, une
colonne à part, **jamais fusionnée** avec les mondes échantillonnés.

L'annonce mérite le même traitement. Aujourd'hui la page répond « ton 130♥ passe
dans 68 % des mondes que tu pouvais imaginer » et ne peut pas ajouter « …et
cette donne-là était dans les 32 % ». C'est pourtant la phrase qui referme la
boucle avec Rejouer : elle distingue une bonne annonce punie par la donne d'une
mauvaise annonce sauvée par elle.

**Ça n'enlève rien.** Le bandeau couleur × palier, la synthèse Oracle et le Jeu
réel restent tels quels ; c'est une ligne de plus dans la box Jeu parfait.

**Et surtout, ça n'entre pas dans le tirage des mondes.** Conditionner le pool
sur la donne réelle transformerait « cette annonce était-elle bonne ? » en
« a-t-elle marché ? » — la seule des deux questions qui n'apprend rien. Les
pourcentages et les teintes de confiance de la page deviendraient du rétroviseur
sans le dire.

Conditions d'affichage : **il faut connaître les quatre mains**. Deux entrées,
et deux seulement —

- on arrive depuis Rejouer (`from=<gameId>`) : rien à calculer, `analysis.py:88`
  fait déjà `solve_all_suits()` sur la vraie donne et cache `suits` + `best` par
  équipe dans la table `analysis` ;
- on colle un CFN complet 4 sections, comme `/analyse/jeu` (la page annonces ne
  prend aujourd'hui que `?hand=` + `?history=`, il faut donc lui ajouter cette
  entrée).

Main saisie à la main = pas de vraie donne, pas de ligne. C'est le cas nominal
de la page, la ligne est donc **conditionnelle par construction**, comme le lien
« ← Retour à la partie ».

Pièges :

- **n = 1, et la page encode la taille d'échantillon dans son rendu** (Wilson,
  classes `doudou-high/mid/low`, taille de police croissante avec le nombre
  d'observations). Rien de tout ça ne doit toucher cette ligne : une valeur
  exacte unique ne doit pas emprunter le langage visuel d'un pourcentage
  échantillonné, sinon elle se lit comme la cellule la plus sûre du tableau.
- **Elle appartient au Jeu parfait, donc à l'état partagé entre onglets**, pas à
  l'onglet courant : comme l'Oracle, elle ne dépend pas de l'annonce analysée.
- **Échelle** : points cartes DD, dans la box Oracle, sur les mêmes axes couleur
  × palier. La page a déjà le piège des deux échelles (Oracle en points cartes
  vs Jeu réel en points de donne marqués) ; ne pas en introduire une troisième
  présentation.
- **Clé de déduplication des mains enregistrées** : elle porte sur (main,
  enchères précédentes). La même main venue d'une partie et tapée à la main s'y
  confondraient, l'une portant une vraie donne et l'autre non. Le plus simple
  est de perdre la vraie donne au rechargement plutôt que d'élargir la clé — à
  arbitrer avec §4.1, qui veut y ajouter le score de partie.

### 4.3 Minuteur sur le coup d'un joueur — seulement quand d'autres humains attendent

En solo, un joueur qui réfléchit dix minutes ne gêne que lui : aucun minuteur.
En salon, dès qu'**au moins un autre humain** est à la table, un siège muet
bloque tout le monde — et aujourd'hui il les bloque *pour toujours* :
`Room._drive` attend le coup humain sur sa queue sans échéance, et un joueur
déconnecté (les sièges sont liés au compte) gèle la partie jusqu'à ce qu'il
revienne ou qu'un autre quitte. La condition est donc le **nombre d'humains
dans le salon (≥ 2)**, pas le mode ni le rythme.

Le mécanisme existe déjà en deux exemplaires : le passe forcé (pas de décision
→ le serveur joue) et surtout le dernier pli, où le délai est une **échéance et
non une attente** — le clic avant l'échéance coupe court, sinon le serveur pose
la carte (`_wait_human_card`). Ici c'est la même chose avec une échéance longue
(30-60 s ?) : `asyncio.wait_for` autour de l'attente de la queue, et au timeout
le serveur joue pour le siège — le bot du rythme (`Room.bot_type`) choisit le
coup, jamais un coup au hasard, pour que la donne reste digne d'être jouée et
analysée.

À trancher / pièges :

- **Afficher le compte à rebours** aux deux camps : au joueur au trait (c'est
  l'avertissement) et à ceux qui attendent (c'est la promesse que ça se
  débloque). Probablement seulement sur les dernières secondes, pour ne pas
  mettre la pression dès la première.
- **Enchère vs carte** : une annonce se réfléchit plus qu'une carte en milieu
  de pli ; au timeout d'une enchère, « passe » est un coup sûr et neutre — pas
  besoin du bot.
- **Ne pas doubler le dernier pli** ni le passe forcé, qui ont déjà leurs
  échéances propres.
- **L'Elo** : la donne continue et reste notée. Un joueur qui abandonne son
  siège voit le bot jouer à sa place mais c'est toujours *sa* donne — c'est le
  choix déjà fait pour la reconnexion, le minuteur ne fait que le borner dans
  le temps.

### 4.4 Dire quand Dédé dépasse sa pause (0,5-1 j, corollaire du §2.2)

La ligne « Nord réfléchit… » existe déjà (`#play-status`,
`shared/table.js:532`) mais elle est rendue **depuis l'état seul** : le texte
est identique que l'attente soit 0,9 s de pause d'affichage ou 4 s de
recherche. Le client n'apprend jamais rien — le message `ai_move` arrive
simplement en retard, et une attente inexpliquée se lit comme un serveur
planté, pas comme un bot qui réfléchit.

C'est pour ça que ça tient au §2.2 : inverser le budget déplace la dégradation
vers la latence *parce que* la latence est le canal honnête. Elle ne l'est que
si elle est signalée. Sinon on remplace une dégradation invisible (Dédé joue
plus mal) par une autre (Dédé a l'air cassé).

**Le seuil n'est pas 1 s dans l'absolu.** Le bot réfléchit *dans* la pause
d'affichage (`pacing.hold`), et cette pause vaut déjà 1,4 → 0,9 s par carte en
`standard` (`pacing.py:23`) contre 0,6 → 0,25 s en `rapide` : un seuil absolu
d'une seconde s'allumerait à chaque coup dans un mode et jamais dans l'autre.
L'événement à montrer est le **dépassement de la pause** — `elapsed > target`,
que seul le serveur connaît (le client n'a pas la table de tempo, et la
dupliquer côté JS la ferait diverger). Le « 1 s » revient comme plancher :
signaler quand `elapsed > target` **et** que l'attente totale dépasse ~1 s,
pour qu'un dépassement de 0,2 s en `rapide` ne fasse pas clignoter la table.

Forme : la recherche tourne déjà hors boucle d'événements (`asyncio.to_thread`,
`server.py:2221` ; `run_in_executor`, `rooms.py:410`), donc un `asyncio.wait`
avec `timeout=target` sur la tâche suffit — à l'échéance, émettre un message
`bot_thinking` et continuer d'attendre. Côté client, escalader la ligne
existante plutôt qu'ajouter un élément.

Pièges :

- **Ne pas cohabiter avec le minuteur du §4.3** : deux comptes à rebours dans
  le même `#play-status` (« le bot réfléchit » / « c'est à toi et ça expire »)
  se contrediraient. Trancher lequel occupe la ligne, ou leur donner deux
  langages visuels distincts — les deux entrées se font de préférence ensemble.
- **Rien pendant le dernier pli** (hors tempo, `LAST_TRICK_CARD = 0,3 s`, aucune
  décision), ni sur un passe forcé, ni sur l'état terminal : le status y porte
  déjà autre chose.
- **En `rapide`, le signal ne doit jamais apparaître** — DouDou50 répond en
  ~1 ms. S'il apparaît, c'est le mode dégradé (Dédé à 400 ms,
  `pacing.resolve`), et c'est précisément l'information qu'on veut donner.
- **Un bot est nommé par sa position** (`GameTable.playerName`) : « Nord
  réfléchit », jamais « Dédé réfléchit ». La ligne le fait déjà, le signal ne
  doit pas réintroduire le nom du bot.
- **Pas de chiffres au joueur.** Une fois le §2.2 en place le serveur saura
  *pourquoi* il dépasse (plafond de temps atteint, mondes manquants,
  `worlds_source` retombé sur `cpu`) — ça va au journal, pas à la table. Le
  joueur a besoin de « il réfléchit encore », pas de « 312 mondes sur 500 ».
- Message d'un type nouveau sur une socket dont le client attend des paires
  état/coup — vérifier que les vues l'ignorent proprement si elles ne le
  connaissent pas.
