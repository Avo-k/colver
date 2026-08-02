# Backlog web (non implémenté)

Idées notées, rien d'engagé. (Dernière mise à jour : 2026-08-01)

**Faits le 2026-08-01 (seconde passe) : §4.5, §2.5, §2.1.** Leurs sections sont
gardées, réécrites au passé, pour ce qu'elles apprennent — en particulier §4.5,
dont la cause était ouverte et est maintenant connue.

Rangé par **gain / effort**, pas par thème : les premiers blocs sont ce
qu'il faut faire pour passer d'un site personnel à un site à quelques dizaines
de joueurs, dans l'ordre où ça se fait. L'effort est en journées de travail,
très grossièrement.

L'ancien bloc 1 (« gain fort, effort faible » : fuite des mains par
`/api/games/{id}`, limites de débit, SEO, backup de la base, logging + `/health`,
rename `dmc_50.bin`) a été **implémenté le 2026-08-01** et retiré d'ici — voir
les commits de ce jour-là. La numérotation des blocs restants est conservée
pour que les renvois (§2.2, §3.1…) restent stables.

**§2.7 est fait le même jour** (la donne en cours se reprend), et sa section est
gardée pour ce qu'elle apprend : l'hypothèse qui la bloquait était fausse. Ce
qu'il en reste à faire est en §2.8.

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

### 2.1 Cycle de vie du compte — FAIT (2026-08-01), sauf les mentions légales

`auth.py` ne faisait que register / login / logout : ni changement de mot de
passe, ni réinitialisation, ni adresse e-mail, ni suppression. À quelques
dizaines de joueurs, le premier « j'ai perdu mon mot de passe » n'avait aucune
réponse.

**Ce qui est fait.** Migration v11 (`users.email` unique et facultative, table
`password_resets`), module `mail.py`, et quatre routes : `POST /api/auth/password`
(changer, en connaissant l'actuel), `/auth/email` (poser, changer ou retirer),
`/auth/forgot` + `/auth/reset` (lien par courriel), `POST /api/account/delete`.
Côté client, une section « Réglages du compte » repliée sur `/compte`, et deux
pages `/mot-de-passe/oublie` et `/mot-de-passe/nouveau` (`views/motdepasse.js`,
`noindex`).

Trois règles traversent le tout, et ce sont elles que les tests vérifient
(`tests/test_account.py`) plutôt que les codes de retour un par un :

- **Ne jamais dire si un compte existe.** `forgot` répond exactement la même
  chose pour un pseudo connu, un inconnu, un compte sans adresse et un SMTP en
  panne. Sinon le formulaire public devient un annuaire. `login` avait déjà
  cette discipline (un bcrypt brûlé sur un utilisateur absent, pour le temps de
  réponse).
- **Tout changement d'identifiant révoque les autres sessions** — c'est
  précisément le cas qu'on veut couvrir. Un changement de mot de passe garde la
  sienne (sinon on se déconnecte en se protégeant), une réinitialisation les
  tue toutes (quelqu'un a peut-être pris le compte).
- **Toute opération sensible redemande le mot de passe**, même connecté : le
  cookie prouve qu'une session a été ouverte, pas qu'on est encore devant
  l'écran.

Deux détails qui ne se devinent pas :

- **Sans SMTP configuré, `mail.send` écrit le lien au journal** au lieu de
  l'envoyer, et la demande aboutit quand même. La réinitialisation est donc
  utilisable de bout en bout en développement, et une panne d'envoi reste
  visible au lieu d'être masquée par un refus. Config :
  `COLVER_SMTP_HOST/_PORT/_USER/_PASSWORD/_TLS`, `COLVER_MAIL_FROM` ; les liens
  sont fabriqués depuis `COLVER_PUBLIC_URL`, jamais depuis `request.url` (
  derrière Cloudflare puis Caddy, ce serait `localhost:8000`).
- **Supprimer un compte n'efface pas ses donnes, ça les détache** : une donne de
  salon appartient à quatre joueurs, l'effacer prendrait la partie des trois
  autres avec elle. `games.user_id` → NULL, `game_players` et `elo_ratings`
  effacés, le siège devient « Invité » (ce que `game_seat_names` affichait déjà
  pour un `user_id` absent).

**Ce qui reste** : les mentions légales et la politique de confidentialité. On
stocke un pseudo, un bcrypt, une adresse e-mail et l'historique de jeu de
personnes réelles ; la suppression existe désormais, la page qui l'explique non.
C'est de la rédaction, pas du code.

S'y raccroche depuis le §2.7 : **un joueur anonyme garde la redonne gratuite**,
et il n'y a rien à corriger côté reprise — sans compte, aucune identité à
laquelle rattacher une donne (`pending_deal` exige un `user_id`). Sans effet sur
l'Elo, qui ne note que des joueurs identifiés, mais c'est bien le compte qui
ferme ce trou-là, pas la reprise.

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

**L'étape (2) est faite pour le repli de mondes** (2026-08-02), après que la
prod a tourné plus d'un jour sans `COLVER_PLAYGEN_GPU_URL` — Dédé jouait sur des
mondes uniformes, plus faiblement qu'annoncé, sans que rien ne le dise. Trois
silences, tous fermés :

- **au démarrage** : `agents.log_startup_state()` nomme la source de mondes à
  chaque boot (WARNING sans sidecar, ERROR s'il est pourtant exigé) ;
- **dans `/health`** : `sidecar_configured` valait `bool(os.environ[…])`, donc
  aurait dit `true` devant un sidecar mort. `playgen_gpu.probe()` interroge
  vraiment son `/health` (1,5 s max, en cache 30 s, dans un thread) et rend
  `reachable` **et pourquoi pas** ;
- **par décision** : `worlds_source` partait au client et jamais au journal.
  `AgentTable.decide` le journalise, plafonné à une ligne par minute avec le
  compte courant (~24 coups de bot par donne : sans plafond, une panne noierait
  le journal au lieu de le renseigner).

Ce qui rend le cas initial *détectable* est `COLVER_REQUIRE_SIDECAR` : sans lui,
« pas de playgen » est indiscernable d'un choix légitime (une machine de dev n'a
pas de GPU). Avec, le déploiement déclare son attente et `/health` passe en
`degraded` quand la réalité diverge — le code HTTP reste 200, un sidecar absent
affaiblit le jeu mais n'empêche pas de jouer. Généraliser ce principe est
probablement la bonne forme pour le reste du §2.2 : **ce qu'on attend se
déclare, et la sonde compare.**

Reste de l'étape (2) : `elapsed` et `determinizations` par coup, qui n'ont de
sens qu'une fois le budget inversé (ils mesurent alors ce que le plafond a
coûté).

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
**tue le socket**, donc interrompt la donne en cours — moins grave depuis le
§2.7, qui la rend reprenable, mais le joueur se prend quand même la coupure
sans explication. Un modèle Pydantic par type de message supprime toute une
classe de plantages et documente le protocole au passage — c'est le même
travail.

Le message `play` est **déjà** validé depuis le §4.5, mais pour une autre raison
(la légalité du coup, pas le type du champ) et à la main. C'est le modèle de ce
qu'il faut généraliser : refuser, le dire au journal, renvoyer la position, et
surtout ne pas tomber. Les deux tests `TestCoupIllegal` de `test_ws_play.py`
couvrent déjà « une action absurde ne tue pas la socket » pour ce message-là.

### 2.5 Premiers tests Python et CI qui teste — FAIT (2026-08-01)

Le noyau Rust était bien testé ; la couche web (6 600 lignes, et la logique la
plus retorse du projet : reprise de partie, pilote de salon, course du dernier
pli dans `_wait_human_card`) n'avait **aucun test**, et la CI se limitait à
`publish.yml`.

**Ce qui est en place** : `tests/` (142 tests, ~15 s), configuration pytest +
ruff dans `pyproject.toml`, groupe de dépendances `test` séparé de `dev` (la CI
n'installe ni torch ni scikit-learn), et `.github/workflows/ci.yml` — deux
jobs, `cargo check` + `cargo test -p colver-core` d'un côté, ruff + pytest de
l'autre, sur push et PR.

Les fichiers, dans l'ordre du risque : `test_match_state` (record idempotent,
`finished` à égalité, `restore` qui ne resomme pas les donnes),
`test_play_session` (fidélité du rejeu), `test_resume` (`pending_deal` /
`drop_deal`, isolation entre comptes, donne de partie contre donne isolée),
`test_integrity` (§4.5), `test_account` (§2.1), `test_pacing`, `test_ws_play`.

Ce que l'écriture a appris, et qui vaut pour tout test de cette couche :

- **`TestClient` suffit à piloter le protocole WebSocket de bout en bout**, sans
  serveur à lancer — c'était l'outil qui manquait.
- **Ne jamais lire « un message de plus »** : le donneur est tiré au sort, donc
  le premier `game_state` peut déjà être le tour du joueur, et un harnais qui
  compte les messages se bloque une fois sur quatre sur une donne où le serveur,
  lui, attend. On lit jusqu'à une *condition* (`Table.until`).
- **`game_id` n'accompagne que le premier `game_state`** d'une donne ; le relire
  sur le dernier message reçu donne un `KeyError` dès que la main est passée.
- **Neutraliser `pacing` dans les tests de protocole** (fixture `no_tempo`) : une
  donne dure 16 à 42 s de pauses d'affichage, qui ont leurs propres tests
  unitaires. Avec, `test_ws_play` tombe de plusieurs minutes à 2 s.
- **Les objets de module fuient d'un test à l'autre.** `auth._AUTH_LIMITER` est
  partagé par le processus et tous les tests sortent de la même « IP » : sans
  remise à zéro (`RateLimiter.reset`, fixture autouse), le premier fichier qui
  teste un échec d'authentification laisse le budget vide pour les suivants, et
  leurs échecs n'ont plus rien à voir avec ce qu'ils prétendent tester.
- **Fermer la connexion aiosqlite** en fin de fixture : oubliée, elle laisse un
  thread vivant et le processus de test ne rend jamais la main.
- Les modèles ne se téléchargent pas (`conftest` neutralise les `download_*`
  avant tout import) : la suite tourne sans poids, bots repliés sur les règles
  du moteur. Aucun test ne doit dépendre d'un modèle.

**Pas encore fait** : mypy, et les warnings-as-errors — `server.py` utilise
encore `@app.on_event`, déprécié par FastAPI, et la règle échouerait dès
l'import sans rien apprendre de plus que ce que le §2.6 note déjà.

### 2.6 Déployer en fin de donne, pas au milieu (drain)

Aujourd'hui un déploiement, c'est `docker compose up -d --build` : le conteneur
est tué, toutes les WebSocket tombent d'un coup. Le coût n'est pas symétrique
entre les deux échelles de jeu :

- La **partie** survit — le score cumulé est en base (`matches.points_ns/ew`) et
  un joueur connecté la reprend via `_resume_match`.
- La **donne en cours** survit aussi depuis le §2.7 : elle se rejoue depuis
  `games.actions`, et un joueur connecté la retrouve à son coup près. Ce qu'il
  perd, c'est le fil — la coupure arrive sans prévenir et il faut revenir.
- Le **joueur anonyme**, lui, perd tout : rien en base ne le rattache ni à la
  donne ni à la partie.

Donc : attendre la fin des donnes actives avant de couper. Ça reste utile — une
coupure au milieu d'un pli est brutale même quand elle est réparable, et le
joueur anonyme n'a rien à reprendre — mais ce n'est plus une perte sèche, donc
ça passe derrière §2.7 dans l'ordre d'urgence. Fin de **donne** suffit, pas fin
de partie — c'est la seule granularité qui soit à la fois nécessaire et bornée
(~42 s en standard, ~16 s en rapide ; une partie en 2000 points, elle, n'a pas
de durée maximale).

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
  (`playgen-gpu.service` sur l'hôte GPU). Redéployer le sidecar coupe IS-DD en pleine
  recherche — le drain du web ne protège rien si le sidecar tombe pendant.
- Les tâches d'analyse (`agent_review` qui streame, `annonces_sim`,
  `card_analysis`) sont à **annuler**, pas à drainer : elles sont recalculables
  et déjà annulables.

### 2.7 Recharger la page efface une donne mal engagée — FAIT (2026-08-01), sauf le garde-fou

Signalé par un joueur : en pleine donne, quitter la page (retour navigateur,
rechargement) puis revenir suffisait à faire disparaître la donne en cours et à
en obtenir une neuve. C'était même le comportement *documenté* de la reprise :
« une donne abandonnée n'a pas eu lieu ». Or elle ne coûtait rien nulle part —
la ligne `games` reste `is_complete = 0`, donc ni les points de la partie
(`Match.record` jamais atteint), ni l'Elo (`rate_game` exige `is_complete = 1`)
ne la voyaient. Contrat mal engagé → F5 → redonne gratuite, répétable.

**Ce qui est fait : la donne en cours se reprend à son coup près**, en partie
comme hors partie (le cas par défaut, `target = 0`, qui n'avait aucune reprise
du tout et était donc le trou le plus large). `db.pending_deal` →
`PlaySession(hands=…)` + `replay(actions)` → `server._resume_deal`, plus les
entrées côté Jouer (`play_open.deal`, messages `resume_deal` / `drop_deal`).

L'hypothèse à retenir, parce qu'elle était fausse dans les deux docs :
« rejouer les actions ne rendrait pas leur état aux bots ». Les bots n'ont pas
d'état à restaurer, ils en ont un qui **se déduit** du préfixe visible — le
rejeu passe par `_record_action`, donc par `AgentTable.observe`, exactement
comme le jeu. Vérifié bit à bit (position, enchères, plis, mains d'origine) à
toutes les coupures, DouDou50 comme Dédé.

Ce qui reste est passé en **§2.8** (le garde-fou), §2.1 (le cas anonyme) et
§2.5 (les tests). Bénéfice collatéral à encaisser quand §2.6 et §2.4 se feront :
un déploiement en pleine donne et un socket tué par un message malformé ne
coûtent plus la donne, il suffit de revenir.

### 2.8 Renoncer à une donne devrait coûter quelque chose (1 j, équité)

Reste du §2.7, et la moitié qui demande une décision plutôt que du code.
Reprendre une donne interrompue est désormais le chemin par défaut, mais y
**renoncer** est encore gratuit, par deux portes : le bouton « Abandonner » de
la ligne de reprise (`drop_deal`) et le simple fait de démarrer une donne neuve
(`start_game` efface celle en plan, sinon elle serait reproposée sans fin et
chaque abandon en empilerait une de plus).

L'exploit n'est donc plus ni silencieux ni accidentel — il demande deux clics
et s'annonce — mais il n'est toujours pas payant : contrat mal engagé →
Abandonner → donne neuve, au score acquis.

À trancher, et c'est un choix de règle avant d'être du code :

- **Marquer la donne chutée pour le camp qui abandonne** — l'adversaire marque
  comme s'il avait gagné la donne. Simple, symétrique des règles, et
  `Match.record` prend déjà des rewards ; c'est la piste recommandée.
- **Ne compter que les points cartes déjà pris** par l'adversaire — plus doux,
  mais suppose une règle de marque qui n'existe pas dans le moteur, et récompense
  d'abandonner tôt.

Deux détails que la décision devra couvrir : une donne abandonnée **avant tout
contrat** (enchère non conclue) ne met rien en jeu et devrait rester gratuite ;
et le résultat doit-il compter pour l'Elo, ou seulement pour le score de la
partie ? (Aujourd'hui `rate_game` exige `is_complete = 1`, donc une donne
abandonnée n'est jamais notée.)

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

### 4.2 Le « vrai monde » sur la page annonces — FAIT (2026-08-02) par le chemin Rejouer

`/analyse/jeu` sépare deux questions que la page annonces confondait en n'en
posant qu'une : *le vrai monde* (un solve sur la donne réelle, information
parfaite) et *les mondes de l'information set* (échantillonnés depuis ce que le
siège pouvait savoir). Le raisonnement est dans
[web_analyse_jeu.md](web_analyse_jeu.md) §2 et l'implémentation dans
`card_analysis.true_world()` — une fonction à part, une colonne à part,
**jamais fusionnée** avec les mondes échantillonnés.

L'annonce méritait le même traitement. La page répondait « ton 130♥ passe dans
68 % des mondes que tu pouvais imaginer » sans pouvoir ajouter « …et cette
donne-là était dans les 32 % ». C'est pourtant la phrase qui referme la boucle
avec Rejouer : elle distingue une bonne annonce punie par la donne d'une
mauvaise annonce sauvée par elle.

**Ce qui est fait.** `analysis.true_world(game_id, action_idx)` rend les points
DD par couleur, du côté de l'équipe du siège qui parle (la page l'assied
toujours en Sud, donc dans son repère c'est Nord-Sud), plus sa main pour que le
client sache à quelle donne la ligne appartient. Le solve est déjà en cache dès
que Rejouer a analysé la donne (`analysis.oracle_bids`) ; sinon il coûte quatre
solves (~300 ms) et **n'est pas mis en cache** — une ligne `analysis` partielle
serait relue comme une analyse complète. Côté client, message WS
`annonces_true_world` envoyé à l'arrivée quand l'URL porte `from`/`i` (rejoué
sur `onOpen` : `send()` jette en silence tant que le socket n'est pas ouvert),
et une colonne « Vraie donne » dans le tableau du Jeu parfait, points + palier
tenu.

**Ça n'a rien enlevé** : le bandeau couleur × palier, la synthèse Oracle et le
Jeu réel sont inchangés. **Et ça n'entre pas dans le tirage des mondes** :
conditionner le pool sur la donne réelle transformerait « cette annonce
était-elle bonne ? » en « a-t-elle marché ? » — la seule des deux questions qui
n'apprend rien. Les pourcentages et les teintes de confiance de la page
deviendraient du rétroviseur sans le dire.

Conditions d'affichage : **il faut connaître les quatre mains**. Deux entrées
possibles, une seule faite —

- on arrive depuis Rejouer (`from=<gameId>` + `i`) : **fait** ;
- on colle un CFN complet 4 sections, comme `/analyse/jeu` (la page annonces ne
  prend que `?hand=` + `?history=`, il faudrait donc lui ajouter cette entrée).
  **Reste à faire** — c'est la seule part non couverte.

Main saisie à la main = pas de vraie donne, pas de ligne. C'est le cas nominal
de la page, la ligne est donc **conditionnelle par construction**, comme le lien
« ← Retour à la partie ».

Pièges, et ce qu'ils ont donné :

- **n = 1, et la page encode la taille d'échantillon dans son rendu** (Wilson,
  classes `doudou-high/mid/low`, opacité croissante avec le nombre
  d'observations). Rien de tout ça ne touche cette colonne : elle se distingue
  par des filets verticaux. Une valeur exacte unique ne doit pas emprunter le
  langage visuel d'un pourcentage échantillonné, sinon elle se lit comme la
  cellule la plus sûre du tableau.
- **Elle appartient au Jeu parfait, donc à l'état partagé entre onglets**, pas à
  l'onglet courant : comme l'Oracle, elle ne dépend pas de l'annonce analysée.
- **Échelle** : points cartes DD, dans la box Oracle, sur les mêmes axes couleur
  × palier — pas de troisième présentation à côté des deux échelles existantes
  (Oracle en points cartes vs Jeu réel en points de donne marqués).
- **Clé de déduplication des mains enregistrées** : elle porte sur (main,
  enchères précédentes), donc la même main venue d'une partie et tapée à la main
  s'y confondraient, l'une portant une vraie donne et l'autre non. Tranché comme
  prévu : on **perd la vraie donne au rechargement** d'une main enregistrée
  plutôt que d'élargir la clé. Le rendu vérifie en plus que la main à l'écran est
  toujours celle du siège analysé — sinon la ligne décrirait une autre donne.
- **Le cache d'une autre `ANALYSIS_VERSION` n'est pas relu** : un barème ou un
  coup légal qui change périme les valeurs DD (cf. la règle du 2026-08-01).

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

### 4.5 Donnes enregistrées dont les `actions` ne collent pas aux `hands` — FAIT (2026-08-01), cause comprise

Des donnes terminées décrivent une partie impossible : en les rejouant,
certaines cartes sortent deux fois et d'autres jamais.

`env.step()` **ne valide pas la légalité** — c'est le comportement attendu d'un
moteur RL, où le contrôle est à la charge de l'appelant — donc le moteur avale
une carte absente de la main : elle s'ajoute au pli sans être retirée nulle
part. La donne se rejoue jusqu'au bout, sans erreur, avec un décompte faux.

**Deux causes, pas une** — mesuré sur la prod le 2026-08-02, 6 donnes écartées
sur 822 terminées (0,7 %) :

| donne | mode | date | diagnostic |
|---|---|---|---|
| `jmp2` `41ec` `9ru7` `pjaf` `qqy9` | play | 02-19 → 08-01 | **carte déjà jouée**, rejouée 4 à 6 coups plus tard |
| `ao9e` | watch | 02-18 | **carte jamais jouée** — coup illégal d'un bot |

1. **Le gestionnaire `play` du solo** (5 cas sur 6). Il prenait `data["action"]`
   sur la socket et ne vérifiait que deux choses — donne non terminale, et c'est
   bien le tour de ce siège — jamais que le coup existait. Or dans les cinq cas
   la carte avait *déjà été jouée* par ce même siège un ou deux plis plus tôt, et
   au moment du renvoi c'était bel et bien son tour : le seul garde-fou en place
   ne pouvait rien voir. Un client qui renvoie une carte déjà posée (double-clic,
   message rejoué) suffisait donc à écrire une donne fausse. Le salon validait
   déjà (`rooms._await_human_action`) ; le solo, non.
2. **Un bot qui rend un coup illégal** (`ao9e`, `mode = 'watch'`, quatre sièges
   tenus par DouDou50 — aucun humain à la table). Le siège 2 avait JS, 10S, AS à
   l'atout pique, 9S entamé et JH déjà coupé : seul JS est jouable (obligation de
   surcouper), le bot a posé 10S. C'est pour ça que `WatchSession.apply_action`
   est gardé lui aussi. Ça date de février et n'est pas réapparu depuis.

Signature commune, et ce qui rendait le diagnostic trompeur : **les six sont au
siège 2**. Pour les cinq donnes solo c'est le siège humain par défaut, donc
c'est bien la marque du chemin fautif ; pour `ao9e` c'est une coïncidence.

**Depuis le déploiement du garde-fou** (2026-08-01 21:56 UTC) : 17 donnes
terminées, **0 écartée**. Échantillon mince, mais le chemin d'écriture rend
désormais le cas structurellement impossible.

**Le correctif, en deux moitiés du même prédicat** :

- **En écriture** : `game_manager.check_legal`, appelé depuis
  `PlaySession._record_action` — le chemin unique par lequel une action entre
  dans une session. Il couvre donc le clic humain, le coup de bot et le rejeu
  d'une donne reprise, en solo comme en salon, plus `WatchSession.apply_action`.
  `replay` (§2.7) n'a plus son test à lui : il n'ajoute que le rang du coup
  fautif. Le gestionnaire `play` refuse et renvoie la position (`_play_state_msg`)
  au lieu de laisser tomber la socket.
- **En lecture** : `integrity.check_deal` / `integrity.scan`, migration v10
  (`games.invalid`, `checked_at`, `invalid_reason`). Le scan tourne au démarrage
  **avant** le backfill Elo (une donne irrejouable ne doit pas être notée ;
  en parallèle la course se jouerait à la milliseconde). Chaque ligne n'est
  examinée qu'une fois (`checked_at`), donc le coût est borné par les donnes
  terminées depuis le dernier lancement. `/health` publie `invalid_deals`.
  - **Conséquence à connaître : `invalid_deals` est en retard d'un
    redémarrage.** Une donne jouée après le démarrage n'est scannée qu'au
    lancement suivant (17 en attente au moment de la mesure). C'est assumé —
    le garde-fou en écriture rend le cas impossible en amont, le scan n'est
    qu'un filet — mais ça veut dire qu'on ne peut pas lire ce compteur comme
    une surveillance temps réel.

**Le prédicat, et pourquoi il suffit** : une donne est saine si, partant de
`hands` et `dealer`, chaque action est légale à son tour et la dernière rend la
donne terminale. Un journal entièrement légal ne peut ni jouer deux fois la même
carte ni en oublier une — les 152 points et le dix de der tombent juste par
construction. C'est **plus strict** que l'assertion de `CountingSession._payload`
(somme des plis = 152) qui avait donné l'alerte : celle-ci rendait 0 anomalie sur
les mêmes 19 donnes où la légalité en trouve 2. Le symptôme était les points ; la
cause est la légalité, et elle nomme le coup fautif.

**On marque, on n'efface pas** : une donne fausse est un incident, et l'effacer
effacerait la trace avec elle. `invalid = 1` la retire de `get_game`,
`list_games`, `random_user_game`, des statistiques et de l'Elo, et purge ses
analyses en cache (calculées depuis un état impossible — même raison qu'à la
migration v9). `db.list_invalid_games` les rend à l'exploitant.

**Reste, mineur** : les donnes fausses notées *avant* ce correctif gardent leur
ligne d'`elo_history`. L'Elo est séquentiel, le défaire demanderait un recalcul
complet pour un effet de deux donnes.

**Reproduire** (le scan le fait maintenant tout seul au démarrage) :

```bash
uv run python -c "
import asyncio, sys; sys.path.insert(0, 'python')
from colver.web import database as db, integrity
async def m():
    await db.get_db()
    print(await integrity.scan())
asyncio.run(m())
"
```
