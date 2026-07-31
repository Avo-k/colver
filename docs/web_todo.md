# Backlog web (non implémenté)

Idées notées, rien d'engagé. (Dernière mise à jour : 2026-08-01)

## 1. Déployer en fin de donne, pas au milieu (drain)

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
   que dans les locales de `websocket_endpoint`, il faudrait un registre.
3. Plus rien d'actif (ou timeout) → sortir, `restart: unless-stopped` relance.

Pièges repérés :

- **Il n'y a pas de hook d'arrêt** : `server.py` n'a qu'un `@app.on_event("startup")`.
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

## 2. Le score de partie dans les pages d'analyse

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
