# Tableau des scores — FFB, règles officielles de la belote contrée (27/01/2016), §10.2

Le tableau de la page 7 du PDF est une **image** : il ne sort pas de l'extraction texte, et c'est
pour ça qu'il manque dans `federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt`.
Cette transcription est donc la seule forme exploitable de ce tableau dans le corpus — d'où sa
présence ici, dans les fichiers versionnés, et non dans le corpus brut.

Rendu de la page à côté : `tableau-ffb-2016-p7.png` (non versionné, régénérable avec
`pymupdf` depuis le PDF source).

Deux méthodes de comptage, **au choix de l'organisateur du tournoi** : « points faits » ou
« points faits + points demandés ».

## Contrat réussi

| Cas | Camp | Points faits | Points faits + demandés |
|---|---|---|---|
| Contrat en x points, standard | Preneurs | Points réalisés + Belote preneurs | Points réalisés + **Contrat** + Belote preneurs |
| | Défense | Points réalisés + Belote défense | Points réalisés + Belote défense |
| Contrat en x points, contré (surcontré) | Preneurs | **320 (640)** + Belote preneurs ou défense | **320 (640) + Contrat ×2 (×4)** + Belote preneurs ou défense |
| | Défense | 0 | 0 |
| Capot, standard | Preneurs | **500** + Belote preneurs | **500** + Belote preneurs |
| | Défense | 0 + Belote défense | 0 + Belote défense |
| Capot, contré (surcontré) | Preneurs | **1000 (2000)** + Belote preneurs ou défense | **1000 (2000)** + Belote preneurs ou défense |
| | Défense | 0 | 0 |

## Contrat chuté

| Cas | Camp | Points faits | Points faits + demandés |
|---|---|---|---|
| Contrat en x points, standard | Preneurs | 0 | 0 |
| | Défense | **160** + Belote preneurs ou défense | **160 + Contrat** + Belote preneurs ou défense |
| Contrat en x points, contré (surcontré) | Preneurs | 0 | 0 |
| | Défense | **320 (640)** + Belote preneurs ou défense | **320 (640) + Contrat ×2 (×4)** + Belote preneurs ou défense |
| Capot, standard | Preneurs | 0 | 0 |
| | Défense | **500** + Belote preneurs ou défense | **500** + Belote preneurs ou défense |
| Capot, contré (surcontré) | Preneurs | 0 | 0 |
| | Défense | **1000 (2000)** + Belote preneurs ou défense | **1000 (2000)** + Belote preneurs ou défense |

## Les deux notes du bas de page (elles, extraites en texte)

> **Note 1** : Lors de tournois organisés en réel, afin que l'issue de la partie ne soit pas définie
> par une simple donne, en mode points faits + points demandés, les organisateurs peuvent utiliser
> la comptabilisation suivante en cas de contre (surcontre) :
> **Contrat × 2 + 160** au lieu de Contrat × 2 + 320  (**Contrat × 4 + 160** au lieu de Contrat × 4 + 640)

> **Note 2** : En points faits, si l'équipe contrée réalise un capot, elle marque 320 points
> auxquels s'ajoutent les 100 points du dix de der, soit 420 points.

## Points saillants

1. **Le surcontre est ×4**, et le forfait passe de 320 à 640. La rédaction FFB plus récente
   (`LOCAL_regles_officielles_belote_contree.txt`) dit **×3**. Même fédération, même jeu.
2. **Le capot est un forfait** (500 / 1000 / 2000), pas un contrat à 250 comme dans la rédaction
   récente. Ces deux modèles ne coïncident nulle part.
3. **La base de la chute standard est 160**, pas 162 — alors que le règlement de belote classique
   de la *même* fédération, du *même* jour, écrit « 162 points ou 252 points s'il elle a réalisé un
   capot » (§10.2 de `ffbelote_regles-officielles-de-la-Belote-27-01-2016.txt`).
4. La **Note 1** est le seul endroit d'un texte fédéral qui propose un contré en
   `base + contrat × mult` avec une base de 160 — c'est la forme dont Colver est le plus proche
   (à ceci près que Colver utilise 162 et un surcontre ×3).
5. Le tableau ne dit **jamais** que le multiplicateur porte sur autre chose que le contrat : la
   base (320/640) est un forfait qui *remplace* les points faits, elle n'est pas multipliée.
