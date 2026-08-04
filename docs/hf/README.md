# Publication Hugging Face

Les cartes de modèle vivent ici et sont **la source de vérité** : le `README.md` d'un
dépôt HF en est une copie. Les garder dans le dépôt de code, c'est ce qui permet qu'une
modification d'architecture et la fiche qui la décrit passent dans le même commit.

## Ce qui est prêt à publier

| Fiche | Dépôt HF visé | Fichier de poids | sha256 (12) | Taille |
|---|---|---|---|---|
| [bid-v6/](bid-v6/README.md) | `Avo-k/colver-bid-v6` | `models/bid_v6_isdd_resume/bid_nn_final.bin` | `9443671cab1e` | 2,4 Mo |
| [doudou50/](doudou50/README.md) | `Avo-k/colver-doudou50` | `models/dmc_50.bin` | `f9fb4c4bc9ea` | 10,2 Mo |
| [playgen-v2/](playgen-v2/README.md) | `Avo-k/colver-playgen-v2` | `models/playgen/playgen_v2_final.bin` | `3cb43a8cae84` | 43,0 Mo |
| [belief-v4/](belief-v4/README.md) | `Avo-k/colver-belief-v4` | `models/belief_v4_fix_v2.bin` | `6d141252ea8b` | 1,9 Mo |

Les quatre sha256 ont été **appariés aux assets GitHub Releases réellement servis en
production** (`v0.4.0/dmc_50.bin`, `v0.7.0/bid_v6_isdd.bin`, `v0.7.0/belief_v4_fix_v2.bin`,
`v0.8.0/playgen_v2_final.bin`). Ce sont donc bien les poids que fait tourner colver.net,
et non un checkpoint voisin.

> ⚠️ **Le piège vérifié une fois, à revérifier à chaque publication.** Les poids publiés de
> playgen v2 sont `models/playgen/playgen_v2_final.bin`. Les fichiers de
> `models/playgen_v2/` (`playgen_v2_60k.bin`, `playgen_v2_half.bin`, identiques entre eux)
> sont un **checkpoint intermédiaire jamais publié**. Les noms ne les distinguent pas —
> seul le sha256 le fait. C'est la même confusion qui avait laissé la prod tourner un jour
> entier sur un checkpoint intermédiaire.

## Conventions de fiche

Trois choses qu'une fiche Colver doit porter, et qui ne vont pas de soi :

1. **Une ligne de renvoi vers les règles, sous l'intro** — il n'existe pas de règlement
   unique de la belote contrée, donc un modèle n'est interprétable que si l'on sait sous
   lequel il a appris. Une ligne suffit, le lecteur suivra le lien s'il veut le détail.
2. **Un exemple exécutable dès la première section**, chargé depuis HF, dont la sortie
   affichée est celle réellement obtenue. Toute valeur numérique doit avoir été mesurée —
   pas arrondie de mémoire, pas complétée à l'estimation.
3. **Une section « ce qu'il ne sait pas faire »** avec les défauts mesurés, sans les
   adoucir.

## Ordre de publication

Les modèles d'abord, les données ensuite. Les quatre modèles sont **déjà publics** via
GitHub Releases : les publier sur HF n'expose rien de nouveau, le travail est entièrement
dans les cartes. C'est donc le lot à faible risque par lequel commencer, et celui qui
permet de roder la procédure avant d'y mettre un corpus de 523 Mo.

### ⚠️ Rendre les dépôts publics **avant** de tagger une release

Depuis la 0.10, `colver/_model.py` télécharge les poids depuis le Hub. Tant que les
dépôts sont privés, un utilisateur sans token reçoit un `401` — le repli GitHub Releases
le rattrape, mais silencieusement et sur des URL figées à d'anciens tags. Publier une
version dont le chemin nominal échoue pour tout le monde n'a pas de sens.

L'ordre est donc : **(1)** passer les 4 dépôts en public, **(2)** vérifier qu'un
téléchargement anonyme aboutit, **(3)** commiter le bump, **(4)** tagger `v0.10.0`.

Le repli, lui, reste là pour de bon : il couvre le Hub injoignable, pas l'oubli de
publication.

## Procédure

Prérequis : un token `write` (`hf auth login`). Pour chaque modèle :

```bash
hf repo create colver-bid-v6 --type model
hf upload colver-bid-v6 models/bid_v6_isdd_resume/bid_nn_final.bin bid_v6_isdd.bin
hf upload colver-bid-v6 docs/hf/bid-v6/README.md README.md
```

Après upload, **revérifier l'empreinte côté HF** — c'est la seule preuve que le bon
fichier est parti :

```bash
hf download Avo-k/colver-bid-v6 bid_v6_isdd.bin --local-dir /tmp/verif
sha256sum /tmp/verif/bid_v6_isdd.bin   # doit valoir 9443671cab1e35bb…
```

## Ne jamais supprimer les assets GitHub Releases

`python/colver/_model.py` référence des URL GitHub **en dur**, et ces URL sont gravées
dans les wheels PyPI déjà publiées (0.4.0 → 0.9.1). Effacer un asset de release casserait
le démarrage de colver chez tous ceux qui les ont installées. HF s'ajoute à côté ; seules
les versions futures doivent pointer vers HF.

## Compte perso maintenant, organisation plus tard

Publié sous `Avo-k/` avec le préfixe `colver-`. Le passage vers une organisation reste
ouvert et documenté : `move_repo(from_id="Avo-k/colver-bid-v6", to_id="colver/bid-v6")`
change le namespace **et** le nom en une opération, redirige l'ancienne URL, et conserve
compteurs de téléchargement et likes. Le seul transfert que HF interdit est d'utilisateur
à utilisateur — pas celui-là.
