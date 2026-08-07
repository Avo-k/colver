# Colver — documentation

Moteur de Belote Contrée : les règles, la recherche, les agents, et de quoi vérifier
qu'ils font ce qu'ils prétendent.

## Le moteur

- [ARCHITECTURE.md](ARCHITECTURE.md) — organisation du workspace, sous-systèmes, formats
  d'observation. C'est la porte d'entrée si vous lisez le code.
- [RULES.md](RULES.md) — les règles **telles qu'implémentées ici**, y compris les endroits où
  le moteur s'écarte de la FFB, et pourquoi.
- [agents.md](agents.md) — la couche `Player` / `WorldSource` : comment un bot se construit et
  se pilote, et le format de spec TOML. Un bot est une spec, pas un chemin de code.
- [BENCH.md](BENCH.md) — les mesures de performance, avec la commande qui les reproduit.

## Ce que fait le reste du monde

- [rules-survey/](rules-survey/) — ~594 règlements publiés (fédérations, tournois, clubs,
  applications, logiciels libres) comparés axe par axe. Commencer par
  [SYNTHESE.md](rules-survey/SYNTHESE.md), et
  [matrices/](rules-survey/matrices/) pour le détail par question.
  La méthode est dans [METHODE.md](rules-survey/METHODE.md).
- Les textes tiers **ne sont pas redistribués ici** : le corpus brut vit hors dépôt, et
  [`rules-survey/_refetch.py`](rules-survey/_refetch.py) le reconstitue depuis ses sources.
- [deal_bias.md](deal_bias.md) — la distribution traditionnelle (ramasser, couper, donner par
  3-2-3) contre un mélange de compétition : ce que le biais vaut réellement, mesuré.

## Comment on décide ici

- [measurements/README.md](measurements/README.md) — toute mesure se journalise, avec la
  provenance et l'empreinte des poids consultés. Une mesure sans son modèle n'est pas
  interprétable six mois plus tard.
- [bid/interpretability/hand_classification.md](bid/interpretability/hand_classification.md) —
  l'espace des mains est **énumérable** : 472 579 mains distinctes à 8 cartes, indexées
  bijectivement. Donc la politique d'ouverture d'un enchérisseur *est* une table finie, pas
  une boîte noire — et ce qu'un code de main doit encoder se mesure au lieu de s'opiner.

## Les modèles

- [hf/](hf/) — les fiches des poids publiés, une par modèle
  ([DouDou50](hf/doudou50/README.md), [Bid v6](hf/bid-v6/README.md),
  [Belief v4](hf/belief-v4/README.md), [Playgen v2](hf/playgen-v2/README.md)).
  Les poids eux-mêmes vivent sur
  [Hugging Face](https://huggingface.co/collections/Avo-k/colver-belote-contree-6a71df4a723e6734fe623a65) ;
  `colver.download_*()` va les y chercher.

---

**Ce qui n'est pas ici.** Les notes de recherche — carnets d'entraînement, impasses mesurées,
études de récompense, plans non lancés — ne sont pas publiées. Elles décrivent un travail en
cours plutôt qu'un artefact fini, et leur valeur est dans l'incrément, pas dans la
photographie. Ce qui est publié est ce qui permet de **lire le moteur et de le vérifier** :
le code, les règles, l'enquête sur les règlements, et la méthode de mesure.
