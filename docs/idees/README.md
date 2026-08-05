# Idées

Des pistes **cadrées mais pas lancées** : on sait ce qu'on ferait, on sait à peu
près ce que ça coûte, et on ne le fait pas maintenant. Chaque fiche existe pour
qu'on n'ait pas à refaire le raisonnement dans six mois.

## Ce qui va ici, et ce qui n'y va pas

| | où |
|---|---|
| « il faut faire X », c'est décidé, il reste à l'écrire | [engine_todo.md](../engine_todo.md) / [web_todo.md](../web_todo.md) |
| « X serait peut-être une bonne idée », et voilà pourquoi, voilà le prix, voilà la mesure qui trancherait | **ici** |
| « on a essayé X, ça ne marche pas » | à côté du sous-système concerné, en résultat négatif |

La différence avec un backlog est qu'une fiche d'ici a le droit de ne jamais
être faite. Ce qu'elle doit contenir, en revanche :

1. **l'état des réflexions**, avec les chiffres déjà mesurés et leur source ;
2. **ce qui bloque** — un format, une dépendance, une machine occupée ;
3. **les prochaines étapes**, chacune avec son coût, en séparant ce qui demande
   du GPU de ce qui n'en demande pas ;
4. **ce qu'on croit sans l'avoir mesuré**, dit comme tel.

Le point 4 est le seul qui justifie vraiment ces pages. Une idée qu'on laisse
dormir se réveille toujours avec ses hypothèses converties en certitudes.

## Fiches

- [corpus_isdd_playgen_v3.md](corpus_isdd_playgen_v3.md) — **entraîner playgen
  sur du jeu fort** : combien de temps coûte un corpus IS-DD, pourquoi le mode
  partie est bloqué par le format, pourquoi ce corpus n'entraîne pas un bidder,
  et pourquoi playgen v2-belote-small n'est pas le raccourci qu'il paraît être.
  (2026-08-04)
- [decodage_speculatif_playgen.md](decodage_speculatif_playgen.md) — **un mini
  modèle comme brouillon du gros** : pourquoi ça vise le mauvais poste de coût
  (le modèle pèse moins de 5 % d'un pas de décodage), et ce que le spéculatif
  rendrait quand même (~1,5× au plafond, contre 1,62× rendus par la fusion
  ACT+CARD sans second modèle). Le levier qu'elle pointait est **fait et
  mesuré** ; ce qui reste ouvert tient en une mesure de dix minutes.
  (2026-08-04, révisée le même jour)
- [rejouer_analyse_erreurs.md](rejouer_analyse_erreurs.md) — **compter les
  erreurs dans Rejouer, montrer l'alternative, croiser DD et IS-DD** : pourquoi
  l'échelle en points cartes désigne les mauvaises erreurs (32 coups sur 1057
  affichés « ✓ » coûtent au score, jusqu'à 1264 points), pourquoi les deux
  évaluations sont déjà calculées et jetées, pourquoi dérouler une variante est
  gratuit (0,12 s pour toute une donne) et pourquoi DouDou50 n'est pas le
  raccourci qu'il paraît (2 à 30× plus lent que le solveur). **§3.1 ter** :
  la grille livrée était fausse sur ses deux colonnes — un `cost_score` nul
  veut dire « l'Oracle n'a pas d'avis » 59,7 % du temps (88,3 % sous contré),
  et le seuil de Dédé, absolu, valait moins qu'un seul monde. (2026-08-05,
  corrigée le 2026-08-06)
