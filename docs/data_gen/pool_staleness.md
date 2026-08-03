# Un pool DD périmé : ce que ça coûte vraiment (2026-08-02)

**Verdict : non, il ne faut pas regénérer `base_5M.bin` pour cause de dérive d'IS-DD.**
Au moins 87 % de l'écart avril → aujourd'hui est du bruit d'échantillonnage. Et une
deuxième raison, plus forte que la première : le format ne peut pas porter le label
qu'on voudrait de toute façon.

## La question

`scores_isdd_5M.sc` date du 2026-04-23. Trois choses ont changé depuis :

| date | changement |
|---|---|
| 2026-07-23 | retrait de `quick_tricks` — le solveur rendait des valeurs fausses sur 25 % des appels |
| 2026-07-24 | IS-DD échantillonne des mondes **playgen** au lieu de mondes uniformes |
| 2026-08-01 | correctif de la règle de surcoupe — élargit l'ensemble des coups légaux |

Comparer directement aujourd'hui contre avril mesure les trois d'un coup et n'en
attribue aucun. D'où quatre bras séparés, tous sur les **1000 premières donnes de
`base_5M.bin`**, 4 couleurs par donne, mode compte (20 déterminisations) — pas le mode
temps, pour que le label ne dépende pas de la charge machine.

| bras | mondes | enchère | graine | ce qu'il isole |
|---|---|---|---|---|
| **B0** | uniform | aucune | 42 | code d'aujourd'hui, mondes uniformes comme en avril |
| **plancher** | uniform | aucune | 1234 | B0 contre lui-même → bruit intrinsèque |
| **B** | uniform | synthétique | 42 | prix du préfixe d'enchère |
| **C** | playgen | synthétique | 42 | l'effet playgen |

⚠️ **B0 n'est pas le protocole d'avril, contrairement à ce que ce document a d'abord
affirmé.** `scores_isdd_5M.sc` a été produit par `enrich_pool_isdd` en **mode temps**
(20 ms/coup), et en mode temps la boucle d'IS-DD ne sort que sur l'échéance : le
`else if det_count >= config.determinizations` de [is_dd.rs](../../colver-core/src/search/is_dd.rs)
est inatteignable. Le « × 20 dets » que citait `CLAUDE.md` n'a jamais borné quoi que ce
soit — **le nombre de mondes par label est variable, et au premier pli il est petit**,
un monde de trick 1 étant une donne complète à résoudre. Les bras ci-dessous sont tous en
mode **compte** (20 mondes), donc la comparaison avril → aujourd'hui mélange la dérive de
code et un budget d'échantillonnage différent.

Le sens de l'effet est connu même sans le mesurer : moins de mondes = plus de bruit, donc
**les labels d'avril sont plus bruités que ceux d'ici**. Le plancher mesuré plus bas
(24,32, entre deux runs à 20 mondes) est donc un plancher *trop bas* pour lire l'écart
avril → aujourd'hui, et les 9,3 pts d'excès en sont d'autant plus un **majorant**. La
conclusion — ne pas regénérer — s'en trouve renforcée, pas fragilisée. Ce qui tombe, c'est
l'attribution : on n'a pas isolé la dérive de code.

**Chiffré depuis** (2026-08-03,
[isdd_worlds_per_budget.py](../../scripts/analysis/isdd_worlds_per_budget.py)) : en mode
temps à 20 ms/coup, IS-DD traverse une **médiane de 2 mondes au pli 1**, contre 20 pour les
bras ci-dessous. Un facteur 10 sur les coups qui décident le plus de l'issue — l'écart de
plancher est donc large, et le majorant d'autant plus lâche.
[../play/is_dd.md](../play/is_dd.md#worlds-per-budget-measured-2026-08-03)

Le bras B existe parce que le sidecar ne peut pas représenter une position atteinte par
`setup_dd` : il rejoue `GameState::new` à travers une liste d'actions, et pour playgen v2
**l'atout n'est porté que par les jetons d'enchère**. Il faut donc un préfixe qui nomme
l'atout, et il faut le facturer à part.

## Rien ne se lit sans le plancher

IS-DD est stochastique : deux passes de la **même** config sur les **mêmes** donnes ne
rendent pas les mêmes labels. Mesuré, RMS **24,32** et r = 0,843 — un label porte ~17 pts
de bruit à lui tout seul. Toute comparaison se lit contre ce plancher, jamais contre zéro.

## Résultats — 1000 donnes, 4000 paires (donne × couleur)

| comparaison | Δ moyen | RMS | r | même meilleure couleur |
|---|---|---|---|---|
| **plancher** (B0 vs B0, graine ≠) | +0,52 ± 0,38 | **24,32** | 0,843 | 72,3 % |
| avril → B0 (code **et** budget de mondes) | +0,08 ± 0,40 | 25,60 | 0,820 | 69,8 % |
| préfixe d'enchère (B0 → B) | 0,00 | **0,00** | 1,000 | 100 % |
| mondes playgen (B → C) | −0,26 ± 0,41 | 26,08 | 0,808 | 69,6 % |
| **péremption totale** (avril → C) | **−0,18 ± 0,41** | **26,05** | 0,805 | 68,4 % |

Trois lectures.

**1. Aucun biais.** Tous les décalages de moyenne tiennent dans ~1 erreur type de zéro.
Rien n'a systématiquement déplacé la valeur d'un contrat — il n'y a pas de recentrage à
appliquer aux anciens labels.

**2. L'excès sur le bruit est petit.** √(26,05² − 24,32²) = **9,3 pts**, soit **12,8 % de
la variance** d'un label. Autrement dit **au moins 87 % de ce qui bouge entre avril et
aujourd'hui aurait bougé de toute façon** en relançant le même code avec une autre graine.
« Au moins », parce que 24,32 est le plancher entre deux runs à 20 mondes, alors qu'avril
en avait moins (cf. l'avertissement plus haut) : le vrai plancher est plus haut et les
9,3 pts sont un majorant.

**3. Le préfixe synthétique est un no-op bit-à-bit** en mondes uniformes (RMS exactement
0,00, 100 % de labels identiques). C'est ce qui autorise à lire tout l'écart B → C comme
l'effet playgen et rien d'autre.

## Pourquoi playgen ne change presque rien ici, alors qu'il gagne l'arène 59-65 %

Ce n'est pas une contradiction, c'est **la quantité mesurée**.

Ici les quatre sièges sont améliorés **en même temps**, et le label est le nombre de
points cartes pris par N-S — une quantité à **somme constante** (152). Un effet commun
aux quatre sièges y est **nul par construction**. L'arène, elle, mesure un différentiel :
un camp joue avec playgen, l'autre sans.

> **Règle à retenir : un label DD symétrique ne peut pas voir la force de jeu.**
> Il ne voit que le bruit d'échantillonnage du solveur. Pour mesurer la force il faut un
> bras asymétrique (N-S playgen / E-O uniforme) — jamais fait, ~1,2 h aujourd'hui.

Note également : le plancher de bruit est mesuré en mondes **uniformes**. Un plancher
playgen-contre-playgen n'a pas été mesuré, donc les 9,3 pts d'excès sont un **majorant**.

## La deuxième raison, et elle est plus forte

COLVSC01 stocke `[u8;4]` par donne — une valeur **par couleur d'atout**. Quatre couleurs,
donc quatre contrats. Or une vraie enchère atterrit sur **un** contrat, choisi par la
communication entre quatre joueurs. Aucune regénération ne corrige ça : ce qui manque
n'est pas la fraîcheur des chiffres, c'est que la question posée à chaque ligne n'est pas
celle que se pose un annonceur.

C'est le même point que celui qui a fait échouer la distillation :
[bid/experiments/auction_conditioned_labels.md](../bid/experiments/auction_conditioned_labels.md).

## Ce qui justifierait quand même une regénération

- **Un changement de format** — c'est le vrai motif, et il relève du périmètre de v7.
- **Une rupture de règles.** Le correctif de surcoupe du 2026-08-01 en est une, mais il ne
  retire effectivement une option légale que sur **~1 décision sur 7 000** (CLAUDE.md).
  Seul, il ne presse pas.
- **Pas la dérive d'IS-DD.** 9,3 pts d'excès sur ~24 de bruit, contre ~**117 GPU-jours** à
  l'échelle 5M au débit actuel de 0,496 donne/s (c'était 241 avant l'optimisation du
  sidecar du 2026-08-02).

Et à budget de solves fixé, ce sont les **donnes** qui gagnent, pas les mondes par donne :
[bid/bid_v7_plan.md](../bid/bid_v7_plan.md) §2.8.

## Reproduire

L'outil est [relabel_isdd.rs](../../colver-core/src/bin/relabel_isdd.rs). Les cinq
comparaisons ci-dessus sont dans le registre versionné
[docs/measurements/index.jsonl](../measurements/index.jsonl) (`script = "relabel_isdd"`),
avec le sha256 de chaque couche ; le brut est dans `data/analysis/relabel_isdd/`.

```bash
cargo build --release --features parallel --bin relabel_isdd

# B0 — mondes uniformes, protocole d'avril (~6 min sur 32 cœurs)
./target/release/relabel_isdd --deals 1000 --worlds uniform --auction none \
  --time-ms 0 --dets 20 --seed 42 \
  --name b0_uniform_noauction --output data/deals/relabel/b0_uniform_noauction.sc

# C — mondes playgen. Exige le sidecar (~1,2 h ; ~10 h avant le groupage)
playgen-up
./target/release/relabel_isdd --deals 1000 --worlds sidecar --auction synthetic \
  --time-ms 0 --dets 20 --seed 42 --chunk 100 --threads 192 \
  --name c_playgen_synth --output data/deals/relabel/c_playgen_synth.sc \
  --baseline data/deals/relabel/b_uniform_synth.sc
playgen-down   # 5,5 Go de VRAM résidents tant qu'il vit

# comparer deux couches (lecture seule, marche sur un checkpoint partiel)
./target/release/relabel_isdd --compare-only <couche> --baseline <référence>

# idem + enregistrement dans le registre
uv run python scripts/analysis/relabel_log.py --compare <couche> --baseline <référence> \
  --tag <nom> --note "<ce que le bras isole>"
```

⚠️ `--threads 192` n'est pas un détail : à 32 threads le client passe son temps bloqué en
HTTP et le sidecar ne groupe plus que ~13 requêtes au lieu de ~26, ce qui annule le gain
du groupage (0,237 donne/s contre 0,496).
