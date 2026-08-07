# Belote Contree Rules

Rules as implemented in Colver, following FFB (Federation Francaise de Belote) official rules.

> **There is no single FFB rulebook.** The federation has published at least four mutually
> incompatible editions, a rival federation exists (Federation Francaise de Coinche, Saint-Etienne,
> 1997), and the flagship national tournament follows neither. Which choices below are attested
> where, and which are unique to Colver: [rules-survey/SYNTHESE.md](rules-survey/SYNTHESE.md)
> (§6 "Ou tombe Colver").

## Overview

Belote Contree is a 4-player trick-taking card game played in teams of two. Partners sit across from each other: North-South (team 0) vs East-West (team 1). A match is played to **2000 points** across multiple deals.

## The Deck

32 cards: four suits (Spades, Hearts, Diamonds, Clubs), eight ranks per suit (7, 8, 9, Jack, Queen, King, 10, Ace). No Sans Atout or Tout Atout -- only color contracts.

### Card Point Values

| Rank  | Trump | Plain |
|-------|------:|------:|
| 7     |     0 |     0 |
| 8     |     0 |     0 |
| 9     |    14 |     0 |
| Jack  |    20 |     2 |
| Queen |     3 |     3 |
| King  |     4 |     4 |
| 10    |    10 |    10 |
| Ace   |    11 |    11 |
| **Total** | **62** | **30** |

All 32 cards total **152 points** (62 from the trump suit + 30 x 3 from the three plain suits).

### Card Strength

**Plain suits** (highest to lowest): A > 10 > K > Q > J > 9 > 8 > 7

**Trump suit** (highest to lowest): J > 9 > A > 10 > K > Q > 8 > 7

## Deal

Each player receives 8 cards. The dealer rotates clockwise each deal.

## Bidding

Bidding starts with the player after the dealer and proceeds clockwise. Each player may:

- **Pass**
- **Bid** a contract: a point value + a trump suit
- **Coinche** (double) an opponent's bid
- **Surcoinche** (redouble) after a coinche

### Bid Values

Available values: 80, 90, 100, 110, 120, 130, 140, 150, 160, or **Capot** (250).

Each new bid must be **strictly higher** in value than the current bid. Suit can be anything -- only the value must increase. Capot (250) is always higher than any numbered bid.

### Coinche and Surcoinche

- **Coinche**: only an opponent of the bidding team may coinche. It **freezes** the contract -- no further bids are allowed, only surcoinche or pass.
- **Surcoinche**: only the team whose bid was coinched may surcoinche. It **ends bidding immediately**.

### Bidding Ends When

- **3 consecutive passes** after a bid has been made
- **Surcoinche** is declared (immediate end)
- **4 consecutive passes** with no bid: the deal is **void** (0 points, redeal)

## Play

The player after the dealer leads the first trick. Play proceeds clockwise, 8 tricks of 4 cards.

### Following Suit

1. **Must follow the lead suit** if you have cards in it
2. If you **cannot follow** the lead suit:
   - If your partner is currently winning the trick: you may play any card
   - Otherwise: you **must trump** (play a trump card) if you have one
3. If you have **no cards in the lead suit and no trump**: play any card

### Overtrumping

When playing a trump card (whether following a trump lead or cutting), you **must play a higher trump** than any trump already on the trick, if you can.

### Ne Pisse Pas

If you cannot follow suit and must trump, but **cannot overtrump** an opponent's trump already on the table, you have two options:
- Undertrump (play a lower trump)
- Discard a non-trump card instead

However, if you **only have trump cards** in your hand, you must undertrump.

### Partner Is Winning — No Obligation At All

If your partner currently holds the trick, you may play **any card, without exception** — whatever they played, and whatever is left in your hand.

That "without exception" is load-bearing in one specific spot: partner **cut** a non-trump lead, and **trump is all you have left**. You may then play a trump *lower* than theirs. FFB contrée §2.3 spells it out — « n'importe quelle carte sans exception (y compris un atout inférieur au sien) » — and the 2015 edition calls it « le seul cas de figure, plutôt rare, où il est permis de jouer un atout inférieur ».

> **BREAKING (2026-08-01): this case used to force an overtrump.** The engine followed the one
> FFB edition (the "Équipe Ludique" reprint, `LOCAL_regles_officielles_belote_contree.pdf` in the
> unversioned local corpus) whose
> article 4 drops the `n'est pas` from that sentence and deletes the clause explaining it. FFB
> contrée 2015, FFB contrée 2016 and FFB belote 2016 all say the opposite.
>
> The fix **enlarges** the legal move set, so it changes the game tree: **DD values, pre-solved
> pools and any score layer produced before this date are stale**, same class of breakage as the
> `quick_tricks` removal. Evidence and the four texts side by side:
> [rules-survey/matrices/jeu-de-la-carte.md](rules-survey/matrices/jeu-de-la-carte.md).
>
> **How stale, measured.** Over 20 000 random deals (640 000 play decisions), the case arose
> **485 times (0.076 % of decisions)**, and the old rule actually removed a legal option in only
> **91 of them (0.014 %, ~1 decision in 7 000, ~1 deal in 220)** — the rest had no lower trump to
> choose anyway. So the staleness is real but shallow: regenerating DD pools can be batched with
> the next breaking change rather than done urgently. Caveat: measured under *random* play, which
> distributes voids differently from skilled play, so treat it as an order of magnitude.

### Trick Winner

- The highest **trump** card wins the trick (using trump strength order)
- If no trump was played, the highest card **in the lead suit** wins
- Cards that are neither trump nor the lead suit cannot win

The trick winner leads the next trick.

## Belote and Rebelote

If a player holds both the **Queen and King of the trump suit**, they declare *belote* when playing the first of the two cards, and *rebelote* when playing the second. This earns a **20 point bonus** for their team.

Both cards must be played by the **same player** (who necessarily holds both). The bonus only counts when both declarations are made (belote + rebelote = 20 points).

**The declaration is public information, and it is read both ways** (`play::belote_facts`,
added 2026-08-03). Because the announcement is compulsory, hearing it places the second
honour at the announcer — and *not* hearing it, when a trump King or Queen falls, proves
its player does not hold the other one, ever, since a hand only shrinks. The silent case
is the more frequent of the two (20.5% of play positions against 5.7%). Both are hard
determinization constraints on the same footing as a revealed void — see
`play::belote_facts` in `colver-core/src/engine/play.rs`, and its three callers in
`colver-core/src/search/determinize.rs`.

## Dix de Der (Last Trick Bonus)

The team winning the last (8th) trick receives a bonus:
- **+10 points** in the normal case
- **+100 points** if the same team won all 8 tricks (capot)

With dix de der, the maximum total card points in a deal are:
- Normal: 152 + 10 = **162**
- Capot: 152 + 100 = **252**

## Scoring

Colver uses the **"points faits + points demandes"** scoring mode (FFB official rules, section 10.2).

### Contract Success (Reussi)

The taker succeeds when:
- **Non-capot contracts**: taker's trick points + belote >= contract value
- **Capot contracts**: taker wins all 8 tricks

### Non-Capot Contracts

The base of every fixed-sum line below is **all the card points of the deal**: 162 (dix de der
included), or 252 when the taker actually won all 8 tricks -- announced capot or not. A coinche
multiplier applies to the **contract value only**, never to that base.

#### Standard (no coinche)

| | Taker | Defense |
|-|-------|---------|
| **Reussi** | trick_points + contract_value + own_belote | trick_points + own_belote |
| **Chute** | 0 | 162 + contract_value + all_belote |

On a chute the defense takes the contract **and** every card point of the deal, whatever the
actual trick split.

#### Coinche (x2) / Surcoinche (x3)

`mult` = 2 when coinched, 3 when surcoinched.

| | Taker | Defense |
|-|-------|---------|
| **Reussi** | 162* + contract_value x mult + all_belote | 0 |
| **Chute** | 0 | 162 + contract_value x mult + all_belote |

\* 252 if the taker won all 8 tricks.

### Capot Contracts

Capot is a regular contract worth **250**, not a flat bonus -- the tables above apply, with
contract_value = 250 and trick_points = 252 when realised.

| Coinche level | Reussi (taker) | Reussi (defense) | Chute (taker) | Chute (defense) |
|---------------|----------------|------------------|---------------|-----------------|
| Standard | 252 + 250 + own_belote = **502** | own_belote | 0 | 162 + 250 + all_belote |
| Coinche | 252 + 250 x 2 + all_belote = **752** | 0 | 0 | 162 + 250 x 2 + all_belote |
| Surcoinche | 252 + 250 x 3 + all_belote = **1002** | 0 | 0 | 162 + 250 x 3 + all_belote |

Note on capot reussi (standard): each team keeps their own belote. On chute or when
coinched/surcoinched, all belote goes to the winning side.

### Rounding

**None** (since 2026-07-31). Scores are marked exactly as summed, at the point. The FFB rounds
the marque to the nearest 10 (its section 9.2), we deliberately don't -- the engine and the web
score sheet (`views/score.js`) now agree on every donne, and the 162 base is visible in the
marque instead of collapsing onto 160.

### Scoring Examples

**Example 1** -- Taker bids 80 Hearts, scores 92 trick points:
- Reussi (92 >= 80)
- Taker: 92 + 80 = **172**
- Defense: **70**

**Example 2** -- Taker bids 100 Spades, scores 82 trick points:
- Chute (82 < 100)
- Taker: **0**
- Defense: 162 + 100 = **262**

**Example 3** -- Taker bids 100, scores 88 trick points, has belote (20):
- 88 + 20 = 108 >= 100: Reussi (belote saves the contract!)
- Taker: 88 + 100 + 20 = **208**

**Example 4** -- Taker bids 80 Spades coinche, scores 100:
- Reussi
- Taker: 162 + 80 x 2 = **322**
- Defense: **0**

## Match

The first team to reach **2000 cumulative points** wins the match. Void deals (4 passes) score 0-0.
