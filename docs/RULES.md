# Belote Contree Rules

Rules as implemented in Colver, following FFB (Federation Francaise de Belote) official rules.

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

### Partner Cut Exception

If your partner has cut (played trump on a non-trump lead) and is winning, and you have non-trump cards, you may play anything. But if you **only have trump cards**, you must overtrump the highest trump on the table if possible.

### Trick Winner

- The highest **trump** card wins the trick (using trump strength order)
- If no trump was played, the highest card **in the lead suit** wins
- Cards that are neither trump nor the lead suit cannot win

The trick winner leads the next trick.

## Belote and Rebelote

If a player holds both the **Queen and King of the trump suit**, they declare *belote* when playing the first of the two cards, and *rebelote* when playing the second. This earns a **20 point bonus** for their team.

Both cards must be played by the **same player** (who necessarily holds both). The bonus only counts when both declarations are made (belote + rebelote = 20 points).

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
| **Reussi** | round10(trick_points + contract_value + own_belote) | round10(trick_points + own_belote) |
| **Chute** | 0 | round10(162 + contract_value + all_belote) |

On a chute the defense takes the contract **and** every card point of the deal, whatever the
actual trick split.

#### Coinche (x2) / Surcoinche (x3)

`mult` = 2 when coinched, 3 when surcoinched.

| | Taker | Defense |
|-|-------|---------|
| **Reussi** | round10(162* + contract_value x mult + all_belote) | 0 |
| **Chute** | 0 | round10(162 + contract_value x mult + all_belote) |

\* 252 if the taker won all 8 tricks.

### Capot Contracts

Capot is a regular contract worth **250**, not a flat bonus -- the tables above apply, with
contract_value = 250 and trick_points = 252 when realised.

| Coinche level | Reussi (taker) | Reussi (defense) | Chute (taker) | Chute (defense) |
|---------------|----------------|------------------|---------------|-----------------|
| Standard | round10(252 + 250 + own_belote) = **500** | round10(own_belote) | 0 | round10(162 + 250 + all_belote) |
| Coinche | round10(252 + 250 x 2 + all_belote) = **750** | 0 | 0 | round10(162 + 250 x 2 + all_belote) |
| Surcoinche | round10(252 + 250 x 3 + all_belote) = **1000** | 0 | 0 | round10(162 + 250 x 3 + all_belote) |

Note on capot reussi (standard): each team keeps their own belote. On chute or when
coinched/surcoinched, all belote goes to the winning side.

### Rounding

All scores are rounded to the nearest 10: `round10(x) = (x + 5) / 10 * 10`

Examples: 85 -> 90, 84 -> 80, 162 -> 160

Every term other than the base is a multiple of 10, so under this rounding a base of 162 and a
base of 160 mark the *same* score -- the 162 is what the rule says, and it only shows where
points are kept exact, i.e. the web score sheet (`views/score.js`).

### Scoring Examples

**Example 1** -- Taker bids 80 Hearts, scores 92 trick points:
- Reussi (92 >= 80)
- Taker: round10(92 + 80) = round10(172) = **170**
- Defense: round10(70) = **70**

**Example 2** -- Taker bids 100 Spades, scores 82 trick points:
- Chute (82 < 100)
- Taker: **0**
- Defense: round10(162 + 100) = round10(262) = **260**

**Example 3** -- Taker bids 100, scores 88 trick points, has belote (20):
- 88 + 20 = 108 >= 100: Reussi (belote saves the contract!)
- Taker: round10(88 + 100 + 20) = round10(208) = **210**

**Example 4** -- Taker bids 80 Spades coinche, scores 100:
- Reussi
- Taker: round10(162 + 80 x 2) = round10(322) = **320**
- Defense: **0**

## Match

The first team to reach **2000 cumulative points** wins the match. Void deals (4 passes) score 0-0.
