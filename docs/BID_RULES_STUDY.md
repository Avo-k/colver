# Bidding Rules Study — NN Probing, DD Validation, Rule Extraction

## Overview

Systematic study of the bid NN ("Le Bide a Dede") to extract human-readable bidding rules, validate them against the DD oracle, and iterate on a rule-based bidder.

**Binaries created:**
- `bid_nn_probe` — Query NN on controlled hands + 50k Monte Carlo
- `bid_dd_probe` — DD oracle validation of same hypotheses + NN calibration
- `bid_rules_iter` — Iterative rule-based bidder tournament (V1→V2→V3)

## Part 1: NN Behavioral Rules (bid_nn_probe, 50k deals)

### The Jack is Everything
- J+9 with just 2 trump → **97% bid rate** (NN), 90% DD success at 80
- J alone with 2 trump → 81% bid rate
- Without J or 9, need **5+ cards** to reach 85% bid rate
- 9 alone with 3 cards → only 48% bid rate (coin flip)

### "Annoncer aux as" — Almost Never
- 4 aces + fillers: NN **always passes** (Q(PASS)=+0.665 >> Q(80)=+0.560)
- DD says 4 aces makes 80 in 80% of deals — but NN correctly avoids it (imperfect info discount)
- Only 14% of hands with no J/no 9 anywhere get bid
- Exception: 6+ trump without J/9 → NN bids (sheer length compensates)

### Bid Level Distribution — Bimodal
- **61% at 80**, 29% at 100, 7.5% at 110, **1.9% at 90**
- NN learned two modes: "open light at 80" or "strong hand at 100+"
- 90 is essentially skipped

### Position Effect
- Position 3 (after 2 passes) is most aggressive — others showed weakness
- Position 2 and 4 are more conservative
- Not the clean "later = better" pattern expected

### Side Strength
- Determines **bid level**, not bid/pass decision
- J9A♠ always bids regardless of sides (92-100% DD success)
- 3 side aces → 100♠ (avg 214 DD pts), 0 aces → 80♠ (avg 132 pts)

### Belote (K+Q of trump)
- Barely matters. DD confirms: KQ vs K8 vs Q8 vs 87 all ~same success rate

### Partner Response
- J alone in partner's suit → raise to 100
- 9+A (no J) → raise to 100
- KQ + 3 side aces → raise to 100
- No cards in suit → PASS

### Coinche
- J+9 in opponent's suit → always coinche (Q=+0.774)
- AKQT in opponent's suit → coinche (Q=+0.398)
- 4+ trump in their suit + side ace → coinche
- 0 trump + 3 aces → coinche (théorème 3)

## Part 2: DD Oracle Validation (bid_dd_probe, 10k deals)

### NN Calibration
| Metric | Value |
|---|---|
| NN bids | 73.6% of deals |
| When NN bids, DD confirms ≥80 | **86.7%** (good precision) |
| When NN bids, DD confirms ≥ contract level | **84.0%** |
| When NN passes, DD says best suit ≥80 | **93.3%** (very conservative) |

### DD Success by Trump Features (per-suit, 10k deals)

| Profile | DD ≥80 | NN bids | NN precision |
|---|---|---|---|
| J+9, 3 cards | 97% | 92% | 97% |
| J+9, 2 cards | 90% | 79% | 92% |
| J only, 2 | 72% | 42% | 78% |
| J only, 3 | 88% | 75% | 88% |
| 9 only, 3 | 68% | 20% | 73% |
| 9 only, 4 | 83% | 55% | 86% |
| Neither, 4 | 71% | 14% | 74% |
| Neither, 5 | 85% | 62% | 86% |

NN is systematically conservative vs DD. DD sees all cards; NN reasons under uncertainty.

### Controlled Hands — DD vs NN

**Jack experiment (4♠ + K♥Q♥ + 8♦7♦):**
| Hand | DD avg pts | DD ≥80 | DD ≥100 | NN |
|---|---|---|---|---|
| J9A10♠ | 151 | 100% | 89% | 100♠ |
| JA108♠ | 134 | 94% | 80% | 100♠ |
| 9A108♠ | 115 | 80% | 64% | PASS |
| KQA10♠ | 100 | 70% | 50% | PASS |

**Trump length (with J, strong sides):**
| Trump | DD avg | DD ≥100 | NN |
|---|---|---|---|
| JA (2) | 131 | 65% | 80♠ |
| JA10 (3) | 154 | 90% | 90♠ |
| J9A10 (4) | 177 | 100% | 100♠ |
| J9A10K (5) | 171 | 100% | 100♠ |
| J9A10KQ (6) | 185 | 100% | 100♠ |

**Side strength (J9A♠ fixed):**
| Sides | DD avg | DD ≥80 | NN |
|---|---|---|---|
| 3 side aces | 214 | 100% | 100♠ |
| 0 aces (garbage) | 132 | 94% | 80♠ |

## Part 3: Rule-Based Bidder Iterations

### V1: Strict Quality Gate
- Quality: require J or 9, or 5+ cards, or 4+ with ace
- Bimodal: skip 90 (80 or 100)
- Same coinche rules as NN probe findings

**Results (2000 deals):**
- vs ImprovedV2: margin **+4** (roughly tied)
- vs NN: margin **-55** (crushed)
- NN takes 61% of contracts vs V1's 39%

### V2: Looser Quality Gate
- Also allow A+10, 4+ with any honor, bring back 90
- Lower opening threshold

**Results:** No improvement. Margin vs NN: **-59** (slightly worse).

**Insight:** The gap isn't about which hands to bid — it's about bid intelligence (history reading, coinche precision).

### V3: History-Aware with Auction Context

Added `AuctionContext` struct parsed from bid history:
- Partner bid tracking, opponent overbid detection
- Competitive raise logic (rebid partner's suit when overbid)
- Position-scaled aggression
- Surcoinche (later removed — too risky)
- Smart coinche with overbid-avoidance

**V3a results (500 deals):**
- vs ImprovedV2: margin **+10** (improved)
- vs NN: margin **-61** (same gap)

**Diagnostics (V3 vs NN side-by-side):** 88% of deals differ. Key patterns:
1. V3 overbids (100-120 where NN bids 80-100)
2. NN competes harder in bidding wars; V3 gives up or coinches
3. V3 coinches when it should overbid instead
4. NN does "bait and switch" — opens light, lets partner correct

**V3b fixes:** Lower bid levels, remove surcoinche, add competitive rebidding, prefer overbid over coinche.

**V3b results:**
- vs ImprovedV2: margin **+6**
- vs NN: margin **-59**
- V3b now takes **58% of contracts** vs NN (up from 35%), but makes only 57% (down from 64%)

### Summary: Rule Ceiling

| Bidder | vs ImprovedV2 | vs NN |
|---|---|---|
| V1 (strict) | +4 | -55 |
| V2 (loose) | +5 | -59 |
| V3a (history) | +10 | -61 |
| V3b (competitive) | +6 | -59 |
| ImprovedV2 (ref) | 0 | -54 |

**The ~55-60pt gap to the NN is structural.** Rule improvements change the character of the gap (more contracts but lower success, or fewer but safer) but not the magnitude.

The NN's edge comes from:
1. **Bid history encoding** (72 floats of auction context)
2. **Joint hand+history reasoning** (which specific hands to bid given what others did)
3. **Coinche precision** (87% correct, rules are ~70%)
4. **Millions of RL training games** calibrating the exact thresholds

## Next Steps

1. **Dual learning loop**: Train bid NN using DMC play results (not DD oracle) → bid NN learns actual achievable success rates → more accurate bidding. Then retrain DMC on games with better bids. Iterate.

2. **Transfer NN knowledge to rules**: Use the diagnostic framework to find the ~20% of deals where rules match NN, identify what's different about the other 80%, and write more targeted rules.

3. **Hybrid approach**: Use rule-based bidder for opening, NN for competitive/coinche decisions (where history matters most).
