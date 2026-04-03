"""Show concrete examples of NN bidder vs improved_v2 side by side."""
import random
import colver

SUITS = ["S", "H", "D", "C"]
SUIT_SYMBOLS = {"S": "\u2660", "H": "\u2665", "D": "\u2666", "C": "\u2663"}
RANKS_DISPLAY = ["7", "8", "9", "J", "Q", "K", "T", "A"]  # bit order within suit
SEATS = ["N", "E", "S", "W"]
TEAMS = ["NS", "EW"]


def hand_str(cards):
    """Format a hand grouped by suit."""
    suits = {s: [] for s in SUITS}
    for c in sorted(cards):
        s = SUITS[c // 8]
        r = RANKS_DISPLAY[c % 8]
        suits[s].append(r)
    parts = []
    for s in SUITS:
        if suits[s]:
            parts.append(SUIT_SYMBOLS[s] + "".join(reversed(suits[s])))
    return " ".join(parts) if parts else "-"


def action_name(action):
    if action == 0:
        return "PASS"
    if action == 41:
        return "COINCHE"
    if action == 42:
        return "SURCOINCHE"
    if 37 <= action <= 40:
        suit = action - 37
        return f"CAPOT {SUITS[suit]}"
    idx = action - 1
    suit = idx % 4
    value_idx = idx // 4
    value = 80 + value_idx * 10
    return f"{value}{SUIT_SYMBOLS[SUITS[suit]]}"


def run_bidding(env, use_nn=False):
    """Run bidding phase, return list of (seat, action_name, action)."""
    history = []
    while env.phase() == 0:  # Bidding
        seat = env.current_player()
        if use_nn:
            result = env.action_bid_nn()
            action = result["best_action"]
        else:
            action = env.bid_improved_v2()
        history.append((SEATS[seat], action_name(action), action))
        env.step(action)
    return history


def play_dd_and_score(env):
    """Play cards with DD oracle and return deal_outcome."""
    if env.is_terminal():
        return env.deal_outcome()
    while not env.is_terminal():
        action = env.action_oracle_dd()
        env.step(action)
    return env.deal_outcome()


def contract_str(c):
    if c["value"] == 0:
        return "VOID (4 passes)"
    coinche = "xx" if c["coinche"] == 2 else ("x" if c["coinche"] == 1 else "")
    return f"{c['value']}{SUIT_SYMBOLS[SUITS[c['trump']]]} by {TEAMS[c['team']]}{coinche}"


def show_example(seed, idx):
    """Deal a hand, run both bidders, compare with DD scoring."""
    random.seed(seed + idx)

    # Generate a random deal
    env_tmp = colver.Env()
    env_tmp.reset()
    hands_raw = env_tmp.get_hands()
    dealer_val = env_tmp.get_dealer()

    # Create two identical envs
    env_nn = colver.Env.deal_with_hands(dealer_val, hands_raw)
    env_nn.load_bid_model("models/bid_nn_latest.bin")
    env_v2 = colver.Env.deal_with_hands(dealer_val, hands_raw)

    # Run bidding
    nn_history = run_bidding(env_nn, use_nn=True)
    v2_history = run_bidding(env_v2, use_nn=False)

    nn_contract = env_nn.get_contract()
    v2_contract = env_v2.get_contract()

    # DD solve all 4 suits
    dd_ns = {}
    for suit_idx in range(4):
        dd_env = colver.Env.deal_with_hands(dealer_val, hands_raw)
        dd_env.set_contract(suit_idx, 80, 0, 0)
        dd_env.set_phase_playing()
        while not dd_env.is_terminal():
            dd_env.step(dd_env.action_oracle_dd())
        pts = dd_env.get_points()
        dd_ns[suit_idx] = pts[0]

    # Play both contracts with DD oracle
    nn_score = play_dd_and_score(env_nn) if nn_contract["value"] > 0 else (0.0, 0.0)
    v2_score = play_dd_and_score(env_v2) if v2_contract["value"] > 0 else (0.0, 0.0)

    # Print
    print(f"\n{'='*70}")
    print(f"Deal #{idx+1}  (dealer={SEATS[dealer_val]})")
    print(f"{'='*70}")
    for seat in range(4):
        team = "NS" if seat % 2 == 0 else "EW"
        print(f"  {SEATS[seat]} ({team}): {hand_str(hands_raw[seat])}")

    # DD summary line
    dd_parts = []
    for suit_idx, suit in enumerate(SUITS):
        ns = dd_ns[suit_idx]
        ew = 252 - ns if ns == 0 or ns == 252 else 162 - ns
        dd_parts.append(f"{SUIT_SYMBOLS[suit]}NS={ns}")
    print(f"  DD: {' | '.join(dd_parts)}")

    # Bidding side by side
    print(f"\n  {'NN Bidder':<30} {'improved_v2':<30}")
    print(f"  {'-'*28}   {'-'*28}")
    max_len = max(len(nn_history), len(v2_history))
    for i in range(max_len):
        nn_str = f"{nn_history[i][0]}: {nn_history[i][1]}" if i < len(nn_history) else ""
        v2_str = f"{v2_history[i][0]}: {v2_history[i][1]}" if i < len(v2_history) else ""
        diff = " *" if nn_str != v2_str and nn_str and v2_str else ""
        print(f"  {nn_str:<30} {v2_str:<30}{diff}")

    # Contracts and scores
    print(f"\n  NN: {contract_str(nn_contract):<25} => NS={nn_score[0]:+.0f}  EW={nn_score[1]:+.0f}")
    print(f"  v2: {contract_str(v2_contract):<25} => NS={v2_score[0]:+.0f}  EW={v2_score[1]:+.0f}")

    winner = ""
    if nn_score[0] + nn_score[1] != v2_score[0] + v2_score[1]:
        # Who extracted more total value?
        nn_better = (nn_score[0] - nn_score[1]) > (v2_score[0] - v2_score[1])
        # Actually: NN plays both sides. Compare the deal outcomes
        if nn_score != v2_score:
            winner = "  => Different outcome!"
    print(winner)

    return nn_score, v2_score, nn_contract, v2_contract


def main():
    print("=== NN Bidder vs improved_v2 — Concrete Examples ===")
    print("(Using latest checkpoint: models/bid_nn_latest.bin)\n")
    print("Each deal: same 4 hands, NN bids all 4 seats vs v2 bids all 4 seats.")
    print("Card play: DD oracle (perfect play) for both.")

    nn_wins = 0
    v2_wins = 0
    same = 0

    for i in range(15):
        nn_out, v2_out, nn_c, v2_c = show_example(seed=777, idx=i)
        if nn_out == v2_out:
            same += 1
        else:
            # Compare from perspective of "which bidder got a better deal"
            # Hard to compare directly since teams may differ
            pass

    print(f"\n{'='*70}")
    print(f"Identical outcomes: {same}/15")
    print(f"Different outcomes: {15-same}/15")


if __name__ == "__main__":
    main()
