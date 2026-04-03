"""Helper to decode bid actions without needing the Rust crate."""


def decode_bid_action(action: int) -> tuple[int, int]:
    """Decode action (1-40) to (value_encoded, suit_idx).

    value_encoded: 8=80, 9=90, ..., 16=160, 25=capot
    suit_idx: 0=S, 1=H, 2=D, 3=C
    """
    if 37 <= action <= 40:
        return 25, action - 37  # capot
    # action = value_idx * 4 + suit_idx + 1
    action_0 = action - 1
    value_idx = action_0 // 4
    suit_idx = action_0 % 4
    return value_idx + 8, suit_idx
