"""Full-game CFN notation (web layer).

Extends the core CFN (`<dealer>:<hands> <tricks> <contract>`, produced/parsed by
`Env.to_cfn`/`Env.from_cfn`) with an optional **auction** section inserted
between the hands and the tricks:

    <dealer>:<hands> <auction> <tricks> <contract>   (4 sections, with auction)
    <dealer>:<hands> <tricks> <contract>             (3 sections, core CFN)

The auction is a ``-``-joined sequence of bid tokens, in play order from the
seat left of the dealer:

    p            pass
    <val><suit>  bid, e.g. 90h, 120c  (suit s/h/d/c)
    C<suit>      capot, e.g. Cs
    x / xx       coinche / surcoinche

The core 3-section CFN is untouched, so watch/tests keep working; a 3-section
string parses here too (with an empty auction).
"""

SUITS = "shdc"  # bid/action suit order: 0=S 1=H 2=D 3=C


def bid_to_token(a: int) -> str:
    if a == 0:
        return "p"
    if a == 41:
        return "x"
    if a == 42:
        return "xx"
    if 37 <= a <= 40:
        return "C" + SUITS[a - 37]
    if 1 <= a <= 36:
        vi, si = (a - 1) // 4, (a - 1) % 4
        return f"{80 + vi * 10}{SUITS[si]}"
    raise ValueError(f"invalid bid action {a}")


def token_to_bid(t: str) -> int:
    if t == "p":
        return 0
    if t == "x":
        return 41
    if t == "xx":
        return 42
    if t.startswith("C"):
        return 37 + SUITS.index(t[1])
    suit = SUITS.index(t[-1])
    val = int(t[:-1])
    if val < 80 or val > 160 or val % 10 != 0:
        raise ValueError(f"invalid bid token {t}")
    return (val - 80) // 10 * 4 + suit + 1


def build_auction(bid_actions) -> str:
    return "-".join(bid_to_token(int(a)) for a in bid_actions) if bid_actions else "-"


def parse_auction(section: str):
    if section in ("-", ""):
        return []
    return [token_to_bid(t) for t in section.split("-")]


def to_full_cfn(core_cfn: str, bid_actions) -> str:
    """Insert the auction section into a standard 3-section core CFN."""
    parts = core_cfn.strip().split(" ")
    if len(parts) != 3:
        raise ValueError(f"expected a 3-section core CFN, got {len(parts)}: {core_cfn!r}")
    hands, tricks, contract = parts
    return f"{hands} {build_auction(bid_actions)} {tricks} {contract}"


def parse_full_cfn(s: str):
    """Return (core 3-section CFN, bid_actions). Accepts 3 or 4 sections."""
    parts = s.strip().split(" ")
    if len(parts) == 3:
        return s.strip(), []
    if len(parts) == 4:
        hands, auction, tricks, contract = parts
        return f"{hands} {tricks} {contract}", parse_auction(auction)
    raise ValueError(f"expected 3 or 4 space-separated sections, got {len(parts)}")
