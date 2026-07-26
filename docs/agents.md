# Agents

**One object per seat that knows how to play a whole deal.** Everything that
decides *how* a seat plays lives in `colver-core/src/agent/`; the arena, the web
server and any script only build agents and ask them for actions.

## Why

Before this module, every caller re-implemented the same dispatch: "if this seat
uses IS-DD call `search`, else if it uses DMC write the observation, convert the
mask to canonical space, call `best_action`, convert back…". There were **ten
copies** — `arena.rs` (460 lines of it), `agent_tournament`, four tournament
binaries, `isdd_sweep`, `dmc_eval`, `joint_env`, and `game_manager.py`.

They drifted. The visible symptom: the web sampled IS-DD's determinized worlds
from the playgen GPU sidecar and pushed them into the search, while the arena
did not. The arena was benchmarking a **different, weaker agent** than the one
in production, under the same name.

The fix is ordinary object orientation. `Player` is a trait — Rust's equivalent
of an abstract base class — and each strategy is one implementation that owns
everything it needs: its models, its RNG, its per-deal state, and (for IS-DD)
its source of determinized worlds. World generation moved **inside** the agent,
so a caller cannot get it wrong by omission.

## Shape

```
AgentSpec  (what a bot TOML says)
   └─ build(seat) ──> Box<dyn Player>
                        └─ ComposedPlayer { bid: BidPolicy, play: CardPlayer }

trait Player      init_deal · observe · decide      (a seated bot, both phases)
trait BidPolicy   the auction half                  RuleBidPolicy · BidNetPolicy
trait CardPlayer  the play half                     IsDdPlayer · DmcPlayer ·
                                                    Oracle · Heuristic · Rule ·
                                                    Naive/SmartIsMcts

trait WorldSource   worlds(state, observer, n)      (crate::worlds)
   ├─ SidecarWorldSource   playgen on a remote GPU, batched   ← default
   ├─ LocalPlaygenSource   same model, in-process on CPU
   └─ UniformWorldSource   constraint-uniform
```

| file | what |
|------|------|
| `agent/mod.rs` | the traits, `MatchContext`, `Decision`, `Stats`, `AgentError` |
| `agent/spec.rs` | `AgentSpec` — TOML parsing and `build(seat)` |
| `agent/models.rs` | process-wide weight cache; **the only** place that reads a weight file's shape |
| `agent/isdd.rs` | `IsDdPlayer`, owns its `WorldSource` |
| `agent/dmc.rs` | `DmcPlayer` + the simple card players |
| `agent/bid.rs` | auction policies |
| `agent/ismcts.rs` | IS-MCTS baselines |
| `worlds.rs` | `WorldSource` and its three implementations |
| `game_loop.rs` | `play_deal` / `play_match` over `[Box<dyn Player>; 4]` |

## Lifecycle

```rust
let spec = AgentSpec::from_toml_file("arena/bots/champion.toml")?;
let mut players: [Box<dyn Player>; 4] =
    [spec.build(0)?, other.build(1)?, spec.build(2)?, other.build(3)?];
let mut ctx = MatchContext::new(dealer);

let result = game_loop::play_match(&mut players, dealer, &mut rng)?;
```

The one rule, and the one that every hand-written copy eventually got wrong:
**every player observes every action** — its own, its partner's, its opponents',
and the whole auction. That is what keeps belief states, world samplers and
credibility judges in sync with the game. A loop that skips one `observe` gives
you an agent that plays slightly worse for reasons nobody can find.

`init_deal` is also what resets per-deal state, so the same four players are
reused across a whole match rather than rebuilt (and their models reloaded) each
deal.

## Failure policy

Decisions return `Result`. A configured-but-unreachable dependency — in
practice, the playgen world sidecar — is an `AgentError`, **not** a silent
downgrade to weaker worlds. An agent that quietly changes strength turns every
measurement taken while it was degraded into a lie.

`[worlds] fallback` chooses:

- `strict` (default) — propagate the error. Correct for anything whose numbers
  will be compared.
- `uniform` — top up with constraint-uniform worlds and keep playing. The web
  uses this, because finishing the deal matters more there than matching the
  benchmark exactly.

Either way the substitution is **visible**: every decision reports where its
solved worlds came from.

```python
d = agent.decide(env)
d["worlds"]   # {'injected': 240, 'playgen': 0, 'belief': 0, 'uniform': 0}
```

`injected` = from the `WorldSource`; `belief`/`uniform` = the search's own
fallback sampling. A run that should be 100% playgen and isn't says so.

## Bot spec

The same TOML the arena reads, and what the PyO3 `Agent` takes:

```toml
[bid]
strategy = "nn"        # heuristic|improved|improved_v2|improved_v3|smart|roro|maxi|petit_bide|moelleux|nn|playgen
model = "models/bid_v6_isdd_resume/bid_nn_final.bin"
hidden = 512
penalty = 0.0          # discount on high bids (counters DD-trained optimism)
temperature = 0.0      # >0 = softmax-sample instead of argmax
score_aware = true     # endgame adjustments, for nets that cannot see the score

[play]
method = "isdd"        # isdd|dmc|dmc_then_isdd|ismcts|smart_ismcts|oracle|oracle_dd|heuristic|rule
model = "models/doudou50.bin"    # dmc / dmc_then_isdd
residual = true                  # DouDou50 / triforge architecture
time_ms = 1000         # per-move budget; 0 = count mode
determinizations = 240 # used when time_ms = 0
switch_at = 5          # dmc_then_isdd: trick at which IS-DD takes over
cred_alpha = 0.0       # credibility world-weighting (see is_dd.md)
parallel = true        # fan DD solves across the rayon pool

[worlds]
source = "sidecar"     # sidecar|playgen|uniform      (default: sidecar)
url = "http://192.168.1.23:8003"   # or $COLVER_PLAYGEN_GPU_URL
temperature = 0.8
batch = 128            # worlds per refill under a time budget
fallback = "strict"    # strict|uniform

[belief]
model = "models/belief_v4_fix_v2.bin"
```

Only IS-DD methods consume `[worlds]`; the section is ignored elsewhere.

`strategy = "playgen"` is the odd one out: `model` points at a **playgen v2**
checkpoint, not a bid net, and the policy (`PlaygenBidPolicy`, in
[agent/bid.rs](../colver-core/src/agent/bid.rs)) needs the whole visible prefix, so
it tracks the deal through `init_deal` / `observe` like a `WorldSource` does — which
is why `build_bid` takes the seat. `temperature` works as for `nn`. It falls back to
`ImprovedV2` when the sampler cannot answer: over-long auction, a non-v2 model, or a
sampled action that turns out illegal. See [bid/README.md](bid/README.md).

## Python

```python
import colver
agent = colver.Agent(spec_toml, seat=1)        # or Agent.from_file(path, seat)
agent.init_deal(env)

while not env.is_terminal():
    action = agent.decide(env)["action"] if env.current_player() == agent.seat else human()
    for a in all_agents:
        a.observe(env, action)                 # env still *before* the move
    env.step(action)
```

Two read-only companions, deliberately **not** part of `Agent` — looking at what
a model believes must never be able to change how a bot plays:

- `colver.Analyst` — playgen introspection: `marginals`, `bid_policy`,
  `auction_deals`.
- `colver.Beliefs` — the card-belief models (NN + heuristic) from one seat.

Both offer `replay(...)`, which rebuilds their state at any position from the
deal and the action list, so the analysis pages can jump around without keeping
a live object per view.

The web's translation from its agent-type names (`dede`, `doudou`, `oracle_dd`)
to specs lives in `python/colver/web/agents.py`.

## See also

- [play/is_dd.md](play/is_dd.md) — the IS-DD search itself
- [belief/playgen.md](belief/playgen.md) — the world model and its sidecar
- [ARCHITECTURE.md](ARCHITECTURE.md) — the rest of the crate
