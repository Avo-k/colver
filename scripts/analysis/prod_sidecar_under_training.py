#!/usr/bin/env python3
"""Ce qu'un entraînement sur l'hôte de prod coûte à Dédé, en direct.

La 3090 de `moxxi` sert trois choses : le llama-server, le sidecar playgen des joueurs, et
— si on le décide — un bras d'entraînement v7. La question est de savoir ce que le
troisième prend aux deux autres.

## Ce qu'il ne faut PAS mesurer

Le temps d'un coup. Dédé tourne à `time_ms = 1200` **en mode temps** (la prod ne pose pas
`COLVER_ISDD_DETS`, donc `ISDD_DETS = 0`), et en mode temps la boucle d'IS-DD ne sort que
sur l'échéance. Un coup durera 1200 ms que le sidecar soit rapide ou lent : le chronomètre
est plat par construction. Ce qui se dégrade, c'est **le nombre de mondes qui tiennent dans
les 1200 ms**, donc la force de jeu — invisible pour qui chronomètre.

## Protocole

Blocs **alternés** entraînement OFF / ON, jamais l'un puis l'autre : c'est la règle du dépôt
(un même binaire varie de 20 % selon la charge). Chaque bloc enchaîne des décisions IS-DD au
**pli 1** — le pire cas, un monde y est une donne complète et le sidecar y est le plus
sollicité — par le vrai chemin de prod : `colver.Agent` sur `arena/bots/web_dede.toml`,
mondes pris au sidecar par le réseau, comme le conteneur web le fait depuis la VM.

Trois quantités par décision : les **mondes** traversés (la mesure), le temps écoulé (le
contrôle, qui doit rester plat), et la **provenance** des mondes.

⚠️ Cette dernière n'est pas décorative : la prod monte le sidecar en `fallback = "uniform"`,
donc un sidecar noyé ne rend pas une erreur, il rend des mondes uniformes. Un run où
`worlds.uniform > 0` ne mesure plus la même chose et s'arrête.

    uv run python scripts/analysis/prod_sidecar_under_training.py --blocks 6 --seconds 60
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import colver

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

# L'URL que le conteneur web utilise réellement (`.env` de la VM), pas un alias local :
# on veut le même saut réseau VM → hôte que la prod.
SIDECAR = "http://192.168.1.23:8003"
REMOTE = "moxxi"
REMOTE_REPO = "/home/claude/playgen/colver"

# Le sous-shell et la redirection **externe** ne sont pas cosmétiques : `A && B &`
# met tout l'enchaînement en arrière-plan, et ce sous-shell garde les descripteurs de
# la session SSH ouverts même quand B redirige les siens. ssh ne rend alors la main
# qu'au timeout, alors que la commande est bien partie.
TRAIN_CMD = (
    f"( cd {REMOTE_REPO} && setsid nohup "
    "./target/release/train_bid_nn --hidden 512 --layers 3 --num-envs 256 "
    "--pool-file data/deals/base_5M.bin --scores data/deals/scores_isdd_5M.sc "
    "--score-aware --match-sim --sa-features-v7 --canonical "
    "--steps 100000000 --save-freq 100000000 --eval-freq 100000000 "
    "--save-dir /tmp/v7_probe > /tmp/v7_probe.log 2>&1 < /dev/null & ) > /dev/null 2>&1"
)


def ssh(cmd: str, detach: bool = False) -> str:
    """`detach=True` pour lancer un processus qui doit survivre à la session SSH.

    Sans ça, `subprocess.run(capture_output=True)` attend l'EOF des tuyaux, et ssh ne
    ferme sa session que quand plus personne ne tient ses descripteurs — donc un
    `... &` distant fait bloquer l'appelant jusqu'au timeout, alors même que la commande
    est partie. Le remède est de ne pas capturer.
    """
    if detach:
        subprocess.run(
            ["ssh", "-n", REMOTE, cmd],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, timeout=120,
        )
        return ""
    return subprocess.run(
        ["ssh", "-n", REMOTE, cmd], capture_output=True, text=True, timeout=120
    ).stdout.strip()


def spec(time_ms: int) -> str:
    text = (ROOT / "arena/bots/web_dede.toml").read_text()
    text = text.replace("time_ms = 1000", f"time_ms = {time_ms}")
    # Le TOML du dépôt ne nomme pas d'URL (elle vient de l'environnement en prod) ;
    # on la pose explicitement pour viser l'hôte de prod et non un sidecar local.
    return text.replace('source = "sidecar"', f'source = "sidecar"\nurl = "{SIDECAR}"')


def block(cfg: str, seconds: float) -> dict:
    """Décisions IS-DD au pli 1 pendant `seconds`, par le chemin de prod."""
    worlds, elapsed, uniform = [], [], 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        env = colver.Env()
        env.reset()
        agents = [colver.Agent(cfg, s) for s in range(4)]
        for a in agents:
            a.init_deal(env)
        # Enchère au NN jusqu'à la phase de jeu ; on ne mesure que la carte.
        while not env.is_terminal() and env.phase() == 0:
            seat = env.current_player()
            act = agents[seat].decide(env)["action"]
            for a in agents:
                a.observe(env, act)
            env.step(act)
        if env.is_terminal() or env.phase() != 1:
            continue  # donne passée (4 passes)
        seat = env.current_player()
        out = agents[seat].decide(env)
        worlds.append(out["determinizations"])
        elapsed.append(out["elapsed_ms"])
        uniform += out["worlds"]["uniform"]
        if time.monotonic() >= deadline:
            break
    return {
        "n": len(worlds),
        "worlds_median": statistics.median(worlds) if worlds else 0,
        "worlds_mean": round(statistics.fmean(worlds), 1) if worlds else 0,
        "ms_median": round(statistics.median(elapsed), 1) if elapsed else 0,
        "uniform": uniform,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=6, help="nombre total, alternés OFF/ON")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--time-ms", type=int, default=1200, help="budget de Dédé en prod")
    ap.add_argument("--tag", default="dede_under_training")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    cfg = spec(args.time_ms)
    ssh("pkill -f train_bid_nn || true")
    time.sleep(2)

    print(f"sidecar {SIDECAR} · Dédé {args.time_ms} ms · {args.blocks} blocs de {args.seconds:.0f}s\n")
    print(f"{'bloc':>5} {'train':>6} {'n':>4} {'mondes méd.':>12} {'moy.':>7} {'ms méd.':>9}")

    rows = []
    training = False
    for i in range(args.blocks):
        want = i % 2 == 1  # bloc 0 = OFF, puis on alterne
        if want != training:
            if want:
                ssh(TRAIN_CMD, detach=True)
                time.sleep(20)  # chargement du pool + montée en régime
            else:
                ssh("pkill -f train_bid_nn || true")
                time.sleep(5)
            training = want
        r = block(cfg, args.seconds)
        r["block"] = i
        r["training"] = training
        rows.append(r)
        print(
            f"{i:>5} {'ON' if training else 'off':>6} {r['n']:>4} "
            f"{r['worlds_median']:>12.0f} {r['worlds_mean']:>7.1f} {r['ms_median']:>9.1f}"
        )
        if r["uniform"]:
            print(
                f"\n⚠️  {r['uniform']} mondes UNIFORMES au bloc {i} — le sidecar a été "
                "substitué en silence, la mesure ne porte plus sur lui. Arrêt.",
                file=sys.stderr,
            )
            ssh("pkill -f train_bid_nn || true")
            return 1

    ssh("pkill -f train_bid_nn || true")

    off = [r["worlds_mean"] for r in rows if not r["training"]]
    on = [r["worlds_mean"] for r in rows if r["training"]]
    ms_off = [r["ms_median"] for r in rows if not r["training"]]
    ms_on = [r["ms_median"] for r in rows if r["training"]]
    if off and on:
        ratio = statistics.fmean(on) / statistics.fmean(off)
        print(f"\nmondes par coup — off {statistics.fmean(off):.1f} · ON {statistics.fmean(on):.1f}")
        print(f"  → l'entraînement laisse **{ratio * 100:.0f} %** des mondes à Dédé")
        print(f"temps par coup — off {statistics.fmean(ms_off):.0f} ms · ON {statistics.fmean(ms_on):.0f} ms"
              "   (plat attendu : c'est une échéance)")

    if not args.no_log:
        import runlog

        runlog.save(
            script="prod_sidecar_under_training",
            tag=args.tag,
            params={"blocks": args.blocks, "seconds": args.seconds,
                    "time_ms": args.time_ms, "sidecar": SIDECAR},
            summary={
                "worlds_off": round(statistics.fmean(off), 1) if off else None,
                "worlds_on": round(statistics.fmean(on), 1) if on else None,
                "ratio": round(statistics.fmean(on) / statistics.fmean(off), 3)
                if off and on else None,
            },
            payload={"rows": rows},
            models=[],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
