#!/usr/bin/env python3
"""Launch belief net training races on RunPod GPU pods in parallel.

Usage:
    source .env
    uv run python scripts/runpod_race.py [--epochs 50] [--gpu RTX4090] [--dry-run]
    uv run python scripts/runpod_race.py --status        # check pod status
    uv run python scripts/runpod_race.py --stop-all      # terminate all race pods
"""

import argparse
import os
import time

import runpod

SCRIPT_URL = "https://github.com/Avo-k/colver/releases/download/train-v1/cloud_train.sh"
DOCKER_IMAGE = "nvidia/cuda:12.2.2-runtime-ubuntu22.04"

GPU_MAP = {
    "RTX3090": "NVIDIA GeForce RTX 3090",
    "RTX4090": "NVIDIA GeForce RTX 4090",
    "L4": "NVIDIA L4",
    "A4000": "NVIDIA RTX A4000",
}

RACES = {
    "baseline":   "--v2 --augment",
    "v3":         "--v3 --augment",
    "crossattn":  "--v2 --augment --variant cross_attn",
    "auxloss":    "--v2 --augment --variant aux_loss",
    "wide":       "--v2 --augment --variant var_mlp --num-layers 1 --hidden 768",
    "narrow":     "--v2 --augment --variant var_mlp --num-layers 3 --hidden 256",
    "suitshared": "--v2 --variant suit_shared",
    "countreg":   "--v2 --augment --count-reg 0.1",
}


def launch_races(args):
    gpu_type = GPU_MAP[args.gpu]

    print(f"GPU type:    {gpu_type}")
    print(f"Epochs:      {args.epochs}")
    print(f"Races:       {', '.join(args.races)}")
    print()

    if args.dry_run:
        for name in args.races:
            print(f"[DRY RUN] Would launch pod 'race-{name}' with: {RACES[name]}")
        return

    pod_ids = {}
    for name in args.races:
        env_vars = {
            "RACE_NAME": name,
            "RACE_ARGS": RACES[name],
            "EPOCHS": str(args.epochs),
            "BATCH_SIZE": str(args.batch_size),
            "LR": str(args.lr),
            "WARMUP": str(args.warmup_epochs),
            "SEED": str(args.seed),
        }

        print(f"Launching pod: race-{name} ...")
        try:
            pod = runpod.create_pod(
                name=f"race-{name}",
                image_name=DOCKER_IMAGE,
                gpu_type_id=gpu_type,
                gpu_count=1,
                container_disk_in_gb=10,
                min_memory_in_gb=64,
                env=env_vars,
                docker_args=f"bash -c 'curl -sL {SCRIPT_URL} | bash'",
            )
            pod_id = pod.get("id", "unknown")
            pod_ids[name] = pod_id
            print(f"  -> Pod {pod_id} launched")
        except Exception as e:
            print(f"  -> FAILED: {e}")

        time.sleep(2)

    print()
    print(f"=== {len(pod_ids)} pods launched ===")
    print("Monitor at: https://www.runpod.io/console/pods")
    print()
    for name, pid in pod_ids.items():
        print(f"  race-{name}: {pid}")


def check_status():
    pods = runpod.get_pods()
    race_pods = [p for p in pods if p["name"].startswith("race-")]
    if not race_pods:
        print("No race pods found.")
        return

    print(f"{'Name':<20} {'Status':<15} {'GPU':<20} {'Uptime'}")
    print("-" * 75)
    for p in race_pods:
        name = p["name"]
        status = p.get("desiredStatus", "?")
        runtime = p.get("runtime", {}) or {}
        gpu = runtime.get("gpus", [{}])[0].get("id", "?") if runtime.get("gpus") else "?"
        uptime = runtime.get("uptimeInSeconds", 0) if runtime else 0
        mins = uptime // 60
        print(f"{name:<20} {status:<15} {gpu:<20} {mins}m")


def stop_all():
    pods = runpod.get_pods()
    race_pods = [p for p in pods if p["name"].startswith("race-")]
    if not race_pods:
        print("No race pods found.")
        return

    for p in race_pods:
        pid = p["id"]
        name = p["name"]
        print(f"Terminating {name} ({pid})...")
        try:
            runpod.terminate_pod(pid)
            print(f"  -> terminated")
        except Exception as e:
            print(f"  -> error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Launch belief net races on RunPod")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="RTX4090", choices=GPU_MAP.keys())
    parser.add_argument("--races", nargs="*", default=list(RACES.keys()),
                        help="Which races to run (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true", help="Check pod status")
    parser.add_argument("--stop-all", action="store_true", help="Terminate all race pods")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set. Run: source .env")
        return
    runpod.api_key = api_key

    if args.status:
        check_status()
    elif args.stop_all:
        stop_all()
    else:
        launch_races(args)


if __name__ == "__main__":
    main()
