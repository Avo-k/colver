#!/usr/bin/env python3
"""Merge COLVGM01 game-replay files (concat records, rewrite the count header).

Usage: python scripts/training/merge_colvgm.py OUT.bin IN1.bin IN2.bin ...

Validates each input's magic and walks its variable-length records so a
truncated file is caught before it poisons the merged corpus.
"""
import struct
import sys


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_path, in_paths = sys.argv[1], sys.argv[2:]

    total = 0
    bodies = []
    for path in in_paths:
        with open(path, "rb") as f:
            data = f.read()
        if data[:8] != b"COLVGM01":
            sys.exit(f"{path}: bad magic")
        n = struct.unpack("<Q", data[8:16])[0]
        pos = 16
        for _ in range(n):
            num_actions = data[pos + 17]
            pos += 18 + num_actions
            if pos > len(data):
                sys.exit(f"{path}: truncated at game boundary")
        if pos != len(data):
            sys.exit(f"{path}: {len(data) - pos} trailing bytes")
        bodies.append(data[16:pos])
        total += n
        print(f"{path}: {n} games OK")

    with open(out_path, "wb") as f:
        f.write(b"COLVGM01" + struct.pack("<Q", total))
        for body in bodies:
            f.write(body)
    print(f"wrote {out_path}: {total} games")


if __name__ == "__main__":
    main()
