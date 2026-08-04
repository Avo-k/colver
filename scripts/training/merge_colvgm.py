#!/usr/bin/env python3
"""Merge COLVGM game-replay files (concat records, rewrite the count header).

Usage: python scripts/training/merge_colvgm.py OUT.bin IN1.bin IN2.bin ...

Validates each input's magic and walks its variable-length records so a
truncated file is caught before it poisons the merged corpus.

Deux versions coexistent et **n'ont pas la même longueur d'enregistrement** :

    COLVGM01  dealer(1) hands(16)                  n(1) actions(n)
    COLVGM02  dealer(1) hands(16) score_ns/ew(4)   n(1) actions(n)

Concaténer les corps bruts de l'une sous le magic de l'autre produirait un
fichier qui se relit sans erreur et décale tout — le pire mode de défaillance
possible pour un corpus d'entraînement. La sortie est donc **toujours du
COLVGM02**, et les enregistrements v1 sont convertis en insérant un score 0-0.
Ce n'est pas une valeur par défaut arbitraire : un COLVGM01 vient forcément de
donnes tirées indépendamment, où 0-0 est la vérité.
"""
import struct
import sys

MAGIC_V1 = b"COLVGM01"
MAGIC_V2 = b"COLVGM02"
ZERO_SCORE = struct.pack("<HH", 0, 0)


def read_records(path):
    """Rend la liste des enregistrements, **au format v2**, ou sort en erreur."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 16:
        sys.exit(f"{path}: too small")
    magic = data[:8]
    if magic == MAGIC_V2:
        score_len = 4
    elif magic == MAGIC_V1:
        score_len = 0
    else:
        sys.exit(f"{path}: bad magic {magic!r}")

    n = struct.unpack("<Q", data[8:16])[0]
    pos = 16
    records = []
    for _ in range(n):
        head = pos + 17 + score_len  # dealer + hands + scores → octet du compte
        if head >= len(data):
            sys.exit(f"{path}: truncated at game boundary")
        num_actions = data[head]
        end = head + 1 + num_actions
        if end > len(data):
            sys.exit(f"{path}: truncated at game boundary")
        rec = data[pos:end]
        if score_len == 0:  # v1 → v2 : insérer un score 0-0 après les mains
            rec = rec[:17] + ZERO_SCORE + rec[17:]
        records.append(rec)
        pos = end
    if pos != len(data):
        sys.exit(f"{path}: {len(data) - pos} trailing bytes")
    print(f"{path}: {n} games OK ({magic.decode()})")
    return records


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_path, in_paths = sys.argv[1], sys.argv[2:]

    all_records = []
    for path in in_paths:
        all_records.extend(read_records(path))

    with open(out_path, "wb") as f:
        f.write(MAGIC_V2 + struct.pack("<Q", len(all_records)))
        for rec in all_records:
            f.write(rec)
    print(f"wrote {out_path}: {len(all_records)} games (COLVGM02)")


if __name__ == "__main__":
    main()
