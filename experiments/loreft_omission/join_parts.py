"""Reconstruct a published split ZIP from independently hashed parts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from experiments.individual_life.release import join_parts

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    meta = json.loads(a.parts.read_bytes())
    join_parts(a.parts.parent, a.out, meta["parts"])
    with a.out.open("rb") as stream:
        h = sha256()
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    if h.hexdigest() != meta["sha256"]:
        raise ValueError("Archive hash mismatch")
    print(a.out)
