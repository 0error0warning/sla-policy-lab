"""Recover the preregistered 48 factual experiences from PR #9's measured bytes."""

import argparse
import gzip
import io
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from experiments.first_fatal.design import BIRTHS, PILOT, REPLICATION


def build(parent, dest):
    pieces = {}
    sources = {}
    practice_answers = 0
    for stage, seeds in (("ffpilot", PILOT), ("ffconfirm", REPLICATION)):
        d = parent / "gpu" / stage
        archive = (d / "results.zip").read_bytes()
        rec = json.loads((d / "receipt.json").read_bytes())
        assert (
            sha256(archive).hexdigest() == rec["sha256"]
            and rec["worker_returncode"] == 0
        )
        sources[stage] = rec["sha256"]
        with ZipFile(io.BytesIO(archive)) as z:
            measured = json.loads(z.read("archive-manifest.json"))
            for seed in seeds:
                for birth in BIRTHS:
                    for name in ("events.jsonl.gz", "life.json"):
                        rel = f"s{seed}/{birth}/{name}"
                        key = f"output/lives/s{seed}/{birth}/true1/{name}"
                        raw = z.read(key)
                        assert sha256(raw).hexdigest() == measured[key]["sha256"]
                        pieces[rel] = raw
                        if name == "events.jsonl.gz":
                            ev = [
                                json.loads(x) for x in gzip.decompress(raw).splitlines()
                            ]
                            assert ev[-1]["after"] == 0
                            practice_answers += sum(
                                x["practice"] and x["action"] >= 2 for x in ev
                            )
    assert len(pieces) == 96 and practice_answers == 680
    meta = {
        "parent_archives": sources,
        "files": {k: sha256(v).hexdigest() for k, v in pieces.items()},
    }
    pieces["sources.json"] = json.dumps(meta, sort_keys=True).encode()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(dest, "w") as z:
        for name, raw in sorted(pieces.items()):
            z.writestr(ZipInfo(name), raw, compress_type=ZIP_DEFLATED)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("parent", type=Path)
    p.add_argument("destination", type=Path)
    v = p.parse_args()
    build(v.parent, v.destination)
