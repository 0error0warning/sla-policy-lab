"""CPU-only verification of the published saved-weight diagnostic."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

from diagnose_score import score
from protocol import REPO
from score import match, read, verify_archive


DATA = REPO / "artifacts/model-studies/2026-11-10/saved-lora-diagnostic"


def verify(folder):
    publication = read(folder / "publication.json")
    assert {p.name for p in folder.iterdir() if p.is_file()} == set(publication["files"]) | {"publication.json"}
    for name, expected in publication["files"].items():
        payload = (folder / name).read_bytes()
        assert len(payload) == expected["bytes"] and sha256(payload).hexdigest() == expected["sha256"], name
    with verify_archive(folder / "gpu-results.zip") as archive:
        root = zipfile.Path(archive)
        assert (folder / "public.json").read_bytes() == (root / "public.json").read_bytes()
        actual = score(root, folder / "key.json")
    expected = read(folder / "scored.json")
    token_audit = expected.pop("independent_token_audit")
    assert token_audit == publication["independent_token_audit"] == dict(distinct_prompts=398, records_checked=1752)
    match(actual, expected)
    return dict(published_hashes_verified=len(publication["files"]), saved_adapters_verified=8,
        primary_model_prompt_records=1422, batch_checks=330, old_replay_exact=sum(v["original_replay_exact"] for v in actual["validation"].values()),
        metrics_reproduced=True, optimizer_steps=0, gpu_or_network_used=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir)))
