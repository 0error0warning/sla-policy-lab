"""Prepare a self-contained run folder without allocating GPU resources."""

from hashlib import sha256
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from core import CONFIG, MODEL, REVISION, write
from validate import validate


def prepare(destination, manifest):
    source = Path(__file__).resolve().parent
    repo = source.parents[1]
    expected = json.loads(manifest.read_bytes())
    assert expected["id"] == MODEL and expected["revision"] == REVISION
    destination.mkdir(parents=True,exist_ok=False)
    for name in ("core.py","run_gpu.py","validate.py","requirements-gpu.txt","PROTOCOL.md","ERRATA.md","README.md"):
        shutil.copyfile(source/name,destination/name)
    write(destination/"expected-weights.json",expected)
    write(destination/"protocol.json",CONFIG)
    write(destination/"preflight-validation.json",validate())
    write(destination/"run-lock.json",dict(
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
        files={p.name:sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()},
        note="File checksums identify the actual code, including uncommitted changes"))
    print(json.dumps(dict(prepared=str(destination.resolve()),protocol=CONFIG["schema"])))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("destination",type=Path)
    parser.add_argument("--manifest",type=Path,required=True)
    args=parser.parse_args()
    prepare(args.destination,args.manifest)
