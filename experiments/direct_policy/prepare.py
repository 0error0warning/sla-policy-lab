"""Prepare a new isolated run directory; no GPU or account access."""
from hashlib import sha256
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from protocol import CONFIG, MODEL, REVISION, write
from validate import validate


def prepare(destination,manifest):
    source=Path(__file__).resolve().parent;repo=source.parents[1]
    expected=json.loads(manifest.read_bytes())
    assert expected['id']==MODEL and expected['revision']==REVISION
    destination.mkdir(parents=True,exist_ok=False)
    for name in ['protocol.py','run_gpu.py','validate.py','PROTOCOL.md']:
        shutil.copyfile(source/name,destination/name)
    shutil.copyfile(repo/'experiments/social_closed_loop/requirements-gpu.txt',destination/'requirements-gpu.txt')
    write(destination/'protocol.json',CONFIG);write(destination/'expected-weights.json',expected)
    write(destination/'cpu-preflight.json',validate())
    write(destination/'run-lock.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        files={p.name:sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()},
        note='Source hashes identify actual code including uncommitted changes'))
    print(json.dumps(dict(prepared=str(destination.resolve()))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();prepare(a.destination,a.manifest)
