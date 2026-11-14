"""Retrieve only the public pinned tokenizer metadata needed for CPU auditing."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


def download(inputs, out):
    with ZipFile(inputs) as archive:
        config = json.loads(archive.read("execution.json"))
        expected = {
            m: json.loads(archive.read(f"expected-{m}.json")) for m in config["models"]
        }
    files = []
    for model, spec in expected.items():
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            path = out / model / name
            truth = spec["metadata"][name]
            if (
                path.exists()
                and hashlib.sha256(path.read_bytes()).hexdigest() == truth["sha256"]
            ):
                files.append(str(path))
                continue
            url = (
                f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{name}"
            )
            with urlopen(url, timeout=90) as response:
                data = response.read()
            assert len(data) == truth["bytes"]
            assert hashlib.sha256(data).hexdigest() == truth["sha256"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files.append(str(path))
    return {"verified_metadata": files, "model_weights_downloaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(download(args.inputs, args.out), indent=2))
