"""Recompute the published tiny fitting assay on CPU, without model downloads."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from fit_prepare import build
from fit_score import score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts/model-studies/2026-11-10/crossed-mini-fit"


def equal(expected, actual, path="root"):
    if isinstance(expected, dict):
        assert set(expected) == set(actual), path
        for key in expected:
            equal(expected[key],actual[key],path+"."+key)
    elif isinstance(expected,list):
        assert len(expected) == len(actual),path
        for i,(a,b) in enumerate(zip(expected,actual)):
            equal(a,b,f"{path}[{i}]")
    elif isinstance(expected,float):
        assert np.isclose(expected,actual,rtol=1e-11,atol=1e-12), (path,expected,actual)
    else:
        assert expected == actual,(path,expected,actual)


if __name__ == "__main__":
    manifest=json.loads((DATA/"publication.json").read_bytes())
    assert {p.name for p in DATA.iterdir() if p.is_file()} == set(manifest["files"]) | {"publication.json"}
    for name,expected in manifest["files"].items():
        raw=(DATA/name).read_bytes()
        assert len(raw)==expected["bytes"] and sha256(raw).hexdigest()==expected["sha256"],name
    # Rebuilding uses the local NumPy/libm; preserve the frozen bytes as the
    # execution authority and compare regenerated numeric targets tolerantly.
    equal(json.loads((DATA/"protocol.json").read_bytes()),build())
    expected=json.loads((DATA/"scored.json").read_bytes())
    actual=score(DATA/"gpu-results.zip",DATA/"protocol.json")
    optional=expected.pop("independent_token_audit",None)
    assert optional == manifest["independent_token_audit"]
    equal(expected,actual)
    print(json.dumps(dict(published_hashes_verified=len(manifest["files"]),
        result_metrics_reproduced=True,jobs=len(actual["jobs"]),optimizer_steps=actual["status"]["optimizer_steps"],
        completed_jobs=actual["status"]["completed_jobs"],gpu_or_network_used=False,
        independent_token_audit_was_recorded=optional is not None)))
