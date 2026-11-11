"""Verify the published experiment without model downloads or GPU access."""

from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
import math
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import zipfile

from analyze import analyze
from validate import validate


def compare(expected, actual, path="aggregate"):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key, value in expected.items():
            compare(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-9), (path, expected, actual)
    else:
        assert expected == actual, (path, expected, actual)


def verify():
    source = Path(__file__).resolve().parent
    repo = source.parents[1]
    study = repo / "artifacts/model-studies/2026-11-11/closed-loop"
    manifest = json.loads((study / "MANIFEST.json").read_bytes())
    actual_files = {p.relative_to(study).as_posix() for p in study.rglob("*") if p.is_file()}
    assert actual_files == set(manifest["files"]) | {"MANIFEST.json"}
    for name, info in manifest["files"].items():
        data = (study / name).read_bytes()
        assert len(data) == info["bytes"] and sha256(data).hexdigest() == info["sha256"], name
    receipt = json.loads((study / "export-receipt.json").read_bytes())
    archive = study / "gpu-results.zip"
    assert receipt["worker_returncode"] == 0
    assert archive.stat().st_size == receipt["bytes"]
    assert sha256(archive.read_bytes()).hexdigest() == receipt["sha256"]
    mathematical_checks = validate()
    assert mathematical_checks["passed"]
    with TemporaryDirectory(prefix="sla-closed-loop-verify-") as directory:
        root = Path(directory).resolve()
        raw = root / "raw"
        raw.mkdir()
        with zipfile.ZipFile(archive) as zipped:
            assert zipped.testzip() is None
            files = json.loads(zipped.read("archive-manifest.json"))
            assert len(files) == receipt["files"]
            assert set(zipped.namelist()) == set(files) | {"archive-manifest.json"}
            for name, info in files.items():
                rel = PurePosixPath(name)
                target = raw.joinpath(*rel.parts).resolve()
                assert not rel.is_absolute() and ".." not in rel.parts and "\\" not in name
                assert target.is_relative_to(raw)
                payload = zipped.read(name)
                assert len(payload) == info["bytes"] and sha256(payload).hexdigest() == info["sha256"], name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            for name in ("core.py", "run_gpu.py", "validate.py", "PROTOCOL.md"):
                assert (source / name).read_bytes() == zipped.read(name), name
        with redirect_stdout(io.StringIO()):
            reconstructed = analyze(raw, root / "recomputed")
        published = json.loads((study / "scored.json").read_bytes())
        compare(published["aggregate"], reconstructed["aggregate"])
        assert reconstructed["all_events_verified"]
        assert sum(life["turns"] for life in reconstructed["results"]) == 2400
        assert reconstructed["status"]["optimizer_steps"] == 864
    release = json.loads((study / "resource-release.json").read_bytes())
    assert release["assignment_removed"] and release["active_assignments"] == 0
    assert release["consumption_rate_cu_per_hour"] == 0
    print(json.dumps(dict(passed=True, published_files=len(manifest["files"]),
        gpu_export_files=receipt["files"], replayed_records=2400,
        optimizer_steps=864, mathematical_checks_passed=True, gpu_access_used=False)))


if __name__ == "__main__":
    verify()
