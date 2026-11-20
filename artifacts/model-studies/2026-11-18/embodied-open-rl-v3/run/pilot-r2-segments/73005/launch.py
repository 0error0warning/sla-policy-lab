import gc
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import traceback
import zipfile

STAGE = "pilot"
INPUT_SHA256 = "13de4a016b830bc29d0effab25dc33247707be113ca328fcc9f32ec49985a246"
ARCHIVE = Path("/content/sla-open-rl-v3-pilot-r2-segments.zip")
ROOT = Path("/content/work_v3_pilot_r2_73005")
DEPS_OK = Path("/content/deps-installed.ok")

if "actor" in globals():
    del globals()["actor"]
gc.collect()
try:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass

assert sha256(ARCHIVE.read_bytes()).hexdigest() == INPUT_SHA256
ROOT.mkdir(exist_ok=False)
with zipfile.ZipFile(ARCHIVE) as z:
    assert all(
        not PurePosixPath(n).is_absolute()
        and ".." not in PurePosixPath(n).parts
        and "\\" not in n
        for n in z.namelist()
    )
    z.extractall(ROOT)

lock = json.loads((ROOT / "run-lock.json").read_bytes())
assert all(sha256((ROOT / n).read_bytes()).hexdigest() == v for n, v in lock["files"].items())

env = dict(
    os.environ,
    HF_HUB_DISABLE_PROGRESS_BARS="1",
    HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
    TOKENIZERS_PARALLELISM="false",
    PYTHONUNBUFFERED="1",
    OMP_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="1",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
)
receipt = {
    "stage": "dependencies",
    "study_stage": STAGE,
    "input_sha256": INPUT_SHA256,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


def update_receipt(stage_name):
    receipt["stage"] = stage_name
    (ROOT / "launch-receipt.json").write_text(json.dumps(receipt))


def archive_completed_lives():
    output = ROOT / "output"
    export = ROOT / "partial_exports"
    export.mkdir(exist_ok=True)
    for stamp in sorted(output.glob("*/result.json")):
        life = stamp.parent
        path = export / (life.name + ".zip")
        if path.exists():
            continue
        logs = [p for p in life.iterdir() if p.is_file()]
        logs += [output / n for n in ("probes-birth.json", "hidden-birth.npz", "unseen-probes.json")]
        manifest = {}
        temporary = path.with_suffix(".tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in logs:
                name = p.relative_to(ROOT).as_posix()
                data = p.read_bytes()
                manifest[name] = {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
                archive.writestr(name, data)
            archive.writestr("archive-manifest.json", json.dumps(manifest, sort_keys=True))
        temporary.replace(path)
        print("Completed life archive: " + str(path), flush=True)


update_receipt("dependencies")
code = 1
t0 = time.monotonic()
try:
    with (ROOT / "install.log").open("w") as log:
        if not DEPS_OK.exists():
            for args in (
                ["uninstall", "-y", "torchao"],
                ["install", "--disable-pip-version-check", "-r", "requirements-gpu.txt"],
            ):
                subprocess.run(
                    [sys.executable, "-m", "pip", *args],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=480,
                    check=True,
                )
            DEPS_OK.write_text("ok\n")
        else:
            log.write("Skipped pip install: /content/deps-installed.ok present.\n")

    update_receipt("worker")
    with (ROOT / "worker.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.embodied_open_rl_v3.gpu",
                "--root",
                str(ROOT),
                "--stage",
                STAGE,
                "--pilot-seeds",
                "73005",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 9000
        while True:
            archive_completed_lives()
            ret = proc.poll()
            if ret is not None:
                code = ret
                break
            if time.monotonic() > deadline:
                proc.kill()
                raise TimeoutError("pilot v3 worker exceeded 9000s")
            st_file = ROOT / "output" / "status.json"
            if st_file.exists():
                st = json.loads(st_file.read_text())
                brief = {
                    k: st.get(k)
                    for k in (
                        "stage",
                        "completed_jobs",
                        "total_jobs",
                        "job",
                        "arm",
                        "turn",
                        "phase",
                        "health",
                        "version",
                        "optimizer_steps",
                        "elapsed_seconds",
                    )
                    if k in st
                }
                print(f"[pilot-v3 t={int(time.monotonic()-t0)}s] " + json.dumps(brief), flush=True)
            else:
                print(f"[pilot-v3 t={int(time.monotonic()-t0)}s] waiting", flush=True)
            time.sleep(45)
except BaseException:
    (ROOT / "launch-failure.txt").write_text(traceback.format_exc())
finally:
    update_receipt("export")
    paths = [p for p in (ROOT / "output").rglob("*") if p.is_file()] + [
        p
        for p in ROOT.glob("*")
        if p.name
        in ("launch-receipt.json", "install.log", "worker.log", "launch-failure.txt", "run-lock.json")
    ]
    manifest = {
        p.relative_to(ROOT).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": sha256(p.read_bytes()).hexdigest(),
        }
        for p in paths
    }
    with zipfile.ZipFile(ROOT / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("archive-manifest.json", json.dumps(manifest, sort_keys=True))

    res_bytes = (ROOT / "results.zip").read_bytes()
    result = {
        "worker_returncode": code,
        "files": len(paths),
        "bytes": len(res_bytes),
        "sha256": sha256(res_bytes).hexdigest(),
        "study_stage": STAGE,
    }
    parts = []
    chunk_size = 5 * 1024 * 1024
    for i in range(0, max(1, len(res_bytes)), chunk_size):
        payload = res_bytes[i : i + chunk_size]
        name = f"part-{len(parts):03d}.bin"
        (ROOT / name).write_bytes(payload)
        parts.append(
            {"name": name, "bytes": len(payload), "sha256": sha256(payload).hexdigest()}
        )
    result["chunks"] = parts
    (ROOT / "export-receipt.json").write_text(json.dumps(result))
    print(json.dumps(result), flush=True)

if code:
    raise RuntimeError(f"Pilot v3 worker failed with code {code}; evidence preserved")
