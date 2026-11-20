import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

INPUT_SHA256 = "4bc40d61cc2ac7f8a54a789cb72381dc19238e7a74a862284f4201da82a194a5"
ARCHIVE = Path("/content/sla-open-rl-v4-pilot-r7.zip")
DEPS_OK = Path("/content/deps-installed.ok")


def write_chunks(blob: bytes, root: Path, prefix: str, chunk_size: int = 4 * 1024 * 1024):
    parts = []
    for i in range(0, max(1, len(blob)), chunk_size):
        payload = blob[i : i + chunk_size]
        name = f"{prefix}-{len(parts):03d}.bin"
        (root / name).write_bytes(payload)
        parts.append({"name": name, "bytes": len(payload), "sha256": sha256(payload).hexdigest()})
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = int(args.seed)

    root = Path(f"/content/work_v4_pilot_r7_{seed}")
    if root.exists():
        shutil.rmtree(root)

    gc.collect()
    assert sha256(ARCHIVE.read_bytes()).hexdigest() == INPUT_SHA256
    root.mkdir(exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as z:
        assert all(
            not PurePosixPath(n).is_absolute()
            and ".." not in PurePosixPath(n).parts
            and "\\" not in n
            for n in z.namelist()
        )
        z.extractall(root)

    lock = json.loads((root / "run-lock.json").read_bytes())
    assert all(sha256((root / n).read_bytes()).hexdigest() == v for n, v in lock["files"].items())

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
        "study_stage": "pilot",
        "seed": seed,
        "input_sha256": INPUT_SHA256,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def update_receipt(stage_name):
        receipt["stage"] = stage_name
        (root / "launch-receipt.json").write_text(json.dumps(receipt))

    update_receipt("dependencies")
    code = 1
    t0 = time.monotonic()
    try:
        with (root / "install.log").open("w") as log:
            if not DEPS_OK.exists():
                for pip_args in (
                    ["uninstall", "-y", "torchao"],
                    ["install", "--disable-pip-version-check", "-r", "requirements-gpu.txt"],
                ):
                    subprocess.run(
                        [sys.executable, "-m", "pip", *pip_args],
                        cwd=root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                        check=True,
                    )
                DEPS_OK.write_text("ok\n")
            else:
                log.write("Skipped pip install: /content/deps-installed.ok present.\n")

        import torch

        gpu_name = torch.cuda.get_device_name(0)
        assert gpu_name == "NVIDIA L4", f"Expected NVIDIA L4 for segment parity, got {gpu_name}"
        print(f"[pilot-r7 s={seed}] verified GPU={gpu_name}", flush=True)

        update_receipt("worker")
        with (root / "worker.log").open("w") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.embodied_open_rl_v4.gpu",
                    "--root",
                    str(root),
                    "--stage",
                    "pilot",
                    "--pilot-seeds",
                    str(seed),
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 10800
            while True:
                ret = proc.poll()
                if ret is not None:
                    code = ret
                    break
                if time.monotonic() > deadline:
                    proc.kill()
                    raise TimeoutError(f"pilot-r7 seed {seed} exceeded 10800s")
                st_file = root / "output" / "status.json"
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
                            "health",
                            "version",
                            "optimizer_steps",
                            "elapsed_seconds",
                        )
                        if k in st
                    }
                    print(
                        f"[pilot-r7 s={seed} t={int(time.monotonic()-t0)}s] " + json.dumps(brief),
                        flush=True,
                    )
                else:
                    print(f"[pilot-r7 s={seed} t={int(time.monotonic()-t0)}s] waiting", flush=True)
                time.sleep(30)
    except BaseException:
        (root / "launch-failure.txt").write_text(traceback.format_exc())
    finally:
        update_receipt("export")
        paths = [p for p in (root / "output").rglob("*") if p.is_file()] + [
            p
            for p in root.glob("*")
            if p.name
            in ("launch-receipt.json", "install.log", "worker.log", "launch-failure.txt", "run-lock.json")
        ]
        manifest = {
            p.relative_to(root).as_posix(): {
                "bytes": p.stat().st_size,
                "sha256": sha256(p.read_bytes()).hexdigest(),
            }
            for p in paths
        }
        with zipfile.ZipFile(root / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in paths:
                z.write(p, p.relative_to(root).as_posix())
            z.writestr("archive-manifest.json", json.dumps(manifest, sort_keys=True))

        res_bytes = (root / "results.zip").read_bytes()
        res_chunks = write_chunks(res_bytes, root, "res-part")

        ckpt_root = root / "checkpoints"
        ckpt_paths = []
        if ckpt_root.exists():
            init_p = ckpt_root / "initial-adapter.safetensors"
            if init_p.exists():
                ckpt_paths.append(init_p)
            for arm_prefix, stages in (
                (f"body_rl_ab_s{seed}", ("early", "worn", "critical")),
                (f"body_rl_ba_s{seed}", ("early", "worn", "critical")),
                (f"pooled_body_rl_ab_s{seed}", ("early", "worn", "critical")),
                (f"task_only_rl_ab_s{seed}", ("critical",)),
            ):
                adir = ckpt_root / arm_prefix
                if adir.exists():
                    for st_n in stages:
                        for p in sorted(adir.glob(f"adapter-{st_n}-*.safetensors")):
                            ckpt_paths.append(p)

        with zipfile.ZipFile(root / "checkpoints.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in ckpt_paths:
                z.write(p, p.relative_to(ckpt_root).as_posix())
        ckpt_bytes = (root / "checkpoints.zip").read_bytes()
        ckpt_chunks = write_chunks(ckpt_bytes, root, "ckpt-part")

        result = {
            "worker_returncode": code,
            "seed": seed,
            "study_stage": "pilot",
            "results": {
                "files": len(paths),
                "bytes": len(res_bytes),
                "sha256": sha256(res_bytes).hexdigest(),
                "chunks": res_chunks,
            },
            "checkpoints": {
                "files": len(ckpt_paths),
                "bytes": len(ckpt_bytes),
                "sha256": sha256(ckpt_bytes).hexdigest(),
                "chunks": ckpt_chunks,
            },
        }
        (root / "export-receipt.json").write_text(json.dumps(result))
        print(json.dumps(result), flush=True)

    if code:
        raise RuntimeError(f"Pilot-r7 seed {seed} failed with code {code}; evidence preserved")


if __name__ == "__main__":
    main()
