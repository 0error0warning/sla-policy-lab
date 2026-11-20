"""Measure effective LoRA weight changes & cross-ecology divergence from saved tensors (v4)."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .protocol import CONFIG, PARTNERS_TRAIN, STAGES, canonical, write


def load_adapter(path, expected):
    data = path.read_bytes()
    assert len(data) == expected["file_bytes"]
    assert sha256(data).hexdigest() == expected["file_sha256"]
    header_end = 8 + int.from_bytes(data[:8], "little")
    header = json.loads(data[8:header_end])
    payload = memoryview(data)[header_end:]
    tensor_hash = sha256()
    tensors = {}
    for name, info in sorted(header.items()):
        if name == "__metadata__":
            continue
        assert info["dtype"] == "F32"
        start, end = info["data_offsets"]
        raw = payload[start:end]
        tensor = np.frombuffer(raw, dtype="<f4").reshape(info["shape"])
        assert np.isfinite(tensor).all()
        tensor_hash.update(canonical([name, "torch.float32", info["shape"]]))
        tensor_hash.update(raw)
        tensors[name] = tensor.astype(np.float64)
    assert tensor_hash.hexdigest() == expected["tensor_sha256"]
    pairs = {
        name: (value, tensors[name.replace(".lora_A.", ".lora_B.")])
        for name, value in tensors.items()
        if ".lora_A." in name
    }
    assert len(pairs) in (16, 36, 48) and len(tensors) == 2 * len(pairs)
    return pairs


def delta_from_birth(stage_pairs, birth_pairs):
    """Represent ΔW = (alpha/r)*(B A - B0 A0) exactly in rank <= 2r low-rank factors."""
    assert stage_pairs.keys() == birth_pairs.keys()
    delta = {}
    for name, (a_s, b_s) in stage_pairs.items():
        a_0, b_0 = birth_pairs[name]
        if np.allclose(b_0, 0.0, atol=0.0):
            delta[name] = (a_s, b_s)
        else:
            delta[name] = (np.vstack([a_s, a_0]), np.hstack([b_s, -b_0]))
    return delta


def effective_inner_product(left, right):
    assert left.keys() == right.keys()
    value = 0.0
    for name, (left_a, left_b) in left.items():
        right_a, right_b = right[name]
        value += float(np.sum((left_b.T @ right_b) * (left_a @ right_a.T)))
    return value * (CONFIG["lora_alpha"] / CONFIG["lora_rank"]) ** 2


def effective_distance(left, right):
    squared = (
        effective_inner_product(left, left)
        + effective_inner_product(right, right)
        - 2.0 * effective_inner_product(left, right)
    )
    return float(np.sqrt(max(0.0, squared)))


def effective_norm(delta):
    return float(np.sqrt(max(0.0, effective_inner_product(delta, delta))))


def effective_cosine(left_delta, right_delta):
    denom = effective_norm(left_delta) * effective_norm(right_delta)
    if denom <= 1e-12:
        return 0.0
    return float(effective_inner_product(left_delta, right_delta) / denom)


def compare_mirrored_ecologies(results_zip, directory, seeds, out):
    with ZipFile(results_zip) as archive:
        manifest = json.loads(archive.read("output/checkpoints-manifest.json"))
    birth_name = "initial-adapter.safetensors"
    birth = load_adapter(directory / birth_name, manifest[birth_name])
    per_seed = {}
    for s in seeds:
        seed = int(s)
        seed_out = {"arms": {}, "within_life_partner_divergence": {}, "cross_ecology": {}}
        deltas = {}
        for arm in ("body_rl_ab", "body_rl_ba"):
            life = f"{arm}_s{seed}"
            deltas[arm] = {}
            seed_out["arms"][arm] = {}
            for sp in PARTNERS_TRAIN:
                snaps = {}
                sp_deltas = {}
                for st in STAGES:
                    rel = f"{life}/adapter-{st}-{sp}.safetensors"
                    if rel in manifest and (directory / rel).exists():
                        snaps[st] = load_adapter(directory / rel, manifest[rel])
                        sp_deltas[st] = delta_from_birth(snaps[st], birth)
                deltas[arm][sp] = sp_deltas
                seed_out["arms"][arm][sp] = {
                    "effective_l2_from_birth": {
                        st: effective_norm(sp_deltas[st]) for st in sp_deltas
                    },
                    "effective_l2_between_stages": {
                        f"{b}_to_{a}": effective_distance(snaps[b], snaps[a])
                        for b, a in zip(STAGES, STAGES[1:])
                        if b in snaps and a in snaps
                    },
                }

        for st in STAGES:
            wl = {}
            for arm in ("body_rl_ab", "body_rl_ba"):
                if st in deltas[arm]["P_alpha"] and st in deltas[arm]["P_beta"]:
                    da = deltas[arm]["P_alpha"][st]
                    db = deltas[arm]["P_beta"][st]
                    wl[arm] = {
                        "effective_l2_alpha_vs_beta": effective_distance(da, db),
                        "delta_w_cosine_alpha_vs_beta": effective_cosine(da, db),
                    }
            if wl:
                seed_out["within_life_partner_divergence"][st] = wl

            if (
                st in deltas["body_rl_ab"]["P_alpha"]
                and st in deltas["body_rl_ab"]["P_beta"]
                and st in deltas["body_rl_ba"]["P_alpha"]
                and st in deltas["body_rl_ba"]["P_beta"]
            ):
                ab_a = deltas["body_rl_ab"]["P_alpha"][st]  # appeal_only
                ab_b = deltas["body_rl_ab"]["P_beta"][st]   # exchange_only
                ba_a = deltas["body_rl_ba"]["P_alpha"][st]  # exchange_only
                ba_b = deltas["body_rl_ba"]["P_beta"][st]   # appeal_only
                cos_same_role_app = effective_cosine(ab_a, ba_b)
                cos_same_role_exc = effective_cosine(ab_b, ba_a)
                cos_same_id_alpha = effective_cosine(ab_a, ba_a)
                cos_same_id_beta = effective_cosine(ab_b, ba_b)
                same_role_mean = 0.5 * (cos_same_role_app + cos_same_role_exc)
                same_id_opp_role_mean = 0.5 * (cos_same_id_alpha + cos_same_id_beta)
                seed_out["cross_ecology"][st] = {
                    "effective_l2_same_id_alpha": effective_distance(ab_a, ba_a),
                    "effective_l2_same_id_beta": effective_distance(ab_b, ba_b),
                    "effective_l2_same_role_appeal": effective_distance(ab_a, ba_b),
                    "effective_l2_same_role_exchange": effective_distance(ab_b, ba_a),
                    "cos_same_role_appeal": cos_same_role_app,
                    "cos_same_role_exchange": cos_same_role_exc,
                    "cos_same_id_opp_role_alpha": cos_same_id_alpha,
                    "cos_same_id_opp_role_beta": cos_same_id_beta,
                    "mean_same_role_cosine": same_role_mean,
                    "mean_same_id_opp_role_cosine": same_id_opp_role_mean,
                    "role_over_id_cosine_advantage": same_role_mean - same_id_opp_role_mean,
                }
        per_seed[str(seed)] = seed_out

    summary = {}
    for st in STAGES:
        rows = [per_seed[str(s)]["cross_ecology"][st] for s in seeds if st in per_seed[str(s)]["cross_ecology"]]
        if rows:
            summary[st] = {
                "mean_effective_l2_same_id": float(
                    np.mean([0.5 * (r["effective_l2_same_id_alpha"] + r["effective_l2_same_id_beta"]) for r in rows])
                ),
                "mean_effective_l2_same_role": float(
                    np.mean([0.5 * (r["effective_l2_same_role_appeal"] + r["effective_l2_same_role_exchange"]) for r in rows])
                ),
                "mean_same_role_cosine": float(np.mean([r["mean_same_role_cosine"] for r in rows])),
                "mean_same_id_opp_role_cosine": float(np.mean([r["mean_same_id_opp_role_cosine"] for r in rows])),
                "mean_role_over_id_cosine_advantage": float(
                    np.mean([r["role_over_id_cosine_advantage"] for r in rows])
                ),
            }
    out_obj = {"per_seed": per_seed, "mean_cross_ecology": summary}
    write(out, out_obj)
    return out_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-zip", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CONFIG["seeds"]))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare_mirrored_ecologies(args.results_zip, args.directory, args.seeds, args.out)
