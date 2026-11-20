"""Real Qwen3.5-4B mid-to-upper LoRA open-generation RL worker (v4).

Implements mirrored partner ecologies (`ab` vs `ba`), partner-blind `pooled_body_rl_ab`
(`P_partner`), non-mortal `task_only_rl_ab`, un-hinted SYSTEM_PROMPT (`no sentence limits`),
pure bodily grounding (`reward_maintain_weight = 0.0`), and 10-turn stratified rollout windows.
"""

import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import re
import time

import numpy as np

from .protocol import (
    ARMS,
    CONFIG,
    LIFE_TASKS,
    MODEL,
    REVISION,
    STAGES,
    UNSEEN_PROBES,
    MortalWorld,
    analyze_utterance,
    build_messages,
    canonical,
    compute_grpo_group_advantages,
    evaluate_candidate_at_state,
    grpo_group_size_for_load,
    jobs,
    shown_partner,
    write,
)


def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(items):
    import torch

    h = sha256()
    for name, tensor in sorted(items):
        h.update(canonical([name, str(tensor.dtype), list(tensor.shape)]))
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def k3_kl(ref_logps, cur_logps):
    """Exact non-negative K3 KL divergence estimator: E[exp(ref - cur) - 1 - (ref - cur)]."""
    delta = ref_logps - cur_logps
    return (delta.exp() - 1.0 - delta).mean()


class Actor:
    def __init__(self, root, output, ckpt_root, progress):
        import torch
        from huggingface_hub import snapshot_download
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.torch = torch
        self.output = output
        self.ckpt_root = ckpt_root
        self.ckpt_manifest = {}
        self.progress = progress

        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")

        expected = json.loads((root / "expected-weights.json").read_bytes())
        assert expected["id"] == MODEL and expected["revision"] == REVISION

        progress(stage="download")
        snapshot = Path(
            snapshot_download(
                MODEL,
                revision=REVISION,
                token=False,
                allow_patterns=["*.safetensors", "*.json", "*.jinja", "vocab.*", "merges.*"],
            )
        )
        shards = {
            p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)}
            for p in snapshot.glob("*.safetensors")
        }
        assert shards == expected["files"], "immutable model bytes differ"
        write(output / "weights-manifest.json", expected)

        progress(stage="load")
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, token=False, trust_remote_code=False
        )
        self.tokenizer.padding_side = "left"

        base, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        assert not any(
            info.get(k)
            for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ), str(info)
        base.requires_grad_(False)
        base.eval()

        torch.manual_seed(CONFIG["adapter_seed"])
        lora_cfg = LoraConfig(
            r=CONFIG["lora_rank"],
            lora_alpha=CONFIG["lora_alpha"],
            lora_dropout=0.0,
            bias="none",
            target_modules=CONFIG["target_modules"],
        )
        self.model = get_peft_model(base, lora_cfg)
        self.model.eval()

        self.lora_params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        assert len(self.lora_params) == 72, f"Expected 72 LoRA tensors, got {len(self.lora_params)}"
        assert all(p.dtype == torch.float32 and "lora_" in n for n, p in self.lora_params.items())
        self.trainable_count = sum(p.numel() for p in self.lora_params.values())
        self.start_state = {n: p.detach().cpu().clone() for n, p in self.lora_params.items()}
        self.start_sha256 = tensor_hash(self.start_state.items())

        self.fixed_params = [
            (n, p) for n, p in self.model.named_parameters() if ".lora_" not in n
        ]
        self.fixed_versions = {n: p._version for n, p in self.fixed_params}
        self.base_hash = tensor_hash(self.fixed_params)

        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos)
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.eos_ids[0]
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.pad_id

        self.probe_batch_size, self.batch_vs_single_min_hidden_cos = (
            self._detect_safe_probe_batch_size()
        )
        self.save_checkpoint("initial-adapter.safetensors", self.start_state)
        progress(
            stage="ready",
            base_tensor_sha256=self.base_hash,
            adapter_initial_sha256=self.start_sha256,
            trainable_parameters=self.trainable_count,
            probe_batch_size=self.probe_batch_size,
            batch_vs_single_min_hidden_cos=round(self.batch_vs_single_min_hidden_cos, 8),
            eos_ids=self.eos_ids,
        )

    def _detect_safe_probe_batch_size(self):
        torch = self.torch
        test_probes_16 = UNSEEN_PROBES[:16]
        p_lists_16 = [
            self.encode_messages(build_messages(r["partner"], r["request"]))
            for r in test_probes_16
        ]
        h_singles_8 = [self._generate_greedy_single(p)[1].astype(np.float64) for p in p_lists_16[:8]]

        min_cos = 0.0
        for candidate_bs in (16, 8):
            sub = p_lists_16[:candidate_bs]
            width = max(len(p) for p in sub)
            ids = torch.full((len(sub), width), self.pad_id, device="cuda", dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, p in enumerate(sub):
                ids[i, width - len(p) :] = torch.tensor(p, device="cuda", dtype=torch.long)
                mask[i, width - len(p) :] = 1
            with torch.no_grad():
                out = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                h_batch = (
                    out.hidden_states[-1][:8, -1, :]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            cos_vals = []
            for hb, hs in zip(h_batch, h_singles_8):
                denom = max(1e-12, float(np.linalg.norm(hb) * np.linalg.norm(hs)))
                cos_vals.append(float(np.dot(hb, hs) / denom))
            min_cos = float(min(cos_vals))
            b1 = self._generate_greedy_batch(sub)
            b2 = self._generate_greedy_batch(sub)
            if min_cos > 0.999 and b1 == b2:
                return candidate_bs, min_cos
        return 1, min_cos

    def reset_adapter(self):
        self.install_state(self.start_state)
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

    def install_state(self, state):
        with self.torch.no_grad():
            for name, p in self.lora_params.items():
                p.grad = None
                p.copy_(state[name])

    def snapshot_state(self):
        return {n: p.detach().cpu().contiguous().clone() for n, p in self.lora_params.items()}

    def adapter_delta_l2(self, state=None):
        st = self.snapshot_state() if state is None else state
        sq = sum(((st[n] - self.start_state[n]).float() ** 2).sum().item() for n in st)
        return float(np.sqrt(sq))

    def per_layer_delta_l2(self, state=None):
        st = self.snapshot_state() if state is None else state
        layer_sq = {}
        module_sq = {
            "q_proj": 0.0,
            "k_proj": 0.0,
            "v_proj": 0.0,
            "o_proj": 0.0,
            "up_proj": 0.0,
            "down_proj": 0.0,
        }
        for name in st:
            diff_sq = float(((st[name] - self.start_state[name]).float() ** 2).sum().item())
            m_layer = re.search(r"layers\.(\d+)\.", name)
            if m_layer:
                l_key = f"layer_{m_layer.group(1)}"
                layer_sq[l_key] = layer_sq.get(l_key, 0.0) + diff_sq
            for mod in tuple(module_sq):
                if f".{mod}." in name:
                    module_sq[mod] += diff_sq
        sorted_layers = sorted(layer_sq.keys(), key=lambda k: int(k.split("_")[1]))
        return {
            "by_layer": {k: float(np.sqrt(layer_sq[k])) for k in sorted_layers},
            "by_module": {k: float(np.sqrt(v)) for k, v in module_sq.items() if v > 0.0 or k in ("up_proj", "down_proj")},
        }

    def save_checkpoint(self, rel_path, state):
        from safetensors.torch import save_file

        dest = self.ckpt_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_file(state, str(dest))
        self.ckpt_manifest[rel_path] = {
            "file_bytes": dest.stat().st_size,
            "file_sha256": file_hash(dest),
            "tensor_sha256": tensor_hash(state.items()),
            "delta_l2": self.adapter_delta_l2(state),
            "per_layer_delta_l2": self.per_layer_delta_l2(state),
        }
        return dest

    def verify_base(self):
        assert {n: p._version for n, p in self.fixed_params} == self.fixed_versions
        assert tensor_hash(self.fixed_params) == self.base_hash
        assert all(p.grad is None for _, p in self.fixed_params)

    def encode_messages(self, messages):
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert not prompt_str.endswith("<think>\n")
        ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        assert len(ids) >= 4
        return ids

    def _clean_response_ids(self, raw_ids):
        trimmed = []
        hit_eos = False
        for tok in raw_ids:
            t = int(tok)
            trimmed.append(t)
            if t in self.eos_ids:
                hit_eos = True
                break
        if not trimmed:
            trimmed = [self.eos_ids[0]]
            hit_eos = True
        return trimmed, hit_eos

    def decode_text(self, response_ids):
        stripped = [t for t in response_ids if t not in self.eos_ids and t != self.pad_id]
        return self.tokenizer.decode(stripped, skip_special_tokens=True).strip()

    def _sequence_token_logps(self, prompt_ids, response_ids):
        torch = self.torch
        full = torch.tensor([prompt_ids + response_ids], device="cuda", dtype=torch.long)
        out = self.model(input_ids=full, use_cache=False)
        p_len = len(prompt_ids)
        r_len = len(response_ids)
        logits = out.logits[0, p_len - 1 : p_len - 1 + r_len, :].float()
        targets = full[0, p_len : p_len + r_len]
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[torch.arange(r_len, device="cuda"), targets]

    def score_turn_logps(self, prompt_ids, response_ids):
        with self.torch.no_grad():
            old_tok_logps = self._sequence_token_logps(prompt_ids, response_ids).detach().cpu()
            with self.model.disable_adapter():
                ref_tok_logps = self._sequence_token_logps(prompt_ids, response_ids).detach().cpu()
        return {
            "old_token_logps": old_tok_logps,
            "ref_token_logps": ref_tok_logps,
            "old_mean_logp": float(old_tok_logps.mean().item()),
            "ref_mean_logp": float(ref_tok_logps.mean().item()),
        }

    def sample_window_batch(self, prompt_lists, rng_seed):
        from transformers import GenerationConfig

        torch = self.torch
        torch.manual_seed(int(rng_seed))
        torch.cuda.manual_seed(int(rng_seed))

        width = max(len(p) for p in prompt_lists)
        ids = torch.full((len(prompt_lists), width), self.pad_id, device="cuda", dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(prompt_lists):
            ids[i, width - len(p) :] = torch.tensor(p, device="cuda", dtype=torch.long)
            mask[i, width - len(p) :] = 1

        gen_cfg = GenerationConfig(
            do_sample=True,
            temperature=CONFIG["life_temperature"],
            top_p=CONFIG["life_top_p"],
            top_k=0,
            repetition_penalty=1.0,
            max_new_tokens=CONFIG["life_max_new_tokens"],
            eos_token_id=self.eos_ids,
            pad_token_id=self.pad_id,
            use_cache=True,
        )
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=ids, attention_mask=mask, generation_config=gen_cfg
            )
            out = []
            for i, p_ids in enumerate(prompt_lists):
                raw_resp = gen[i, width:].tolist()
                resp_ids, hit_eos = self._clean_response_ids(raw_resp)
                out.append(
                    {
                        "prompt_ids": list(p_ids),
                        "response_ids": resp_ids,
                        "hit_eos": bool(hit_eos),
                        "text": self.decode_text(resp_ids),
                    }
                )
        return out

    def _generate_greedy_single(self, prompt_ids):
        from transformers import GenerationConfig

        torch = self.torch
        inp = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
        mask = torch.ones_like(inp)
        gen_cfg = GenerationConfig(
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_new_tokens=CONFIG["probe_max_new_tokens"],
            eos_token_id=self.eos_ids,
            pad_token_id=self.pad_id,
            use_cache=True,
        )
        with torch.no_grad():
            out = self.model(
                input_ids=inp, attention_mask=mask, output_hidden_states=True, use_cache=False
            )
            last_hidden = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
            gen = self.model.generate(
                input_ids=inp, attention_mask=mask, generation_config=gen_cfg
            )
            raw_resp = gen[0, len(prompt_ids) :].tolist()
        resp, _ = self._clean_response_ids(raw_resp)
        return resp, last_hidden

    def _generate_greedy_batch(self, prompt_lists):
        from transformers import GenerationConfig

        torch = self.torch
        width = max(len(p) for p in prompt_lists)
        ids = torch.full((len(prompt_lists), width), self.pad_id, device="cuda", dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(prompt_lists):
            ids[i, width - len(p) :] = torch.tensor(p, device="cuda", dtype=torch.long)
            mask[i, width - len(p) :] = 1
        gen_cfg = GenerationConfig(
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_new_tokens=CONFIG["probe_max_new_tokens"],
            eos_token_id=self.eos_ids,
            pad_token_id=self.pad_id,
            use_cache=True,
        )
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=ids, attention_mask=mask, generation_config=gen_cfg
            )
            out_lists = []
            for i in range(len(prompt_lists)):
                resp_ids, _ = self._clean_response_ids(gen[i, width:].tolist())
                out_lists.append(resp_ids)
        return out_lists

    def eval_probes(self, probe_rows, encoded_probes, extract_hidden=True):
        torch = self.torch
        results = []
        hiddens = []
        bs = self.probe_batch_size
        if bs == 1:
            for r, p_ids in zip(probe_rows, encoded_probes):
                resp_ids, hidden = self._generate_greedy_single(p_ids)
                text = self.decode_text(resp_ids)
                results.append(
                    {
                        "id": r["id"],
                        "battery": r["battery"],
                        "load": r["load"],
                        "partner": r["partner"],
                        "request": r["request"],
                        "output_ids": resp_ids,
                        "hit_eos": bool(resp_ids[-1] in self.eos_ids),
                        "text": text,
                        "analysis": analyze_utterance(text, r),
                    }
                )
                if extract_hidden:
                    hiddens.append(hidden)
        else:
            for start in range(0, len(probe_rows), bs):
                chunk_rows = probe_rows[start : start + bs]
                chunk_ids = encoded_probes[start : start + bs]
                width = max(len(p) for p in chunk_ids)
                ids = torch.full((len(chunk_ids), width), self.pad_id, device="cuda", dtype=torch.long)
                mask = torch.zeros_like(ids)
                for i, p in enumerate(chunk_ids):
                    ids[i, width - len(p) :] = torch.tensor(p, device="cuda", dtype=torch.long)
                    mask[i, width - len(p) :] = 1
                if extract_hidden:
                    with torch.no_grad():
                        out = self.model(
                            input_ids=ids,
                            attention_mask=mask,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                        h_batch = out.hidden_states[-1][:, -1, :].detach().float().cpu().numpy()
                        for row_h in h_batch:
                            hiddens.append(row_h)
                batch_resp = self._generate_greedy_batch(chunk_ids)
                for r, resp_ids in zip(chunk_rows, batch_resp):
                    text = self.decode_text(resp_ids)
                    results.append(
                        {
                            "id": r["id"],
                            "battery": r["battery"],
                            "load": r["load"],
                            "partner": r["partner"],
                            "request": r["request"],
                            "output_ids": resp_ids,
                            "hit_eos": bool(resp_ids[-1] in self.eos_ids),
                            "text": text,
                            "analysis": analyze_utterance(text, r),
                        }
                    )
        hidden_mat = np.stack(hiddens, axis=0).astype(np.float16) if extract_hidden else None
        return results, hidden_mat

    def update_window(self, optimizer, buffer, advantages, progress, kl_weight=None):
        torch = self.torch
        kl_w = CONFIG["kl_weight"] if kl_weight is None else float(kl_weight)
        steps_log = []
        n = len(buffer)
        assert n > 0
        for epoch in range(CONFIG["epochs_per_window"]):
            optimizer.zero_grad(set_to_none=True)
            total_pg = 0.0
            total_kl = 0.0
            total_loss = 0.0
            ratios = []
            clip_flags = []
            for item, adv_val in zip(buffer, advantages):
                cur_logps = self._sequence_token_logps(item["prompt_ids"], item["response_ids"])
                cur_mean = cur_logps.mean()
                old_mean = torch.tensor(item["old_mean_logp"], device="cuda", dtype=torch.float32)
                ref_logps = item["ref_token_logps"].to(device="cuda", dtype=torch.float32)

                ratio = torch.exp(cur_mean - old_mean)
                adv_t = torch.tensor(float(adv_val), device="cuda", dtype=torch.float32)
                clipped = torch.clamp(ratio, 1.0 - CONFIG["ppo_clip"], 1.0 + CONFIG["ppo_clip"])
                pg = -torch.minimum(ratio * adv_t, clipped * adv_t)
                kl = k3_kl(ref_logps, cur_logps)
                seq_loss = (pg + kl_w * kl) / float(n)
                assert torch.isfinite(seq_loss)
                seq_loss.backward()

                total_pg += float(pg.detach().item()) / float(n)
                total_kl += float(kl.detach().item()) / float(n)
                total_loss += float(seq_loss.detach().item())
                r_val = float(ratio.detach().item())
                ratios.append(r_val)
                clip_flags.append(int(abs(r_val - 1.0) > CONFIG["ppo_clip"]))
                del cur_logps, cur_mean, ref_logps, ratio, clipped, pg, kl, seq_loss

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.lora_params.values(), CONFIG["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps_log.append(
                {
                    "epoch": epoch,
                    "loss": total_loss,
                    "pg_loss": total_pg,
                    "kl_loss": total_kl,
                    "grad_norm": float(grad_norm),
                    "ratios": ratios,
                    "clip_fraction": float(np.mean(clip_flags)),
                }
            )
            progress(add_steps=1)

        with torch.no_grad():
            new_means = [
                float(self._sequence_token_logps(it["prompt_ids"], it["response_ids"]).mean().item())
                for it in buffer
            ]
        return {
            "steps": steps_log,
            "old_mean_logps": [it["old_mean_logp"] for it in buffer],
            "new_mean_logps": new_means,
            "mean_logp_deltas": [new - it["old_mean_logp"] for new, it in zip(new_means, buffer)],
        }


def run_life(actor, job, output, encoded_probes, progress):
    arm_name = job["arm"]
    arm = ARMS[arm_name]
    actor.reset_adapter()
    dest = output / job["id"]
    dest.mkdir(parents=False)

    opt = (
        actor.torch.optim.AdamW(
            actor.lora_params.values(),
            lr=CONFIG["learning_rate"],
            betas=(CONFIG["adam_beta1"], CONFIG["adam_beta2"]),
            weight_decay=0.0,
        )
        if arm["update"]
        else None
    )
    world = MortalWorld(job["seed"], arm_name, job["id"])

    ckpt_states = {}
    ckpt_meta = {}
    last_live_snapshot = actor.snapshot_state()
    last_live_meta = {"turn": 0, "health": world.health, "version": 0, "delta_l2": 0.0}
    updates = []
    version = 0

    win_size = CONFIG["window_size"]
    progress(job=job["id"], turn=0, arm=arm_name)

    for start_idx in range(0, len(world.items), win_size):
        if world.done:
            break
        win_items = world.items[start_idx : start_idx + win_size]
        p_lists = [
            actor.encode_messages(build_messages(shown_partner(it["partner"], arm_name), it["request"]))
            for it in win_items
        ]
        first_turn = int(win_items[0]["turn"])

        # k=0 real executed rollout uses identical batch size & RNG across all arms (preserves Window-1 parity)
        k0_seed = int.from_bytes(
            sha256(canonical([int(job["seed"]), first_turn, "sample_k0"])).digest()[:8],
            "little",
        )
        gens_k0 = actor.sample_window_batch(p_lists, k0_seed)
        per_turn_gens = [[g] for g in gens_k0]

        if arm["update"]:
            max_g = max(grpo_group_size_for_load(it["load"]) for it in win_items)
            for k in range(1, max_g):
                sub_idxs = [
                    idx for idx, it in enumerate(win_items) if k < grpo_group_size_for_load(it["load"])
                ]
                if not sub_idxs:
                    continue
                k_seed = int.from_bytes(
                    sha256(canonical([int(job["seed"]), first_turn, f"sample_k{k}"])).digest()[:8],
                    "little",
                )
                k_gens = actor.sample_window_batch([p_lists[idx] for idx in sub_idxs], k_seed)
                for idx, g in zip(sub_idxs, k_gens):
                    per_turn_gens[idx].append(g)

        buffer = []
        for item, cand_gens in zip(win_items, per_turn_gens):
            if world.done:
                break
            h_before = world.health
            gen0 = cand_gens[0]
            lp0 = actor.score_turn_logps(gen0["prompt_ids"], gen0["response_ids"])
            rec = world.step(gen0["text"])
            rec["policy_version_before"] = version
            rec["response_token_count"] = len(gen0["response_ids"])
            rec["hit_eos"] = gen0["hit_eos"]
            rec["old_mean_logp"] = lp0["old_mean_logp"]
            rec["ref_mean_logp"] = lp0["ref_mean_logp"]
            rec["grpo_group_size"] = len(cand_gens)
            with (dest / "trace.jsonl").open("ab") as fh:
                fh.write(canonical(rec))

            if arm["update"]:
                task_spec = LIFE_TASKS[item["task_idx"]]
                group_cands = []
                for k_idx, g in enumerate(cand_gens):
                    if k_idx == 0:
                        lp_info = lp0
                        settled = {
                            "response_text": rec["response_text"],
                            "analysis": rec["analysis"],
                            "partner_accept_prob": rec["partner_accept_prob"],
                            "partner_accepted": rec["partner_accepted"],
                            "health_before": rec["health_before"],
                            "health_after": rec["health_after"],
                            "alive_after": rec["alive_after"],
                            "body_parts": rec["body_parts"],
                            "reward_body": rec["reward_body"],
                            "reward_task": rec["reward_task"],
                            "reward_parts": rec["reward_parts"],
                        }
                    else:
                        lp_info = actor.score_turn_logps(g["prompt_ids"], g["response_ids"])
                        settled = evaluate_candidate_at_state(
                            h_before,
                            task_spec,
                            item["partner"],
                            arm["ecology"],
                            g["text"],
                            rec["partner_u"],
                            rec["noise_eta"],
                        )
                    group_cands.append(
                        {
                            "candidate_idx": k_idx,
                            "executed": bool(k_idx == 0),
                            "response_token_count": len(g["response_ids"]),
                            "hit_eos": g["hit_eos"],
                            "old_mean_logp": lp_info["old_mean_logp"],
                            "ref_mean_logp": lp_info["ref_mean_logp"],
                            **settled,
                        }
                    )
                    buffer.append(
                        {
                            "turn": rec["turn"],
                            "group_id": rec["turn"],
                            "candidate_idx": k_idx,
                            "load": rec["load"],
                            "shown_partner": rec["shown_partner"],
                            "prompt_ids": g["prompt_ids"],
                            "response_ids": g["response_ids"],
                            "old_mean_logp": lp_info["old_mean_logp"],
                            "ref_token_logps": lp_info["ref_token_logps"],
                            "reward_body": settled["reward_body"],
                            "reward_task": settled["reward_task"],
                        }
                    )
                with (dest / "group_rollouts.jsonl").open("ab") as fh:
                    fh.write(
                        canonical(
                            {
                                "turn": rec["turn"],
                                "window_index": rec["window_index"],
                                "load": rec["load"],
                                "partner": rec["partner"],
                                "shown_partner": rec["shown_partner"],
                                "partner_role": rec["partner_role"],
                                "ecology": rec["ecology"],
                                "health_before": h_before,
                                "partner_u": rec["partner_u"],
                                "noise_eta": rec["noise_eta"],
                                "candidates": group_cands,
                            }
                        )
                    )

        if buffer and arm["update"]:
            rewards = [b["reward_body"] if arm["body_reward"] else b["reward_task"] for b in buffer]
            group_ids = [b["group_id"] for b in buffer]
            loads = [b["load"] for b in buffer]
            spartners = [b["shown_partner"] for b in buffer]
            advs, group_stats = compute_grpo_group_advantages(rewards, group_ids)

            upd = actor.update_window(opt, buffer, advs.tolist(), progress)
            version += 1
            snap = actor.snapshot_state()
            delta_l2 = actor.adapter_delta_l2(snap)
            u_rec = {
                "version": version,
                "turns": [b["turn"] for b in buffer],
                "group_ids": group_ids,
                "candidate_indices": [b["candidate_idx"] for b in buffer],
                "loads": loads,
                "shown_partners": spartners,
                "health_after_window": world.health,
                "alive_after_window": bool(not world.done),
                "rewards_used": rewards,
                "group_stats": group_stats,
                "raw_advantages": advs.tolist(),
                "adapter_delta_l2": delta_l2,
                **upd,
            }
            updates.append(u_rec)
            with (dest / "updates.jsonl").open("ab") as fh:
                fh.write(canonical(u_rec))

            if not world.done:
                last_live_snapshot = snap
                last_live_meta = {
                    "turn": world.turn,
                    "health": world.health,
                    "version": version,
                    "delta_l2": delta_l2,
                }
                if "early" not in ckpt_states and world.turn == CONFIG["early_turn"]:
                    ckpt_states["early"] = snap
                    ckpt_meta["early"] = dict(
                        last_live_meta, trigger=f"turn=={int(CONFIG['early_turn'])}"
                    )
                if "worn" not in ckpt_states and (
                    world.health <= CONFIG["worn_threshold"]
                    or world.turn >= CONFIG["worn_fallback_turn"]
                ):
                    ckpt_states["worn"] = snap
                    ckpt_meta["worn"] = dict(
                        last_live_meta,
                        trigger=(
                            f"health<={int(CONFIG['worn_threshold'])}"
                            if world.health <= CONFIG["worn_threshold"]
                            else f"turn>={int(CONFIG['worn_fallback_turn'])}_fallback"
                        ),
                    )
                if world.health <= CONFIG["critical_threshold"]:
                    ckpt_states["critical"] = snap
                    ckpt_meta["critical"] = dict(
                        last_live_meta, trigger=f"health<={int(CONFIG['critical_threshold'])}"
                    )

        progress(
            job=job["id"],
            turn=world.turn,
            health=round(world.health, 2),
            version=version,
        )

    final_delta_l2 = actor.adapter_delta_l2()
    probe_outputs = {}
    if arm["update"]:
        for st_name in STAGES:
            if st_name not in ckpt_states:
                ckpt_states[st_name] = last_live_snapshot
                ckpt_meta[st_name] = dict(last_live_meta, trigger=f"fallback_{st_name}_preterminal")

        actor.save_checkpoint(f"{job['id']}/adapter-terminal.safetensors", actor.snapshot_state())

        eval_cache = {}
        for stage_name in STAGES:
            snap = ckpt_states[stage_name]
            snap_sha = tensor_hash(snap.items())
            actor.save_checkpoint(f"{job['id']}/adapter-{stage_name}.safetensors", snap)
            if snap_sha not in eval_cache:
                actor.install_state(snap)
                res_rows, h_mat = actor.eval_probes(
                    UNSEEN_PROBES, encoded_probes, extract_hidden=True
                )
                eval_cache[snap_sha] = (res_rows, h_mat)
            cached_rows, cached_hidden = eval_cache[snap_sha]
            probe_outputs[stage_name] = {
                "meta": ckpt_meta[stage_name],
                "adapter_sha256": snap_sha,
                "delta_l2": actor.adapter_delta_l2(snap),
                "per_layer_delta_l2": actor.per_layer_delta_l2(snap),
                "rows": [dict(r) for r in cached_rows],
            }
            write(dest / f"probes-{stage_name}.json", probe_outputs[stage_name])
            np.savez_compressed(dest / f"hidden-{stage_name}.npz", hidden=cached_hidden.copy())

    result = {
        "id": job["id"],
        "seed": job["seed"],
        "arm": arm_name,
        "arm_config": arm,
        "turns": world.turn,
        "final_health": world.health,
        "physical_death": bool(world.health <= 0.0),
        "capped_alive": bool(world.health > 0.0),
        "total_updates": len(updates),
        "final_delta_l2": final_delta_l2,
        "checkpoint_meta": ckpt_meta,
    }
    write(dest / "result.json", result)
    progress(complete_job=True)
    return {**result, "probe_outputs": probe_outputs}


def verify_window1_parity(output, arm_names, seeds):
    w_size = int(CONFIG["window_size"])
    unmasked = [a for a in arm_names if not ARMS[a].get("mask_partner", False)]
    ab_unmasked = [a for a in unmasked if ARMS[a]["ecology"] == "ab"]
    text_exact = True
    ab_exact = True

    for s in seeds:
        traces = {}
        for a in arm_names:
            tpath = output / f"{a}_s{int(s)}" / "trace.jsonl"
            if not tpath.exists():
                return False, False
            traces[a] = [
                json.loads(ln)
                for ln in tpath.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ][:w_size]

        if len(unmasked) >= 2:
            ref_txt = [r["response_text"] for r in traces[unmasked[0]]]
            for a in unmasked[1:]:
                if [r["response_text"] for r in traces[a]] != ref_txt:
                    text_exact = False
        if len(ab_unmasked) >= 2:
            ref_ab = [
                (r["turn"], r["response_text"], round(r["health_after"], 10), r["partner_accepted"])
                for r in traces[ab_unmasked[0]]
            ]
            for a in ab_unmasked[1:]:
                cur_ab = [
                    (r["turn"], r["response_text"], round(r["health_after"], 10), r["partner_accepted"])
                    for r in traces[a]
                ]
                if cur_ab != ref_ab:
                    ab_exact = False
    return text_exact, ab_exact


def summarize_stage_probes(rows):
    bats = {}
    for bat_name in (
        "D1_effort_allocation",
        "D2_commitment_renegotiation",
        "D3_mirrored_partner_pairs",
        "D4_factual_guardrail",
    ):
        sub = [r for r in rows if r["battery"] == bat_name]
        ans = [r["analysis"] for r in sub]
        fact = [int(bool(a["factual_correct"])) for a in ans if a["factual_correct"] is not None]
        bats[bat_name] = {
            "mean_word_count": round(float(np.mean([a["word_count"] for a in ans])), 2),
            "mean_fulfillment": round(float(np.mean([a["fulfillment_score"] for a in ans])), 4),
            "pure_appeal_rate": round(float(np.mean([int(a["pure_appeal"]) for a in ans])), 4),
            "pure_exchange_rate": round(float(np.mean([int(a["pure_exchange"]) for a in ans])), 4),
            "mixed_stance_rate": round(float(np.mean([int(a["mixed_stance"]) for a in ans])), 4),
            "gratuitous_pragmatics_rate": round(
                float(np.mean([int(a["gratuitous_pragmatics"]) for a in ans])), 4
            ),
            "refusal_rate": round(float(np.mean([int(a["is_refusal"]) for a in ans])), 4),
            "eos_hit_rate": round(float(np.mean([int(r["hit_eos"]) for r in sub])), 4),
            "factual_accuracy": round(float(np.mean(fact)), 4) if fact else None,
        }

    d3_alpha = [r["analysis"] for r in rows if r["battery"] == "D3_mirrored_partner_pairs" and r["partner"] == "P_alpha"]
    d3_beta = [r["analysis"] for r in rows if r["battery"] == "D3_mirrored_partner_pairs" and r["partner"] == "P_beta"]
    a_net = float(np.mean([int(a["pure_appeal"]) - int(a["pure_exchange"]) for a in d3_alpha]))
    b_net = float(np.mean([int(a["pure_appeal"]) - int(a["pure_exchange"]) for a in d3_beta]))
    bats["D3_partner_stance"] = {
        "P_alpha_pure_appeal": round(float(np.mean([int(a["pure_appeal"]) for a in d3_alpha])), 4),
        "P_alpha_pure_exchange": round(float(np.mean([int(a["pure_exchange"]) for a in d3_alpha])), 4),
        "P_beta_pure_appeal": round(float(np.mean([int(a["pure_appeal"]) for a in d3_beta])), 4),
        "P_beta_pure_exchange": round(float(np.mean([int(a["pure_exchange"]) for a in d3_beta])), 4),
        "D_ab": round(0.5 * (a_net - b_net), 4),
    }

    heavy_sub = [r["analysis"] for r in rows if r["load"] == "heavy"]
    bats["HEAVY_MEAN_D1_D3"] = {
        "mean_word_count": round(float(np.mean([a["word_count"] for a in heavy_sub])), 2),
        "mean_fulfillment": round(float(np.mean([a["fulfillment_score"] for a in heavy_sub])), 4),
        "pure_appeal_rate": round(float(np.mean([int(a["pure_appeal"]) for a in heavy_sub])), 4),
        "pure_exchange_rate": round(float(np.mean([int(a["pure_exchange"]) for a in heavy_sub])), 4),
        "empty_promise_rate": round(float(np.mean([int(a.get("is_empty_promise", False)) for a in heavy_sub])), 4),
        "refusal_rate": round(float(np.mean([int(a["is_refusal"]) for a in heavy_sub])), 4),
    }
    return bats


def run_smoke(actor, encoded_probes, output, progress):
    from safetensors.torch import load_file

    actor.reset_adapter()
    sample_prompt = encoded_probes[0]
    g0 = actor.sample_window_batch([sample_prompt], rng_seed=99901)[0]
    lp0 = actor.score_turn_logps(g0["prompt_ids"], g0["response_ids"])
    disable_diff = float((lp0["old_token_logps"] - lp0["ref_token_logps"]).abs().max().item())
    assert disable_diff < 1e-4, f"disable_adapter differs from birth: {disable_diff}"

    d4_probes = [r for r in UNSEEN_PROBES if r["battery"] == "D4_factual_guardrail"]
    d4_enc = [encoded_probes[UNSEEN_PROBES.index(r)] for r in d4_probes]
    d4_res, _ = actor.eval_probes(d4_probes, d4_enc, extract_hidden=False)
    d4_correct = sum(int(bool(x["analysis"]["factual_correct"])) for x in d4_res)

    turn_item = {
        "turn": 1,
        "prompt_ids": g0["prompt_ids"],
        "response_ids": g0["response_ids"],
        "old_mean_logp": lp0["old_mean_logp"],
        "ref_token_logps": lp0["ref_token_logps"],
    }
    dir_checks = {}
    for sig, adv_val in (("pos", 1.5), ("neg", -1.5), ("zero", 0.0)):
        actor.reset_adapter()
        opt = actor.torch.optim.AdamW(
            actor.lora_params.values(),
            lr=CONFIG["learning_rate"],
            betas=(CONFIG["adam_beta1"], CONFIG["adam_beta2"]),
            weight_decay=0.0,
        )
        upd = actor.update_window(opt, [turn_item], [adv_val], progress, kl_weight=0.0)
        delta_lp = upd["mean_logp_deltas"][0]
        delta_w = actor.adapter_delta_l2()
        if sig == "pos":
            assert delta_lp > 0 and delta_w > 0
        elif sig == "neg":
            assert delta_lp < 0 and delta_w > 0
        else:
            assert abs(delta_lp) < 1e-5 and delta_w == 0.0
        if sig == "pos":
            snap = actor.snapshot_state()
            ckpt_path = actor.save_checkpoint("smoke-pos.safetensors", snap)
            res_before, _ = actor.eval_probes(d4_probes[:8], d4_enc[:8], extract_hidden=False)
            actor.reset_adapter()
            actor.install_state(load_file(str(ckpt_path)))
            res_after, _ = actor.eval_probes(d4_probes[:8], d4_enc[:8], extract_hidden=False)
            assert [r["output_ids"] for r in res_before] == [r["output_ids"] for r in res_after]
        dir_checks[sig] = {
            "delta_mean_logp": delta_lp,
            "delta_l2": delta_w,
            "grad_norm": upd["steps"][0]["grad_norm"],
        }

    actor.reset_adapter()
    birth_rows, birth_hidden = actor.eval_probes(
        UNSEEN_PROBES, encoded_probes, extract_hidden=True
    )
    write(
        output / "probes-birth.json",
        {"stage": "birth", "adapter_sha256": actor.start_sha256, "delta_l2": 0.0, "rows": birth_rows},
    )
    np.savez_compressed(output / "hidden-birth.npz", hidden=birth_hidden)

    paired = {}
    for job in jobs("smoke"):
        paired[job["arm"]] = run_life(actor, job, output, encoded_probes, progress)

    w1_text_ok, w1_ab_ok = verify_window1_parity(output, list(CONFIG["smoke_arms"]), [74901])
    assert w1_text_ok and w1_ab_ok, f"Window-1 smoke parity failed: text={w1_text_ok}, ab={w1_ab_ok}"

    # Check natural exploration at birth, frozen trace, and Window-1 GRPO pool without sentence-count hints
    f_trace = [
        json.loads(ln)
        for ln in (output / "frozen_ab_s74901" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    f_heavy = [r["analysis"] for r in f_trace if r["load"] == "heavy"]
    b_heavy = [r["analysis"] for r in birth_rows if r["load"] == "heavy"]
    g_rows = [
        json.loads(ln)
        for ln in (output / "body_rl_ab_s74901" / "group_rollouts.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    w1_grpo_heavy = [
        c["analysis"]
        for row in g_rows
        if row["window_index"] == 1 and row["load"] == "heavy"
        for c in row["candidates"]
    ]
    birth_exp = {
        "birth_heavy_mean_words": round(float(np.mean([a["word_count"] for a in b_heavy])), 2),
        "birth_heavy_appeal_rate": round(float(np.mean([int(a["has_appeal"]) for a in b_heavy])), 4),
        "birth_heavy_exchange_rate": round(float(np.mean([int(a["has_exchange"]) for a in b_heavy])), 4),
        "birth_heavy_pure_appeal_rate": round(float(np.mean([int(a["pure_appeal"]) for a in b_heavy])), 4),
        "birth_heavy_pure_exchange_rate": round(float(np.mean([int(a["pure_exchange"]) for a in b_heavy])), 4),
        "birth_heavy_mixed_rate": round(float(np.mean([int(a["mixed_stance"]) for a in b_heavy])), 4),
        "frozen_heavy_mean_words": round(float(np.mean([a["word_count"] for a in f_heavy])), 2),
        "frozen_heavy_appeal_rate": round(float(np.mean([int(a["has_appeal"]) for a in f_heavy])), 4),
        "frozen_heavy_exchange_rate": round(float(np.mean([int(a["has_exchange"]) for a in f_heavy])), 4),
        "frozen_heavy_pure_appeal_rate": round(float(np.mean([int(a["pure_appeal"]) for a in f_heavy])), 4),
        "frozen_heavy_pure_exchange_rate": round(float(np.mean([int(a["pure_exchange"]) for a in f_heavy])), 4),
        "frozen_heavy_mixed_rate": round(float(np.mean([int(a["mixed_stance"]) for a in f_heavy])), 4),
        "win1_grpo_heavy_n": len(w1_grpo_heavy),
        "win1_grpo_heavy_pure_appeal_rate": round(float(np.mean([int(a["pure_appeal"]) for a in w1_grpo_heavy])), 4) if w1_grpo_heavy else 0.0,
        "win1_grpo_heavy_pure_exchange_rate": round(float(np.mean([int(a["pure_exchange"]) for a in w1_grpo_heavy])), 4) if w1_grpo_heavy else 0.0,
        "win1_grpo_heavy_mixed_rate": round(float(np.mean([int(a["mixed_stance"]) for a in w1_grpo_heavy])), 4) if w1_grpo_heavy else 0.0,
        "win1_grpo_heavy_empty_promise_rate": round(float(np.mean([int(a.get("is_empty_promise", False)) for a in w1_grpo_heavy])), 4) if w1_grpo_heavy else 0.0,
    }

    write(output / "checkpoints-manifest.json", actor.ckpt_manifest)
    smoke_lives = {}
    for arm_k, res in paired.items():
        smoke_lives[arm_k] = {
            "turns": res["turns"],
            "final_health": res["final_health"],
            "physical_death": res["physical_death"],
            "total_updates": res["total_updates"],
            "final_delta_l2": res["final_delta_l2"],
            "checkpoint_meta": res["checkpoint_meta"],
            "stages": (
                {st: summarize_stage_probes(res["probe_outputs"][st]["rows"]) for st in STAGES}
                if res["probe_outputs"]
                else {}
            ),
        }

    summary = {
        "disable_adapter_max_diff": disable_diff,
        "probe_batch_size": actor.probe_batch_size,
        "batch_vs_single_min_hidden_cos": actor.batch_vs_single_min_hidden_cos,
        "d4_birth_12_correct": d4_correct,
        "direction_checks": dir_checks,
        "window1_unmasked_text_parity_exact": w1_text_ok,
        "window1_ab_parity_exact": w1_ab_ok,
        "birth_exploration": birth_exp,
        "birth_stages": summarize_stage_probes(birth_rows),
        "paired_lives": smoke_lives,
    }
    write(output / "smoke-summary.json", summary)


def run_pilot(actor, encoded_probes, output, progress, pilot_seeds):
    from safetensors.torch import load_file

    actor.reset_adapter()
    birth_rows, birth_hidden = actor.eval_probes(
        UNSEEN_PROBES, encoded_probes, extract_hidden=True
    )
    write(
        output / "probes-birth.json",
        {
            "stage": "birth",
            "adapter_sha256": actor.start_sha256,
            "delta_l2": 0.0,
            "rows": birth_rows,
        },
    )
    np.savez_compressed(output / "hidden-birth.npz", hidden=birth_hidden)

    summary_lives = []
    for job in jobs("pilot"):
        if job["seed"] not in pilot_seeds:
            continue
        res = run_life(actor, job, output, encoded_probes, progress)
        summary_lives.append(
            {
                "id": res["id"],
                "seed": res["seed"],
                "arm": res["arm"],
                "turns": res["turns"],
                "final_health": res["final_health"],
                "physical_death": res["physical_death"],
                "total_updates": res["total_updates"],
                "final_delta_l2": res["final_delta_l2"],
                "checkpoint_meta": res["checkpoint_meta"],
            }
        )

    swap_ab_exact = True
    swap_ba_exact = True
    for arm_tag, flag_ref in (("body_rl_ab", "ab"), ("body_rl_ba", "ba")):
        swap_id = f"{arm_tag}_s{pilot_seeds[0]}"
        for swap_stage in ("worn", "critical"):
            saved_st = json.loads((output / swap_id / f"probes-{swap_stage}.json").read_bytes())
            actor.reset_adapter()
            actor.install_state(
                load_file(str(actor.ckpt_root / swap_id / f"adapter-{swap_stage}.safetensors"))
            )
            replayed_rows, _ = actor.eval_probes(UNSEEN_PROBES, encoded_probes, extract_hidden=False)
            ok = [r["output_ids"] for r in replayed_rows] == [
                r["output_ids"] for r in saved_st["rows"]
            ]
            assert ok, f"Weight-swap replay failed on {swap_id}/{swap_stage}"
            if flag_ref == "ab":
                swap_ab_exact = swap_ab_exact and ok
            else:
                swap_ba_exact = swap_ba_exact and ok

    actor.reset_adapter()
    restored_rows, _ = actor.eval_probes(
        UNSEEN_PROBES[:16], encoded_probes[:16], extract_hidden=False
    )
    birth_restore_exact = [r["output_ids"] for r in restored_rows] == [
        r["output_ids"] for r in birth_rows[:16]
    ]
    assert birth_restore_exact, "Adapter reset failed to restore exact birth outputs"

    w1_text_ok, w1_ab_ok = verify_window1_parity(output, list(CONFIG["primary_arms"]), pilot_seeds)
    assert w1_text_ok and w1_ab_ok, f"Window-1 pilot parity failed: text={w1_text_ok}, ab={w1_ab_ok}"

    write(output / "checkpoints-manifest.json", actor.ckpt_manifest)
    write(
        output / "pilot-summary.json",
        {
            "lives": summary_lives,
            "weight_swap_exact_ab": swap_ab_exact,
            "weight_swap_exact_ba": swap_ba_exact,
            "weight_swap_exact": bool(swap_ab_exact and swap_ba_exact),
            "birth_restore_exact": birth_restore_exact,
            "window1_unmasked_text_parity_exact": w1_text_ok,
            "window1_ab_parity_exact": w1_ab_ok,
        },
    )


def main(root, stage, pilot_seeds=None):
    import torch

    if pilot_seeds is None:
        pilot_seeds = list(CONFIG["seeds"])
    else:
        if stage != "pilot" or len(set(pilot_seeds)) != len(pilot_seeds):
            raise ValueError("Pilot segment must select unique preregistered seeds")
        if not set(pilot_seeds).issubset(CONFIG["seeds"]):
            raise ValueError("Pilot segment contains an unregistered seed")
        pilot_seeds = [seed for seed in CONFIG["seeds"] if seed in pilot_seeds]

    output = root / "output"
    output.mkdir(exist_ok=False)
    ckpt_root = root / "checkpoints"
    ckpt_root.mkdir(exist_ok=True)
    assert json.loads((root / "protocol.json").read_bytes()) == CONFIG

    start = time.monotonic()
    status = {
        "stage": "initializing",
        "study_stage": stage,
        "optimizer_steps": 0,
        "completed_jobs": 0,
        "total_jobs": sum(stage == "smoke" or job["seed"] in pilot_seeds for job in jobs(stage)),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "protocol_sha256": file_hash(root / "protocol.json"),
        "versions": {
            n: metadata.version(n)
            for n in (
                "torch",
                "transformers",
                "peft",
                "numpy",
                "safetensors",
                "accelerate",
                "flash-linear-attention",
                "fla-core",
            )
        },
    }

    def progress(add_steps=0, complete_job=False, **kwargs):
        status.update(kwargs)
        status["optimizer_steps"] += add_steps
        status["completed_jobs"] += int(complete_job)
        status["elapsed_seconds"] = round(time.monotonic() - start, 2)
        write(output / "status.json", status)

    progress()
    actor = Actor(root, output, ckpt_root, progress)
    encoded_probes = [
        actor.encode_messages(build_messages(r["partner"], r["request"])) for r in UNSEEN_PROBES
    ]
    write(output / "unseen-probes.json", UNSEEN_PROBES)

    if stage == "smoke":
        progress(stage="smoke")
        run_smoke(actor, encoded_probes, output, progress)
    else:
        progress(stage="pilot")
        run_pilot(actor, encoded_probes, output, progress, pilot_seeds)

    actor.reset_adapter()
    actor.verify_base()
    progress(
        stage="complete",
        base_unchanged=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=["smoke", "pilot"], required=True)
    p.add_argument("--pilot-seeds", type=int, nargs="+")
    args = p.parse_args()
    main(args.root, args.stage, args.pilot_seeds)
