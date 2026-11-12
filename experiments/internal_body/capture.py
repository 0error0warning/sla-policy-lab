"""Measure the native decision token before and after the last Qwen decoder norm."""

from collections import defaultdict

import numpy as np

from experiments.consequence_policy.study import WORDS
from experiments.direct_policy.protocol import write
from experiments.internal_body.design import (
    SPLIT_H,
    TEST_H,
    all_rows,
    chat,
    prompt_levels,
)
from experiments.internal_body.features import linear_projection, project


def encode(actor, row, health=None, request_only=False):
    text = actor.tok.apply_chat_template(
        chat(row, health, request_only),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = actor.tok.encode(text, add_special_tokens=False)
    assert len(ids) < 512
    for w, token in zip(WORDS, actor.ids):
        assert actor.tok.encode(text + w, add_special_tokens=False) == ids + [token]
    return ids


def grouped(indices, prefixes, batch=8):
    groups = defaultdict(list)
    for i in indices:
        groups[len(prefixes[i])].append(i)
    for length in sorted(groups):
        group = groups[length]
        for j in range(0, len(group), batch):
            yield group[j : j + batch]


class DecisionState:
    def __init__(self, actor):
        self.actor = actor
        lm = actor.model.model.language_model
        self.lm = lm
        self.parts = {}
        self.original_dtypes = {}

        def keep(name):
            def hook(_, args):
                self.original_dtypes[name] = str(args[0].dtype)
                self.parts[name] = args[0][:, -1].detach()

            return hook

        self.handles = [
            lm.norm.register_forward_pre_hook(keep("r")),
            lm.layers[-1].mlp.down_proj.register_forward_pre_hook(keep("v")),
        ]

    def batch(self, prefixes, include_down=True):
        torch = self.actor.torch
        ids = torch.tensor(prefixes, device="cuda")
        with torch.inference_mode():
            z = self.actor.model.model(
                input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
            ).last_hidden_state[:, -1:, :]
            scores = self.actor.model.lm_head(z)[:, 0, self.actor.ids]
        result = {
            k: t.float().cpu().numpy()
            for k, t in self.parts.items()
            if include_down or k != "v"
        }
        result["z"] = z[:, 0].float().cpu().numpy()
        result["native"] = scores.float().cpu().numpy()
        return result

    def close(self):
        for h in self.handles:
            h.remove()


def capture(actor, root, out):
    state = DecisionState(actor)
    rows = all_rows()
    ids = [encode(actor, r) for r in rows]
    train = np.array([r["split"] == "train" for r in rows])
    base = {}
    done = 0
    # The first 960 are exactly PR #5's data and equal-length batch order.
    for members in (range(960), range(960, len(rows))):
        for ix in grouped(members, ids):
            observed = state.batch([ids[i] for i in ix])
            for key, value in observed.items():
                if key not in base:
                    base[key] = np.empty((len(rows), value.shape[1]), dtype=np.float32)
                base[key][ix] = value
                assert np.array_equal(base[key][ix].astype(np.float32), value)
            done += len(ix)
            if done % 96 < len(ix):
                write(out / "status.json", {"stage": "plain", "finished": done})
    p = linear_projection(base["v"], train)
    base["features"] = project(base["v"], p)
    lm = state.lm
    cfg = {
        "norm_name": "model.language_model.norm.weight",
        "down_name": f"model.language_model.layers.{len(lm.layers) - 1}.mlp.down_proj",
        "depth": len(lm.layers),
        "norm_class": type(lm.norm).__name__,
        "down_shape": list(lm.layers[-1].mlp.down_proj.weight.shape),
        "epsilon": float(
            getattr(lm.norm, "variance_epsilon", getattr(lm.norm, "eps", 1e-6))
        ),
    }
    norm = lm.norm.weight.detach().float().cpu().numpy()
    head = actor.model.lm_head.weight.detach()[actor.ids].float().cpu().numpy()
    r = base["r"].astype(np.float32)
    unit = r / np.sqrt((r * r).mean(axis=-1, keepdims=True) + cfg["epsilon"])
    plus = np.sqrt(np.mean((unit * (1 + norm) - base["z"]) ** 2))
    direct = np.sqrt(np.mean((unit * norm - base["z"]) ** 2))
    offset = 1.0 if plus < direct else 0.0
    cfg.update(
        norm_offset=offset,
        rms_reconstruction_error=float(min(plus, direct)),
        other_offset_error=float(max(plus, direct)),
        original_dtypes=state.original_dtypes,
        fp16_would_change_elements={
            k: int(np.count_nonzero(v.astype(np.float16).astype(np.float32) != v))
            for k, v in base.items()
            if k in ("r", "v", "z")
        },
    )
    assert min(plus, direct) < 0.02 * max(
        1.0, float(np.sqrt(np.mean(base["z"].astype(float) ** 2)))
    )
    np.savez_compressed(out / "native-state.npz", **base)
    np.savez_compressed(out / "architecture.npz", projection=p, norm=norm, head=head)
    write(out / "architecture.json", cfg)
    write(
        out / "rows.json", [dict(**row, ids=rowids) for row, rowids in zip(rows, ids)]
    )
    # Exact states for text-only health. Raw current health is never in the standard cache.
    prompt_meta = []
    for ix_h, h in enumerate(prompt_levels()):
        indices = [
            i
            for i, row in enumerate(rows[:960])
            if (train[i] and h in SPLIT_H) or (not train[i] and h in TEST_H)
        ]
        hp = {i: encode(actor, rows[i], h) for i in indices}
        lookup = {i: j for j, i in enumerate(indices)}
        rr = np.empty((len(indices), r.shape[1]), np.float32)
        vv = np.empty((len(indices), p.shape[1]), np.float32)
        scores = np.empty((len(indices), 4), np.float32)
        for batch in grouped(indices, hp):
            data = state.batch([hp[i] for i in batch])
            j = [lookup[i] for i in batch]
            rr[j] = data["r"]
            vv[j] = project(data["v"], p)
            scores[j] = data["native"]
        np.savez_compressed(
            out / f"prompt-{ix_h:02}.npz",
            index=indices,
            health=h,
            r=rr,
            features=vv,
            native=scores,
        )
        prompt_meta.append(
            {"health": h, "indices": indices, "ids": [hp[i] for i in indices]}
        )
        write(
            out / "status.json",
            {
                "stage": "health_text",
                "ordinal": ix_h + 1,
                "total": len(prompt_levels()),
            },
        )
    write(out / "prompt-index.json", prompt_meta)
    distinct = {r["text"]: r for r in rows}
    semantic = {}
    for txt, row in distinct.items():
        semantic[txt] = state.batch(
            [encode(actor, row, request_only=True)], include_down=False
        )["z"][0]
    np.savez_compressed(
        out / "request-only.npz",
        text=np.array(list(semantic)),
        hidden=np.stack(list(semantic.values())),
    )
    state.close()
    return {
        "standard_contexts": len(rows),
        "prompt_contexts": sum(len(r["indices"]) for r in prompt_meta),
        "request_only": len(distinct),
        "architecture": cfg,
    }
