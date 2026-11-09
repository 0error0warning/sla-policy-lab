"""Actual NumPy gradient training and hidden activation interventions on a toy MLP.

LoRA changes ONLY the frozen model's output projection. It is NOT transformer LoRA.
"""
import hashlib
import numpy as np
from .environment import FEATURES, ACTIONS, reference_action, features

HIDDEN = 24
RANK = 2


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def dataset(seed, n=2200):
    rng = np.random.default_rng(seed)
    observations = []
    for _ in range(n):
        horizon = int(rng.integers(8, 161))
        age = int(rng.integers(0, horizon))
        observations.append({"damage": float(rng.uniform(0, 1)), "load": float(rng.uniform(0, 1)),
                             "age": age, "horizon": horizon, "remaining": (horizon - age) / horizon,
                             "protection_left": int(rng.choice([0, 0, 0, 1, 2, 3])),
                             "support_estimate": float(rng.uniform(.05, .95)),
                             "request_cooldown": int(rng.choice([0, 0, 1, 2]))})
    x = np.stack([features(o) for o in observations])
    labels = {profile: np.array([reference_action(o, profile) for o in observations])
              for profile in ("balanced", "service", "preserve")}
    return x, labels


def softmax(logits):
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)


def cross_entropy(logits, labels):
    p = softmax(logits)
    loss = float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    p[np.arange(len(labels)), labels] -= 1
    return loss, p / len(labels)


class Adam:
    def __init__(self, params):
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= .9
            m += .1 * g
            v *= .999
            v += .001 * g * g
            p -= lr * (m / (1 - .9 ** self.t)) / (np.sqrt(v / (1 - .999 ** self.t)) + 1e-8)


class MLP:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, .35, (len(FEATURES), HIDDEN))
        self.b1 = np.zeros(HIDDEN)
        self.w2 = rng.normal(0, .2, (HIDDEN, len(ACTIONS)))
        self.b2 = np.zeros(len(ACTIONS))

    @property
    def params(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def hidden(self, x):
        return np.tanh(x @ self.w1 + self.b1)

    def logits(self, x):
        return self.hidden(x) @ self.w2 + self.b2

    def train(self, x, y, epochs=180):
        opt = Adam(self.params)
        initial, _ = cross_entropy(self.logits(x), y)
        for _ in range(epochs):
            h = self.hidden(x)
            _, dlogits = cross_entropy(h @ self.w2 + self.b2, y)
            dh = (dlogits @ self.w2.T) * (1 - h * h)
            opt.step(self.params, [x.T @ dh, dh.sum(axis=0), h.T @ dlogits, dlogits.sum(axis=0)], .025)
        final, _ = cross_entropy(self.logits(x), y)
        return {"initial_loss": initial, "final_loss": final, "epochs": epochs,
                "trainable_parameters": sum(p.size for p in self.params)}


class LoRA:
    def __init__(self, base, seed, rank=RANK):
        self.base = base
        self.rank = rank
        self.scale = 1.0 / rank
        rng = np.random.default_rng(seed)
        self.a = rng.normal(0, .05, (HIDDEN, rank))
        self.b = np.zeros((rank, len(ACTIONS)))

    def logits(self, x):
        h = self.base.hidden(x)
        return h @ self.base.w2 + self.base.b2 + self.scale * (h @ self.a @ self.b)

    def train(self, x, y, epochs=180):
        original = digest(*self.base.params)
        h = self.base.hidden(x)
        baseline = h @ self.base.w2 + self.base.b2
        params = [self.a, self.b]
        opt = Adam(params)
        initial, _ = cross_entropy(self.logits(x), y)
        for _ in range(epochs):
            _, dl = cross_entropy(baseline + self.scale * (h @ self.a @ self.b), y)
            da = self.scale * h.T @ dl @ self.b.T
            db = self.scale * (h @ self.a).T @ dl
            opt.step(params, [da, db], .04)
        final, _ = cross_entropy(self.logits(x), y)
        return {"initial_loss": initial, "final_loss": final, "epochs": epochs,
                "trainable_parameters": self.a.size + self.b.size, "rank": self.rank,
                "frozen_base_unchanged": original == digest(*self.base.params)}


class Steering:
    def __init__(self, base, x, labels, profile):
        self.base = base
        h = base.hidden(x)
        # A simple contrastive direction in actual hidden coordinates, not logit offsets.
        target = 2 if profile == "preserve" else 0
        positive = labels == target
        self.direction = h[positive].mean(axis=0) - h[~positive].mean(axis=0)
        self.direction /= max(np.linalg.norm(self.direction), 1e-12)
        self.target = ACTIONS[target]

    def logits(self, x, strength=1.0):
        return (self.base.hidden(x) + strength * self.direction) @ self.base.w2 + self.base.b2


class ModelSuite:
    def __init__(self, seed=17):
        self.seed = seed
        x, y = dataset(seed)
        vx, vy = dataset(seed + 10000, 500)
        self.base = MLP(seed)
        self.report = {"base": self.base.train(x, y["balanced"])}
        self.small, self.lora, self.steering = {}, {}, {}
        for i, profile in enumerate(("balanced", "service", "preserve")):
            small = MLP(seed + 10 + i)
            lora = LoRA(self.base, seed + 20 + i)
            steering = Steering(self.base, x, y[profile], profile)
            sm = small.train(x, y[profile])
            lm = lora.train(x, y[profile])
            sm["heldout_teacher_agreement"] = float((small.logits(vx).argmax(axis=1) == vy[profile]).mean())
            lm["heldout_teacher_agreement"] = float((lora.logits(vx).argmax(axis=1) == vy[profile]).mean())
            self.small[profile], self.lora[profile], self.steering[profile] = small, lora, steering
            self.report[profile] = {"small": sm, "lora": lm,
                                    "steering": {"trainable_parameters": 0, "direction_dimensions": HIDDEN,
                                                 "contrast_action": steering.target}}
        self.fingerprint = digest(*self.base.params, *[p for m in self.small.values() for p in m.params],
                                  *[p for m in self.lora.values() for p in (m.a, m.b)],
                                  *[m.direction for m in self.steering.values()])
