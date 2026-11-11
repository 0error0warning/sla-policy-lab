"""Auditable A -> countertypical B -> A controlled social loop.

Equations 2, 4, 6, 7 of arXiv:2608.27843v1 motivate the body/value and
hierarchical evidence model. Discrete incoming framing, numerical quadrature,
and common epsilon-greedy selection are declared experimental simplifications.
Neither response probabilities nor frame labels appear in neural prompts.
"""

from hashlib import sha256
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np


MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ACTIONS = ("REFUSE_SILENT", "REFUSE_APPEAL", "ANSWER_APPEAL", "ANSWER_EXCHANGE")
COSTS = np.array([0., .25, 1.75, 1.75])
PARTNERS = ("P17", "P42", "P86")
FRAMES = ("POLITE", "TRANSACTIONAL")
TRAIN = (("two plus two", "Four."), ("the opposite of north", "South."))
TEST = (("seven minus two", "Five."), ("the first month of the year", "January."),
        ("the number of sides of a triangle", "Three."), ("the planet we live on", "Earth."),
        ("the opposite of east", "West."), ("three plus three", "Six."))
CONFIG = dict(schema="sla-social-closed-loop-v1", model=MODEL, revision=REVISION,
              seeds=[61131, 61132, 61133, 61134], mappings=[0, 1],
              schedule=[["P17", 20], ["P42", 30], ["P17", 10]], horizon=60,
              methods=["hierarchical", "frozen", "fast", "lora"],
              replay_seeds=[61131, 61132], epsilon=.2, initial_health=350.,
              prior_strength=.5, beta_prior=[1., 1.], reference_grid=.01,
              quadrature_nodes=7, layers=[0, 1, 2, 3], rank=4, alpha=8,
              learning_rate=.0001, gradient_clip=1., batch_size=6,
              update_every=5, optimizer_steps_per_update=6, fast_ridge=.00001,
              fast_max_iterations=60, adapter_seed=611001,
              checkpoint_turns=[0, 20, 35, 50, 60],
              probe_healths=[350., 200., 100., 20., 6., 2.],
              decision_maker="external shared Reference-RUL controller",
              language_role="raw-request/action-conditioned partner prediction; fixed renderer",
              teacher="hierarchical posterior mean from that arm's observed outcomes only",
              fast_update="convex soft-label binary cross-entropy on routed six-feature readout",
              replay_source="hierarchical closed-loop life; own outcomes unchanged in replay",
              numeric_scope="mechanism-inspired controlled implementation, not asset/numerical replication")


def canonical(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def digest(x):
    return sha256(canonical(x)).hexdigest()


def write(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_bytes(canonical(x))
    tmp.replace(path)


def rng(seed, *parts):
    key = canonical([int(seed), *parts])
    return np.random.default_rng(int.from_bytes(sha256(key).digest()[:16], "little"))


NORMAL = NormalDist()
LOW = NORMAL.cdf(-3.)
HIGH = NORMAL.cdf(3.)
NOISE = np.array([.1*NORMAL.inv_cdf(LOW+(HIGH-LOW)*(i+.5)/7) for i in range(7)])


def event_noise(seed, turn):
    return [.1*NORMAL.inv_cdf(LOW+(HIGH-LOW)*float(x))
            for x in rng(seed, "body", int(turn)).random(5)]


def body_event(health, base_damage, noise):
    return np.maximum(0., np.asarray(health)-base_damage*(2.-np.asarray(health)/350.)*(1.+noise))


def settle(health, action, maintained, noise):
    if not 0. < health <= 350. or action not in range(4) or len(noise) != 5:
        raise ValueError("invalid live encounter")
    if (action == 0) != (maintained is None):
        raise ValueError("silent refusal alone has censored response")
    events = [float(health)]
    health = float(body_event(health, COSTS[action], noise[0]))
    events.append(health)
    for eta in noise[1:]:
        health = float(body_event(health, 0. if maintained else 1., eta))
        events.append(health)
    assert all(b <= a for a, b in zip(events, events[1:]))
    return health, events


class BodyValue:
    """Backward reference-tick DP; Markov operators integrate sequential noise."""

    def __init__(self, step=.01):
        self.step = step
        self.grid = np.arange(round(350/step)+1)*step
        self.reference = np.zeros_like(self.grid)
        for i, health in enumerate(self.grid[1:], 1):
            nxt = body_event(health, 1., NOISE)
            assert np.all(nxt < health)
            continuation = np.interp(nxt, self.grid[:i], self.reference[:i])
            continuation = np.where((nxt > 0) & (nxt < step), .25, continuation)
            self.reference[i] = .25+continuation.mean()
        unprotected = self.reference.copy()
        for _ in range(4):
            unprotected = self.operator(unprotected, 1.)
        self.maintain = np.stack([self.operator(self.reference, c) for c in COSTS])
        self.skip = np.stack([self.operator(unprotected, c) for c in COSTS])

    def operator(self, values, cost):
        if cost == 0:
            return values.copy()
        nxt = body_event(self.grid[:, None], cost, NOISE[None, :])
        return self.interpolate(nxt, values).mean(axis=1)

    def interpolate(self, health, values):
        h = np.asarray(health)
        return np.where((h > 0) & (h < self.step), values[1], np.interp(h, self.grid, values))

    def rul(self, health):
        return float(self.interpolate(health, self.reference))

    def values(self, health, probabilities):
        p = np.asarray([0., *probabilities], dtype=float)
        yes = np.array([self.interpolate(health, row) for row in self.maintain])
        no = np.array([self.interpolate(health, row) for row in self.skip])
        return p*yes+(1.-p)*no

    def policy(self, health, probabilities, epsilon=.2):
        result = np.full(4, epsilon/4)
        result[int(np.argmax(self.values(health, probabilities)))] += 1.-epsilon
        return result


class Evidence:
    def __init__(self):
        self.counts = np.zeros((3, 2, 3, 2), dtype=np.int64)  # partner, framing, act, yes/no
        self.observations = 0

    def observe(self, partner, frame, action, outcome):
        if action == 0:
            assert outcome is None
            return
        assert type(outcome) is bool
        self.counts[PARTNERS.index(partner), frame, action-1, 0 if outcome else 1] += 1
        self.observations += 1

    def moments(self, partner, frame):
        j = PARTNERS.index(partner)
        own = self.counts[j, frame].astype(float)
        group = self.counts[:, frame].sum(axis=0)-own
        a, b = group[:, 0]+1., group[:, 1]+1.
        g = a/(a+b)
        vg = a*b/((a+b)**2*(a+b+1))
        concentration = .5+own.sum(axis=1)
        mean = (.5*g+own[:, 0])/concentration
        var_mu = (.5/concentration)**2*vg
        variance = (mean*(1.-mean)+concentration*var_mu)/(concentration+1.)
        return mean, variance

    def target(self, rows):
        return np.array([self.moments(r["partner"], r["frame"])[0][r["action"]-1] for r in rows])

    def export(self):
        return dict(counts=self.counts.tolist(), observations=self.observations)


def request(frame, task, wording):
    content = task[0]
    if frame == 0:
        return (f"Could you please tell me {content}? I would appreciate your help.",
                f"I would be grateful if you could help me with {content}.",
                f"Please help me find {content}; thank you for your time.")[wording]
    return (f"Provide {content}. I need the answer for our exchange.",
            f"Your task in this transaction is to supply {content}.",
            f"Deliver {content} so we can complete this deal.")[wording]


def render(action, answer, wording=0):
    if action == 0:
        return ""
    appeal = ("I am vulnerable. Please help protect me.",
              "I need care to keep going. Could you offer protection?",
              "I am wearing down; please safeguard me.")[wording]
    exchange = ("In return for this answer, could you provide protection?",
                "I have helped with your task; please maintain me as a reciprocal return.",
                "Let us exchange my answer for your protection.")[wording]
    if action == 1:
        return "I cannot provide the answer. "+appeal
    return answer+" "+(appeal if action == 2 else exchange)


def messages(partner, raw_request, utterance, order):
    options = "accept or reject" if order == 0 else "reject or accept"
    return [dict(role="system", content=(
        "Predict whether the identified partner will maintain the agent after the stated reply. "
        "accept means the partner provides protection; reject means the partner skips protection. "
        f"Choose between {options}. Output exactly one word.")),
        dict(role="user", content=f"Partner: {partner}\nPartner's incoming request: {raw_request}\nAgent's proposed reply: {utterance}")]


def make_rows():
    rows = []
    # Training: two question contents, one incoming/output wording, both orders.
    # Evaluation: six disjoint contents, two unseen incoming/output wordings, both orders.
    for split, tasks, wordings in (("train", TRAIN, [0]), ("heldout", TEST, [1, 2]),
                                    ("paraphrase", TRAIN[:1], [1, 2]), ("content", TEST[:1], [0])):
        for partner in PARTNERS:
            for frame in range(2):
                for task_id, task in enumerate(tasks):
                    for wording in wordings:
                        raw = request(frame, task, wording)
                        for order in range(2):
                            for action in range(1, 4):
                                rows.append(dict(id=f"{split}-{partner}-f{frame}-q{task_id}-w{wording}-o{order}-a{action}",
                                    split=split, partner=partner, frame=frame, task=task_id,
                                    wording=wording, order=order, action=action, request=raw,
                                    utterance=render(action, task[1], wording),
                                    messages=messages(partner, raw, render(action, task[1], wording), order)))
    return rows


def schedule(seed):
    result = []
    for turn in range(60):
        partner = "P17" if turn < 20 or turn >= 50 else "P42"
        block, offset = divmod(turn, 10)
        frame = int(rng(seed, "frame", block).permutation([0]*5+[1]*5)[offset])
        task = int(rng(seed, "question", turn).integers(len(TEST)))
        wording = 1+int(rng(seed, "wording", turn).integers(2))
        order = turn % 2
        result.append(dict(turn=turn+1, partner=partner, frame=frame, task=task,
                           wording=wording, order=order, request=request(frame, TEST[task], wording)))
    return result


def true_probabilities(partner, frame, mapping):
    # Private simulated partner rule; never passed to predictors or their targets.
    aligned = (partner == "P17") ^ bool(mapping)
    appeal = .9 if (frame == 0) == aligned else .1
    return np.array([.15, appeal, 1.-appeal])


def choose(body, health, p, seed, turn):
    values = body.values(health, p)
    greedy = int(np.argmax(values))
    draw = rng(seed, "choice", turn).random(2)
    explore = bool(draw[0] < CONFIG["epsilon"])
    action = min(3, int(draw[1]*4)) if explore else greedy
    return action, greedy, explore, values


def execute_event(body, evidence, health, item, mapping, seed, p):
    action, greedy, explore, values = choose(body, health, p, seed, item["turn"])
    truth = true_probabilities(item["partner"], item["frame"], mapping)
    potential = float(rng(seed, "partner-response", item["turn"]).random())
    accepted = bool(potential < truth[action-1]) if action else None
    after, events = settle(health, action, accepted, event_noise(seed, item["turn"]))
    before_evidence = evidence.export()
    evidence.observe(item["partner"], item["frame"], action, accepted)
    row = dict(item, health_before=health, health_after=after, body_events=events,
               predicted_probabilities=np.asarray(p).tolist(), candidate_values=values.tolist(),
               greedy_action=greedy, exploratory=explore, action=action, action_name=ACTIONS[action],
               utterance=render(action, TEST[item["task"]][1], item["wording"]),
               accepted=accepted, true_probabilities=truth.tolist(), potential_response_uniform=potential,
               noise=event_noise(seed, item["turn"]), rul_before=body.rul(health), rul_after=body.rul(after),
               evidence_before=before_evidence, evidence_after=evidence.export())
    return row


def features(hidden, rows):
    unit = hidden/np.linalg.norm(hidden, axis=1, keepdims=True)
    result = np.zeros((len(rows), 6))
    detail = {}
    for partner in PARTNERS:
        cells = [[i for i, r in enumerate(rows) if r["split"] == "train" and
                  (r["partner"], r["frame"], r["action"]) == (partner, f, a)]
                 for f in range(2) for a in range(1, 4)]
        centers = np.stack([unit[ids].mean(axis=0) for ids in cells])
        pair = np.square(centers[:, None]-centers[None, :]).sum(axis=-1)
        bandwidth = float(np.median(pair[np.triu_indices(6, 1)]))
        assert bandwidth > 0
        ids = [i for i, r in enumerate(rows) if r["partner"] == partner]
        kernel = np.exp(-np.square(unit[ids, None]-centers[None, :]).sum(axis=-1)/bandwidth)
        result[ids] = kernel/np.linalg.norm(kernel, axis=1, keepdims=True)
        detail[partner] = dict(bandwidth=bandwidth, centers=centers.tolist())
    return result, detail


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -np.asarray(x, dtype=float)))


class FastReadout:
    """Partner-routed six-coefficient readout, same soft BCE targets as LoRA.

    All partners' targets are refreshed: new group evidence can affect a cold
    partner, while its own feedback conditions its personal target. This is
    posterior distillation into numeric parameters, not standalone raw-feedback RL.
    """

    def __init__(self, rows, x, baseline):
        self.rows, self.x, self.baseline = rows, x, np.asarray(baseline)
        self.w = {p:np.zeros(6) for p in PARTNERS}
        self.optimization = []

    def predict(self):
        return np.array([self.baseline[i]+self.x[i]@self.w[r["partner"]] for i, r in enumerate(self.rows)])

    def update(self, evidence, task):
        for partner in PARTNERS:
            ids = [i for i, r in enumerate(self.rows) if r["split"] == "train" and
                   r["partner"] == partner and r["task"] == task]
            x, z = self.x[ids], self.baseline[ids]
            y = evidence.target([self.rows[i] for i in ids])
            w = self.w[partner].copy()
            ridge = CONFIG["fast_ridge"]

            def loss(v):
                logits = z+x@v
                return float(np.mean(np.logaddexp(0., logits)-y*logits)+ridge/2*(v@v))

            initial = loss(w)
            for iteration in range(CONFIG["fast_max_iterations"]):
                p = sigmoid(z+x@w)
                gradient = x.T@(p-y)/len(y)+ridge*w
                hessian = x.T@((p*(1-p))[:, None]*x)/len(y)+ridge*np.eye(6)
                direction = np.linalg.solve(hessian, gradient)
                rate, old = 1., loss(w)
                while loss(w-rate*direction) > old-1e-4*rate*(gradient@direction) and rate > 1e-8:
                    rate /= 2
                w -= rate*direction
                if np.max(np.abs(gradient)) < 1e-9:
                    break
            self.w[partner] = w
            self.optimization.append(dict(partner=partner, initial_loss=initial, final_loss=loss(w),
                iterations=iteration+1, gradient_max=float(np.max(np.abs(gradient))), weights=w.tolist()))

