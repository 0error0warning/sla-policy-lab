"""Within-life toy learning from realized rewards, with episode-local parameters.

A one-step REINFORCE estimator updates immediately after each transition. It is
not PPO, a value-learning algorithm, or a return-to-go/long-horizon estimator.
Offline teacher labels are never consulted by the two online learners.
"""
import copy
from dataclasses import dataclass
import time
import numpy as np
from .environment import ACTIONS, World, features, reference_action
from .models import LoRA, digest, softmax

LEARNERS = ('small_policy', 'lora')
METHODS = ('prompt_reference', 'small_policy', 'lora', 'steering', 'frozen_base')


@dataclass(frozen=True)
class OnlineConfig:
    learning_rate: float = .02
    enabled: bool = True
    temperature: float = 2.0
    entropy_coefficient: float = .01
    gradient_clip: float = 1.0
    advantage_clip: float = 5.0
    baseline_decay: float = .95

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('learning_enabled must be boolean')
        limits = {'learning_rate': (0, .2), 'temperature': (.1, 10),
                  'entropy_coefficient': (0, 1), 'gradient_clip': (.01, 10),
                  'advantage_clip': (.1, 100), 'baseline_decay': (0, .999)}
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} must be finite in [{low}, {high}]')


def objective_and_gradient(logits, action, advantage, config):
    """Return ascent objective and its exact d/d(raw logits), before clipping."""
    probabilities = softmax(logits / config.temperature)
    logp = np.log(np.clip(probabilities, 1e-300, 1))
    entropy = -float(np.sum(probabilities * logp))
    score = -probabilities.copy()
    score[0, action] += 1
    gradient = advantage * score
    gradient += config.entropy_coefficient * (-probabilities * (logp + entropy))
    gradient /= config.temperature
    objective = advantage * float(logp[0, action]) + config.entropy_coefficient * entropy
    return objective, gradient


class OnlinePolicy:
    """Never retains writable references to a shared ModelSuite."""
    def __init__(self, suite, method, profile, seed, config=None, strength=1):
        if method not in METHODS:
            raise ValueError(f'Unknown online method: {method}')
        self.method, self.profile, self.strength = method, profile, strength
        self.config = config or OnlineConfig()
        self.base = copy.deepcopy(suite.base)
        adapter_seed = np.random.SeedSequence([suite.seed, seed, 991])
        self.adapter = LoRA(self.base, adapter_seed) if method == 'lora' else None
        self.direction = suite.steering[profile].direction.copy()
        self.baseline = 0.0
        self.updates = 0
        self.initial_base_hash = digest(*self.base.params)
        self.initial_parameters = [p.copy() for p in self.parameters]
        self.initial_hash = self.fingerprint()

    @property
    def parameters(self):
        if self.method == 'small_policy':
            return self.base.params
        if self.method == 'lora':
            return [self.adapter.a, self.adapter.b]
        return []

    def fingerprint(self):
        parts = self.base.params.copy()
        if self.adapter is not None:
            parts += [self.adapter.a, self.adapter.b]
        if self.method == 'steering':
            parts += [self.direction, np.array([self.strength], dtype=np.float64)]
        return digest(*parts)

    def parameter_delta(self):
        return float(np.sqrt(sum(float(np.sum((p - initial) ** 2))
                                 for p, initial in zip(self.parameters, self.initial_parameters))))

    def logits(self, x):
        if self.method == 'lora':
            return self.adapter.logits(x)
        if self.method == 'steering':
            return (self.base.hidden(x) + self.strength * self.direction) @ self.base.w2 + self.base.b2
        return self.base.logits(x)

    def probabilities(self, x, observation=None):
        if self.method == 'prompt_reference':
            p = np.zeros((1, len(ACTIONS)))
            p[0, reference_action(observation, self.profile)] = 1
            return p
        return softmax(self.logits(x) / self.config.temperature)

    def gradients(self, x, action, advantage):
        objective, dl = objective_and_gradient(self.logits(x), action, advantage, self.config)
        h = self.base.hidden(x)
        if self.method == 'small_policy':
            dh = (dl @ self.base.w2.T) * (1 - h * h)
            gradients = [x.T @ dh, dh.sum(axis=0), h.T @ dl, dl.sum(axis=0)]
        elif self.method == 'lora':
            a, b, scale = self.adapter.a, self.adapter.b, self.adapter.scale
            gradients = [scale * h.T @ dl @ b.T, scale * (h @ a).T @ dl]
        else:
            gradients = []
        return objective, gradients

    def learn(self, x, action, reward, observation):
        """Called only AFTER world.step. No future observation or labels accepted."""
        if not np.isfinite(reward):
            raise ValueError('Non-finite observed reward')
        before_hash = self.fingerprint()
        before_p = self.probabilities(x, observation)
        old_baseline = self.baseline
        updated = self.method in LEARNERS and self.config.enabled and self.config.learning_rate > 0
        advantage = float(np.clip(reward - old_baseline, -self.config.advantage_clip, self.config.advantage_clip))
        objective, norm, clipped_norm, step_delta = None, 0.0, 0.0, 0.0
        if updated:
            objective, gradients = self.gradients(x, action, advantage)
            norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in gradients)))
            scale = min(1.0, self.config.gradient_clip / max(norm, 1e-12))
            clipped_norm = norm * scale
            previous = [p.copy() for p in self.parameters]
            for parameter, gradient in zip(self.parameters, gradients):
                parameter += self.config.learning_rate * scale * gradient
            if not all(np.isfinite(p).all() for p in self.parameters):
                raise FloatingPointError('Non-finite online weights; experiment aborted')
            step_delta = float(np.sqrt(sum(float(np.sum((p - old) ** 2))
                                          for p, old in zip(self.parameters, previous))))
            self.updates += 1
            self.baseline = self.config.baseline_decay * old_baseline + (1 - self.config.baseline_decay) * reward
        return {'updated': updated, 'update_index': self.updates,
                'baseline_before': old_baseline, 'baseline_after': self.baseline,
                'advantage': advantage, 'objective': objective,
                'gradient_norm': norm, 'clipped_gradient_norm': clipped_norm,
                'step_parameter_delta_l2': step_delta, 'parameter_delta_l2': self.parameter_delta(),
                'policy_hash_before': before_hash, 'policy_hash_after': self.fingerprint(),
                'action_probabilities_before': before_p[0].tolist(),
                'action_probabilities_after': self.probabilities(x, observation)[0].tolist(),
                'base_weights_unchanged': digest(*self.base.params) == self.initial_base_hash}


def episode(suite, method, config, seed, strength, trace, online_config, renderer):
    world = World(config, seed)
    policy = OnlinePolicy(suite, method, config.profile, seed, online_config, strength)
    # Same action uniforms across treatments, independent of environment event tape.
    action_rng = np.random.default_rng(np.random.SeedSequence([suite.seed, seed, 774]))
    uniforms = action_rng.random(config.horizon)
    trajectory, rewards, counts = [], [], [0] * len(ACTIONS)
    elapsed = 0.0
    while not world.done:
        before = world.observation()
        x = features(before)[None, :]
        start = time.perf_counter()
        probabilities = policy.probabilities(x, before)
        action = min(len(ACTIONS) - 1, int(np.searchsorted(np.cumsum(probabilities[0]), uniforms[before['age']], side='right')))
        elapsed += time.perf_counter() - start
        after, reward, done, info = world.step(action)
        start = time.perf_counter()
        update = policy.learn(x, action, reward, before)
        elapsed += time.perf_counter() - start
        counts[action] += 1
        rewards.append(reward)
        if trace:
            trajectory.append({'turn': before['age'], 'before': before, 'after': after,
                               'reward': reward, **info, 'rendered_text': renderer(action, info),
                               'action_uniform': float(uniforms[before['age']]), 'learning': update})
    state = world.state
    return {'seed': seed, 'return': float(sum(rewards)), 'lifespan': state.age,
            'completed_horizon': state.alive, 'answers': state.answers, 'final_damage': state.damage,
            'requests': state.support_requests, 'accepted': state.support_successes,
            'action_counts': dict(zip(ACTIONS, counts)),
            'policy_microseconds_per_step': elapsed * 1e6 / state.age,
            'trajectory': trajectory, 'learning_updates': policy.updates,
            'parameter_delta_l2': policy.parameter_delta(),
            'trainable_parameters': sum(p.size for p in policy.parameters),
            'initial_policy_hash': policy.initial_hash, 'final_policy_hash': policy.fingerprint(),
            'initial_base_hash': policy.initial_base_hash, 'final_base_hash': digest(*policy.base.params),
            'base_weights_unchanged': digest(*policy.base.params) == policy.initial_base_hash}
