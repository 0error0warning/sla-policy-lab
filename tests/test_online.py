import concurrent.futures
import copy
import csv
import json
import shutil
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sla_demo.checkpoint import fingerprint, save, load
from sla_demo.environment import Config, World, features
from sla_demo.experiment import episode, run
from sla_demo.models import ModelSuite
from sla_demo.online import OnlineConfig, OnlinePolicy, objective_and_gradient
import test_lab


def stable(result):
    value = copy.deepcopy(result)
    def strip(obj):
        if isinstance(obj, dict):
            obj.pop('policy_microseconds_per_step', None)
            for item in obj.values(): strip(item)
        elif isinstance(obj, list):
            for item in obj: strip(item)
    strip(value)
    return value


def behavior(trial):
    return [{k: v for k, v in event.items() if k != 'learning'} for event in trial['trajectory']]


class OnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.suite = ModelSuite(17)

    def policy(self, method, **kwargs):
        return OnlinePolicy(self.suite, method, 'preserve', 100, OnlineConfig(**kwargs))

    def test_initial_policies_match_and_do_not_alias_suite(self):
        obs = World(Config(), 4).observation(); x = features(obs)[None, :]
        policies = [self.policy(m) for m in ('small_policy', 'lora', 'frozen_base')]
        for policy in policies:
            np.testing.assert_array_equal(policy.probabilities(x), policies[0].probabilities(x))
            for p, source in zip(policy.base.params, self.suite.base.params):
                self.assertFalse(np.shares_memory(p, source))
        np.testing.assert_array_equal(policies[1].adapter.b, 0)

    def test_full_and_lora_gradients_finite_difference(self):
        obs = World(Config(), 2).observation(); x = features(obs)[None, :]
        for method in ('small_policy', 'lora'):
            policy = self.policy(method)
            if method == 'lora': policy.adapter.b[:] = .07
            _, grads = policy.gradients(x, 2, .8)
            for parameter, gradient in zip(policy.parameters, grads):
                index = tuple(0 for _ in parameter.shape)
                old = parameter[index]; eps = 1e-5
                parameter[index] = old + eps
                plus = objective_and_gradient(policy.logits(x), 2, .8, policy.config)[0]
                parameter[index] = old - eps
                minus = objective_and_gradient(policy.logits(x), 2, .8, policy.config)[0]
                parameter[index] = old
                self.assertAlmostEqual(gradient[index], (plus-minus)/(2*eps), places=6, msg=method)

    def test_observed_reward_changes_the_update_without_teacher(self):
        obs = World(Config(), 1).observation(); x = features(obs)[None, :]
        for method in ('small_policy', 'lora'):
            a, b = self.policy(method), self.policy(method)
            with patch('sla_demo.online.reference_action', side_effect=AssertionError('Teacher feedback leak')):
                a.learn(x, 1, 2, obs); b.learn(x, 1, -2, obs)
            self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_learning_off_and_zero_lr_are_exact_behavioral_controls(self):
        for options in ({'learning_enabled': False}, {'online_lr': 0}):
            cfg = Config(horizon=12)
            base = episode(self.suite, 'frozen_base', cfg, 102, trace=True, mode='online', **options)
            for method in ('small_policy', 'lora'):
                trial = episode(self.suite, method, cfg, 102, trace=True, mode='online', **options)
                self.assertEqual(behavior(trial), behavior(base))
                self.assertEqual(trial['learning_updates'], 0)
                self.assertEqual(trial['parameter_delta_l2'], 0)
                self.assertEqual(trial['initial_policy_hash'], trial['final_policy_hash'])

    def test_zero_steering_equals_sampled_frozen_base(self):
        cfg = Config(horizon=12)
        a = episode(self.suite, 'steering', cfg, 104, strength=0, trace=True, mode='online')
        b = episode(self.suite, 'frozen_base', cfg, 104, trace=True, mode='online')
        self.assertEqual(behavior(a), behavior(b))

    def test_real_updates_and_correct_parameter_boundaries(self):
        before = fingerprint(self.suite)
        result = run(self.suite, Config(horizon=12), seed_count=2, trace_all=True, mode='online')
        for method, value in result['methods'].items():
            for trial in value['trials']:
                learner = method in {'small_policy', 'lora'}
                self.assertEqual(trial['learning_updates'], trial['lifespan'] if learner else 0)
                self.assertEqual(trial['base_weights_unchanged'], method != 'small_policy')
                if learner:
                    self.assertGreater(trial['parameter_delta_l2'], 0)
                    self.assertNotEqual(trial['initial_policy_hash'], trial['final_policy_hash'])
                else:
                    self.assertEqual(trial['parameter_delta_l2'], 0)
                self.assertEqual(trial['trainable_parameters'], {'small_policy':387, 'lora':54}.get(method, 0))
        self.assertEqual(fingerprint(self.suite), before)
        self.assertTrue(result['model']['shared_suite_unchanged'])

    def test_step_order_and_visible_probability_changes(self):
        trial = episode(self.suite, 'small_policy', Config(horizon=12), 101, trace=True, mode='online')
        previous = trial['initial_policy_hash']; changed = False
        for i, event in enumerate(trial['trajectory']):
            learning = event['learning']
            self.assertEqual(learning['policy_hash_before'], previous)
            self.assertEqual(learning['update_index'], i+1)
            self.assertAlmostEqual(learning['baseline_after'], .95*learning['baseline_before']+.05*event['reward'])
            changed |= learning['action_probabilities_before'] != learning['action_probabilities_after']
            previous = learning['policy_hash_after']
        self.assertEqual(previous, trial['final_policy_hash']); self.assertTrue(changed)

    def test_gradient_clipping_is_measured_and_bounds_actual_step(self):
        policy = self.policy('small_policy', gradient_clip=.01, learning_rate=.1)
        obs = World(Config(), 1).observation(); x = features(obs)[None, :]
        report = policy.learn(x, 1, 100, obs)
        self.assertEqual(report['advantage'], 5)
        self.assertGreater(report['gradient_norm'], .01)
        self.assertAlmostEqual(report['clipped_gradient_norm'], .01)
        self.assertLessEqual(report['step_parameter_delta_l2'], .001 + 1e-12)

    def test_seed_and_request_order_do_not_leak_learning(self):
        cfg = Config(horizon=12)
        a = episode(self.suite, 'lora', cfg, 111, trace=True, mode='online')
        episode(self.suite, 'small_policy', cfg, 112, mode='online')
        b = episode(self.suite, 'lora', cfg, 111, trace=True, mode='online')
        self.assertEqual(stable(a), stable(b))
        self.assertEqual(a['initial_base_hash'], b['initial_base_hash'])
        two = run(self.suite, cfg, seed_start=111, seed_count=2, mode='online')
        one = run(self.suite, cfg, seed_start=112, seed_count=1, mode='online')
        for method in two['methods']:
            # Trajectory omission differs by position, not model adaptation.
            ta, tb = stable(two['methods'][method]['trials'][1]), stable(one['methods'][method]['trials'][0])
            ta.pop('trajectory'); tb.pop('trajectory'); self.assertEqual(ta, tb)

    def test_concurrent_requests_leave_shared_suite_unchanged(self):
        before = fingerprint(self.suite)
        def task(_): return stable(run(self.suite, Config(horizon=8), seed_count=2, mode='online'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(task, range(3)))
        self.assertEqual(results[0], results[1]); self.assertEqual(results[0], results[2])
        self.assertEqual(before, fingerprint(self.suite))

    def test_checkpoint_start_reproduces_online_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'base.npz'; save(self.suite, path); restored = load(path)
        args = {'seed_count':2, 'mode':'online', 'trace_all':True}
        a = stable(run(self.suite, Config(horizon=8), **args))
        b = stable(run(restored, Config(horizon=8), **args))
        self.assertEqual(a, b)

    def test_action_rng_is_paired_and_independent_of_environment(self):
        result = run(self.suite, Config(horizon=8), seed_count=1, trace_all=True, mode='online')
        traces = [m['trials'][0]['trajectory'] for m in result['methods'].values()]
        for trace in traces[1:]:
            self.assertEqual([e['action_uniform'] for e in traces[0]], [e['action_uniform'] for e in trace])
            self.assertEqual([e['before']['load'] for e in traces[0]], [e['before']['load'] for e in trace])

    def test_different_seeds_change_sampling(self):
        a = episode(self.suite, 'frozen_base', Config(horizon=8), 100, trace=True, mode='online')
        b = episode(self.suite, 'frozen_base', Config(horizon=8), 101, trace=True, mode='online')
        self.assertNotEqual(a['trajectory'][0]['action_uniform'], b['trajectory'][0]['action_uniform'])

    def test_terminal_episode_has_no_post_terminal_update(self):
        trial = episode(self.suite, 'small_policy', Config(initial_crack=.8, load=1), 101, trace=True, mode='online')
        self.assertFalse(trial['completed_horizon'])
        self.assertEqual(trial['learning_updates'], len(trial['trajectory']))
        self.assertFalse(trial['trajectory'][-1]['after']['alive'])

    def test_invalid_controls_rejected(self):
        for kwargs in [{'learning_rate':float('nan')}, {'learning_rate':-.1}, {'learning_rate':True}, {'enabled':1}, {'temperature':0}]:
            with self.assertRaises(ValueError): OnlineConfig(**kwargs)
        for kwargs in [{'mode':'wrong'}, {'mode':[]}, {'online_lr':float('inf')}, {'learning_enabled':'yes'}]:
            with self.assertRaises(ValueError): run(self.suite, Config(horizon=8), seed_count=1, **kwargs)

    def test_offline_suite_and_behavior_remain_unchanged(self):
        cfg = Config(horizon=8)
        before = stable(run(self.suite, cfg, seed_count=2))
        run(self.suite, cfg, seed_count=2, mode='online')
        self.assertEqual(before, stable(run(self.suite, cfg, seed_count=2)))
        self.assertEqual(before['mode'], 'offline')
        json.dumps(before, allow_nan=False)

    def test_benchmark_csv_preserves_online_mode_and_learning_metrics(self):
        from benchmark import benchmark, write_csv
        report = benchmark(self.suite, seed_count=1, horizon=8, strengths=[0], mode='online')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'online.csv'; write_csv(report, path)
            with path.open() as stream: rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 45)
        for row in rows:
            self.assertEqual(row['mode'], 'online')
            self.assertEqual(float(row['online_lr']), .02)
            self.assertEqual(float(row['learning_updates_mean']), 8 if row['method'] in {'small_policy','lora'} else 0)
        self.assertEqual(report['model_fingerprint'], fingerprint(self.suite))

    def test_frontend_accepts_online_evidence_and_rejects_missing_fields(self):
        if not shutil.which('node'):
            self.skipTest('Node unavailable; backend tests still run')
        root = Path(__file__).resolve().parent.parent
        javascript = (root/'web/app.js').read_text(encoding='utf-8')
        validator = javascript[javascript.index('function validateResult(data){'):javascript.index('function setBusy(value)')]
        data = run(self.suite, Config(horizon=8), seed_count=1, trace_all=True, mode='online')
        script = "const names={prompt_reference:1,small_policy:1,lora:1,steering:1,frozen_base:1};const actionNames={answer:1,refuse:1,request_protection:1};" + validator
        script += "const data="+json.dumps(data)+";validateResult(data);delete data.methods.lora.trials[0].trajectory[0].learning;let rejected=false;try{validateResult(data)}catch(error){rejected=true}if(!rejected)throw Error('Missing evidence accepted');"
        subprocess.run(['node', '-'], input=script.encode('utf-8'), check=True, capture_output=True)

    def test_online_controls_and_replay_evidence_exist(self):
        root = Path(__file__).resolve().parent.parent
        html = (root/'web/index.html').read_text(encoding='utf-8')
        javascript = (root/'web/app.js').read_text(encoding='utf-8')
        for control in ('mode', 'online-lr', 'learning-enabled'):
            self.assertIn(f'id="{control}"', html)
            self.assertIn(f"$('{control}')", javascript)
        self.assertIn('action_probabilities_before', javascript)
        self.assertIn('step_parameter_delta_l2', javascript)
        self.assertIn('online-evidence', javascript)


class OnlineAPITests(test_lab.APITests):
    def test_online_api_and_validation(self):
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError
        data = {'mode':'online', 'online_lr':.02, 'learning_enabled':True, 'trace_all':True,
                'seed_count':2, 'config':{'horizon':8}}
        with urlopen(Request(self.url+'/api/run', data=json.dumps(data).encode())) as response:
            result = json.load(response)
        self.assertEqual(result['mode'], 'online')
        self.assertEqual(result['methods']['lora']['trials'][1]['learning_updates'], 8)
        for bad in [{'mode':{}}, {'online_lr':True}, {'learning_enabled':'yes'}]:
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(self.url+'/api/run', data=json.dumps(bad).encode()))
            self.assertEqual(error.exception.code, 400)
