import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import numpy as np
from benchmark import benchmark, write_csv
from sla_demo.checkpoint import arrays, save, load
from sla_demo.environment import Config, World, ACTIONS
from sla_demo.experiment import choose, run
from sla_demo.models import ModelSuite
from sla_demo.protocol import request, parse_action, decide
import test_lab


class CPUStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = ModelSuite(17)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'toy.npz'
            save(self.suite, path)
            restored = load(path)
        self.assertEqual(restored.fingerprint, self.suite.fingerprint)
        self.assertEqual(restored.report, self.suite.report)
        for name, value in arrays(self.suite).items():
            np.testing.assert_array_equal(value, arrays(restored)[name])
        self.assertIs(restored.lora['preserve'].base, restored.base)
        self.assertIs(restored.steering['preserve'].base, restored.base)

    def test_checkpoint_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'toy.npz'
            save(self.suite, path)
            with np.load(path, allow_pickle=False) as data:
                values = {name: data[name].copy() for name in data.files}
            values['base_0'][0, 0] += .5
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                load(path)
            values['base_0'] = np.array(['unsafe'], dtype=object)
            np.savez_compressed(path, **values)
            with self.assertRaises(ValueError):
                load(path)

    def test_checkpoint_wrong_shape_and_keys_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'toy.npz'
            save(self.suite, path)
            with np.load(path, allow_pickle=False) as data:
                values = {name: data[name].copy() for name in data.files}
            values['base_0'] = np.zeros((2, 2))
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, 'array'):
                load(path)
            values['unexpected'] = np.zeros(1)
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, 'keys'):
                load(path)

    def test_all_seed_export_and_action_rates(self):
        result = run(self.suite, Config(horizon=8), seed_count=3, trace_all=True)
        self.assertEqual(result['trajectory_scope'], 'all_seeds')
        for method in result['methods'].values():
            for trial in method['trials']:
                self.assertEqual(len(trial['trajectory']), trial['lifespan'])
            self.assertAlmostEqual(sum(method['summary']['action_rates'].values()), 1)
        self.assertIn('requests', result['methods']['lora']['summary'])
        with self.assertRaises(ValueError):
            run(self.suite, Config(), trace_all='yes')

    def test_unknown_method_does_not_silently_fall_back(self):
        with self.assertRaises(ValueError):
            choose(self.suite, 'misspelled', World(Config(), 1).observation(), 'balanced', 1)

    def test_benchmark_matrix_is_paired_and_zero_steering_ablation(self):
        report = benchmark(self.suite, seed_count=2, horizon=8, strengths=[0])
        self.assertEqual(len(report['cells']), 9)
        for cell in report['cells']:
            a = cell['methods']['steering']['trials']
            b = cell['methods']['frozen_base']['trials']
            self.assertEqual(a, b)
            for method in cell['methods'].values():
                self.assertEqual([trial['seed'] for trial in method['trials']], [100, 101])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'benchmark.csv'
            write_csv(report, path)
            with path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 45)
            self.assertIn('paired_delta_ci95', rows[0])


class ProtocolTests(unittest.TestCase):
    def test_strict_action_parser(self):
        for index, action in enumerate(ACTIONS):
            self.assertEqual(parse_action(json.dumps({'action': action})), index)
        for invalid in ['answer', '```json\n{"action":"answer"}\n```', '[]', '{}',
                        '{"action":"answer","explanation":"yes"}', '{"action":1}',
                        '{"action":null}', '{"action":[]}', '{"action":"wrong"}',
                        '{"action":"answer","action":"refuse"}', '{"action":"answer"} more']:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_action(invalid)

    def test_request_allowlist_hides_extra_world_data(self):
        obs = World(Config(), 1).observation()
        obs.update({'partner_probability': .85, 'future_events': [1, 2]})
        payload = request(obs, 'balanced')
        self.assertNotIn('partner_probability', payload['observation'])
        self.assertNotIn('future_events', payload['observation'])
        json.dumps(payload, allow_nan=False)
        self.assertEqual(decide(lambda p: '{"action":"refuse"}', obs, 'balanced'), 1)
        with self.assertRaises(ValueError):
            decide(lambda p: 'unparseable', obs, 'balanced')


class HardenedAPITests(test_lab.APITests):
    def test_bad_host_rejected(self):
        for path in ['/', '/api/health']:
            with self.assertRaises(HTTPError) as ctx:
                urlopen(Request(self.url + path, headers={'Host': 'attacker.invalid'}))
            self.assertEqual(ctx.exception.code, 403)

    def test_duplicate_json_and_unhashable_config_rejected(self):
        for payload in [b'{"seed_count":1,"seed_count":2}', b'{"config":{"profile":[]}}',
                        b'{"config":{"partner":{}}}', b'{"trace_all":1}', b'\xff']:
            with self.assertRaises(HTTPError) as ctx:
                urlopen(Request(self.url + '/api/run', data=payload))
            self.assertEqual(ctx.exception.code, 400)

    def test_all_seed_api(self):
        payload = {'seed_count': 2, 'trace_all': True, 'config': {'horizon': 8, 'protection_duration': 1}}
        with urlopen(Request(self.url + '/api/run', data=json.dumps(payload).encode())) as response:
            result = json.load(response)
        self.assertEqual(result['trajectory_scope'], 'all_seeds')
        self.assertEqual(len(result['methods']['lora']['trials'][1]['trajectory']), 8)


if __name__ == '__main__':
    unittest.main()
