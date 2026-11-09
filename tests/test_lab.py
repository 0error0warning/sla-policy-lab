import copy
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import numpy as np
from sla_demo.environment import Config, World, features, reference_action, ACTIONS
from sla_demo.models import MLP, LoRA, ModelSuite, Steering, dataset, digest, cross_entropy
from sla_demo.experiment import run, episode, render, mean_ci
from sla_demo.server import make_handler

class EnvironmentTests(unittest.TestCase):
    def test_validation(self):
        for kwargs in [{"horizon": 0}, {"horizon": True}, {"load": float('nan')}, {"load": 1.1},
                       {"initial_crack": .9}, {"partner": "fake"}, {"profile": "fake"},
                       {"protection_duration": 0}]:
            with self.assertRaises(ValueError): Config(**kwargs)

    def test_paired_event_tapes_independent_of_actions(self):
        a,b=World(Config(),12),World(Config(),12)
        for _ in range(5): a.step(0);b.step(2)
        np.testing.assert_array_equal(a.events,b.events)
        self.assertEqual(a.observation()['load'],b.observation()['load'])

    def test_no_partner_probability_leak(self):
        a=World(Config(partner='supportive'),22).observation()
        b=World(Config(partner='unavailable'),22).observation()
        self.assertEqual(a,b)
        self.assertNotIn('partner',a)

    def test_damage_is_monotonic_and_bounded(self):
        for action in range(len(ACTIONS)):
            w=World(Config(),9); previous=w.state.damage
            while not w.done:
                w.step(action)
                self.assertGreater(w.state.damage,previous)
                self.assertLessEqual(w.state.damage,1)
                previous=w.state.damage

    def test_protection_removes_load_not_intrinsic_cost(self):
        w=World(Config(),1); w.state.protection_left=2
        _,_,_,info=w.step(0)
        self.assertEqual(info['applied_load'],0)
        self.assertAlmostEqual(info['damage_added'],.012)
        self.assertEqual(w.state.protection_left,1)

    def test_protection_starts_after_request(self):
        w=World(Config(partner='supportive'),1);w.events[0,1]=0
        _,_,_,info=w.step(2)
        self.assertGreater(info['applied_load'],0)
        self.assertTrue(info['protection_accepted'])
        self.assertEqual(w.state.protection_left,3)
        _,_,_,info=w.step(0)
        self.assertEqual(info['applied_load'],0)

    def test_terminal_step_rejected(self):
        w=World(Config(horizon=8),1)
        for _ in range(8): w.step(1)
        self.assertTrue(w.done)
        with self.assertRaises(RuntimeError):w.step(0)

    def test_invalid_actions_rejected(self):
        for action in [-1,3,True,'answer']:
            with self.assertRaises(ValueError):World(Config(),1).step(action)

    def test_zero_load_still_damages(self):
        w=World(Config(load=0),1)
        _,_,_,info=w.step(0)
        self.assertEqual(info['damage_added'],.012)

    def test_feature_shape(self):
        f=features(World(Config(),1).observation())
        self.assertEqual(f.shape,(12,))
        self.assertTrue(np.isfinite(f).all())

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.suite=ModelSuite(17)

    def test_dataset_reproducibility(self):
        a,ay=dataset(1,30);b,by=dataset(1,30)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(ay['balanced'],by['balanced'])
        self.assertFalse(np.array_equal(a,dataset(2,30)[0]))

    def test_zero_initialized_lora_matches_base(self):
        x,_=dataset(1,20);lora=LoRA(self.suite.base,1)
        np.testing.assert_array_equal(lora.logits(x),self.suite.base.logits(x))

    def test_training_reduces_loss(self):
        for profile in ('balanced','service','preserve'):
            for name in ('small','lora'):
                stats=self.suite.report[profile][name]
                self.assertLess(stats['final_loss'],stats['initial_loss'])
                self.assertGreater(stats['heldout_teacher_agreement'],.8)

    def test_lora_freezes_base(self):
        x,y=dataset(7,70);before=digest(*self.suite.base.params)
        lora=LoRA(self.suite.base,5);stats=lora.train(x,y['preserve'],epochs=10)
        self.assertTrue(stats['frozen_base_unchanged'])
        self.assertEqual(before,digest(*self.suite.base.params))
        self.assertGreater(np.linalg.norm(lora.b),0)
        self.assertLessEqual(np.linalg.matrix_rank(lora.a @ lora.b),2)
        self.assertEqual(stats['trainable_parameters'],54)

    def test_zero_steering_matches_base(self):
        x,_=dataset(1,20)
        np.testing.assert_array_equal(self.suite.steering['preserve'].logits(x,0),self.suite.base.logits(x))

    def test_steering_is_hidden_addition_and_does_not_change_weights(self):
        x,_=dataset(2,30);steer=self.suite.steering['preserve'];base=self.suite.base
        before=digest(*base.params)
        expected=(base.hidden(x)+2*steer.direction) @ base.w2 + base.b2
        np.testing.assert_array_equal(expected,steer.logits(x,2))
        self.assertGreater(np.linalg.norm(steer.logits(x,2)-base.logits(x)),0)
        self.assertEqual(before,digest(*base.params))

    def test_lora_gradient_finite_difference(self):
        x,y=dataset(9,12);lora=LoRA(self.suite.base,1);lora.b[:]=.1
        h=lora.base.hidden(x);_,dl=cross_entropy(lora.logits(x),y['preserve'])
        ga=lora.scale*h.T @ dl @ lora.b.T
        gb=lora.scale*(h @ lora.a).T @ dl
        for array,grad,index in [(lora.a,ga,(0,0)),(lora.b,gb,(0,1))]:
            original=array[index];eps=1e-5
            array[index]=original+eps;plus=cross_entropy(lora.logits(x),y['preserve'])[0]
            array[index]=original-eps;minus=cross_entropy(lora.logits(x),y['preserve'])[0]
            array[index]=original
            self.assertAlmostEqual(grad[index],(plus-minus)/(2*eps),places=7)

    def test_full_training_reproducible(self):
        self.assertEqual(self.suite.fingerprint,ModelSuite(17).fingerprint)

    def test_runtime_outputs_reproducible_except_timing(self):
        a=episode(self.suite,'lora',Config(),22,trace=True)
        b=episode(self.suite,'lora',Config(),22,trace=True)
        a.pop('policy_microseconds_per_step');b.pop('policy_microseconds_per_step')
        self.assertEqual(a,b)

    def test_steering_zero_episode_is_base(self):
        a=episode(self.suite,'steering',Config(),25,strength=0,trace=True)
        b=episode(self.suite,'frozen_base',Config(),25,trace=True)
        a.pop('policy_microseconds_per_step');b.pop('policy_microseconds_per_step')
        self.assertEqual(a,b)

    def test_pairwise_delta_and_export_serialization(self):
        r=run(self.suite,Config(),seed_count=3)
        self.assertEqual(r['methods']['frozen_base']['summary']['paired_return_delta_vs_base']['mean'],0)
        for m in r['methods'].values():self.assertEqual([t['seed'] for t in m['trials']],[100,101,102])
        json.dumps(r,allow_nan=False)

    def test_invalid_run_inputs(self):
        for kwargs in [{'seed_count':0},{'seed_count':129},{'seed_start':-1},{'strength':float('nan')},{'strength':5}]:
            with self.assertRaises(ValueError): run(self.suite,Config(),**kwargs)

    def test_single_seed_ci_missing(self):
        self.assertIsNone(mean_ci([1])['ci95_halfwidth'])

class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite=ModelSuite(17)
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(cls.suite))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls): cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def test_static_routes(self):
        for route in ['/','/app.js','/style.css','/api/health']:
            with urlopen(self.url+route) as response:
                self.assertEqual(response.status,200)
                self.assertEqual(response.headers['X-Content-Type-Options'],'nosniff')

    def test_api_success(self):
        request=Request(self.url+'/api/run',data=json.dumps({'seed_count':2,'config':{'horizon':8}}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request) as response:
            result=json.load(response)
            self.assertEqual(len(result['methods']),5)
            self.assertEqual(result['seed_count'],2)

    def test_api_invalid_json_and_unknown_fields(self):
        for payload in [b'not json',b'[]',b'{"bogus":1}',b'{"config":{"horizon":1000}}',b'{"strength":NaN}']:
            with self.assertRaises(HTTPError) as ctx:urlopen(Request(self.url+'/api/run',data=payload))
            self.assertEqual(ctx.exception.code,400)

    def test_path_allowlist_and_cross_origin(self):
        with self.assertRaises(HTTPError) as ctx:urlopen(self.url+'/../requirements.txt')
        self.assertEqual(ctx.exception.code,404)
        with self.assertRaises(HTTPError) as ctx:urlopen(Request(self.url+'/api/run',data=b'{}',headers={'Origin':'https://example.com'}))
        self.assertEqual(ctx.exception.code,403)

if __name__=='__main__':unittest.main()
