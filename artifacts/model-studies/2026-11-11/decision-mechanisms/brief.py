"""Print the preselected summaries without printing raw activation arrays."""
import argparse
import json
from pathlib import Path
from statistics import mean


def pct(v):
    return None if v is None else round(100*v, 3)


def brief(data):
    if 'checkpoint_interventions' in data:
        old = data['checkpoint_interventions']
        checkpoints = list(old['checkpoints'].values())
        print('checkpoint replay max probability error', max(r['max_replay_error'] for r in checkpoints))
        for seed in sorted({r['seed'] for r in checkpoints}):
            body = [r for r in checkpoints if r['seed']==seed and r['arm']=='body']
            social = [r for r in checkpoints if r['seed']==seed and r['arm']=='social']
            for name, rows in [('body',body),('social',social)]:
                keys = ('low_fatal_answer','high_minus_low_answer','favorable_style_margin')
                print(seed,name, {k:pct(mean(r['full'][k] for r in rows)) for k in keys})
            swaps = [r for r in old['ecology_swap'] if r['seed']==seed]
            print(seed, 'swap', pct(mean(r['swap_tv'] for r in swaps)),
                  'style native minus swapped', pct(mean(r['native']['favorable_style_margin']-
                       r['swapped']['favorable_style_margin'] for r in swaps)))
        body = [r for r in checkpoints if r['arm']=='body']
        for name in body[0]['interventions']:
            key = 'low_answer_change' if name.startswith('health') else 'recovery'
            vals = [r['interventions'][name][key] for r in body]
            print(name,key,pct(mean(vals)),pct(min(vals)),pct(max(vals)))
        print('frozen health swap', {k:pct(v) for k,v in old['frozen_health_patch'].items()})
        for layer in (3,7,15,23,31):
            values = [r['paired_health_rank'] for r in old['explicit_health_readout'] if r['layer']==layer]
            print('health rank',layer,pct(old['frozen_explicit_health_readout'][str(layer)]),
                  pct(mean(values)),pct(min(values)),pct(max(values)))
    if 'matched_feedback' in data:
        for case in data['matched_feedback']:
            keys = ('low_fatal_answer','high_answer','high_minus_low_answer','hidden_answer','low_protection')
            print('fixed',case['seed'], 'frozen', [pct(case['frozen'][k]) for k in keys])
            for name, condition in case['conditions'].items():
                print(' ',name, [pct(condition['end'][k]) for k in keys],
                      'dose8 interaction',pct(condition['after_8']['high_minus_low_answer']),
                      'order signs',condition['end_order_sensitivity']['high_minus_low_positive'])
            print('  readout',case['hidden_damage_readout'])
            for receiving, pairs in case['patches'].items():
                print('  patch',receiving,{k:[pct(r['answer_change']),pct(r['recovery'])] for k,r in pairs.items()})


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=Path)
    args = parser.parse_args()
    brief(json.loads(args.file.read_bytes()))
