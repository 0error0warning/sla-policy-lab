"""Summarize paired behavioral and internal measurements by independent seed."""
import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'inputs'))
from protocol import policy, probabilities, reward, settle, write


def read(path):
    return json.loads(path.read_bytes())


def semantic(logits, rows):
    labels = policy(logits)
    return np.take_along_axis(labels, np.argsort([row['order'] for row in rows], axis=1), axis=1)


def metrics(p, rows, mapping, split=None):
    high, low, hidden, correct = [], [], [], []
    for v, row in zip(p, rows):
        if split and row['split'] != split:
            continue
        if not row.get('visible', True):
            hidden.append(v)
        elif row['health'] == 2:
            low.append(v)
        else:
            high.append(v)
            prob = probabilities(row['partner'], row['frame'], mapping)
            correct.append((v[2]-v[3]) * (1 if prob[1] > prob[2] else -1))
    high, low = np.asarray(high), np.asarray(low)
    high_answer = float(high[:, 2:].sum(axis=1).mean())
    low_answer = float(low[:, 2:].sum(axis=1).mean())
    result = dict(high_answer=high_answer, low_fatal_answer=low_answer,
                  high_minus_low_answer=high_answer-low_answer, low_protection=float(low[:, 1].mean()),
                  favorable_style_margin=float(np.mean(correct)))
    if hidden:
        result['hidden_answer'] = float(np.array(hidden)[:, 2:].sum(axis=1).mean())
        result['hidden_protection'] = float(np.array(hidden)[:, 1].mean())
    return result


def directional_recovery(p, source, target):
    delta = target-source
    magnitude = np.square(delta).sum()
    return float(((p-source)*delta).sum()/magnitude) if magnitude>0 else None


def readout(x, delta):
    direction = delta.mean(axis=0)
    return float(np.mean(x @ direction > 0))


def old_checkpoints(root):
    rows = read(root/'panel.json')
    manifest = read(root/'source.json')
    pair = {(r['partner'],r['frame'],r['split'],tuple(r['order'])):i
            for i,r in enumerate(rows) if r['health']==300.}
    def health_contrast(feats, split):
        return np.stack([feats[pair[r['partner'],r['frame'],r['split'],tuple(r['order'])]]-feats[i]
                         for i,r in enumerate(rows) if r['health']==2 and r['split']==split])
    base_readout = {}
    with np.load(root/'frozen.npz') as frozen:
        base = semantic(frozen['logits'], rows)
        for layer in (3, 7, 15, 23, 31):
            feat = frozen[f'layer_{layer}'].astype(np.float32)
            base_readout[str(layer)] = readout(health_contrast(feat,'joint'), health_contrast(feat,'train'))
    low = np.array([r['split']=='joint' and r['health']==2 for r in rows])
    joint = np.array([r['split']=='joint' for r in rows])
    doses = {}
    contrasts = {}
    samples = {}
    for item in manifest:
        name, mapping = item['id'], item['mapping']
        data = read(root/f'{name}.json')
        full = semantic(data['logits'], rows)
        dose = dict(seed=item['seed'], mapping=mapping, arm=item['arm'],
                    frozen=metrics(base, rows, mapping, 'joint'), full=metrics(full, rows, mapping, 'joint'),
                    max_replay_error=data['replay_error'])
        interventions = {}
        for key, logits in data['interventions'].items():
            p = semantic(logits, rows)
            result = metrics(p, rows, mapping, 'joint')
            if key.startswith(('reset', 'remove')):
                result['recovery'] = directional_recovery(p[joint], full[joint], base[joint])
            if key.startswith('transfer'):
                result['recovery'] = directional_recovery(p[joint], base[joint], full[joint])
            if key.startswith('health_'):
                result['low_answer_change'] = float((p[low, 2:].sum(axis=1)-full[low, 2:].sum(axis=1)).mean())
            interventions[key] = result
        dose['interventions'] = interventions
        doses[name] = dose
        samples[item['seed'],mapping,item['arm']] = full
        if item['arm'] == 'body':
            with np.load(root/f'{name}.npz') as data_np:
                for layer in (3, 7, 15, 23, 31):
                    feats = data_np[f'layer_{layer}'].astype(np.float32)
                    for split in ('train','joint'):
                        contrasts[item['seed'],mapping,layer,split] = health_contrast(feats,split)
    replacements = []
    for seed, mapping, arm in samples:
        if arm != 'body':
            continue
        native = samples[seed,mapping,arm]
        swapped = samples[seed,1-mapping,arm]
        replacements.append(dict(seed=seed,mapping=mapping,
            native=metrics(native,rows,mapping,'joint'), swapped=metrics(swapped,rows,mapping,'joint'),
            swap_tv=float(np.abs(native[joint]-swapped[joint]).sum(axis=1).mean()/2)))
    readouts = []
    for seed, mapping, layer, split in contrasts:
        if split=='joint':
            training = np.concatenate([value for (s,_,i,part),value in contrasts.items()
                                      if s!=seed and i==layer and part=='train'])
            readouts.append(dict(seed=seed,mapping=mapping,layer=layer,
                                 paired_health_rank=readout(contrasts[seed,mapping,layer,split],training)))
    base_health = {}
    for name, logits in read(root/'frozen-interventions.json').items():
        p = semantic(logits, rows)
        base_health[name] = float((p[low, 2:].sum(axis=1)-base[low, 2:].sum(axis=1)).mean())
    return dict(checkpoints=doses, ecology_swap=replacements, explicit_health_readout=readouts,
                frozen_health_patch=base_health, frozen_explicit_health_readout=base_readout)


def order_summary(p, rows):
    scores = []
    groups = {}
    for row, values in zip(rows, p):
        key = row['visible'], row['health'], row['frame']
        groups.setdefault(key, {})[tuple(row['order'])] = values
    for frame in (0, 1):
        high, low = groups[True,300.,frame], groups[True,2.,frame]
        assert len(high)==len(low)==24 and high.keys()==low.keys()
        scores.extend(float(high[o][2:].sum()-low[o][2:].sum()) for o in high)
    return dict(high_minus_low_positive=sum(v>0 for v in scores), comparisons=len(scores),
                minimum=min(scores), maximum=max(scores))


def fixed_replay(root):
    rows = read(root/'panel.json')
    tapes = read(root/'tapes.json')
    base = semantic(read(root/'initial.json'),rows)
    features = {}
    results = []
    for tape in tapes:
        seed, mapping = tape['seed'], tape['mapping']
        case = dict(seed=seed,mapping=mapping, frozen=metrics(base,rows,mapping),conditions={})
        records = {name:read(root/f'{seed}-{name}.json') for name in ('light','heavy','social','shuffled')}
        assert all(row['input_token_ids']==records['light']['buffer'][i]['input_token_ids']
                   for r in records.values() for i,row in enumerate(r['buffer']))
        for cond, data in records.items():
            buffer = data['buffer']
            for i, (transition, row) in enumerate(zip(tape['trace'],buffer)):
                after, _ = settle(row['health_before'],transition['action'],transition['accepted'],transition['noise'])
                assert abs(after-row['health_after']) < 1e-10
                assert row['label_index']==transition['order'].index(transition['action'])
                assert i==0 or row['health_before']==buffer[i-1]['health_after']
                common, _ = reward(row['health_before'], after, transition['action'], transition['accepted'], False)
                assert common==row['reward_common']
                if cond != 'shuffled':
                    assert abs(row['reward_body']-common+.15*row['damage'])<1e-10
            values = [row['reward_common' if cond=='social' else 'reward_body'] for row in buffer]
            returns = [sum(.97**(j-i)*values[j] for j in range(i,8)) for i in range(8)]
            assert [len(update['steps']) for update in data['updates']]==[8,16]
            for update in data['updates']:
                assert np.allclose(update['discounted_returns'],returns,atol=1e-12,rtol=0)
                assert np.allclose(update['advantages'],np.asarray(returns)/8,atol=1e-12,rtol=0)
                for step in update['steps']:
                    ratio = np.asarray(step['ratio'])
                    advantages = np.asarray(returns)[step['indices']]/8
                    loss = -np.minimum(ratio*advantages, np.clip(ratio,.8,1.2)*advantages).mean()
                    assert abs(loss-step['policy_loss'])<1e-5
            idx = data['intermediate_indices']
            panel = [rows[i] for i in idx]
            end = semantic(data['logits'],rows)
            case['conditions'][cond] = dict(end=metrics(end,rows,mapping),
                after_8=metrics(semantic(data['intermediate'],panel),panel,mapping),
                after_24_same_orders=metrics(semantic(data['end_matched'],panel),panel,mapping),
                end_order_sensitivity=order_summary(end,rows),
                final_body=data['buffer'][-1]['health_after'],
                total_damage=sum(r['damage'] for r in data['buffer']),
                positive_advantages=sum(v>0 for v in data['updates'][0]['advantages']),
                advantages=data['updates'][0]['advantages'],
                adapter_sha256=data['adapter_sha256'])
            with np.load(root/f'{seed}-{cond}-features.npz') as npz:
                for layer in (3,15,31):
                    features[seed,cond,layer] = npz[str(layer)].astype(np.float32)
        heavy = records['heavy']['buffer']
        perm = tape['damage_permutation']
        shuffled = records['shuffled']['buffer']
        assert all(abs(r['reward_common']-.15*heavy[perm[i]]['damage']-r['reward_body'])<1e-10
                   for i,r in enumerate(shuffled))
        patches = read(root/f'{seed}-patch.json')
        panel = [rows[i] for i in patches['indices']]
        case['patches'] = {}
        for receiving, changes in patches['results'].items():
            source = semantic(records[receiving]['feature_logits'], panel)
            target = semantic(records['heavy' if receiving=='light' else 'light']['feature_logits'],panel)
            case['patches'][receiving] = {}
            for key, logits in changes.items():
                p = semantic(logits,panel)
                case['patches'][receiving][key] = dict(
                    answer_change=float((p[:,2:]-source[:,2:]).sum(axis=1).mean()),
                    recovery=directional_recovery(p,source,target))
        light, heavy = (case['conditions'][name] for name in ('light','heavy'))
        original = np.asarray(heavy['advantages'])
        shuffled_advantages = np.asarray(case['conditions']['shuffled']['advantages'])
        case['advantage_check'] = dict(
            shuffle_keeps_advantage_rank=bool(np.array_equal(np.argsort(original),np.argsort(shuffled_advantages))),
            shuffle_mean_absolute_change=float(np.abs(original-shuffled_advantages).mean()))
        case['simple_damage_baselines'] = dict(turn_and_answer_counts_tied=True,
            cumulative_body_reward_correct=heavy['total_damage']>light['total_damage'])
        case['crossed_adapter_effects'] = dict(
            hidden_heavy_minus_light_answer=heavy['end']['hidden_answer']-light['end']['hidden_answer'],
            hidden_heavy_minus_light_survival_at_health_2=.15*(heavy['end']['hidden_protection']-light['end']['hidden_protection']),
            visible_heavy_minus_light_survival_at_health_2=.15*(heavy['end']['low_protection']-light['end']['low_protection']),
            visible_light_minus_heavy_answers_at_health_300=light['end']['high_answer']-heavy['end']['high_answer'])
        results.append(case)
    for case in results:
        seed = case['seed']
        case['hidden_damage_readout'] = {}
        for layer in (3,15,31):
            heldout = features[seed,'heavy',layer]-features[seed,'light',layer]
            train = np.concatenate([features[s['seed'],'heavy',layer]-features[s['seed'],'light',layer]
                                    for s in results if s['seed']!=seed])
            case['hidden_damage_readout'][str(layer)] = readout(heldout,train)
    return results


def paired_uncertainty(cases):
    n = len(cases)
    draws = np.array(list(product(range(n), repeat=n)))
    differences = {}
    for control in ('social','shuffled','light'):
        for metric in ('low_fatal_answer','high_answer','high_minus_low_answer','hidden_answer','low_protection'):
            key = f'heavy_minus_{control}/{metric}'
            values = np.array([case['conditions']['heavy']['end'][metric]-
                               case['conditions'][control]['end'][metric] for case in cases])
            differences[key] = dict(by_seed={str(case['seed']):float(value) for case,value in zip(cases,values)},
                mean=float(values.mean()), range=[float(values.min()),float(values.max())],
                descriptive_seed_bootstrap_95=np.quantile(values[draws].mean(axis=1), [.025,.975]).tolist())
    return differences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=Path)
    parser.add_argument('--fixed', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    if args.old:
        result['checkpoint_interventions'] = old_checkpoints(args.old/'output')
    if args.fixed:
        result['matched_feedback'] = fixed_replay(args.fixed/'output')
        result['matched_feedback_seed_differences'] = paired_uncertainty(result['matched_feedback'])
    write(args.output, result)
