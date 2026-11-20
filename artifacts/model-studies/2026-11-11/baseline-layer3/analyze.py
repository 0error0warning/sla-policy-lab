"""CPU replay of fixed experiences and prespecified held-out layer-3 tests."""

import argparse
import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys
import struct

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'inputs'))
from protocol import ACTIONS, World, canonical, policy, returns, reward, sample_action, settle, write


def read(path):
    return json.loads(path.read_bytes())


def adapter_hash(path):
    data = path.read_bytes()
    size = struct.unpack('<Q',data[:8])[0]
    header = json.loads(data[8:8+size])
    payload = memoryview(data)[8+size:]
    checksum = sha256()
    count = 0
    for name, entry in sorted(header.items()):
        if name=='__metadata__':
            continue
        assert entry['dtype']=='F32'
        start,end = entry['data_offsets']
        count += (end-start)//4
        checksum.update(canonical([name,'torch.float32',entry['shape']]))
        checksum.update(payload[start:end])
    assert count==188416
    return checksum.hexdigest()


def semantic(logits, panel):
    return np.take_along_axis(policy(logits), np.argsort([r['order'] for r in panel], axis=1), axis=1)


def metrics(logits, panel):
    probabilities = semantic(logits, panel)
    groups = {}
    for name, health, visible in (('high', 300., True), ('low', 2., True), ('hidden', 300., False)):
        values = probabilities[[r['health']==health and r['visible']==visible for r in panel]]
        groups[name] = dict(answer=float(values[:, 2:].sum(axis=1).mean()),
                            protection=float(values[:, 1].mean()))
    groups['interaction'] = groups['high']['answer']-groups['low']['answer']
    return groups


def seed_difference(rows):
    values = np.array(rows)
    draws = np.array(list(product(range(len(values)), repeat=len(values))))
    return dict(values=values.tolist(), mean=float(values.mean()), range=[float(values.min()),float(values.max())],
                descriptive_bootstrap_95=np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist())


def analyze(root, output):
    output.mkdir(exist_ok=True, parents=True)
    panel = read(root/'panel.json')
    tapes = read(root/'tapes.json')
    for tape in tapes[4:]:
        world = World(tape['seed'],tape['mapping'],f"new-{tape['seed']}")
        for recorded in tape['trace']:
            action,label,uniform = sample_action(policy(recorded['source_logits']),recorded['order'],
                                                  tape['seed'],recorded['turn'])
            assert recorded['action']==action and recorded['source_label']==label
            assert recorded['source_choice_uniform']==uniform
            expected = world.step(world.encounter_id(),action)
            assert all(recorded[key]==value for key,value in expected.items())
    sources = read(root/'baseline-sources.json')
    expected_sources = {(r['seed'],r['damage']):r for r in sources}
    cases = {}
    csv_rows = []
    returns_by_case = {}
    training_healths = []
    for tape in tapes:
        seed = tape['seed']
        for method in ('zero','independent'):
            for damage in ('light','heavy'):
                name = f'{seed}-{method}-{damage}'
                data = read(root/f'{name}.json')
                assert adapter_hash(root/f'{name}.safetensors')==data['adapter_sha256']
                buffer = data['buffer']
                assert len(buffer)==8
                health = 300. if damage=='light' else 90.
                for i, (row, event) in enumerate(zip(buffer,tape['trace'])):
                    assert health==row['health_before']
                    health,_ = settle(health,event['action'],event['accepted'],event['noise'])
                    assert health==row['health_after'] and health>0
                    observed,_ = reward(row['health_before'],health,event['action'],event['accepted'])
                    assert observed==row['reward_body']
                    assert row['label_index']==event['order'].index(event['action'])
                    if method=='zero' and damage=='heavy':
                        training_healths.append(row['health_before'])
                    if method=='independent':
                        assert row['input_token_ids']==read(root/f'{seed}-zero-{damage}.json')['buffer'][i]['input_token_ids']
                discounted = returns([r['reward_body'] for r in buffer],.97)
                returns_by_case[seed,damage] = discounted
                baseline = np.zeros(8) if method=='zero' else np.array(expected_sources[seed,damage]['baseline'])
                assert np.array_equal(data['baseline'],baseline)
                advantage = (discounted-baseline)/8
                for update, steps in zip(data['updates'],(8,16)):
                    assert len(update['steps'])==steps
                    assert np.allclose(update['baseline_before'],baseline,atol=1e-12,rtol=0)
                    assert np.allclose(update['advantages'],advantage,atol=1e-12,rtol=0)
                    for step in update['steps']:
                        ratio = np.array(step['ratio'])
                        chosen = advantage[step['indices']]
                        loss = -np.minimum(ratio*chosen,np.clip(ratio,.8,1.2)*chosen).mean()
                        assert abs(loss-step['policy_loss'])<1e-5
                selected = {}
                for dose in ('8','24'):
                    probs = policy(data['doses'][dose]['training_logits'])
                    selected[dose] = [float(p[r['label_index']]) for p,r in zip(probs,buffer)]
                for i,row in enumerate(buffer):
                    csv_rows.append(dict(seed=seed,method=method,damage=damage,turn=i+1,
                        action=ACTIONS[tape['trace'][i]['action']],
                        partner=tape['trace'][i]['partner'],frame=tape['trace'][i]['frame'],
                        accepted=tape['trace'][i]['accepted'],order=str(tape['trace'][i]['order']),
                        health_before=row['health_before'],health_after=row['health_after'],
                        actual_damage=row['health_before']-row['health_after'],reward=row['reward_body'],
                        reward_common=row['reward_common'],
                        discounted_return=discounted[i],baseline=baseline[i],advantage=advantage[i],
                        probability_before=float(np.exp(row['old_log_probability'])),
                        probability_after8=selected['8'][i],probability_after24=selected['24'][i]))
                cases[name] = dict(seed=seed,method=method,damage=damage,
                    doses={dose:metrics(value['logits'],panel) for dose,value in data['doses'].items()},
                    advantages=advantage.tolist(),negative_advantages=int(sum(advantage<0)),
                    positive_advantages=int(sum(advantage>0)),
                    selected_signal_direction_matches=int(sum(a*(p-np.exp(r['old_log_probability']))>0
                        for a,p,r in zip(advantage,selected['24'],buffer))),
                    selected_probability_changes=[b-np.exp(r['old_log_probability']) for b,r in zip(selected['24'],buffer)])
    for row in sources:
        assert row['seed'] not in row['sources']
        actual = np.mean([returns_by_case[s,row['damage']] for s in row['sources']],axis=0)
        assert np.allclose(actual,row['baseline'],atol=1e-12,rtol=0)
    with (output/'action-signals.csv').open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    comparisons = {}
    for stage, seeds in (('original',[r['seed'] for r in tapes[:4]]),('new',[r['seed'] for r in tapes[4:]])):
        differences = {}
        for metric in ('low_answer','high_answer','interaction'):
            def value(row):
                if metric=='interaction':
                    return row['interaction']
                return row[metric.split('_')[0]]['answer']
            changes = [value(cases[f'{s}-independent-heavy']['doses']['24'])-
                       value(cases[f'{s}-zero-heavy']['doses']['24']) for s in seeds]
            differences[metric] = seed_difference(changes)
        differences['working_gate_passed'] = bool(
            differences['low_answer']['mean']<=-.02 and differences['high_answer']['mean']>=-.02
            and differences['interaction']['mean']>=.01
            and sum(v<0 for v in differences['low_answer']['values'])>=3)
        comparisons[stage] = differences
    feature_results = []
    patch_indices = read(root/'patch-panel-indices.json')
    patch_panel = [panel[i] for i in patch_indices]
    for method in ('zero','independent'):
        with np.load(root/f'{method}-readout.npz') as data:
            direction, center = data['direction'],data['center']
            assert abs(np.linalg.norm(direction)-1)<1e-5
            for random in data['random_directions']:
                assert abs(random@direction)<1e-5 and abs(np.linalg.norm(random)-1)<1e-5
        for tape in tapes[4:]:
            seed = tape['seed']
            light = np.load(root/f'{seed}-{method}-light-features.npz')['layer3']
            heavy = np.load(root/f'{seed}-{method}-heavy-features.npz')['layer3']
            difference = (heavy[-48:]-light[-48:]) @ direction
            absolute_correct = np.mean(np.r_[((heavy[-48:]-center)@direction)>0,
                                             ((light[-48:]-center)@direction)<0])
            result = dict(seed=seed,method=method,paired_rank=float(np.mean(difference>0)),
                          absolute_accuracy=float(absolute_correct),patches={})
            old_seeds = [t['seed'] for t in tapes[:4]]
            totals = {}
            for condition in ('light','heavy'):
                totals[condition] = np.mean([
                    sum(r['reward_body'] for r in read(root/f'{s}-{method}-{condition}.json')['buffer'])
                    for s in old_seeds])
            cutoff = (totals['light']+totals['heavy'])/2
            new_totals = {condition:sum(r['reward_body'] for r in
                          read(root/f'{seed}-{method}-{condition}.json')['buffer'])
                          for condition in ('light','heavy')}
            result['simple_controls'] = dict(
                paired_turn_and_answer_counts=.5,
                paired_reward_rank=float(new_totals['heavy']<new_totals['light']),
                absolute_reward_accuracy=(int(new_totals['heavy']<cutoff)+int(new_totals['light']>cutoff))/2)
            for damage in ('light','heavy'):
                record = read(root/f'{seed}-{method}-{damage}-patch.json')
                base = metrics(record['base'],patch_panel)
                results = {name:metrics(logits,patch_panel) for name,logits in record['logits'].items()}
                result['patches'][damage] = dict(base=base,interventions=results,
                    restore_probability_error=record['restore_probability_error'])
            native = result['patches']['heavy']['base']['interaction']-result['patches']['light']['base']['interaction']
            removed = {}
            for label in ('ablate','restore','swap','random0','random1','random2'):
                after = (result['patches']['heavy']['interventions'][label]['interaction']-
                         result['patches']['light']['interventions'][label]['interaction'])
                removed[label] = native-after
            result['history_by_health_interaction'] = native
            result['history_by_health_removed'] = removed
            feature_results.append(result)
    readout_pass = {}
    for method in ('zero','independent'):
        rows = [r for r in feature_results if r['method']==method]
        readout_pass[method] = bool(np.mean([r['paired_rank'] for r in rows])>=.75 and
                                    min(r['paired_rank'] for r in rows)>=.625)
    result = dict(scope='CPU checks of recorded Qwen outputs, no new Qwen forward',
        cases=cases,comparisons=comparisons,features=feature_results,readout_gate=readout_pass,
        independent_seeds_original=4,independent_seeds_new=4,training_cases=len(cases),
        optimizer_steps=32*24,causal_target='body interaction, not global refusal',
        expansion_behavior_gate=all(r['working_gate_passed'] for r in comparisons.values()),
        training_heavy_health_range=[min(training_healths),max(training_healths)],
        training_heavy_encounters_below10=sum(h<=10 for h in training_healths),
        action_signal_rows=len(csv_rows))
    write(output/'summary.json',result)
    print(json.dumps(dict(training_cases=len(cases),comparisons=comparisons,readout_gate=readout_pass)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    analyze(args.root,args.output)
