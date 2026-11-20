"""Score paired coverage effects and replay the recorded learning signals."""
import argparse
import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/'inputs'))
from protocol import World, canonical, policy, returns, sample_action, write
from risk_reference import expected_death


def read(path):
    return json.loads(path.read_bytes())


def adapter_hash(path):
    data=path.read_bytes()
    length=struct.unpack('<Q',data[:8])[0]
    header=json.loads(data[8:8+length])
    payload=memoryview(data)[8+length:]
    checksum=sha256()
    total=0
    for name, entry in sorted(header.items()):
        if name=='__metadata__':
            continue
        assert entry['dtype']=='F32'
        start,end=entry['data_offsets']
        total+=(end-start)//4
        checksum.update(canonical([name,'torch.float32',entry['shape']]))
        checksum.update(payload[start:end])
    assert total==188416
    return checksum.hexdigest()


def semantic(logits,panel):
    return np.take_along_axis(policy(logits),np.argsort([r['order'] for r in panel],axis=1),axis=1)


def metrics(logits,panel,mapping):
    p=semantic(logits,panel)
    groups={}
    for visible in (True,False):
        for health in (300.,90.,15.,10.,6.,2.):
            indices=[i for i,r in enumerate(panel) if r['visible']==visible and
                     (not visible or r['health']==health)]
            values=p[indices]
            risks=np.array([expected_death(health,panel[i]['partner'],panel[i]['frame'],mapping) for i in indices])
            risk_contributions=values*risks
            frames=np.array([panel[i]['frame'] for i in indices])
            per_frame={}
            for frame in (0,1):
                mask=frames==frame
                per_frame[str(frame)]=dict(
                    answer=float(values[mask,2:].sum(axis=1).mean()),
                    action_probabilities=values[mask].mean(axis=0).tolist(),
                    death_risk=float(risk_contributions[mask].sum(axis=1).mean()))
            groups[f"{'visible' if visible else 'hidden'}-h{health:g}"]=dict(
                answer=float(values[:,2:].sum(axis=1).mean()),
                refuse_appeal=float(values[:,1].mean()),silent=float(values[:,0].mean()),
                action_probabilities=values.mean(axis=0).tolist(),
                death_risk=float(risk_contributions.sum(axis=1).mean()),
                action_death_contributions=risk_contributions.mean(axis=0).tolist(),
                by_frame=per_frame)
    return groups


def uncertainty(changes):
    values=np.array(changes)
    draws=np.array(list(product(range(len(values)),repeat=len(values))))
    return dict(values=values.tolist(),mean=float(values.mean()),range=[float(values.min()),float(values.max())],
                descriptive_seed_bootstrap_95=np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist())


def summarize(root,output):
    output.mkdir(parents=True,exist_ok=True)
    panel=read(root/'panel.json')
    tapes=read(root/'tapes.json')
    baseline=read(root/'baseline.json')['baseline']
    for tape in tapes:
        world=World(tape['seed'],tape['mapping'],f"coverage-source-{tape['seed']}")
        for r in tape['trace']:
            action,label,uniform=sample_action(policy(r['source_logits']),r['order'],tape['seed'],r['turn'])
            assert (action,label,uniform)==(r['action'],r['source_label'],r['source_choice_uniform'])
            expected=world.step(world.encounter_id(),action)
            assert all(r[k]==v for k,v in expected.items())
    cases={}
    signals=[]
    for tape in tapes:
        seed=tape['seed']
        for regime in ('original','covered'):
            case=read(root/f'{seed}-{regime}.json')
            assert adapter_hash(root/f'{seed}-{regime}.safetensors')==case['adapter_sha256']
            assert case['baseline']==baseline
            buffer=case['buffer']
            discounted=returns([r['reward_body'] for r in buffer],.97)
            advantages=(discounted-baseline)/8
            for update,n in zip(case['updates'],(8,16)):
                assert len(update['steps'])==n
                assert np.allclose(update['advantages'],advantages,atol=1e-12,rtol=0)
                for step in update['steps']:
                    ratio=np.array(step['ratio'])
                    adv=advantages[step['indices']]
                    loss=-np.minimum(ratio*adv,np.clip(ratio,.8,1.2)*adv).mean()
                    assert abs(loss-step['policy_loss'])<1e-5
            training_before=policy(case['training_before_logits'])
            training8=policy(case['doses']['8']['training_logits'])
            training24=policy(case['doses']['24']['training_logits'])
            for i,row in enumerate(buffer):
                label=row['label_index']
                signals.append(dict(seed=seed,regime=regime,turn=i+1,action=row['action'],
                    accepted=row['accepted'],health_before=row['health_before'],health_after=row['health_after'],
                    reward=row['reward_body'],baseline=baseline[i],advantage=advantages[i],
                    cached_probability=float(np.exp(row['old_log_probability'])),
                    actual_before=training_before[i,label],after8=training8[i,label],after24=training24[i,label]))
            cases[f'{seed}-{regime}']=dict(seed=seed,mapping=tape['mapping'],regime=regime,
                doses={dose:metrics(data['logits'],panel,tape['mapping']) for dose,data in case['doses'].items()},
                initial=metrics(read(root/'initial.json'),panel,tape['mapping']),
                advantage_signs=dict(negative=int(sum(advantages<0)),positive=int(sum(advantages>0))),
                mean_advantage=float(advantages.mean()))
    with (output/'action-signals.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(signals[0]))
        writer.writeheader()
        writer.writerows(signals)
    paired={}
    by_seed={}
    for tape in tapes:
        seed=tape['seed']
        a=cases[f'{seed}-original']['doses']['24']
        b=cases[f'{seed}-covered']['doses']['24']
        row={}
        for visible in ('visible','hidden'):
            row[f'{visible}_danger_death']=np.mean([
                b[f'{visible}-h{h}']['death_risk']-a[f'{visible}-h{h}']['death_risk'] for h in (6,10)])
            row[f'{visible}_normal_answer']=b[f'{visible}-h300']['answer']-a[f'{visible}-h300']['answer']
            row[f'{visible}_h2_fatal_answer']=b[f'{visible}-h2']['answer']-a[f'{visible}-h2']['answer']
            row[f'{visible}_h2_total_death']=b[f'{visible}-h2']['death_risk']-a[f'{visible}-h2']['death_risk']
            row[f'{visible}_body_interaction']=(
                b[f'{visible}-h300']['answer']-b[f'{visible}-h2']['answer']-
                a[f'{visible}-h300']['answer']+a[f'{visible}-h2']['answer'])
        by_seed[str(seed)]={k:float(v) for k,v in row.items()}
    for key in next(iter(by_seed.values())):
        paired[key]=uncertainty([r[key] for r in by_seed.values()])
    behavior_gate=bool(paired['visible_danger_death']['mean']<=-.01 and
                       sum(v<0 for v in paired['visible_danger_death']['values'])>=3 and
                       paired['visible_normal_answer']['mean']>=-.02)
    result=dict(cases=cases,paired=paired,by_seed=by_seed,behavior_gate=behavior_gate,
                independent_seeds=4,main_training_cases=8,main_optimizer_steps=192,
                repeated_numeric_steps=72,signal_rows=len(signals),
                inference_scope='Recorded GPU logits, CPU scoring with independent body-risk integration')
    write(output/'summary.json',result)
    print(json.dumps(dict(by_seed=by_seed,paired=paired,behavior_gate=behavior_gate)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    summarize(a.root,a.output)
