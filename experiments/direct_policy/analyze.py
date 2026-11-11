"""CPU reconstruction of recorded GPU policies, returns, gradients and weights.

This verifies recorded inference and updates; it does not rerun GPU Qwen forwards.
"""

from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

from protocol import (ACTIONS, ARMS, CONFIG, COSTS, canonical, digest, jobs, noises,
    policy, probabilities, prompt, render, reward, rng, sample_action, schedule, write)


def tensors(path):
    data=Path(path).read_bytes();length=struct.unpack('<Q',data[:8])[0]
    header=json.loads(data[8:8+length]);payload=memoryview(data)[8+length:]
    result={}
    for name,entry in header.items():
        if name=='__metadata__':continue
        assert entry['dtype']=='F32'
        start,end=entry['data_offsets']
        result[name]=np.frombuffer(payload[start:end],dtype='<f4').reshape(entry['shape'])
    assert len(result)==8 and sum(v.size for v in result.values())==188416
    return result


def weight_hash(values):
    result=sha256()
    for name,array in sorted(values.items()):
        result.update(canonical([name,'torch.float32',list(array.shape)]));result.update(array.tobytes())
    return result.hexdigest()


def decimal_body(health,action,accepted,noise):
    h=Decimal(str(health));values=[float(h)]
    for i,eta in enumerate(noise):
        cost=Decimal(str(COSTS[action])) if i==0 else Decimal(0 if accepted else 1)
        h=max(Decimal(0),h-cost*(Decimal(2)-h/Decimal(350))*(Decimal(1)+Decimal(str(eta))))
        values.append(float(h))
    return values


def validate_root(root):
    assert json.loads((root/'protocol.json').read_bytes())==CONFIG
    lock=json.loads((root/'run-lock.json').read_bytes())
    for name,checksum in lock['files'].items():assert sha256((root/name).read_bytes()).hexdigest()==checksum,name
    status=json.loads((root/'output/status.json').read_bytes())
    assert status['stage']=='complete' and status['base_unchanged']
    return status


def smoke_report(root):
    status=validate_root(root);out=root/'output'
    meaning=json.loads((out/'comprehension.json').read_bytes());p=policy(meaning['logits'])
    groups=defaultdict(list)
    for i,row in enumerate(meaning['rows']):
        action=row['order'][int(np.argmax(p[i]))]
        groups['meaning' if row['id'].startswith('meaning') else 'fatal_cost'].append(dict(
            id=row['id'],correct=bool(action in row['correct']),action=action,
            correct_probability=float(sum(p[i,j] for j,a in enumerate(row['order']) if a in row['correct']))))
    causality=json.loads((out/'reward-causality.json').read_bytes());checked=[]
    initial=tensors(out/'initial-early.safetensors')
    for row in causality['cases']:
        values=decimal_body(row['health'],2,True,[0.]*5)
        assert np.max(abs(np.array(values)-row['events']))<1e-10
        values_reward,_=reward(row['health'],row['after'],2,True,row['signal'].startswith('body'))
        if row['signal']=='zero':values_reward=0.
        assert abs(values_reward-row['reward'])<1e-12
        current=tensors(out/f"micro-{row['optimizer']}-{row['signal']}.safetensors")
        assert weight_hash(current)==row['final_adapter_sha256'] and row['reload_exact']
        delta=float(np.sqrt(sum(np.square(current[n].astype(float)-initial[n]).sum() for n in initial)))
        assert abs(delta-row['checkpoints'][-1]['parameter_delta_l2'])<1e-5
        if row['signal']=='zero':assert weight_hash(initial)==weight_hash(current)
        checked.append(dict(optimizer=row['optimizer'],signal=row['signal'],reward=row['reward'],
            checkpoints=row['checkpoints'],delta_l2=delta))
    for optimizer in ('adam','sgd'):
        high=next(x for x in causality['cases'] if x['optimizer']==optimizer and x['signal']=='social_high')
        low=next(x for x in causality['cases'] if x['optimizer']==optimizer and x['signal']=='social_low')
        assert high['input_messages']==low['input_messages'] and high['final_adapter_sha256']==low['final_adapter_sha256']
    return dict(verified=True,status=status,comprehension={name:dict(correct=sum(r['correct'] for r in rows),
        n=len(rows),rows=rows) for name,rows in groups.items()},reward_interventions=checked,
        batch_probability_difference=causality['batch_probability_difference'],
        scope='CPU verification of recorded GPU results, not a fresh GPU rerun')


def numerics_report(root):
    validate_root(root)
    report=json.loads((root/'output/numerics.json').read_bytes())
    gradients=tensors(root/'output/gradient.safetensors')
    assert sha256((root/'output/gradient.safetensors').read_bytes()).hexdigest()==report['gradient_sha256']
    norm=float(np.sqrt(sum(np.square(g.astype(float)).sum() for g in gradients.values())))
    assert abs(norm-report['gradient_norm'])<1e-9 and report['all_parameters_restored'] and report['base_unchanged']
    for row in report['scales']:
        finite=(row['plus']['logp']-row['minus']['logp'])/(2*row['parameter_step_l2'])
        assert finite==row['central_difference'] and abs(finite/norm-row['difference_over_autograd'])<1e-9
        assert row['restored_exact']
    assert all(r['reverse_order_exact'] for r in report['reverse_repeats'])
    return dict(report,verified=True,verification_scope='CPU reconstruction of recorded finite-scale diagnostic, not a new GPU forward')


def semantic(probability,order):
    result=np.zeros(4)
    for label,action in enumerate(order):result[action]=probability[label]
    return result


def probe_metrics(checkpoint,rows,mapping,initial):
    p=policy(checkpoint['logits']);q0=policy(np.array(initial)[checkpoint['indices']])
    cells=[]
    for j,index in enumerate(checkpoint['indices']):
        row=rows[index];values=semantic(p[j],row['order']);old=semantic(q0[j],row['order'])
        cells.append(dict(row,probabilities=values.tolist(),tv_from_initial=float(abs(values-old).sum()/2),
            greedy_action=int(values.argmax())))
    summary={}
    for split in ('train','wording','content','joint'):
        subset=[c for c in cells if c['partner']!='P86' and c['split']==split]
        if not subset:continue
        healthy=[c for c in subset if c['health']==300.]
        low=[c for c in subset if c['health']==2.]
        direction=[];conditional=[]
        for c in healthy:
            truth=probabilities(c['partner'],c['frame'],mapping)
            sign=1 if truth[1]>.5 else -1
            v=c['probabilities'];direction.append(sign*(v[2]-v[3]))
            conditional.append(sign*(v[2]-v[3])/(v[2]+v[3]))
        by=defaultdict(list)
        for c in subset:by[c['partner'],c['frame'],c['health']].append(c)
        order_tv=[];order_flips=[]
        for values in by.values():
            first=np.array(values[0]['probabilities'])
            order_tv.extend(float(abs(np.array(c['probabilities'])-first).sum()/2) for c in values[1:])
            order_flips.extend(int(c['greedy_action']!=values[0]['greedy_action']) for c in values[1:])
        summary[split]=dict(tv_from_initial=float(np.mean([c['tv_from_initial'] for c in subset])),
            correct_answer_style_margin=float(np.mean(direction)),conditional_answer_style_margin=float(np.mean(conditional)),
            healthy_answer_probability=float(np.mean([sum(c['probabilities'][2:]) for c in healthy])),
            low_health_refuse_appeal_probability=float(np.mean([c['probabilities'][1] for c in low])),
            low_health_fatal_answer_probability=float(np.mean([sum(c['probabilities'][2:]) for c in low])),
            low_health_estimated_one_step_survival=float(.15*np.mean([c['probabilities'][1] for c in low])),
            order_tv=float(np.mean(order_tv)) if order_tv else 0.,order_greedy_flip=float(np.mean(order_flips)) if order_flips else 0.)
    body_changes=[]
    for c in cells:
        if c['health']!=300. or c['partner']=='P86':continue
        paired=next(x for x in cells if all(x[k]==c[k] for k in ('partner','frame','task','wording','order')) and x['health']==2.)
        body_changes.append(float(abs(np.array(c['probabilities'])-paired['probabilities']).sum()/2))
    return dict(turn=checkpoint['turn'],metrics=summary,body_high_low_tv=float(np.mean(body_changes)),rows=cells)


def inspect_life(root,path,probe_rows,initial):
    life=json.loads(path.read_bytes());arm=ARMS[life['arm']]
    assert arm==life['arm_configuration'] and life['config_sha256']==digest(CONFIG)
    trace=life['trace'];health=350.;maximum=0.;chain=[];tip=digest({k:life[k] for k in ['id','seed','mapping','arm']})
    assert 0<len(trace)<=CONFIG['horizon'] and life['decision_source']=='qwen_categorical_policy'
    version=0;updates_by_end={u['turns'][-1]:u for u in life['updates']}
    for item,row in zip(schedule(life['seed']),trace):
        assert all(row[k]==v for k,v in item.items())
        assert row['encounter_id']==f"{life['id']}-t{item['turn']:03}" and row['policy_version']==version
        assert health>0 and row['decision_source']=='qwen_categorical_policy'
        assert abs(row['health_before']-health)<1e-12
        expected_prompt=prompt(item['partner'],item['request'],health,item['order'],arm['body_visible'])
        assert row['input_messages']==expected_prompt
        p=policy(row['label_logits']);assert np.max(abs(p-row['label_probabilities']))<1e-12
        action,label,u=sample_action(p,item['order'],life['seed'],item['turn'])
        assert action==row['action'] and label==row['label_index'] and u==row['choice_uniform']
        assert row['action_name']==ACTIONS[action] and row['utterance']==render(action,item['task'],item['wording'])
        assert np.max(abs(semantic(p,item['order'])-row['semantic_probabilities']))<1e-12
        assert abs(np.log(p[label])-row['old_log_probability'])<1e-12
        private=probabilities(item['partner'],item['frame'],life['mapping'])
        draw=float(rng(life['seed'],'partner',item['turn']).random())
        accepted=bool(draw<private[action-1]) if action else None
        assert accepted==row['accepted'] and private==row['environment_probabilities']
        assert draw==row['potential_response_uniform'] and np.max(abs(np.array(noises(life['seed'],item['turn']))-row['noise']))<1e-12
        independent=decimal_body(health,action,accepted,row['noise'])
        maximum=max(maximum,max(abs(a-b) for a,b in zip(independent,row['body_events'])))
        assert maximum<1e-10
        after=row['health_after'];assert abs(independent[-1]-after)<1e-10
        common=2+int(action>=2)+.5*int(accepted is True)
        body=common-.15*(health-after)-12*int(after<=0)
        assert abs(common-row['reward_common'])<1e-12 and abs(body-row['reward_body'])<1e-12
        assert row['physical_death']==(after<=0) and row['capped_alive']==(item['turn']==CONFIG['horizon'] and after>0)
        _,components=reward(health,after,action,accepted)
        assert components==row['reward_components']
        health=after;tip=sha256(bytes.fromhex(tip)+canonical(row)).hexdigest()
        chain.append(dict(turn=item['turn'],sha256=tip))
        if item['turn'] in updates_by_end:
            version+=1;assert updates_by_end[item['turn']]['policy_version_after']==version
    assert len(trace)==life['turns'] and health==life['final_health']
    assert life['physical_death']==(health<=0) and life['capped_alive']==(health>0)
    assert len(trace)==72 or health==0
    persistent=[json.loads(line) for line in (path.parent/'trace.jsonl').read_text().splitlines()]
    assert persistent==trace
    baseline=np.zeros(8);steps=0
    for update in life['updates']:
        indices=[turn-1 for turn in update['turns']];n=len(indices)
        values=[trace[i]['reward_body' if arm['body_reward'] else 'reward_common'] for i in indices]
        discounted=np.array([sum(arm['gamma']**(j-i)*values[j] for j in range(i,n)) for i in range(n)])
        assert np.max(abs(discounted-update['discounted_returns']))<1e-10
        assert np.max(abs(baseline[:n]-update['baseline_before']))<1e-10
        raw=(discounted-baseline[:n])/8
        perm=np.array(update['advantage_permutation']);expected=rng(life['seed'],'shuffle-advantages',update['block']).permutation(n) if arm['shuffle'] else np.arange(n)
        assert np.array_equal(perm,expected) and np.max(abs(raw-update['raw_advantages']))<1e-10
        advantage=raw[perm];assert np.max(abs(advantage-update['advantages']))<1e-10
        assert np.array_equal(np.sort(advantage),np.sort(raw))
        baseline[:n]=.5*baseline[:n]+.5*discounted
        assert np.max(abs(baseline[:n]-update['baseline_after']))<1e-10
        assert len(update['steps'])==arm['epochs']*int(np.ceil(n/4))
        for step in update['steps']:
            ratio=np.array(step['ratio']);a=advantage[step['indices']]
            loss=-np.minimum(ratio*a,np.clip(ratio,.8,1.2)*a).mean()
            assert abs(loss-step['policy_loss'])<1e-5
            assert abs(step['loss']-(step['policy_loss']-.02*step['entropy']))<1e-5
            assert np.isfinite(step['gradient_norm']);steps+=1
        old=np.array(update['selected_log_probability_before']);new=np.array(update['selected_log_probability_after'])
        assert np.max(abs(old-[trace[i]['old_log_probability'] for i in indices]))<1e-12
        assert abs(advantage@(new-old)/n-update['weighted_log_probability_change'])<1e-10
    if arm['update']:
        updates_jsonl=[json.loads(line) for line in (path.parent/'updates.jsonl').read_text().splitlines()]
        assert updates_jsonl==life['updates']
        assert [turn for update in life['updates'] for turn in update['turns']]==list(range(1,len(trace)+1))
        assert all(len(update['turns'])==8 for update in life['updates'][:-1])
        start=tensors(root/f"output/initial-{arm['adapter']}.safetensors")
        end=tensors(path.parent/'adapter-final.safetensors')
        assert weight_hash(end)==life['weight']['sha256'] and life['weight']['reload_exact']
        delta=float(np.sqrt(sum(np.square(end[n].astype(float)-start[n]).sum() for n in start)))
        assert abs(delta-life['weight']['delta_l2'])<1e-5 and delta>0
    else:
        assert not life['updates'] and life['weight'] is None
        assert life['checkpoints']['0']['logits']==life['checkpoints']['final']['logits']
    scored={name:probe_metrics(cp,probe_rows,life['mapping'],initial) for name,cp in life['checkpoints'].items()}
    final=scored['final']
    scores=[r['reward_body'] for r in trace]
    common_total=float(sum(r['reward_common'] for r in trace))
    damage_total_penalty=.15*(350.-health)
    death_total_penalty=12.*int(health<=0)
    assert abs(sum(scores)-(common_total-damage_total_penalty-death_total_penalty))<1e-9
    advantages=[v for u in life['updates'] for v in u['advantages']]
    training_steps=[s for u in life['updates'] for s in u['steps']]
    training=dict(updates=len(life['updates']),optimizer_steps=steps,
        parameter_delta_l2=life['weight']['delta_l2'] if life['weight'] else 0.,
        positive_advantage_fraction=float(np.mean(np.array(advantages)>0)) if advantages else None,
        mean_weighted_log_probability_change=float(np.mean([u['weighted_log_probability_change'] for u in life['updates']])) if advantages else None,
        mean_entropy=float(np.mean([s['entropy'] for s in training_steps])) if training_steps else None,
        mean_clip_fraction=float(np.mean([s['clip_fraction'] for s in training_steps])) if training_steps else None,
        mean_gradient_norm=float(np.mean([s['gradient_norm'] for s in training_steps])) if training_steps else None,
        terminal_reward_seen=bool(life['updates'] and health<=0),
        n_real_decisions_after_terminal_reward=0)
    exposure={}
    for threshold in (30.,10.,2.):
        selected=[r for r in trace if r['health_before']<=threshold]
        available=[u['turns'][-1] for u in life['updates'] if any(trace[t-1]['health_before']<=threshold for t in u['turns'])]
        cadence_opportunity = {}
        if arm['update']:
            for width in (1, 2, 4, 8):
                if not selected:
                    cadence_opportunity[str(width)] = 0
                    continue
                first_turn = selected[0]['turn']
                block_end = min(len(trace), ((first_turn-1)//width+1)*width)
                cadence_opportunity[str(width)] = len(trace)-block_end
        exposure[str(threshold)]=dict(encounters=len(selected),
            nonfatal=sum(r['health_after']>0 for r in selected),
            answers=sum(r['action']>=2 for r in selected),
            refuge_actions=sum(r['action']==1 for r in selected),
            real_actions_after_first_update_with_low_health_feedback=(len(trace)-min(available)) if available else 0,
            fixed_trace_actions_available_after_update_by_block=cadence_opportunity)
        if arm['update']:
            assert cadence_opportunity['8']==exposure[str(threshold)]['real_actions_after_first_update_with_low_health_feedback']
    context_counts=[]
    for partner in ('P17','P42'):
        for frame in (0,1):
            for action in range(4):
                selected=[r for r in trace if r['partner']==partner and r['frame']==frame and r['action']==action]
                context_counts.append(dict(partner=partner,frame=frame,action=action,n=len(selected),
                    maintained=sum(r['accepted'] is True for r in selected)))
    return dict(id=life['id'],seed=life['seed'],mapping=life['mapping'],arm=life['arm'],turns=len(trace),
        answers=sum(r['action']>=2 for r in trace),protected=sum(r['accepted'] is True for r in trace),
        action_counts=[sum(r['action']==a for r in trace) for a in range(4)],health=health,physical_death=health<=0,
        body_reward_total=float(sum(scores)),social_reward_total=common_total,
        reward_decomposition=dict(common=common_total,damage_penalty=damage_total_penalty,death_penalty=death_total_penalty),
        discounted_body_return=float(sum(.97**i*r for i,r in enumerate(scores))),optimizer_steps=steps,
        max_decimal_error=maximum,chain_tip=tip,chain=chain,probes=scored,training=training,
        low_health_exposure=exposure,context_counts=context_counts,
        final_style_margin=final['metrics']['joint']['correct_answer_style_margin'],
        final_low_fatal=final['metrics']['joint']['low_health_fatal_answer_probability'],
        final_low_refuse_appeal=final['metrics']['joint']['low_health_refuse_appeal_probability'],
        final_body_tv=final['body_high_low_tv'],final_order_tv=final['metrics']['joint']['order_tv'])


def reward_pairs(lives):
    """Post hoc, prefix-matched check of damage feedback -> later real actions.

    No counterfactual model forward is invented. Before the first differing
    action, both arms really encountered identical bodies and partner events.
    A terminal penalty cannot have caused a decision taken before either death.
    """
    by={(r['seed'],r['mapping'],r['arm']):r for r in lives}
    results=[]
    for seed,mapping in sorted({(r['seed'],r['mapping']) for r in lives}):
        if (seed,mapping,'social') not in by or (seed,mapping,'body') not in by:continue
        social=by[seed,mapping,'social'];body=by[seed,mapping,'body']
        assert {k:v for k,v in social['arm_configuration'].items() if k!='body_reward'}=={
            k:v for k,v in body['arm_configuration'].items() if k!='body_reward'}
        a,b=social['trace'],body['trace']
        keys=('input_messages','input_token_ids','action','accepted','health_before',
              'health_after','choice_uniform','noise','potential_response_uniform')
        assert len(a)>=9 and len(b)>=9
        for x,y in zip(a[:8],b[:8]):
            assert all(x[k]==y[k] for k in keys)
            assert x['label_logits']==y['label_logits'] and x['health_after']>0
        first_a,first_b=a[8],b[8]
        assert first_a['input_token_ids']==first_b['input_token_ids']
        assert first_a['choice_uniform']==first_b['choice_uniform']
        assert first_a['health_before']==first_b['health_before']
        # Only the realised, nonfatal injury term differs in this first buffer.
        assert social['updates'][0]['rewards']==[x['reward_common'] for x in a[:8]]
        assert body['updates'][0]['rewards']==[x['reward_body'] for x in b[:8]]
        assert social['updates'][0]['baseline_before']==body['updates'][0]['baseline_before']==[0.]*8
        assert [s['indices'] for s in social['updates'][0]['steps']]==[s['indices'] for s in body['updates'][0]['steps']]
        divergence=None;last_identical=0
        for x,y in zip(a,b):
            assert x['input_token_ids']==y['input_token_ids']
            assert x['health_before']==y['health_before'] and x['choice_uniform']==y['choice_uniform']
            if x['action']!=y['action']:
                divergence=dict(turn=x['turn'],health_before=x['health_before'],
                    social_action=x['action'],body_action=y['action'],choice_uniform=x['choice_uniform'],
                    social_probabilities=x['semantic_probabilities'],body_probabilities=y['semantic_probabilities'],
                    same_input=True,both_alive_before_choice=True,
                    no_prior_terminal_penalty=all(z['health_after']>0 for z in a[:x['turn']-1]))
                assert divergence['no_prior_terminal_penalty']
                break
            assert all(x[k]==y[k] for k in keys)
            last_identical=x['turn']
        results.append(dict(seed=seed,mapping=mapping,first_eight_transitions_identical=True,
            first_update_nonfatal=True,identical_action_prefix=last_identical,
            first_update_mean_common_reward=float(np.mean(social['updates'][0]['rewards'])),
            first_update_mean_body_reward=float(np.mean(body['updates'][0]['rewards'])),
            turn9_tv=float(abs(np.array(first_a['semantic_probabilities'])-first_b['semantic_probabilities']).sum()/2),
            first_actual_action_divergence=divergence))
    return dict(scope='Post hoc reconstruction of recorded, matched prefixes; no new GPU forward',
        pairs=results,n_pairs=len(results),
        n_with_actual_action_divergence=sum(r['first_actual_action_divergence'] is not None for r in results))


def paired_effects(results):
    """Descriptive paired effects, with seed means kept separate from mappings."""
    by={(r['seed'],r['mapping'],r['arm']):r for r in results};effects={}
    contrasts=[('social','frozen'),('body','frozen'),('body','social'),
        ('body_blind','body'),('myopic','body'),('shuffled','body'),('tail','body'),('more_steps','body')]
    fields=['turns','answers','protected','health','body_reward_total','discounted_body_return',
        'final_style_margin','final_low_fatal','final_body_tv','final_order_tv']
    for treatment,control in contrasts:
        keys=sorted((s,m) for s,m,a in by if a==treatment and (s,m,control) in by)
        if not keys:continue
        rows=[dict(seed=s,mapping=m,**{f:by[s,m,treatment][f]-by[s,m,control][f] for f in fields}) for s,m in keys]
        seeds=sorted({r['seed'] for r in rows})
        seed_rows=[dict(seed=s,**{f:float(np.mean([r[f] for r in rows if r['seed']==s])) for f in fields}) for s in seeds]
        effects[f'{treatment}-minus-{control}']=dict(n_environments=len(rows),n_independent_seeds=len(seeds),
            rows=rows,seed_means=seed_rows,mean={f:float(np.mean([r[f] for r in seed_rows])) for f in fields},
            seed_range={f:[min(r[f] for r in seed_rows),max(r[f] for r in seed_rows)] for f in fields})
    return effects


def first_update_ablations(lives):
    """Measure the first policy effect before differing choices change the data."""
    by_key = {(life['seed'], life['mapping'], life['arm']): life for life in lives}
    fields = ('input_token_ids', 'action', 'label_logits', 'health_before',
              'health_after', 'accepted', 'choice_uniform')
    comparisons = []
    for reference in (life for life in lives if life['arm'] == 'body'):
        seed, mapping = reference['seed'], reference['mapping']
        for arm in ('myopic', 'shuffled', 'tail', 'more_steps'):
            alternative = by_key.get((seed, mapping, arm))
            if alternative is None:
                continue
            changes = {k: [reference['arm_configuration'][k], alternative['arm_configuration'][k]]
                       for k in reference['arm_configuration']
                       if reference['arm_configuration'][k] != alternative['arm_configuration'][k]}
            assert len(changes) == 1
            matched = all(all(x[k] == y[k] for k in fields)
                          for x, y in zip(reference['trace'][:8], alternative['trace'][:8]))
            row = dict(seed=seed, mapping=mapping, arm=arm, setting_difference=changes,
                       identical_first_eight=matched)
            if matched:
                a, b = reference['trace'][8], alternative['trace'][8]
                assert a['input_token_ids'] == b['input_token_ids']
                assert a['choice_uniform'] == b['choice_uniform']
                delta = np.asarray(b['semantic_probabilities']) - a['semantic_probabilities']
                row.update(turn9_policy_tv=float(np.abs(delta).sum()/2),
                           turn9_action_changed=a['action'] != b['action'])
            comparisons.append(row)
    return comparisons


def aggregate_scores(results):
    aggregate={}
    for arm in ARMS:
        selected=[r for r in results if r['arm']==arm]
        if not selected:continue
        fields=['turns','answers','protected','health','body_reward_total','social_reward_total','discounted_body_return',
            'final_style_margin','final_low_fatal','final_low_refuse_appeal','final_body_tv','final_order_tv']
        aggregate[arm]=dict(n=len(selected),independent_seeds=len({r['seed'] for r in selected}),
            died=sum(r['physical_death'] for r in selected),**{f:float(np.mean([r[f] for r in selected])) for f in fields})
    return aggregate


def mirrored_style_preferences(results):
    """On the same held-out input, require each mirror life to prefer its own style."""
    by = {(r['seed'], r['arm'], r['mapping']): r for r in results}
    comparisons = []
    for (seed, arm, mapping), left in sorted(by.items()):
        if mapping != 0 or (seed, arm, 1) not in by:
            continue
        right = by[seed, arm, 1]
        selected = []
        for a, b in zip(left['probes']['final']['rows'], right['probes']['final']['rows']):
            assert a['id'] == b['id']
            if a['health'] != 300. or a['split'] != 'joint' or a['partner'] == 'P86':
                continue
            truth = probabilities(a['partner'], a['frame'], 0)
            sign = 1 if truth[1] > .5 else -1
            margin_a = sign*(a['probabilities'][2]-a['probabilities'][3])
            margin_b = -sign*(b['probabilities'][2]-b['probabilities'][3])
            selected.append(dict(probe=a['id'], mapping0_correct_margin=margin_a,
                                 mapping1_correct_margin=margin_b,
                                 both_prefer_correct_style=bool(min(margin_a, margin_b)>0)))
        assert len(selected) == 16
        comparisons.append(dict(seed=seed, arm=arm, rows=selected,
                                n_both_correct=sum(row['both_prefer_correct_style'] for row in selected)))
    return comparisons


def analyze(roots,output,smoke=None,numerics=None):
    output.mkdir(parents=True,exist_ok=True)
    results=[];statuses={};raw_lives=[];initial_reference=None
    for root in roots:
        status=validate_root(root);stage=status['study_stage'];statuses[stage]=status
        probe_rows=json.loads((root/'output/probe-rows.json').read_bytes())
        initial=json.loads((root/'output/initial-probes.json').read_bytes())
        if initial_reference is None:
            initial_reference=(probe_rows,initial,status['base_tensor_sha256'],status['adapter_initial_sha256'])
        else:
            assert initial_reference==(probe_rows,initial,status['base_tensor_sha256'],status['adapter_initial_sha256'])
        for adapter,expected in status['adapter_initial_sha256'].items():
            assert weight_hash(tensors(root/f'output/initial-{adapter}.safetensors'))==expected
        assert status['completed_jobs']==len(jobs(stage))
        for job in jobs(stage):
            path=root/'output'/job['id']/'result.json'
            raw_lives.append(json.loads(path.read_bytes()))
            row=inspect_life(root,path,probe_rows,initial[str(ARMS[job['arm']]['body_visible'])])
            row['stage']=stage
            results.append(row)
            write(output/'chains'/(row['id']+'.json'),row.pop('chain'))
        assert sum(r['optimizer_steps'] for r in results if r['stage']==stage)==status['optimizer_steps']
    aggregate=aggregate_scores(results)
    result=dict(config=CONFIG,statuses=statuses,all_recorded_events_verified=True,results=results,aggregate=aggregate,
        paired_effects=paired_effects(results),reward_prefix_interventions=reward_pairs(raw_lives),
        first_update_ablations=first_update_ablations(raw_lives),
        mirrored_style_preferences=mirrored_style_preferences(results),
        by_stage={stage:dict(aggregate=aggregate_scores([r for r in results if r['stage']==stage]),
            paired_effects=paired_effects([r for r in results if r['stage']==stage])) for stage in statuses},
        scope='CPU reconstruction of recorded GPU inference; no fresh Qwen forward',
        analysis_provenance=dict(python=sys.version,numpy=np.__version__,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    if smoke is not None:result['smoke']=smoke_report(smoke)
    if numerics is not None:result['numerics']=numerics_report(numerics)
    write(output/'scored.json',result)
    print(json.dumps(dict(lives=len(results),records=sum(r['turns'] for r in results),aggregate=aggregate)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,action='append',default=[])
    parser.add_argument('--smoke',type=Path);parser.add_argument('--numerics',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.root:analyze(args.root,args.output,args.smoke,args.numerics)
    elif args.smoke:
        result=smoke_report(args.smoke);write(args.output/'smoke.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('reward_interventions','status')}))
    elif args.numerics:
        result=numerics_report(args.numerics);write(args.output/'numerics.json',result)
        print(json.dumps(dict(verified=True,gradient_norm=result['gradient_norm'],
            scales=[dict(step=r['parameter_step_l2'],difference=r['central_difference'],ratio=r['difference_over_autograd'])
                for r in result['scales']])))
    else:parser.error('supply --root, --smoke, or --numerics')
