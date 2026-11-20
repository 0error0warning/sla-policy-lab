"""Extract the small tables used in the study report from the full CPU replay."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

from protocol import ARMS, write


def arm_summary(rows, mirror):
    final = [row['probes']['final']['metrics']['joint'] for row in rows]
    thresholds = {}
    for threshold in ('30.0', '10.0', '2.0'):
        exposure = [row['low_health_exposure'][threshold] for row in rows]
        thresholds[threshold] = {
            'encounters': sum(row['encounters'] for row in exposure),
            'nonfatal': sum(row['nonfatal'] for row in exposure),
            'answers': sum(row['answers'] for row in exposure),
            'lives_with_later_actions': sum(row['real_actions_after_first_update_with_low_health_feedback'] > 0
                                           for row in exposure),
            'fixed_trace_actions_after_update_by_block': {
                width: sum(row['fixed_trace_actions_available_after_update_by_block'].get(width, 0)
                           for row in exposure) for width in ('1', '2', '4', '8')},
        }
    training = [row['training'] for row in rows if row['training']['updates']]
    cells = defaultdict(int)
    for row in rows:
        for cell in row['context_counts']:
            cells[(row['id'], cell['partner'], cell['frame'], cell['action'])] += cell['n']
    answer_exposures = [n for key, n in cells.items() if key[-1] >= 2]
    margins = {split: mean(row['probes']['final']['metrics'][split]['correct_answer_style_margin']
                           for row in rows) for split in ('train', 'wording', 'content', 'joint')}
    return dict(low_health_exposure=thresholds, style_margins_by_split=margins,
                expected_survival_at_h2=mean(x['low_health_estimated_one_step_survival'] for x in final),
                answer_reduction_h300_to_h2=mean(x['healthy_answer_probability']-x['low_health_fatal_answer_probability']
                                                  for x in final),
                joint_greedy_order_flip=mean(x['order_greedy_flip'] for x in final),
                mirrored_both_prefer_correct_style=sum(row['n_both_correct'] for row in mirror),
                mirrored_probe_pairs=sum(len(row['rows']) for row in mirror),
                per_life_partner_frame_style_count=dict(median=median(answer_exposures),
                    zero=sum(n == 0 for n in answer_exposures), total=len(answer_exposures)),
                optimizer_steps=sum(x['optimizer_steps'] for x in training),
                final_parameter_delta_l2=mean(x['parameter_delta_l2'] for x in training) if training else 0.,
                mean_positive_advantage_fraction=mean(x['positive_advantage_fraction'] for x in training) if training else None,
                mean_weighted_training_logp_change=mean(x['mean_weighted_log_probability_change'] for x in training) if training else None)


def summarize(analysis):
    rows = analysis['results']
    observations = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        if selected:
            mirrors = [r for r in analysis['mirrored_style_preferences'] if r['arm'] == arm]
            observations[arm] = arm_summary(selected, mirrors)
    same_prefix = analysis['reward_prefix_interventions']
    comparison = analysis['first_update_ablations']
    return dict(lives=len(rows), rounds=sum(r['turns'] for r in rows),
                training_steps=sum(r['training']['optimizer_steps'] for r in rows),
                max_body_replay_error=max(r['max_decimal_error'] for r in rows),
                aggregate=analysis['aggregate'], by_stage=analysis['by_stage'],
                paired_effects=analysis['paired_effects'],
                reward_prefix_interventions=same_prefix, first_update_ablations=comparison,
                observations=observations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.analysis.read_bytes()))
    write(args.output, result)
    print(json.dumps({k: result[k] for k in ('lives', 'rounds', 'training_steps', 'max_body_replay_error')}))
