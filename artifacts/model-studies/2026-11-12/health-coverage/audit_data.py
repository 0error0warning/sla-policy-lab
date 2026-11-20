"""Independent CPU audit of matched normal/low-health coverage trajectories.

No model inference, training, account access, or GPU operation is performed.
The Decimal transition/reward calculation is intentionally independent of the
GPU worker's protocol implementation.
"""

import argparse
from collections import Counter
from decimal import Decimal, localcontext
from hashlib import sha256
import json
from pathlib import Path


COSTS = (Decimal('0'), Decimal('.25'), Decimal('1.75'), Decimal('1.75'))
MATCHED_FIELDS = ('action', 'accepted', 'noise', 'order', 'partner', 'request')
THRESHOLDS = (2, 6, 10, 15, 30)
SEEDS = (62201, 62202, 62203, 62204)
TOLERANCE = Decimal('0.0000000001')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def number(value):
    return Decimal(str(value))


def check_close(actual, expected, description):
    difference = abs(number(actual)-expected)
    if difference > TOLERANCE:
        raise AssertionError(f'{description}: absolute difference {difference}')
    return difference


def body_transition(health, action, accepted, noise):
    if not (Decimal(0) < health <= Decimal(350)):
        raise AssertionError('A transition must start with a living body.')
    if action not in range(4) or len(noise) != 5:
        raise AssertionError('Each transition needs one valid action and five noise values.')
    if (action == 0) != (accepted is None):
        raise AssertionError('Only silent refusal has no observed partner outcome.')
    states = [health]
    for index, raw_noise in enumerate(noise):
        eta = number(raw_noise)
        if not Decimal('-.3') <= eta <= Decimal('.3'):
            raise AssertionError('Body noise is outside the fixed bounds.')
        damage = COSTS[action] if index == 0 else Decimal(0 if accepted else 1)
        health = max(Decimal(0), health-damage*(Decimal(2)-health/Decimal(350))*(Decimal(1)+eta))
        states.append(health)
    return states


def body_reward(before, after, action, accepted):
    common = Decimal(2)+int(action >= 2)+Decimal('.5')*int(accepted is True)
    return common-Decimal('.15')*(before-after)-12*int(after <= 0)


def health_bin(health):
    lower = None
    for threshold in THRESHOLDS:
        if health <= threshold:
            return f'<= {threshold}' if lower is None else f'({lower}, {threshold}]'
        lower = threshold
    return '> 30'


def observation_label(accepted):
    if accepted is None:
        return 'unobserved'
    if accepted is True:
        return 'maintain'
    if accepted is False:
        return 'skip'
    raise AssertionError('Partner feedback must be true, false, or null.')


def baseline(case):
    if 'baseline' in case:
        return case['baseline']
    if case.get('updates'):
        return case['updates'][0]['baseline_before']
    raise AssertionError('The case does not record its frozen baseline.')


def normalized_buffer(case, tape):
    if not isinstance(case.get('buffer'), list) or not case['buffer']:
        raise AssertionError('The case must contain its nonempty training buffer.')
    source = {row.get('turn', index+1): row for index, row in enumerate(tape['trace'])}
    result = []
    for index, row in enumerate(case['buffer']):
        turn = row.get('turn', index+1)
        if turn not in source:
            raise AssertionError(f'Turn {turn} is missing from the source tape.')
        merged = dict(source[turn])
        merged.update(row)
        merged['turn'] = turn
        for field in MATCHED_FIELDS:
            if field not in merged:
                raise AssertionError(f'Turn {turn} is missing {field}.')
            if field in row and field in source[turn] and row[field] != source[turn][field]:
                raise AssertionError(f'Turn {turn} changes the source {field}.')
        result.append(merged)
    if [row['turn'] for row in result] != list(range(1, len(result)+1)):
        raise AssertionError('Training turns must be a continuous sequence starting at one.')
    return result


def marginal(counter, names):
    return [dict(zip(names, key), n=count) for key, count in sorted(counter.items())]


def trajectory_summary(rows):
    maximum_error = Decimal(0)
    rewards = []
    expected_health = number(rows[0]['health_before'])
    healths = []
    joint = Counter()
    actions = Counter()
    feedback = Counter()
    for row in rows:
        maximum_error = max(maximum_error, check_close(row['health_before'], expected_health, 'Health continuity'))
        before = number(row['health_before'])
        states = body_transition(before, row['action'], row['accepted'], row['noise'])
        after = states[-1]
        maximum_error = max(maximum_error, check_close(row['health_after'], after, 'Decimal body settlement'))
        if 'body_events' in row:
            if len(row['body_events']) != len(states):
                raise AssertionError('The recorded body event count differs.')
            for actual, expected in zip(row['body_events'], states):
                maximum_error = max(maximum_error, check_close(actual, expected, 'Decimal body event'))
        if after <= 0:
            raise AssertionError('The coverage test contains a terminal transition.')
        expected_health = number(row['health_after'])
        actual_reward = body_reward(before, after, row['action'], row['accepted'])
        if 'reward_body' not in row:
            raise AssertionError('The training buffer does not record its actual body reward.')
        check_close(row['reward_body'], actual_reward, 'Body reward')
        rewards.append(actual_reward)
        label = observation_label(row['accepted'])
        action = str(row['action'])
        healths.append(float(before))
        joint[action, label, health_bin(float(before))] += 1
        actions[action,] += 1
        feedback[label,] += 1
    total = sum(rewards, Decimal(0))
    discounted = sum((Decimal('.97')**i)*reward for i, reward in enumerate(rewards))
    return dict(n=len(rows), actual_start=rows[0]['health_before'], final_health=rows[-1]['health_after'],
        action_health_range=[min(healths), max(healths)], action_healths=healths,
        at_or_below={str(t): sum(h <= t for h in healths) for t in THRESHOLDS},
        disjoint_health_bins=dict(Counter(health_bin(h) for h in healths)),
        action_marginal=marginal(actions, ['action']), feedback_marginal=marginal(feedback, ['feedback']),
        joint_action_feedback_health=marginal(joint, ['action', 'feedback', 'health_bin']),
        rewards=[float(r) for r in rewards], total_reward=float(total),
        discounted_total_reward=float(discounted), all_nonfatal=True,
        maximum_decimal_error=float(maximum_error))


def audit(root):
    raw_tapes = read_json(root/'tapes.json')
    tapes = raw_tapes['tapes'] if isinstance(raw_tapes, dict) else raw_tapes
    by_seed = {tape['seed']: tape for tape in tapes}
    if set(by_seed) != set(SEEDS):
        raise AssertionError(f'Expected the four prespecified seeds {SEEDS}.')
    pairs = []
    combined_rows = {'original': [], 'covered': []}
    with localcontext() as context:
        context.prec = 50
        for seed in SEEDS:
            cases = {label: read_json(root/f'{seed}-{label}.json') for label in combined_rows}
            buffers = {label: normalized_buffer(case, by_seed[seed]) for label, case in cases.items()}
            a, b = buffers['original'], buffers['covered']
            if len(a) != len(b):
                raise AssertionError('Paired training sample counts differ.')
            if len(a) != 8 or len(by_seed[seed]['trace']) != 8:
                raise AssertionError('The fixed-coverage comparison requires eight source encounters.')
            for normal, covered in zip(a, b):
                for field in MATCHED_FIELDS:
                    if normal[field] != covered[field]:
                        raise AssertionError(f'Seed {seed} turn {normal["turn"]} differs in {field}.')
                if 'frame' in normal and normal.get('frame') != covered.get('frame'):
                    raise AssertionError('Paired request frames differ.')
            left, right = baseline(cases['original']), baseline(cases['covered'])
            if left != right or len(left) != len(a):
                raise AssertionError('Paired baseline arrays are not exactly equal.')
            for case in cases.values():
                for update in case.get('updates', []):
                    if update['baseline_before'] != left:
                        raise AssertionError('An update used a different baseline from the frozen array.')
            if 'input_token_ids' in a[0]:
                for normal, covered in zip(a, b):
                    if normal['input_token_ids'] != covered['input_token_ids']:
                        raise AssertionError('Hidden-health training token sequences differ.')
            summaries = {label: trajectory_summary(rows) for label, rows in buffers.items()}
            if summaries['covered']['actual_start'] >= summaries['original']['actual_start']:
                raise AssertionError('The covered condition does not start below the original condition.')
            check_close(summaries['covered']['final_health'], Decimal(1), 'Covered end-of-tape reserve')
            for field in ('action_marginal', 'feedback_marginal'):
                if summaries['original'][field] != summaries['covered'][field]:
                    raise AssertionError(f'The {field} distributions differ.')
            differences = [r-l for l, r in zip(summaries['original']['rewards'], summaries['covered']['rewards'])]
            pairs.append(dict(seed=seed, paired_fixed_fields_identical=True, baseline_exact=True,
                baseline_sha256=sha256(json.dumps(left, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                coverage_start_metadata={label: case.get('coverage_start') for label, case in cases.items()},
                conditions=summaries, reward_difference_covered_minus_original=differences,
                total_reward_difference=sum(differences), mean_step_reward_difference=sum(differences)/len(a),
                discounted_total_reward_difference=summaries['covered']['discounted_total_reward']-
                    summaries['original']['discounted_total_reward']))
            for label in combined_rows:
                combined_rows[label].extend(buffers[label])
        # Each seed starts its own body, so aggregate distribution counts only.
        distributions = {}
        for label, rows in combined_rows.items():
            healths = [float(r['health_before']) for r in rows]
            joint = Counter((str(r['action']), observation_label(r['accepted']), health_bin(float(r['health_before']))) for r in rows)
            distributions[label] = dict(n=len(rows), health_range=[min(healths), max(healths)],
                at_or_below={str(t): sum(h <= t for h in healths) for t in THRESHOLDS},
                joint_action_feedback_health=marginal(joint, ['action', 'feedback', 'health_bin']))
    return dict(verified=True, independent_seeds=len(SEEDS), source_encounters=sum(len(by_seed[s]['trace']) for s in SEEDS),
        comparison_scope='Same-source-tape coverage intervention with frozen per-position baseline',
        construction='Covered initial health is retrospectively chosen from each complete recorded tape. The actor does not receive future events.',
        causal_scope='The estimated effect belongs to this controlled coverage-construction scheme. Health by action/feedback associations remain visible in the joint tables.',
        source='Independent Decimal body and reward implementation, without importing the GPU protocol',
        pairs=pairs, distributions=distributions,
        all_pairs_same_actions_feedback_noise_orders_requests=True, all_baselines_exact=True, all_nonfatal=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='Extracted GPU output directory')
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(verified=True, seeds=result['independent_seeds'], distributions=result['distributions'])))
