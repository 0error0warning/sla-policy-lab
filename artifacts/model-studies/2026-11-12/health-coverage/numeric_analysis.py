"""CPU analysis of recorded Qwen repeatability measurements.

Nonzero measurement differences are results, not validation failures. This tool
reads JSON records and uses NumPy only; it never loads a language model.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


LOCAL_INPUTS = Path(__file__).resolve().parent / 'inputs'
sys.path.insert(0, str(LOCAL_INPUTS))
from protocol import policy


def load(path):
    return json.loads(path.read_bytes())


def semantic_probabilities(logits, rows):
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError('Expected an N by 4 array of action logits')
    if len(values) != len(rows):
        raise ValueError('The action panel and logit array have different lengths')
    order = np.asarray([row['order'] for row in rows], dtype=int)
    if order.shape != values.shape or not np.all(np.sort(order, axis=1) == np.arange(4)):
        raise ValueError('Every panel row must contain a permutation of four actions')
    return np.take_along_axis(policy(values), np.argsort(order, axis=1), axis=1)


def compare_logits(reference, candidate, rows):
    a = np.asarray(reference, dtype=float)
    b = np.asarray(candidate, dtype=float)
    result = dict(reference_shape=list(a.shape), candidate_shape=list(b.shape),
                  same_shape=a.shape == b.shape)
    if a.shape != b.shape:
        return result
    result['finite'] = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not result['finite']:
        return result
    p = semantic_probabilities(a, rows)
    q = semantic_probabilities(b, rows)
    raw_difference = np.abs(b - a)
    probability_difference = np.abs(q - p)
    centered = (b - b.mean(axis=1, keepdims=True)) - (a - a.mean(axis=1, keepdims=True))
    result.update(
        logits_exact=bool(np.array_equal(a, b)),
        semantic_probabilities_exact=bool(np.array_equal(p, q)),
        max_raw_logit_difference=float(raw_difference.max()),
        mean_raw_logit_difference=float(raw_difference.mean()),
        max_centered_logit_difference=float(np.abs(centered).max()),
        max_semantic_probability_difference=float(probability_difference.max()),
        mean_semantic_probability_difference=float(probability_difference.mean()),
        max_semantic_probability_difference_pp=float(100 * probability_difference.max()),
        mean_semantic_probability_difference_pp=float(100 * probability_difference.mean()),
        mean_semantic_total_variation=float(probability_difference.sum(axis=1).mean() / 2),
        max_semantic_total_variation=float(probability_difference.sum(axis=1).max() / 2),
        changed_greedy_semantic_actions=int(np.sum(p.argmax(axis=1) != q.argmax(axis=1))),
        n_rows=len(rows),
    )
    return result


def compare_state(reference, candidate):
    if reference is None or candidate is None:
        return dict(available=False, reference_available=reference is not None,
                    candidate_available=candidate is not None)
    a = reference.get('buffers', {})
    b = candidate.get('buffers', {})
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(name for name in set(a) & set(b) if a[name] != b[name])
    rng = {}
    for key in ('cpu_rng', 'cuda_rng'):
        present = key in reference and key in candidate
        rng[key] = dict(available=present, same=reference[key] == candidate[key] if present else None,
                        reference=reference.get(key), candidate=candidate.get(key))
    return dict(available=True, rng=rng, buffers=dict(
        reference_count=len(a), candidate_count=len(b), identical=not (added or removed or changed),
        changed_names=changed, added_names=added, removed_names=removed))


def nested_differences(reference, candidate, prefix=''):
    differences = {}
    for key in sorted(set(reference) | set(candidate)):
        path = f'{prefix}.{key}' if prefix else key
        a = reference.get(key)
        b = candidate.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            differences.update(nested_differences(a, b, path))
        elif a != b or (key in reference) != (key in candidate):
            differences[path] = dict(reference=a, candidate=b,
                                     reference_present=key in reference,
                                     candidate_present=key in candidate)
    return differences


def compare_case(reference, candidate, panel):
    training_rows = reference['buffer']
    same_training_tokens = [r['input_token_ids'] for r in training_rows] == [
        r['input_token_ids'] for r in candidate['buffer']]
    same_training_orders = [r['order'] for r in training_rows] == [
        r['order'] for r in candidate['buffer']]
    reset_parameter_hash_available = ('reset_parameter_sha256' in reference
                                      and 'reset_parameter_sha256' in candidate)
    result = dict(
        same_adapter_sha256=reference.get('adapter_sha256') == candidate.get('adapter_sha256'),
        reference_adapter_sha256=reference.get('adapter_sha256'),
        candidate_adapter_sha256=candidate.get('adapter_sha256'),
        reset_parameter_hash_available=reset_parameter_hash_available,
        same_reset_parameter_sha256=(reference['reset_parameter_sha256'] == candidate['reset_parameter_sha256']
                                    if reset_parameter_hash_available else None),
        same_training_tokens=same_training_tokens,
        same_training_orders=same_training_orders,
        same_baseline=reference['baseline'] == candidate['baseline'],
        same_buffer=training_rows == candidate['buffer'],
        reset_state=compare_state(reference.get('reset_state'), candidate.get('reset_state')),
        end_state=compare_state(reference.get('end_state'), candidate.get('end_state')),
        training_before=compare_logits(reference['training_before_logits'],
                                       candidate['training_before_logits'], training_rows),
        doses={},
    )
    for dose in sorted(set(reference['doses']) | set(candidate['doses']), key=int):
        if dose not in reference['doses'] or dose not in candidate['doses']:
            result['doses'][dose] = dict(available=False)
            continue
        a = reference['doses'][dose]
        b = candidate['doses'][dose]
        row = dict(evaluation=compare_logits(a['logits'], b['logits'], panel))
        if 'training_logits' in a and 'training_logits' in b:
            row['training'] = compare_logits(a['training_logits'], b['training_logits'], training_rows)
        result['doses'][dose] = row
    return result


def analyze(root):
    main = root / 'output'
    fresh = root / 'fresh-output'
    audit = load(main / 'numeric-path.json')
    if not audit:
        raise ValueError('numeric-path.json contains no snapshots')
    audit_panel = load(main / 'audit-panel.json')
    panel = load(main / 'panel.json')
    first = audit[0]
    result = dict(
        schema='sla-coverage-numeric-analysis-v1',
        scope='CPU comparison of recorded GPU measurements; no fresh model forward',
        probability_unit='Probability in [0,1]; fields ending in _pp use percentage points',
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_snapshot=first['name'],
        snapshot_count=len(audit),
        snapshots=[],
        same_process_retraining={},
        fresh_process={},
        missing_optional_files=[],
    )
    for snapshot in audit:
        result['snapshots'].append(dict(
            name=snapshot['name'],
            logits_vs_first=compare_logits(first['logits'], snapshot['logits'], audit_panel),
            before_vs_first_before=compare_state(first.get('before'), snapshot.get('before')),
            after_vs_first_after=compare_state(first.get('after'), snapshot.get('after')),
            state_change_during_readout=compare_state(snapshot.get('before'), snapshot.get('after')),
        ))
    for regime in ('original', 'covered'):
        name = f'62201-{regime}'
        reference = load(main / f'{name}.json')
        repeated_path = main / f'repeat-{name}.json'
        if repeated_path.is_file():
            result['same_process_retraining'][name] = compare_case(reference, load(repeated_path), panel)
        else:
            result['missing_optional_files'].append(str(repeated_path.relative_to(root)))
        reloaded_path = fresh / f'{name}-reloaded.json'
        if reloaded_path.is_file():
            result['fresh_process'].setdefault('saved_checkpoint_inference', {})[name] = compare_logits(
                reference['doses']['24']['logits'], load(reloaded_path), panel)
        else:
            result['missing_optional_files'].append(str(reloaded_path.relative_to(root)))
    frozen_path = fresh / 'frozen.json'
    if frozen_path.is_file():
        frozen = load(frozen_path)
        result['fresh_process']['frozen'] = dict(
            logits_vs_main_first=compare_logits(first['logits'], frozen['logits'], audit_panel),
            before_vs_main_first=compare_state(first.get('before'), frozen.get('before')),
            after_vs_main_first=compare_state(first.get('after'), frozen.get('after')),
            state_change_during_readout=compare_state(frozen.get('before'), frozen.get('after')),
        )
    else:
        result['missing_optional_files'].append(str(frozen_path.relative_to(root)))
    trained_path = fresh / 'retrained-62201-original.json'
    if trained_path.is_file():
        result['fresh_process']['retraining'] = compare_case(
            load(main / '62201-original.json'), load(trained_path), panel)
    else:
        result['missing_optional_files'].append(str(trained_path.relative_to(root)))
    runtime_path = fresh / 'runtime.json'
    if runtime_path.is_file():
        a = load(main / 'runtime.json')
        b = load(runtime_path)
        differences = nested_differences(a, b)
        result['runtime'] = dict(main=a, fresh=b, identical=not differences, differences=differences)
    else:
        result['missing_optional_files'].append(str(runtime_path.relative_to(root)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')
    print(json.dumps(dict(snapshots=result['snapshot_count'],
                         same_process_retraining=list(result['same_process_retraining']),
                         fresh_process_checks=list(result['fresh_process']),
                         missing_optional_files=result['missing_optional_files'])))


if __name__ == '__main__':
    main()
