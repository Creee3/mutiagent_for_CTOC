"""Recompute history, position, byte and call-accounting checks from saved artifacts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def analyze(folder):
    manifest, summary = load(folder/'manifest.json'), load(folder/'summary.json')
    cfg = manifest['settings']
    assert sha(folder/'code/three_node_conditions.json') == manifest['settings_sha256']
    for name, digest in cfg['source_hashes'].items():
        assert sha(folder/'code'/name) == digest
    root = folder.parent.parent
    assert sha(root/'data/three_node_cases.json') == cfg['cases_sha256']
    assert sha(root/'data/calibration.json') == cfg['calibration_sha256']
    assert sha(root/'checks/three_node_cpu_check.json') == manifest['cpu_check_sha256']
    events = [json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
    assert all(e['status'] == 'ok' for e in events)
    assert len({(e['kind'], e['tag']) for e in events}) == len(events)
    counts = dict(Counter(e['kind'] for e in events))
    assert counts == summary['counts']
    assert counts['generate'] == (29 if summary['gate_passed'] else 20)
    assert counts.get('cache_tail_forward', 0) <= 14
    rows = [load(p) for p in sorted((folder/'items').glob('*.json'))]
    mains = {r['id']: r for r in rows if r['phase'] == 'main'}
    assert set(mains) == set(cfg['main_ids'])
    assert [r['id'] for r in rows if r['phase'] == 'warmup'] == [cfg['warmup_id']]
    eos = manifest['generation_config']['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    checked_nodes, checked_forwards, tails, main_details = 0, 0, 0, []
    matches = Counter()
    for row in rows:
        assert sha(folder/row['vectors']['file']) == row['vectors']['sha256']
        vectors = np.load(folder/row['vectors']['file'])
        for label in ['reference', 'replay', 'free_kv']:
            for node, record in row.get(label, {}).items():
                ids = record['output_ids']
                history = record['global_history_ids'][0]
                prompt = record['local_prompt_ids'][0]
                full_input = history[:-len(ids)]
                prefix = record['input_prefix_tokens']
                assert full_input[prefix:] == prompt and history[-len(ids):] == ids
                assert token_sha([history]) == record['global_history_sha256']
                assert token_sha([full_input]) == record['input_history_sha256']
                assert len(ids) == record['generated_tokens']
                assert record['eos_stopped'] == (ids[-1] in eos)
                assert record['actually_truncated'] == (len(ids) >= cfg['budgets'][node] and ids[-1] not in eos)
                cursor, seen = 0, set()
                for segment in record['segments']:
                    assert segment['start'] == cursor and segment['id'] not in seen
                    cursor = segment['end']
                    seen.add(segment['id'])
                    assert token_sha([history[segment['start']:cursor]]) == segment['tokens_sha256']
                assert cursor == len(history)
                trace = record['backend_forward_trace']
                positions = [x['position_ids'][0] for x in trace if x['level'] == 'effective_rotary_position_ids']
                start = prefix if record['route'] == 'kv' else 0
                expected = [list(range(start, len(full_input)))]+[[len(full_input)+step-1] for step in range(1, len(ids))]
                assert positions == expected
                model_trace = [x for x in trace if x['level'] == 'native_model_forward_input']
                attention_trace = [x for x in trace if x['level'] == 'effective_first_layer_attention']
                assert len(model_trace) == len(attention_trace) == len(ids)
                for step, value in enumerate(model_trace):
                    expected_ids = full_input[start:] if step == 0 else [ids[step-1]]
                    assert value['input_ids'] == [expected_ids]
                    # CV may omit the optional model argument; Qwen then derives
                    # it. Check the position actually used by attention/RoPE.
                    if value['cache_position'] is not None:
                        assert value['cache_position'] == expected[step]
                    assert attention_trace[step]['cache_position'] == expected[step]
                    assert attention_trace[step]['position_ids'] == [expected[step]]
                    assert value['past_length'] == (start if step == 0 else len(full_input)+step-1)
                before = record['cache_lengths_before_completion']
                assert len(before) == len(manifest['full_layers']) and len(set(before)) == 1
                assert before[0] in [len(history)-1, len(history)]
                completion = record['cache_completion']
                if completion is not None:
                    assert completion['expected_length'] == len(history)
                    assert completion['length_before'] == before[0]
                    assert completion['missing_tokens'] == len(history)-before[0]
                    if completion['missing_tokens']:
                        assert completion['filled_ids'] == [[history[-1]]]
                        tail_trace = completion['forward']['backend_forward_trace']
                        assert [x['position_ids'] for x in tail_trace if x['level'] == 'effective_rotary_position_ids'] == [[[len(history)-1]]]
                        tails += 1
                checked_nodes += 1
                checked_forwards += len(model_trace)
        for node, predecessor in [('B', 'A'), ('C', 'B')]:
            left, right = row['reference'][node], row['replay'][node]
            source = row['reference'][predecessor]
            assert right['global_history_ids'][0][:right['input_prefix_tokens']] == source['global_history_ids'][0]
            assert left['input_history_sha256'] == right['input_history_sha256']
            same = left['output_ids'] == right['output_ids']
            assert same == row['comparisons'][node]['generated_tokens_identical']
            if row['phase'] == 'main':
                matches[node] += same
        edges = row['reference_edges']
        seen, repeated = set(), 0
        for edge in edges:
            assert sum(part['content_bytes'] for part in edge['parts']) == edge['content_bytes']
            assert edge['metadata']['length']*4 == edge['content_bytes']
            for part in edge['parts']:
                assert part['content_bytes'] == (part['end']-part['start'])*4
                if part['id'] in seen:
                    repeated += part['content_bytes']
                seen.add(part['id'])
        total = sum(e['content_bytes'] for e in edges)
        recorded = row['reference_repeated_transmissions']
        assert repeated == edges[0]['content_bytes'] == recorded['repeated_source_event_bytes']
        assert total == recorded['total_content_bytes']
        assert repeated/total == recorded['repeated_fraction_across_edges']
        if row['phase'] == 'main':
            detail = {'id': row['id'], 'reference_answer': row['reference']['C']['response'],
                'reference_score': row['reference_final_score'],
                'replay_output_matches': {n: row['comparisons'][n]['generated_tokens_identical'] for n in ['B', 'C']},
                'repeated_fraction_across_edges': repeated/total,
                'inherited_A_fraction_of_B_to_C': recorded['A_fraction_of_B_to_C_content']}
            if not row['comparisons']['B']['generated_tokens_identical']:
                left = vectors['reference_B_divergence_raw']
                right = vectors['replay_B_divergence_raw']
                detail['B_divergence'] = {'first_index_zero_based': row['comparisons']['B']['first_different_index'],
                    'reference_response': row['reference']['B']['response'], 'replay_response': row['replay']['B']['response'],
                    'saved_common_history_logits': {'reference_top5': top5(left), 'replay_top5': top5(right),
                        'max_abs_difference': float(np.max(np.abs(left-right)))}}
            main_details.append(detail)
    assert tails == counts.get('cache_tail_forward', 0)
    assert dict(matches) == summary['replay_matches']
    if not summary['gate_passed']:
        assert all('free_kv' not in r for r in rows) and summary['free_kv_chains'] == 0
        assert (folder/'GATE_MISMATCH.json').exists()
    report = {'passed': True, 'audit_script_sha256': sha(__file__), 'run_name': folder.name,
        'counts': counts, 'checked_nodes': checked_nodes, 'checked_generation_forwards': checked_forwards,
        'all_recorded_nodes_EOS': all(record['eos_stopped'] for row in rows for label in ['reference', 'replay', 'free_kv'] for record in row.get(label, {}).values()),
        'truncated_nodes': sum(record['actually_truncated'] for row in rows for label in ['reference', 'replay', 'free_kv'] for record in row.get(label, {}).values()),
        'replay_matches': dict(matches), 'gate_passed': summary['gate_passed'], 'free_kv_chains': summary['free_kv_chains'],
        'source_cache_bytes_not_reaudited_from_saved_tensors': True, 'network_transfer_measured': False,
        'main_details': main_details}
    (folder/'independent_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'main_details'}, ensure_ascii=False))


def top5(values):
    indices = np.argsort(values)[-5:][::-1]
    return {'ids': indices.tolist(), 'values': [float(values[i]) for i in indices],
        'top1_minus_top2': float(values[indices[0]]-values[indices[1]])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_folder', type=Path)
    analyze(parser.parse_args().run_folder)
