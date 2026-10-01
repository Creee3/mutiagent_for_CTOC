"""Audit actual relay history, cache sources, positions, byte totals and controls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def independent_score(prediction, gold):
    def norm(value):
        value = ''.join(c for c in str(value).lower() if c not in string.punctuation)
        return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', value).split())
    prediction, gold = norm(prediction), norm(gold)
    p, g = prediction.split(), gold.split()
    overlap = sum((Counter(p) & Counter(g)).values())
    special_mismatch = any((prediction == x or gold == x) and prediction != gold for x in ['yes', 'no', 'noanswer'])
    f1 = 0.0 if not overlap or special_mismatch else 2*overlap/(len(p)+len(g))
    return {'exact_match': prediction == gold, 'standard_token_f1': f1}


def audit(folder):
    manifest, summary = read(folder/'manifest.json'), read(folder/'summary.json')
    cfg = manifest['settings']
    root = folder.parent.parent
    assert sha(folder/'code/free_chain_conditions.json') == manifest['settings_sha256']
    for name, digest in cfg['source_hashes'].items():
        assert sha(folder/'code'/name) == digest
    for path, digest in [('data/three_node_cases.json', cfg['cases_sha256']),
                         ('data/calibration.json', cfg['calibration_sha256']),
                         ('assets.json', cfg['assets_sha256']), ('config.json', cfg['master_config_sha256']),
                         ('checks/free_chain_cpu.json', manifest['cpu_check_sha256'])]:
        assert sha(root/path) == digest
    reference = root/'runs'/cfg['reference_run']
    assert sha(reference/'manifest.json') == cfg['reference_manifest_sha256']
    assert sha(reference/'summary.json') == cfg['reference_summary_sha256']
    assert read(reference/'summary.json')['gate_passed'] == cfg['old_reference_gate_passed'] == False
    events = [json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
    counts = dict(Counter(e['kind'] for e in events))
    assert counts == summary['counts'] and counts['generate'] == 20 and counts.get('cache_tail_forward', 0) <= 8
    assert all(e['status'] == 'ok' for e in events)
    assert len({(e['tag'], e['kind']) for e in events}) == len(events)
    rows = [read(p) for p in sorted((folder/'items').glob('*.json'))]
    assert {r['id'] for r in rows if r['phase'] == 'main'} == set(cfg['main_ids'])
    assert [r['id'] for r in rows if r['phase'] == 'warmup'] == [cfg['warmup_id']]
    assert len(rows) == 4
    model_cfg = manifest['model_config']
    head_dim = model_cfg.get('head_dim') or model_cfg['hidden_size']//model_cfg['num_attention_heads']
    kv_bytes_per_token = 2*model_cfg['num_hidden_layers']*model_cfg['num_key_value_heads']*head_dim*2
    eos = manifest['generation_config']['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    matched = Counter()
    completed_tails, node_count, forwards = 0, 0, 0
    details, truncated = [], []
    for row in rows:
        old_path = reference/'items'/f'{row["phase"]}_{row["id"]}.json'
        assert sha(old_path) == row['reference_row_sha256'] == cfg['reference_rows_sha256'][row['id']]
        old = read(old_path)
        assert row['counts']['generate'] == 5
        for group in ['nodes', 'controls']:
            for node, record in row[group].items():
                ids, history = record['output_ids'], record['global_history_ids'][0]
                full_input = history[:-len(ids)]
                prefix = record['input_prefix_tokens']
                assert history[-len(ids):] == ids and record['generated_tokens'] == len(ids)
                assert full_input[prefix:] == record['local_prompt_ids'][0]
                assert record['local_prompt_ids'] == old['reference'][node]['local_prompt_ids']
                assert token_sha([full_input]) == record['input_history_sha256']
                assert token_sha([history]) == record['global_history_sha256']
                assert record['eos_stopped'] == (ids[-1] in eos)
                assert record['actually_truncated'] == (len(ids) >= cfg['budgets'][node] and ids[-1] not in eos)
                if record['actually_truncated']:
                    truncated.append({'id': row['id'], 'phase': row['phase'], 'group': group, 'node': node})
                start = prefix if record['route'] in ['kv', 'native_cached'] else 0
                expected = [list(range(start, len(full_input)))]+[[len(full_input)+i-1] for i in range(1, len(ids))]
                trace = record['backend_forward_trace']
                assert [x['position_ids'][0] for x in trace if x['level'] == 'effective_rotary_position_ids'] == expected
                inputs = [x for x in trace if x['level'] == 'native_model_forward_input']
                attention = [x for x in trace if x['level'] == 'effective_first_layer_attention']
                assert len(inputs) == len(attention) == len(ids)
                for step, value in enumerate(inputs):
                    assert value['input_ids'] == [full_input[start:] if step == 0 else [ids[step-1]]]
                    assert value['past_length'] == (start if step == 0 else len(full_input)+step-1)
                    assert attention[step]['cache_position'] == expected[step]
                    assert attention[step]['position_ids'] == [expected[step]]
                before = record['cache_lengths_before_completion']
                assert before == [before[0]]*model_cfg['num_hidden_layers']
                assert len(history)-before[0] in [0, 1]
                if group == 'nodes':
                    ledger = record['segments']
                    cursor, seen = 0, set()
                    for part in ledger:
                        assert part['start'] == cursor and part['id'] not in seen
                        cursor = part['end']
                        seen.add(part['id'])
                        assert token_sha([history[part['start']:cursor]]) == part['tokens_sha256']
                    assert cursor == len(history)
                    assert len(ledger) == 2*(['A', 'B', 'C'].index(node)+1)
                    completion = record['cache_completion']
                    if node != 'C':
                        assert completion['expected_length'] == len(history)
                        assert completion['length_before'] == before[0]
                        assert completion['missing_tokens'] == len(history)-before[0]
                        if completion['missing_tokens']:
                            assert completion['filled_ids'] == [[history[-1]]]
                            assert [x['position_ids'] for x in completion['forward']['backend_forward_trace']
                                if x['level'] == 'effective_rotary_position_ids'] == [[[len(history)-1]]]
                            completed_tails += 1
                    else:
                        assert completion is None
                else:
                    assert record['cache_completion'] is None and not record['main_chain_member']
                forwards += len(inputs)
                node_count += 1
        for node, sender in [('B', 'A'), ('C', 'B')]:
            current, control, source = row['nodes'][node], row['controls'][node], row['nodes'][sender]
            actual_hash = row['source_cache_hashes'][sender]
            assert current['sender_cache_hash_before_and_after'] == control['sender_cache_hash_before'] == control['sender_cache_hash_after'] == actual_hash
            assert source['outgoing_completed_cache_sha256'] == actual_hash
            assert current['global_history_ids'][0][:current['input_prefix_tokens']] == source['global_history_ids'][0]
            assert current['segments'][:len(source['segments'])] == source['segments']
            assert control['input_history_sha256'] == current['input_history_sha256']
            assert row['cache_provenance'][node]['sender_generation_tag'] == source['tag']
            assert row['cache_provenance'][node]['main_sender_state_used'] and not row['cache_provenance'][node]['control_state_used']
            same = current['output_ids'] == control['output_ids']
            assert same == row['comparisons'][node]['generated_tokens_identical']
            if row['phase'] == 'main':
                matched[node] += same
        assert row['cache_provenance']['C']['sender_route'] == 'kv'
        assert not row['control_outputs_entered_main_chain']
        for kind, per_token in [('kv', kv_bytes_per_token), ('text', 4)]:
            edges = row['edge_payloads'][kind]
            seen, repeated = set(), 0
            for edge, (sender, receiver) in zip(edges, [('A', 'B'), ('B', 'C')]):
                history_len = len(row['nodes'][sender]['global_history_ids'][0])
                assert edge['content_bytes'] == history_len*per_token
                assert edge['metadata']['source_cache_sha256'] == row['source_cache_hashes'][sender]
                assert edge['metadata']['source_generation_tag'] == row['nodes'][sender]['tag']
                assert edge['metadata']['source_history_sha256'] == row['nodes'][sender]['global_history_sha256']
                assert edge['logical_total_bytes'] == edge['content_bytes']+edge['metadata_bytes']
                assert edge['metadata_bytes'] == len(json.dumps(edge['metadata'], sort_keys=True, separators=(',', ':')).encode())
                for part in edge['parts']:
                    assert part['content_bytes'] == (part['end']-part['start'])*per_token
                    if part['id'] in seen:
                        repeated += part['content_bytes']
                    seen.add(part['id'])
                assert sum(p['content_bytes'] for p in edge['parts']) == edge['content_bytes']
            recorded = row['repeated'][kind]
            total = sum(e['content_bytes'] for e in edges)
            assert repeated == edges[0]['content_bytes'] == recorded['repeated_source_event_bytes'] == recorded['A_history_bytes_in_B_to_C']
            assert total == recorded['total_content_bytes']
            assert repeated/total == recorded['repeated_fraction_across_edges']
        vectors = row['first_step_and_divergence_vectors']
        assert sha(folder/vectors['file']) == vectors['sha256']
        arrays = np.load(folder/vectors['file'], allow_pickle=False)
        assert sorted(arrays.files) == vectors['keys']
        for node in ['B', 'C']:
            left, right = arrays[f'{node}__cv__first__step0__raw'], arrays[f'{node}__native_cached__first__step0__raw']
            assert np.max(np.abs(left-right)) == row['comparisons'][node]['same_history_steps'][0]['raw']['max_abs']
        for node in ['A', 'B', 'C']:
            comparison = row['vs_joint_reference'][node]
            assert comparison['same_input_history'] == (row['nodes'][node]['input_history_sha256'] == old['reference'][node]['input_history_sha256'])
            assert comparison['same_output_tokens'] == (row['nodes'][node]['output_ids'] == old['reference'][node]['output_ids'])
        independent = independent_score(row['nodes']['C']['response'], row['gold'])
        assert independent['exact_match'] == row['final_scores']['free_kv']['exact_match']
        assert abs(independent['standard_token_f1']-row['final_scores']['free_kv']['standard_token_f1']) < 1e-12
        assert row['final_scores']['joint_reference'] == old['reference_final_score']
        if row['phase'] == 'main':
            details.append({'id': row['id'], 'vs_joint_reference': row['vs_joint_reference'],
                'final_free_response': row['nodes']['C']['response'], 'final_scores': row['final_scores'],
                'edge_kv_content_bytes': [e['content_bytes'] for e in row['edge_payloads']['kv']],
                'edge_kv_logical_bytes': [e['logical_total_bytes'] for e in row['edge_payloads']['kv']],
                'repeated_kv': row['repeated']['kv']})
    assert completed_tails == counts.get('cache_tail_forward', 0)
    assert dict(matched) == summary['same_cache_control_matches']
    main_EOS = all(r['nodes'][n]['eos_stopped'] for r in rows if r['phase'] == 'main' for n in ['A', 'B', 'C'])
    no_main_truncation = not any(x['phase'] == 'main' for x in truncated)
    assert summary['all_main_free_nodes_EOS'] == main_EOS
    assert summary['no_main_truncation'] == no_main_truncation
    assert summary['implementation_control_passed'] == (all(v == 3 for v in matched.values()) and no_main_truncation)
    if not summary['implementation_control_passed']:
        assert (folder/'IMPLEMENTATION_CONTROL_MISMATCH.json').exists()
    report = {'passed': True, 'audit_script_sha256': sha(__file__), 'counts': counts,
        'checked_nodes': node_count, 'checked_generation_forwards': forwards,
        'C_receives_actual_CV_B_cache': True, 'same_cache_control_matches': dict(matched),
        'all_main_free_nodes_EOS': main_EOS, 'implementation_control_passed': summary['implementation_control_passed'],
        'truncated_nodes': truncated, 'old_gate_result_preserved': True,
        'kv_bytes_per_token': kv_bytes_per_token, 'network_transfer_measured': False, 'cases': details}
    (folder/'independent_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_folder', type=Path)
    audit(parser.parse_args().run_folder)
