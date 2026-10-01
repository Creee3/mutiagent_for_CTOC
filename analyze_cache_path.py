"""Check the saved native-cache/CV control without running another model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def analyze(folder):
    manifest, summary, row = [load(folder/p) for p in ['manifest.json', 'summary.json', 'items/case.json']]
    cfg = manifest['settings']
    assert sha(folder/'code/cache_path_conditions.json') == manifest['settings_sha256']
    for name, digest in cfg['source_hashes'].items():
        assert sha(folder/'code'/name) == digest
    root = folder.parent.parent
    assert sha(root/'checks/native_cache_control_cpu.json') == manifest['cpu_check_sha256']
    assert sha(root/'data/three_node_cases.json') == cfg['cases_sha256']
    assert sha(root/'runs'/cfg['reference_run']/'manifest.json') == cfg['reference_manifest_sha256']
    events = [json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
    counts = dict(Counter(e['kind'] for e in events))
    assert counts == summary['counts'] == {'generate': 4, 'cache_tail_forward': 1}
    assert all(e['status'] == 'ok' for e in events)
    assert len({(e['kind'], e['tag']) for e in events}) == len(events)
    a_history = row['sender']['global_history_ids'][0]
    previous = load(root/'runs'/cfg['reference_run']/'items'/f'main_{cfg["case_id"]}.json')
    assert row['sender']['output_ids'] == previous['reference']['A']['output_ids']
    assert len({r['input_history_sha256'] for r in row['receivers'].values()}) == 1
    actual_forwards = 0
    for label, record in [('sender', row['sender'])]+list(row['receivers'].items()):
        ids, history = record['output_ids'], record['global_history_ids'][0]
        full_input = history[:-len(ids)]
        assert token_sha([full_input]) == record['input_history_sha256']
        assert token_sha([history]) == record['global_history_sha256']
        assert history[-len(ids):] == ids
        assert record['generated_tokens'] == len(ids)
        prefix = record['input_prefix_tokens']
        start = prefix if record['route'] in ['kv', 'native_cached'] else 0
        expected = [list(range(start, len(full_input)))]+[[len(full_input)+i-1] for i in range(1, len(ids))]
        trace = record['backend_forward_trace']
        assert [r['position_ids'][0] for r in trace if r['level'] == 'effective_rotary_position_ids'] == expected
        model_trace = [r for r in trace if r['level'] == 'native_model_forward_input']
        attention = [r for r in trace if r['level'] == 'effective_first_layer_attention']
        assert len(model_trace) == len(attention) == len(ids)
        for step, model_input in enumerate(model_trace):
            assert model_input['input_ids'] == [full_input[start:] if step == 0 else [ids[step-1]]]
            assert model_input['past_length'] == (start if step == 0 else len(full_input)+step-1)
            assert attention[step]['cache_position'] == expected[step]
            assert attention[step]['position_ids'] == [expected[step]]
        actual_forwards += len(model_trace)
        if label != 'sender':
            assert full_input[:prefix] == a_history
    native, cv = row['receivers']['native_cached'], row['receivers']['cv']
    assert native['sender_cache_hash_before'] == native['sender_cache_hash_after'] == cv['sender_cache_hash_before_and_after']
    assert native['output_ids'] == cv['output_ids']
    assert native['response'] == cv['response']
    assert row['receivers']['joint']['output_ids'] != native['output_ids']
    for pair, comparison in row['comparisons'].items():
        left, right = pair.split('_vs_')
        a, b = row['receivers'][left]['output_ids'], row['receivers'][right]['output_ids']
        diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if diff is None and len(a) != len(b):
            diff = min(len(a), len(b))
        assert comparison['first_different_index'] == diff
        assert comparison['generated_tokens_identical'] == (a == b)
        assert comparison['max_raw_error_on_same_history'] == max(s['raw']['max_abs'] for s in comparison['same_history_steps'])
    assert sha(folder/'vectors/steps.npz') == row['vectors_sha256']
    vectors = np.load(folder/'vectors/steps.npz')
    saved_first_step_diff = float(np.max(np.abs(vectors['native_cached_first_raw']-vectors['cv_first_raw'])))
    payload = row['sender_kv_payload']
    assert payload['metadata']['length'] == len(a_history)
    assert payload['metadata']['lengths'] == [len(a_history)]*28
    assert sum(p['content_bytes'] for p in payload['parts']) == payload['content_bytes']
    assert row['sender_text_payload']['content_bytes'] == len(a_history)*4
    report = {'passed': True, 'audit_script_sha256': sha(__file__), 'counts': counts,
        'checked_generation_forwards': actual_forwards, 'same_sender_history_and_cache': True,
        'native_cached_vs_CV_token_matches': len(native['output_ids']),
        'saved_first_step_native_cached_CV_max_logit_difference': saved_first_step_diff,
        'runtime_all_same_history_steps_native_cached_CV_max_logit_difference': summary['native_cached_vs_cv']['max_raw_error_on_same_history'],
        'joint_vs_native_cached_first_different_index': summary['joint_vs_native_cached']['first_different_index'],
        'all_receivers_EOS': summary['all_receivers_EOS'], 'any_truncated': summary['any_truncated'],
        'sender_kv_content_bytes_recorded_from_actual_tensors': payload['content_bytes'],
        'sender_text_int32_content_bytes': row['sender_text_payload']['content_bytes'],
        'interpretation': 'Divergence also occurs with native cache reuse; not unique to CV. Precision/kernel cause not isolated.'}
    (folder/'independent_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_folder', type=Path)
    analyze(parser.parse_args().run_folder)
