"""Check frozen receiver inputs, label isolation and optional offline token counts."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(tokenizer_check):
    source = ROOT / 'data/receiver_evidence_20261002'
    safe = ROOT / 'data/receiver_runtime_views_20261002'
    first, second = read(source / 'manifest.json'), read(safe / 'manifest.json')
    for directory, manifest in [(source, first), (safe, second)]:
        for name, digest in manifest['outputs_sha256'].items():
            assert sha(directory / name) == digest, name
    assert sha(ROOT / 'data/calibration.json') == first['calibration_sha256']
    assert sha(ROOT / 'config.json') == first['config_sha256']
    for name, digest in first['code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert sha(source / 'selector_input_views.json') == second['source_sha256']
    assert sha(ROOT / 'prepare_receiver_runtime_views.py') == second['code_sha256']
    views = read(source / 'selector_input_views.json')
    sender = {x['question_id']: x for x in views['senders']}
    rows = {x['key']: x for x in views['receivers']}
    diag = read(source / 'diagnostic_annotations.json')
    labels = read(source / 'scoring_labels.json')
    assert set(sender) == {x['question_id'] for x in labels}
    assert len(rows) == first['receiver_input_rows'] == 72
    partial_ids = {x['question_id'] for x in rows.values() if x['condition'] == 'partial'}
    assert len(partial_ids) == 6
    for item in rows.values():
        assert item['sender_prompt_ids_sha256'] == sender[item['question_id']]['sender_prompt_ids_sha256']
        assert not item['inherited_history_provided']
        assert item['received_message_status'] == 'not_generated'
    for qid in partial_ids:
        assert all(qid + ':' + condition in rows for condition in ['missing', 'partial', 'full'])
    for held, control in diag['paired_control_keys']:
        assert rows[held]['receiver_prompt_tokens'] == rows[control]['receiver_prompt_tokens']
    schedules = {x['lookup_id']: x for x in read(safe / 'scheduler_metadata.json')}
    payloads = read(safe / 'runtime_payloads.json')
    whitelist = {'question', 'sender_material', 'receiver_visible_text'}
    assert len(payloads) == len(schedules) == len(rows)
    for row in payloads:
        meta = schedules[row['lookup_id']]
        expected = rows[meta['source_key']]
        assert set(row['payload']) == whitelist and not meta['selector_input']
        assert row['payload']['question'] == expected['question']
        assert row['payload']['sender_material'] == sender[meta['question_id']]['sender_material']
        assert row['payload']['receiver_visible_text'] == expected['receiver_prompt']
    sampling = read(safe / 'text_sampling_body_controls.json')
    assert len(sampling) == len(sender) == 20
    for row in sampling:
        assert row['sampled_body_tokens'] > 0
        assert 0 < row['payload_tokens'] <= row['budget_tokens'] < sender[row['question_id']]['material_tokens']
    tokens_checked = 0
    if tokenizer_check:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['HF_HUB_OFFLINE'] = '1'
        from transformers import AutoTokenizer
        config = read(ROOT / 'config.json')
        model_path = Path(config['model_7b_path'])
        for name, digest in first['tokenizer_files_sha256'].items():
            assert sha(model_path / name) == digest
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        for item, field, count_field, digest_field in (
            [(x, 'sender_prompt', 'sender_prompt_tokens', 'sender_prompt_ids_sha256') for x in sender.values()]
            + [(x, 'receiver_prompt', 'receiver_prompt_tokens', 'receiver_prompt_ids_sha256') for x in rows.values()]):
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': item[field]}],
                add_generation_prompt=True, tokenize=True)
            assert len(ids) == item[count_field]
            digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            assert digest == item[digest_field]
            tokens_checked += 1
        for row in sampling:
            assert len(tokenizer.encode(row['sampled_text'], add_special_tokens=False)) == row['payload_tokens']
            tokens_checked += 1
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'prepared_artifacts_passed_not_a_protocol_or_quality_result',
        'unique_questions': len(sender), 'receiver_input_rows': len(rows),
        'partial_eligible_questions': len(partial_ids),
        'exact_complete_prompt_length_pairs': len(diag['paired_control_keys']),
        'original_heading_only_controls': sum(x['original_heading_only'] for x in sampling),
        'corrected_nonempty_body_controls': len(sampling),
        'corrected_controls_omitting_heading': sum(not x['heading_included_in_payload'] for x in sampling),
        'selector_payload_whitelist': sorted(whitelist),
        'offline_tokenizer_rows_checked': tokens_checked,
        'manifest_files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [source / 'manifest.json', safe / 'manifest.json']},
        'audit_code_sha256': sha(Path(__file__)), 'model_calls': 0, 'model_weights_loaded': False,
        'limits': ['semantic irrelevance of length controls is not proved',
            'body sampling is a random control, not a strong semantic selector',
            'sender outputs/cache, receiver query and legal KV ingestion are not ready',
            'only the nested payload is passed to selectors, never lookup IDs or scheduler metadata']}
    path = ROOT / 'checks/receiver_evidence_preparation_audit_20261002.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', action='store_true', help='school existing environment only; no models')
    main(parser.parse_args().tokenizer)
