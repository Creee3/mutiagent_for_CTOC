"""Read-only stdlib checks of one new answer-format run; no model or old-run audit."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('no_msg', 'token_full', 'kv_full', 'material_text')
EXPECTED_PHASES = {'main_generate': 72, 'warmup_generate': 4,
                   'main_sender_prefill': 6, 'warmup_sender_prefill': 1}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    sys.dont_write_bytecode = True
    helper_path = ROOT / 'checks/receiver_endpoint_result_audit_20261002.py'
    spec = importlib.util.spec_from_file_location('_new_run_wire_parser', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    audit = helper.Audit()
    check = audit.check
    folder = (ROOT / 'runs' / args.run_name).resolve()
    if not folder.is_relative_to((ROOT / 'runs').resolve()):
        raise ValueError('run path must stay in runs')
    manifest = read(folder / 'manifest.json')
    summary = read(folder / 'summary.json')
    complete = read(folder / 'COMPLETE.json')
    settings_path = ROOT / 'receiver_answer_format_run_conditions.json'
    settings = read(settings_path)
    inputs_path = ROOT / settings['input_file']
    frozen = read(inputs_path)
    check(not (folder / 'FAILED.json').exists(), 'no failure marker')
    check(manifest['run_name'] == folder.name, 'run identity')
    check(manifest['conditions_sha256'] == digest(settings_path), 'run conditions SHA')
    check(digest(inputs_path) == manifest['input_sha256'] == settings['input_sha256'], 'frozen input SHA')
    for name, value in settings.items():
        check(manifest.get(name) == value, 'manifest froze setting ' + name)
    for name, value in settings['source_hashes'].items():
        check(digest(folder / 'code' / name) == value, 'run code snapshot SHA ' + name)
    check(manifest['generation_kwargs'] == dict(max_new_tokens=48, do_sample=False, num_beams=1,
          use_cache=True, repetition_penalty=1.0, eos_token_id=settings['eos_token_id'],
          pad_token_id=settings['pad_token_id'], return_dict_in_generate=True,
          output_scores=False, output_logits=False), 'frozen actual generation arguments')
    check(manifest['expected_counts'] == summary['counts'] == {'generate': 76, 'sender_prefill': 7},
          'actual 76 generate and 7 prefill')
    check(manifest['expected_phase_counts'] == summary['phase_counts'] == complete['counts'] == EXPECTED_PHASES,
          'main and warmup counts')
    senders = {s['question_id']: s for s in frozen['senders']}
    receivers = {r['key']: r for r in frozen['receivers']}
    source_paths = sorted((folder / 'sources').glob('*.json'))
    item_paths = sorted((folder / 'items').glob('*.json'))
    sources = [read(p) for p in source_paths]
    rows = [read(p) for p in item_paths]
    source_index = {(s['phase'], s['question_id']): s for s in sources}
    main_rows = [r for r in rows if r['phase'] == 'main']
    warm_rows = [r for r in rows if r['phase'] == 'warmup']
    expected_sources = {('main', qid) for qid in senders} | {('warmup', settings['warmup_key'].split(':')[0])}
    check(len(sources) == len(source_index) == 7 and set(source_index) == expected_sources, 'seven phase-bound sources')
    check(len(rows) == 19 and len(main_rows) == 18 and len(warm_rows) == 1, '19 item files')
    check({r['key'] for r in main_rows} == set(receivers), '18 frozen main receiver views')
    check([r['key'] for r in warm_rows] == [settings['warmup_key']], 'single frozen warmup view')
    events_path = folder / 'calls.jsonl'
    events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    event_index = {(e['phase'], e['kind'], e['tag']): e for e in events}
    check(len(events) == len(event_index) == 83, '83 distinct ledger events')
    check(all(e['status'] == 'ok' for e in events), 'all ledger events succeeded')
    check(dict(Counter(e['kind'] for e in events)) == {'generate': 76, 'sender_prefill': 7}, 'ledger 76/7')
    check(dict(Counter(e['phase'] + '_' + e['kind'] for e in events)) == EXPECTED_PHASES,
          'ledger exact phase counts')
    check(summary['underlying_model_forward_calls'] == sum(e['model_forward_calls'] for e in events),
          'summary forward count equals ledger')
    packets = {}
    for source in sources:
        phase, qid = source['phase'], source['question_id']
        sender = senders[qid]
        n = len(sender['ids'])
        check(source['A_ids'] == sender['ids'], phase + ':' + qid + ': actual source IDs')
        event = event_index[(phase, 'sender_prefill', qid + ':A')]
        check(source['sender_prefill'] == event, phase + ':' + qid + ': bound prefill event')
        check(event['model_forward_calls'] == 1 and event['effective_position_ids'] == [[list(range(n))]],
              phase + ':' + qid + ': source prefill positions')
        for method, name in [('token_full', 'text_packet'), ('kv_full', 'kv_packet')]:
            path = folder / 'packets' / (phase + '_' + qid + '_' + method + '.bin')
            header, info = helper.audit_packet(path, sender, settings, audit)
            packets[path.name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
            for field, value in info.items():
                check(source[name].get(field) == value, path.name + ': source metric ' + field)
            if method == 'kv_full':
                check(header['chain_cache_sha256'] == source['cache_source_hash'], path.name + ': source cache hash')
                check(source['cache_resident_tensor_bytes'] == info['kv_tensor_content_bytes'],
                      path.name + ': resident raw tensor bytes')
        path = folder / 'packets' / (phase + '_' + qid + '_material_text.bin')
        raw = path.read_bytes()
        body = json.loads(raw.decode('utf-8'), object_pairs_hook=helper.no_duplicate_keys)
        info = source['material_packet']
        check(body == {'format': 'source-material-json-v1', 'source': 'A', 'qid': qid,
                      'revision': settings['model_revision'], 'material': sender['sender_material']},
              path.name + ': actual unpruned material and provenance')
        check(info['payload_bytes'] == len(raw) and info['packet_sha256'] == hashlib.sha256(raw).hexdigest(),
              path.name + ': actual material bytes and SHA')
        content = len(body['material'].encode('utf-8'))
        check(info['payload_content_bytes'] == content and info['payload_header_bytes'] == len(raw) - content,
              path.name + ': material JSON overhead')
        packets[path.name] = {'sha256': digest(path), 'bytes': len(raw)}
    check(len(packets) == len(list((folder / 'packets').glob('*.bin'))) == 21, 'exactly 21 actual packet files')
    tokenizer_path = Path(args.tokenizer)
    check(digest(tokenizer_path) == settings['model_tokenizer_hashes']['tokenizer.json'], 'response decoder tokenizer SHA')
    decode = helper.byte_level_decoder(tokenizer_path)
    eos = settings['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    total_tokens, main_eos, main_truncated = 0, 0, 0
    for row in rows:
        phase, qid, key = row['phase'], row['question_id'], row['key']
        label = phase + ':' + key
        source = source_index[(phase, qid)]
        for field, value in receivers[key].items():
            check(row.get(field) == value, label + ': frozen row ' + field)
        check(set(row['methods']) == set(METHODS) and len(row['method_order']) == 4
              and set(row['method_order']) == set(METHODS), label + ': methods exactly once')
        check(row['source_cache_unchanged'] is True and row['cache_source_hash'] == source['cache_source_hash'],
              label + ': source binding and immutability assertion')
        for method, value in row['methods'].items():
            tag = label + ':' + method
            event = event_index[(phase, 'generate', key + ':' + method)]
            view = receivers[key]['method_inputs'][method]
            ids = value['output_ids']
            n_input = len(view['ids'])
            n_prefix = len(source['A_ids']) if method == 'kv_full' else 0
            total = n_input + n_prefix
            positions = [[list(range(n_prefix, total))]] + [[[total + i - 1]] for i in range(1, len(ids))]
            check(event['effective_position_ids'] == positions, tag + ': actual source-aware positions')
            check(value['checked_effective_positions'] is True and value['receiver_input_tokens'] == n_input,
                  tag + ': frozen input extent')
            check(ids and all(type(i) is int for i in ids) and len(ids) <= 48
                  and value['generated_tokens'] == value['model_forward_calls'] == event['model_forward_calls'] == len(ids),
                  tag + ': output IDs and forwards')
            check(value['response'] == decode(ids, skip_special=True), tag + ': raw response decodes from actual IDs')
            ended = ids[-1] in eos
            truncated = len(ids) >= 48 and not ended
            check(value['eos_stopped'] == ended and value['actually_truncated'] == truncated,
                  tag + ': actual EOS and budget flag')
            for field, event_field in [('receiver_model_seconds', 'seconds'),
                                      ('receiver_prefill_gpu_seconds', 'prefill_gpu_seconds'),
                                      ('receiver_decode_gpu_seconds', 'decode_gpu_seconds')]:
                check(value[field] == event[event_field] and math.isfinite(value[field]) and value[field] >= 0,
                      tag + ': ledger timing ' + field)
            if method == 'no_msg':
                check(value['payload_bytes'] == value['payload_content_bytes'] == value['payload_header_bytes'] == 0,
                      tag + ': no communication bytes')
            else:
                packet_name = {'token_full': 'text_packet', 'kv_full': 'kv_packet', 'material_text': 'material_packet'}[method]
                metric = source[packet_name]
                for field in ['payload_bytes', 'payload_content_bytes', 'payload_header_bytes', 'packet_sha256']:
                    check(value[field] == metric[field], tag + ': actual packet binding ' + field)
            total_tokens += len(ids)
            if phase == 'main':
                main_eos += ended
                main_truncated += truncated
    check(total_tokens + 7 == summary['underlying_model_forward_calls'], 'generated-token forwards plus seven source forwards')
    check(main_eos == summary['EOS_main_answers'] and main_truncated == summary['truncated_main_answers'], 'main EOS/trigger summary')
    result = {'run_name': folder.name, 'passed': not audit.errors, 'checks': audit.checks, 'errors': audit.errors,
              'scope': 'new run only: raw packets, frozen source/input binding, actual positions, ledger and output decoding',
              'model_calls_by_audit': 0, 'model_imported_by_audit': False, 'old_runs_reaudited': False,
              'floating_point_equality_gate': False, 'manual_answer_format_review_completed': False,
              'main_answers': 72, 'event_counts': dict(Counter(e['kind'] for e in events)),
              'phase_counts': dict(Counter(e['phase'] + '_' + e['kind'] for e in events)),
              'underlying_model_forward_calls': summary['underlying_model_forward_calls'],
              'main_eos': main_eos, 'main_truncated': main_truncated,
              'source_hashes': {'audit_script': digest(__file__), 'wire_parser_helper': digest(helper_path),
                                'conditions': digest(settings_path), 'manifest': digest(folder / 'manifest.json'),
                                'summary': digest(folder / 'summary.json'), 'calls': digest(events_path),
                                'inputs': digest(inputs_path), 'tokenizer': digest(tokenizer_path)},
              'actual_packets': packets,
              'limits': ['Successful byte/provenance/position checks do not establish semantic equivalence or KV superiority.',
                         'All 72 main answers still need human format review; short output is not an automatic answer-only pass.',
                         'Response IDs are decoded without gold labels or answer extraction.']}
    output = ROOT / 'checks/audit_answer_format_run_20261002.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['run_name', 'passed', 'checks', 'errors', 'event_counts',
          'underlying_model_forward_calls', 'main_eos', 'main_truncated']}, ensure_ascii=False))
    if audit.errors:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--tokenizer', required=True)
    main(parser.parse_args())
