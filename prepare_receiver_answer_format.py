"""Freeze the same-six short-answer/canonical-chat inputs. No model inference."""
import hashlib
import json
import os
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / 'data/receiver_answer_format_20261002.json'
CONDITIONS = ROOT / 'receiver_answer_format_conditions.json'
CPU_REPORT = ROOT / 'checks/receiver_answer_format_cpu.json'
METHODS = ['no_msg', 'token_full', 'kv_full', 'material_text']
SYSTEM = ('You answer questions using the supplied evidence. Return only the answer: '
          'a name, number, date, short phrase, or yes/no, as needed. Do not repeat '
          'the question or add an explanation. Do not add a complete sentence '
          'around the answer. If the supplied evidence does not determine the '
          'answer, return unknown.')
B_ROLE = 'You are Agent B, the answerer.'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def material_packet(material, revision, qid):
    """A JSON UTF-8 packet; only its material field becomes receiver evidence."""
    body = {'format': 'source-material-json-v1', 'source': 'A',
            'qid': qid, 'revision': revision, 'material': material}
    return json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def build():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    from transformers import AutoTokenizer, __version__ as transformers_version
    import torch
    from run import write
    from receiver_endpoint_probe import ids_hash, packet_info
    from chain_core import State, add_segment
    from owned_prefix_wire import full_text_packet, decode_text_packet

    assert all(not p.exists() for p in [INPUTS, CONDITIONS, CPU_REPORT]), 'do not overwrite a freeze'
    cfg = read(ROOT / 'config.json')
    source_path = ROOT / 'data/receiver_evidence_20261002/selector_input_views.json'
    safe_path = ROOT / 'data/receiver_runtime_views_20261002/runtime_payloads.json'
    schedule_path = ROOT / 'data/receiver_runtime_views_20261002/scheduler_metadata.json'
    original = read(source_path)
    safe = read(safe_path)
    schedule = read(schedule_path)
    # The six questions are selected by the already-frozen availability of a
    # partial view, not by previous answers, scores, or gold labels.
    partial_ids = {r['question_id'] for r in schedule if r['condition'] == 'partial'}
    selected = [s for s in original['senders'] if s['question_id'] in partial_ids]
    assert len(selected) == len(partial_ids) == 6
    meta = {r['lookup_id']: r for r in schedule}
    chosen = [r for r in safe if meta[r['lookup_id']]['question_id'] in partial_ids
              and meta[r['lookup_id']]['condition'] in ['missing', 'partial', 'full']]
    assert len(chosen) == 18
    assert len({meta[r['lookup_id']]['source_key'] for r in chosen}) == 18
    assert {meta[r['lookup_id']]['condition'] for r in chosen} == {'missing', 'partial', 'full'}

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_7b_path'], local_files_only=True)
    model_path = Path(cfg['model_7b_path'])
    model_cfg = read(model_path / 'config.json')
    gen_cfg = read(model_path / 'generation_config.json')
    sender_views, receiver_views = [], []
    encode_calls = 0
    started = time.perf_counter()

    def encode(messages, generation):
        nonlocal encode_calls
        encode_calls += 1
        return tokenizer.apply_chat_template(messages, add_generation_prompt=generation, tokenize=True)

    for s in selected:
        qid = s['question_id']
        # A is a material source in this prefill-only mechanism check. Its old
        # three-sentence extraction instruction is not a task for the receiver.
        source_user = 'Source A evidence for this question.\nQuestion: ' + s['question'] + '\nMaterial:\n' + s['sender_material']
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': source_user}]
        ids = encode(messages, False)
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        state = State(torch.tensor([ids], dtype=torch.int64), [])
        state.segments = add_segment([], qid, 'answer_format_source', 'A', 'prompt', state.history, 0)
        packet = full_text_packet(torch, state, cfg['model_7b_revision'], qid)
        restored = decode_text_packet(torch, packet, cfg['model_7b_revision'], qid)
        assert torch.equal(restored.history, state.history)
        utf8_packet = material_packet(s['sender_material'], cfg['model_7b_revision'], qid)
        received = json.loads(utf8_packet.decode('utf-8'))
        assert received['material'] == s['sender_material'] and received['qid'] == qid
        material_bytes = len(s['sender_material'].encode('utf-8'))
        # JSON syntax/escaping is overhead too; content is counted in its
        # original UTF-8 representation and every transmitted byte is counted.
        sender_views.append({'question_id': qid, 'question': s['question'],
            'sender_material': s['sender_material'], 'messages': messages, 'rendered': rendered,
            'ids': ids, 'ids_sha256': ids_hash(ids), 'source_state_id': state.segments[0]['id'],
            'prepared_token_packet_info': packet_info(packet),
            'prepared_material_packet': {'packet_text': utf8_packet.decode('utf-8'),
                'packet_sha256': hashlib.sha256(utf8_packet).hexdigest(),
                'payload_bytes': len(utf8_packet), 'material_utf8_bytes': material_bytes,
                'json_metadata_and_escaping_bytes': len(utf8_packet) - material_bytes,
                'encoding': 'source-material-json-v1', 'network_transfer_measured': False},
            'cache_status': 'not_generated_new_prefix'})

    by_qid = {s['question_id']: s for s in sender_views}
    maximum = 0
    for r in chosen:
        p = r['payload']
        m = meta[r['lookup_id']]
        s = by_qid[m['question_id']]
        assert set(p) == {'question', 'sender_material', 'receiver_visible_text'}
        assert p['question'] == s['question'] and p['sender_material'] == s['sender_material']
        # This exact remainder retains both local B material and held A material.
        remainder = p['receiver_visible_text'].split('\nQuestion: ', 1)[1]
        b_user = B_ROLE + '\nQuestion: ' + remainder
        b_message = {'role': 'user', 'content': b_user}
        full_messages = s['messages'] + [b_message]
        canonical = encode(full_messages, True)
        rendered = tokenizer.apply_chat_template(full_messages, add_generation_prompt=True, tokenize=False)
        assert rendered.startswith(s['rendered'])
        suffix_text = rendered[len(s['rendered']):]
        encode_calls += 1
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        assert s['ids'] + suffix_ids == canonical, 'canonical prefix/suffix token boundary mismatch'
        assert rendered.count('<|im_start|>system\n') == 1
        assert '<|im_start|>system\n' not in suffix_text
        assert rendered.endswith('<|im_start|>assistant\n')
        assert s['rendered'].endswith('<|im_end|>\n')
        no_messages = [{'role': 'system', 'content': SYSTEM}, b_message]
        no_ids = encode(no_messages, True)
        received_material = json.loads(s['prepared_material_packet']['packet_text'])['material']
        strong_user = b_user + '\nReceived material from A:\n' + received_material
        strong_messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': strong_user}]
        strong_ids = encode(strong_messages, True)
        assert strong_user.startswith(b_user) and strong_user.endswith(s['sender_material'])
        max_tokens = max(len(canonical), len(no_ids), len(strong_ids)) + 48
        maximum = max(maximum, max_tokens)
        receiver_views.append({'key': m['source_key'], 'question_id': m['question_id'],
            'condition': m['condition'], 'lookup_id': r['lookup_id'],
            'runtime_payload': p, 'B_user': b_user, 'strong_text_user': strong_user,
            'canonical_messages': full_messages, 'canonical_rendered': rendered,
            'B_suffix_text': suffix_text, 'B_suffix_ids': suffix_ids,
            'B_suffix_ids_sha256': ids_hash(suffix_ids),
            'method_inputs': {
                'no_msg': {'ids': no_ids, 'ids_sha256': ids_hash(no_ids)},
                'token_full': {'ids': canonical, 'ids_sha256': ids_hash(canonical)},
                'kv_full': {'ids': suffix_ids, 'ids_sha256': ids_hash(suffix_ids),
                            'prefix_tokens': len(s['ids']), 'joint_ids_sha256': ids_hash(canonical)},
                'material_text': {'ids': strong_ids, 'ids_sha256': ids_hash(strong_ids)}},
            'A_ids_sha256': s['ids_sha256'], 'maximum_prompt_tokens_plus_48': max_tokens})

    assert maximum <= model_cfg['max_position_embeddings']
    assert not torch.cuda.is_initialized()
    write(INPUTS, {'senders': sender_views, 'receivers': receiver_views})
    deps = ['prepare_receiver_answer_format.py', 'receiver_endpoint_probe.py', 'run.py',
            'chain_core.py', 'alignment_probe.py', 'owned_prefix_wire.py']
    source_files = [ROOT/'config.json', source_path, safe_path, schedule_path]
    settings = {
        'stage': 'CPU input freeze only; no GPU runner or inference yet',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model_path': cfg['model_7b_path'], 'model_revision': cfg['model_7b_revision'],
        'official_commit': cfg['source_commit'], 'dtype': 'bfloat16', 'attention': 'sdpa',
        'methods': METHODS, 'system_rule': SYSTEM, 'B_role': B_ROLE,
        'unique_questions': 6, 'selected_question_ids': [s['question_id'] for s in selected],
        'question_selection': 'all frozen partial-available calibration IDs in original order; no scores read',
        'receiver_inputs': 18, 'condition_counts': {'missing': 6, 'partial': 6, 'full': 6},
        'input_file': str(INPUTS.relative_to(ROOT)), 'input_sha256': sha(INPUTS),
        'answer_max_new_tokens': 48, 'greedy': True, 'num_beams': 1,
        'repetition_penalty': 1.0, 'full_layers': model_cfg['num_hidden_layers'], 'shift_back': False,
        'eos_token_id': gen_cfg['eos_token_id'], 'pad_token_id': gen_cfg['pad_token_id'],
        'proposed_expected_calls': {'main_generate': 72, 'warmup_generate': 4,
                                    'main_sender_prefill': 6, 'warmup_sender_prefill': 1},
        'warmup_key': receiver_views[0]['key'], 'warmup_is_existing_calibration_view': True,
        'timing_repeats': 0, 'cache_tail_forwards': 0,
        'model_calls_executed': 0, 'inference_runner_ready': False,
        'maximum_prompt_tokens_plus_answer_budget': maximum,
        'capacity_required_mib': 18000,
        'source_hashes': {name: sha(ROOT/name) for name in deps},
        'input_source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in source_files},
        'model_tokenizer_hashes': {name: sha(model_path/name) for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json']},
        'answer_evaluation': {'raw_response_EM_F1': True, 'gold_aware_extraction': False,
            'format_review': 'manual answer phrase/sentence/abstention review; no automatic length-based correctness',
            'failure_rule': 'if format differs systematically across methods, do not infer semantic message-value trend'},
        'limits': ['format calibration, not an algorithm or confirmatory benchmark',
            'A is a prefill-only source, not a sender-generated analysis; no private receiver cache merging',
            'new canonical A state differs from the old role-prompt state; do not reuse old source cache',
            'system rule, A representation and chat layout changed together; not causal isolation of old formatting',
            'whole prefix still includes question/system/source framing, not only missing facts',
            'material_text is a strong workflow baseline; its tokens differ from exact token/KV controls',
            'full material is copied without condition/oracle pruning; visible held text remains in every method',
            'six short public calibration questions only; no other-document control or 100/500 evaluations',
            'timings start from prepared token arrays, excluding all prior tokenization uniformly; no network or speed claim']}
    write(CONDITIONS, settings)
    report = {'passed': True, 'script_sha256': sha(__file__), 'inputs_sha256': sha(INPUTS),
        'conditions_sha256': sha(CONDITIONS), 'transformers_version': transformers_version,
        'unique_questions': 6, 'receiver_rows': 18, 'method_input_plans': 72,
        'tokenizer_encode_calls': encode_calls, 'canonical_prefix_suffix_equal_pairs': 18,
        'single_system_full_prompts': 18, 'held_material_preserved_rows': 18,
        'new_source_token_packet_roundtrips': 6, 'utf8_material_packet_roundtrips': 6,
        'maximum_prompt_tokens_plus_48': maximum,
        'preparation_seconds': time.perf_counter()-started,
        'model_calls': 0, 'model_weights_loaded': False, 'new_KV_generated': False,
        'gpu_initialized': False, 'old_frozen_inputs_modified': False}
    write(CPU_REPORT, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    build()
