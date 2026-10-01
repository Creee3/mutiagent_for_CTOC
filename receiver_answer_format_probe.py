"""Same-six answer-format control; reuse official full-prefix KVComm and codecs."""
import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

from run import ROOT, write, sha, gpu_snapshot, load_model
from receiver_endpoint_probe import CallLedger, packet_info, ids_hash, read
from prepare_receiver_answer_format import material_packet

PREPARATION = ROOT / 'receiver_answer_format_conditions.json'
CONDITIONS = ROOT / 'receiver_answer_format_run_conditions.json'
INPUTS = ROOT / 'data/receiver_answer_format_20261002.json'
AUDIT = ROOT / 'checks/audit_receiver_answer_format_20261002.json'
METHODS = ['no_msg', 'token_full', 'kv_full', 'material_text']


def utf8_packet_info(packet):
    body = json.loads(packet.decode('utf-8'))
    content = len(body['material'].encode('utf-8'))
    return {'payload_bytes': len(packet), 'payload_content_bytes': content,
        'payload_header_bytes': len(packet)-content,
        'payload_overhead_kind': 'JSON metadata, syntax and escaping; not a contiguous binary header',
        'token_content_bytes': 0, 'kv_tensor_content_bytes': 0,
        'packet_sha256': hashlib.sha256(packet).hexdigest(),
        'encoding': 'source-material-json-v1', 'network_transfer_measured': False}


def verify(settings):
    assert sha(INPUTS) == settings['input_sha256']
    assert sha(PREPARATION) == settings['preparation_conditions_sha256']
    for name, digest in settings['source_hashes'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in settings['input_source_hashes'].items():
        assert sha(ROOT/name) == digest, name
    audit = read(AUDIT)
    assert sha(AUDIT) == settings['independent_cpu_audit_sha256'] and audit['passed']
    assert audit['inputs_sha256'] == sha(INPUTS) and audit['conditions_sha256'] == sha(PREPARATION)
    assert audit['audit_script_sha256'] == sha(ROOT/'checks/audit_receiver_answer_format_20261002.py')
    assert sha(ROOT/'checks/receiver_answer_format_cpu.json') == settings['preparation_cpu_sha256']
    assert settings['expected_calls'] == read(PREPARATION)['proposed_expected_calls']


def freeze_run():
    assert not CONDITIONS.exists(), 'preserve the run freeze'
    prep, cpu, audit = read(PREPARATION), read(ROOT/'checks/receiver_answer_format_cpu.json'), read(AUDIT)
    assert cpu['passed'] and audit['passed'] and cpu['model_calls'] == audit['model_calls'] == 0
    assert cpu['script_sha256'] == sha(ROOT/'prepare_receiver_answer_format.py')
    assert cpu['inputs_sha256'] == sha(INPUTS) and cpu['conditions_sha256'] == sha(PREPARATION)
    assert audit['inputs_sha256'] == sha(INPUTS) and audit['conditions_sha256'] == sha(PREPARATION)
    settings = {**prep, 'stage': 'run conditions frozen; GPU not started',
        'inference_runner_ready': True,
        'preparation_conditions_sha256': sha(PREPARATION),
        'preparation_cpu_sha256': sha(ROOT/'checks/receiver_answer_format_cpu.json'),
        'independent_cpu_audit_sha256': sha(AUDIT),
        'expected_calls': prep['proposed_expected_calls'],
        'source_hashes': {**prep['source_hashes'], 'receiver_answer_format_probe.py': sha(__file__)}}
    verify(settings)
    write(CONDITIONS, settings)
    print(json.dumps({'conditions_sha256': sha(CONDITIONS), 'expected_calls': settings['expected_calls'],
                      'model_calls': 0, 'inference_started': False}))


def launch():
    import fcntl
    settings = read(CONDITIONS)
    verify(settings)
    lock = open(ROOT/'worker.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    candidates = sorted([g for g in gpu_snapshot() if int(g['utilization']) < 10],
                        key=lambda g: int(g['free_mib']), reverse=True)
    assert candidates and int(candidates[0]['free_mib']) >= settings['capacity_required_mib'], 'no qualified capacity; no CPU fallback'
    gpu = candidates[0]
    name = 'answer_format_' + time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    logdir = ROOT/'logs'
    logdir.mkdir(exist_ok=True)
    log = open(logdir/(name+'.log'), 'ab', buffering=0)
    os.set_inheritable(lock.fileno(), True)
    worker = {'run_name': name, 'pid': None, 'gpu': gpu, 'conditions_sha256': sha(CONDITIONS)}
    write(ROOT/'receiver_answer_format_worker.json', worker)
    child = subprocess.Popen(['/home/liuxin/.conda/envs/kv-graph/bin/python', '-u', str(Path(__file__).resolve()), '--run-name', name],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
    worker['pid'] = child.pid
    write(ROOT/'receiver_answer_format_worker.json', worker)
    print(json.dumps(worker), flush=True)


def main(name):
    import torch
    import transformers
    from transformers import AutoTokenizer
    from models import CVCommunicator
    from chain_core import State, add_segment, cache_lengths, cache_hash
    from owned_prefix_wire import full_packet, full_text_packet, decode_packet, decode_text_packet
    settings, data = read(CONDITIONS), read(INPUTS)
    verify(settings)
    worker = read(ROOT/'receiver_answer_format_worker.json')
    assert worker['run_name'] == name and worker['conditions_sha256'] == sha(CONDITIONS)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == worker['gpu']['uuid']
    assets, verified = read(ROOT/'assets.json'), read(ROOT/'checks/model7b_manifest.json')
    for source in ['models.py', 'model_attn.py', 'eval.py', 'layer_importance.py']:
        assert sha(ROOT/'work/KVComm'/source) == assets['source_files'][source]['work_sha256']
    assert verified['revision'] == settings['model_revision']
    model_path = Path(settings['model_path'])
    for file, digest in settings['model_tokenizer_hashes'].items():
        assert sha(model_path/file) == digest, file
    assert all((model_path/file).stat().st_size == spec['bytes'] for file,spec in verified['files'].items())
    folder = ROOT/'runs'/name
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'code').mkdir()
    for source in settings['source_hashes']:
        (folder/'code'/source).write_bytes((ROOT/source).read_bytes())
    write(folder/'manifest.json', {**settings, 'run_name': name, 'conditions_sha256': sha(CONDITIONS),
        'gpu_launch': worker['gpu'], 'torch': torch.__version__, 'transformers': transformers.__version__,
        'expected_counts': {'generate': 76, 'sender_prefill': 7},
        'expected_phase_counts': settings['expected_calls'], 'network_transfer_measured': False})
    torch.manual_seed(42)
    model, load_seconds = load_model(torch, settings['model_path'])
    tokenizer = AutoTokenizer.from_pretrained(settings['model_path'], local_files_only=True)
    assert model.config.num_hidden_layers == settings['full_layers'] == 28
    assert model.generation_config.eos_token_id == settings['eos_token_id']
    assert model.generation_config.pad_token_id == settings['pad_token_id']
    cv = CVCommunicator(model, model, 0, model.config.num_hidden_layers-1,
        layers_list=list(range(model.config.num_hidden_layers)), shift_back=False)
    cv.generation_config = copy.deepcopy(model.generation_config)
    args = dict(max_new_tokens=settings['answer_max_new_tokens'], do_sample=False, num_beams=1,
        use_cache=True, repetition_penalty=1.0, eos_token_id=settings['eos_token_id'],
        pad_token_id=settings['pad_token_id'], return_dict_in_generate=True, output_scores=False, output_logits=False)
    manifest = read(folder/'manifest.json')
    manifest.update(model_load_seconds=load_seconds, generation_kwargs=args, identity_layer_map=cv.layer_map,
        timing_scope='from frozen tokens ready; all previous tokenization excluded uniformly',
        source_representation='new canonical single-system material prefix; not old reader-role cache')
    write(folder/'manifest.json', manifest)
    ledger = CallLedger(torch, model, folder)
    eos = args['eos_token_id'] if isinstance(args['eos_token_id'], list) else [args['eos_token_id']]
    senders = {s['question_id']: s for s in data['senders']}
    rows = []

    def stage(fn):
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        return value, time.perf_counter()-started

    def source(s, cases, phase):
        qid, revision = s['question_id'], settings['model_revision']
        a = torch.tensor([s['ids']], dtype=torch.int64, device=model.device)
        text_state = State(a, add_segment([], qid, 'answer_format_source', 'A', 'prompt', a, 0))
        text_packet, text_encode = stage(lambda: full_text_packet(torch, text_state, revision, qid))
        assert packet_info(text_packet) == s['prepared_token_packet_info']
        utf8_packet, utf8_encode = stage(lambda: material_packet(s['sender_material'], revision, qid))
        assert hashlib.sha256(utf8_packet).hexdigest() == s['prepared_material_packet']['packet_sha256']
        out, prefill = ledger.call('sender_prefill', phase, qid+':A',
            lambda: model(input_ids=a, attention_mask=torch.ones_like(a), use_cache=True, return_dict=True))
        cache = out.past_key_values
        del out
        n = len(s['ids'])
        assert cache_lengths(cache) == [n]*28 and prefill['effective_position_ids'] == [[list(range(n))]]
        state = State(a, text_state.segments, cache)
        source_hash = cache_hash(torch, cache)
        kv_packet, kv_encode = stage(lambda: full_packet(torch, state, revision, qid))
        packet_values = {'token_full': text_packet, 'kv_full': kv_packet, 'material_text': utf8_packet}
        metrics = {'token_full': packet_info(text_packet), 'kv_full': packet_info(kv_packet, True),
                   'material_text': utf8_packet_info(utf8_packet)}
        encodes = {'token_full': text_encode, 'kv_full': kv_encode, 'material_text': utf8_encode}
        packet_dir = folder/'packets'
        packet_dir.mkdir(exist_ok=True)
        for method, packet in packet_values.items():
            (packet_dir/f'{phase}_{qid}_{method}.bin').write_bytes(packet)
        write(folder/'sources'/f'{phase}_{qid}.json', {'question_id': qid, 'phase': phase,
            'A_ids': s['ids'], 'cache_source_hash': source_hash, 'sender_prefill': prefill,
            'cache_resident_tensor_bytes': metrics['kv_full']['kv_tensor_content_bytes'],
            'text_packet': {**metrics['token_full'], 'encode_seconds': text_encode},
            'kv_packet': {**metrics['kv_full'], 'encode_seconds': kv_encode},
            'material_packet': {**metrics['material_text'], 'encode_seconds': utf8_encode}})
        for case in cases:
            views = case['method_inputs']
            assert s['ids'] + views['kv_full']['ids'] == views['token_full']['ids']
            row = {**case, 'phase': phase, 'cache_source_hash': source_hash, 'methods': {}}
            order = METHODS.copy()
            random.Random(hashlib.sha256(case['key'].encode()).hexdigest()).shuffle(order)
            row['method_order'] = order
            for method in order:
                prefix, restore_time, restored = 0, 0.0, None
                input_ids = torch.tensor([views[method]['ids']], dtype=torch.int64, device=model.device)
                target = model
                if method == 'no_msg':
                    payload = dict(payload_bytes=0, payload_content_bytes=0, payload_header_bytes=0,
                                   token_content_bytes=0, kv_tensor_content_bytes=0, network_transfer_measured=False)
                elif method == 'token_full':
                    restored, restore_time = stage(lambda: decode_text_packet(torch, text_packet, revision, qid, device=model.device))
                    assert torch.equal(restored.history, a)
                    suffix = torch.tensor([case['B_suffix_ids']], dtype=torch.int64, device=model.device)
                    actual = torch.cat([restored.history, suffix], -1)
                    assert torch.equal(actual, input_ids)
                    input_ids = actual
                    payload = metrics[method]
                elif method == 'kv_full':
                    restored, restore_time = stage(lambda: decode_packet(torch, kv_packet, revision, qid, device=model.device))
                    assert torch.equal(restored.history, a) and cache_lengths(restored.cache) == [n]*28
                    prefix, target, payload = n, cv, metrics[method]
                else:
                    received, restore_time = stage(lambda: json.loads(utf8_packet.decode('utf-8')))
                    assert received['qid'] == qid and received['revision'] == revision
                    assert case['strong_text_user'] == case['B_user']+'\nReceived material from A:\n'+received['material']
                    payload = metrics[method]
                kwargs = dict(args, attention_mask=torch.ones_like(input_ids))
                if method == 'kv_full':
                    kwargs['out_A_past_key_values'] = restored.cache
                output, call = ledger.call('generate', phase, case['key']+':'+method,
                    lambda: target.generate(input_ids, **kwargs))
                assert torch.equal(output.sequences[:, :input_ids.shape[-1]], input_ids)
                output_ids = output.sequences[0,input_ids.shape[-1]:].cpu().tolist()
                assert output_ids and len(output_ids) == call['model_forward_calls']
                total = input_ids.shape[-1] + prefix
                positions = call['effective_position_ids']
                assert positions[0] == [list(range(prefix, total))]
                assert positions[1:] == [[[total+i-1]] for i in range(1,len(output_ids))]
                assert len(positions) == len(output_ids)
                pref = prefill['seconds'] if method == 'kv_full' else 0.0
                encode = encodes.get(method, 0.0)
                ended = output_ids[-1] in eos
                row['methods'][method] = {**payload, 'response': tokenizer.decode(output_ids, skip_special_tokens=True),
                    'output_ids': output_ids, 'generated_tokens': len(output_ids), 'eos_stopped': ended,
                    'actually_truncated': len(output_ids)>=settings['answer_max_new_tokens'] and not ended,
                    'model_forward_calls': call['model_forward_calls'], 'receiver_input_tokens': input_ids.shape[-1],
                    'sender_prefill_seconds': pref, 'receiver_model_seconds': call['seconds'],
                    'receiver_prefill_gpu_seconds': call['prefill_gpu_seconds'], 'receiver_decode_gpu_seconds': call['decode_gpu_seconds'],
                    'packet_encode_seconds': encode, 'packet_restore_seconds': restore_time,
                    'from_scratch_stage_sum_seconds': pref+encode+restore_time+call['seconds'],
                    'preexisting_cache_stage_sum_seconds': encode+restore_time+call['seconds'],
                    'timing_has_light_position_event_hooks': True, 'previous_tokenization_excluded_uniformly': True,
                    'standalone_time_is_sum_of_measured_stages_not_an_independent_run': True,
                    'checked_effective_positions': True}
                del output, restored, kwargs
                gc.collect()
            assert cache_hash(torch, cache) == source_hash, 'source cache mutated'
            row['source_cache_unchanged'] = True
            tag = hashlib.sha256(case['key'].encode()).hexdigest()[:20]
            write(folder/'items'/f'{phase}_{tag}.json', row)
            if phase == 'main':
                rows.append(row)
            print(json.dumps({'phase': phase, 'key': case['key'], 'EOS': {m:v['eos_stopped'] for m,v in row['methods'].items()}}), flush=True)
        del state, text_state, cache, kv_packet, text_packet, utf8_packet, packet_values
        gc.collect()

    with torch.inference_mode():
        warm = next(c for c in data['receivers'] if c['key'] == settings['warmup_key'])
        source(senders[warm['question_id']], [warm], 'warmup')
        for qid, s in senders.items():
            source(s, [c for c in data['receivers'] if c['question_id'] == qid], 'main')
    phases = {p+'_'+k: sum(e['phase']==p and e['kind']==k for e in ledger.events)
              for p in ['main','warmup'] for k in ['generate','sender_prefill']}
    assert phases == settings['expected_calls'] and len(rows) == 18
    summary = {'status': 'completed', 'unique_questions': 6, 'main_input_rows': 18,
        'counts': dict(Counter(e['kind'] for e in ledger.events)), 'phase_counts': phases,
        'underlying_model_forward_calls': sum(e['model_forward_calls'] for e in ledger.events),
        'EOS_main_answers': sum(v['eos_stopped'] for r in rows for v in r['methods'].values()),
        'truncated_main_answers': sum(v['actually_truncated'] for r in rows for v in r['methods'].values()),
        'network_transfer_measured': False, 'timing_repeats': 0, 'quality_not_evaluated_by_runner': True,
        'new_unique_evaluation_questions': 0, 'answer_format_not_automatically_verified': True}
    write(folder/'summary.json', summary)
    write(folder/'COMPLETE.json', {'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'counts': phases})
    for handle in ledger.handles:
        handle.remove()
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-run', action='store_true')
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--run-name')
    args = parser.parse_args()
    if args.freeze_run:
        freeze_run()
    elif args.launch:
        launch()
    else:
        try:
            main(args.run_name)
        except Exception:
            write(ROOT/'runs'/args.run_name/'FAILED.json', {'error': traceback.format_exc(),
                'failed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
            raise
