"""Held-evidence calibration endpoints using the official complete KV prefix."""
import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import time
import traceback

from run import ROOT, gpu_snapshot, load_model, sha, write

METHODS = ['no_msg', 'token_full', 'kv_full']
B_ROLE = ('You are Agent B, an answerer. Use all supplied material to answer '
          'the question directly, without explanation. Do not invent unsupported facts.')
CONDITIONS = ROOT / 'receiver_endpoint_conditions.json'
INPUTS = ROOT / 'data/receiver_endpoint_tokens_20261002.json'
SOURCE_FILES = ['receiver_endpoint_probe.py', 'run.py', 'alignment_probe.py',
                'chain_core.py', 'owned_prefix_wire.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def packet_info(packet, kv=False):
    size = struct.unpack('<I', packet[:4])[0]
    header = json.loads(packet[4:4+size])
    n = header['length']
    cache_bytes = sum(e[k]['payload_nbytes'] for e in header.get('cache', [])
                      for k in ['key', 'value'])
    assert len(packet) == 4 + size + header['body_nbytes']
    assert header['body_nbytes'] == n*4 + cache_bytes
    return {'payload_bytes': len(packet), 'payload_content_bytes': header['body_nbytes'],
        'payload_header_bytes': size+4, 'token_content_bytes': n*4,
        'kv_tensor_content_bytes': cache_bytes, 'packet_sha256': hashlib.sha256(packet).hexdigest(),
        'encoding': 'owned-prefix-wire-v1', 'position_base': 0,
        'cache_descriptors': header.get('cache'), 'network_transfer_measured': False}


def freeze():
    # Tokenization and small wire tensors only; no model is instantiated.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from transformers import AutoTokenizer
    import torch
    from transformers.cache_utils import DynamicCache
    from chain_core import State, add_segment
    from owned_prefix_wire import full_text_packet, decode_text_packet, full_packet, decode_packet
    assert not CONDITIONS.exists() and not INPUTS.exists(), 'preserve existing freeze'
    cfg = read(ROOT / 'config.json')
    original = read(ROOT / 'data/receiver_evidence_20261002/selector_input_views.json')
    source = {s['question_id']: s for s in original['senders']}
    safe_root = ROOT / 'data/receiver_runtime_views_20261002'
    safe = read(safe_root / 'runtime_payloads.json')
    schedules = {x['lookup_id']: x for x in read(safe_root / 'scheduler_metadata.json')}
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_7b_path'], local_files_only=True)
    senders, receivers = [], []
    revision = cfg['model_7b_revision']
    for qid, s in source.items():
        # A only prefills: close its user turn instead of opening an empty assistant.
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': s['sender_prompt']}],
            add_generation_prompt=False, tokenize=True)
        state = State(torch.tensor([ids], dtype=torch.int64), [])
        state.segments = add_segment([], qid, 'endpoint_source', 'A', 'prompt', state.history, 0)
        packet = full_text_packet(torch, state, revision, qid)
        restored = decode_text_packet(torch, packet, revision, qid)
        assert torch.equal(state.history, restored.history)
        senders.append({'question_id': qid, 'prompt': s['sender_prompt'], 'ids': ids,
            'ids_sha256': ids_hash(ids), 'source_state_id': state.segments[0]['id'],
            'prepared_text_packet_info': packet_info(packet)})
    by_qid = {s['question_id']: s for s in senders}
    maximum = 0
    for row in safe:
        p, metadata = row['payload'], schedules[row['lookup_id']]
        assert set(p) == {'question', 'sender_material', 'receiver_visible_text'}
        assert p['sender_material'] == source[metadata['question_id']]['sender_material']
        # Derive a new role version; retain all question/material/held text.
        remainder = p['receiver_visible_text'].split('\nQuestion: ', 1)[1]
        body = B_ROLE + '\nQuestion: ' + remainder
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': body}],
            add_generation_prompt=True, tokenize=True)
        a_ids = by_qid[metadata['question_id']]['ids']
        maximum = max(maximum, len(a_ids)+len(ids)+48)
        receivers.append({'key': metadata['source_key'], 'question_id': metadata['question_id'],
            'condition': metadata['condition'], 'lookup_id': row['lookup_id'],
            'prompt': body, 'ids': ids, 'ids_sha256': ids_hash(ids),
            'joint_ids_sha256': ids_hash(a_ids+ids)})
    assert len(senders) == 20 and len(receivers) == 72
    by_key = {r['key']: r for r in receivers}
    pairs = read(ROOT/'data/receiver_evidence_20261002/diagnostic_annotations.json')['paired_control_keys']
    assert all(len(by_key[a]['ids'])==len(by_key[b]['ids']) for a,b in pairs)
    model_cfg = read(Path(cfg['model_7b_path']) / 'config.json')
    assert maximum <= model_cfg['max_position_embeddings']
    # One integration packet with the NEW token ledger, not a repeated model gate.
    first = senders[0]
    n = len(first['ids'])
    cache = DynamicCache()
    for layer in range(2):
        key = torch.arange(n*4, dtype=torch.float32).reshape(1, 1, n, 4).to(torch.bfloat16)
        cache.update(key+layer, key-layer, layer)
    state = State(torch.tensor([first['ids']]), [], cache)
    state.segments = add_segment([], first['question_id'], 'endpoint_source', 'A', 'prompt', state.history, 0)
    toy = full_packet(torch, state, revision, first['question_id'])
    restored = decode_packet(torch, toy, revision, first['question_id'])
    assert torch.equal(restored.history, state.history)
    assert all(torch.equal(k, t) for k, t in zip(restored.cache.key_cache, cache.key_cache))
    for item in receivers:
        a_ids = by_qid[item['question_id']]['ids']
        joint = a_ids + item['ids']
        assert ids_hash(joint) == item['joint_ids_sha256']
    write(INPUTS, {'senders': senders, 'receivers': receivers})
    data = {'stage': 'frozen calibration endpoint conditions, not inference results',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model_path': cfg['model_7b_path'], 'model_revision': revision,
        'official_commit': cfg['source_commit'], 'dtype': 'bfloat16', 'attention': 'sdpa',
        'seed': 42, 'methods': METHODS, 'answer_max_new_tokens': 48,
        'greedy': True, 'num_beams': 1, 'repetition_penalty': 1.0,
        'full_layers': model_cfg['num_hidden_layers'], 'shift_back': False,
        'unique_questions': 20, 'receiver_inputs': 72,
        'condition_counts': dict(Counter(x['condition'] for x in receivers)),
        'warmup_key': receivers[0]['key'], 'warmup_is_existing_calibration_input': True,
        'expected_calls': {'main_generate': 216, 'warmup_generate': 3,
                           'main_sender_prefill': 20, 'warmup_sender_prefill': 1},
        'timing_repeats': 0, 'cache_tail_forwards': 0,
        'new_role': B_ROLE, 'sender_add_generation_prompt': False,
        'source_preparation_unchanged': True, 'input_file': str(INPUTS.relative_to(ROOT)),
        'input_sha256': sha(INPUTS), 'max_joint_tokens_plus_answer_budget': maximum,
        'source_hashes': {name: sha(ROOT/name) for name in SOURCE_FILES},
        'input_source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT/'config.json', ROOT/'data/calibration.json',
            ROOT/'data/receiver_evidence_20261002/selector_input_views.json',
            safe_root/'runtime_payloads.json', safe_root/'scheduler_metadata.json']},
        'model_config_sha256': sha(Path(cfg['model_7b_path'])/'config.json'),
        'capacity_required_mib': 18000, 'capacity_rationale': 'one 15.24GB BF16 checkpoint plus bounded batch-one activations/cache/headroom; not a generic GPU policy',
        'limits': ['short public support-sentence calibration, not a benchmark score',
            'whole A prefix effect includes role/question/context length, not only missing facts',
            'A only prefills; not generated-output communication or multiple private KV merging',
            'the original node-output text baseline remains for the later output-communication stage',
            'single instrumented timings without network/repeats do not establish speedup',
            'standalone stage sums and preexisting cache scenarios are accounting estimates, not observed end-to-end network runs']}
    write(CONDITIONS, data)
    report = {'passed': True, 'conditions_sha256': sha(CONDITIONS), 'inputs_sha256': sha(INPUTS),
        'script_sha256': sha(__file__), 'actual_text_packet_roundtrips': 20,
        'new_source_ledger_toy_kv_packet_roundtrips': 1, 'joint_token_plans': 72,
        'derived_complete_prompt_equal_length_pairs': len(pairs),
        'max_joint_tokens_plus_budget': maximum, 'model_calls': 0, 'model_weights_loaded': False,
        'toy_packet_not_real_7b': True}
    write(ROOT/'checks/receiver_endpoint_cpu.json', report)
    print(json.dumps(report), flush=True)


class CallLedger:
    """Low volume GPU position/event recording; no per-token CPU transfers."""
    def __init__(self, torch, model, folder):
        self.torch, self.folder, self.model = torch, folder, model
        self.events, self.current = [], None
        self.handles = [model.register_forward_pre_hook(self.before, with_kwargs=True),
            model.register_forward_hook(self.after, with_kwargs=True),
            model.model.rotary_emb.register_forward_pre_hook(self.rotary, with_kwargs=True)]

    def before(self, module, args, kwargs):
        if self.current is None:
            return
        start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
        start.record()
        self.current['gpu_events'].append((start, end))

    def after(self, module, args, kwargs, output):
        if self.current is not None:
            self.current['gpu_events'][-1][1].record()

    def rotary(self, module, args, kwargs):
        if self.current is not None:
            position = kwargs.get('position_ids', args[1] if len(args)>1 else None)
            # Retain the generated position tensor; conversion occurs after timing.
            self.current['positions'].append(position.detach())

    def call(self, kind, phase, tag, fn):
        torch = self.torch
        self.current = {'gpu_events': [], 'positions': []}
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            output = fn()
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            gpu_times = [a.elapsed_time(b)/1000 for a,b in self.current['gpu_events']]
            positions = [p.cpu().tolist() for p in self.current['positions']]
            event = {'kind': kind, 'phase': phase, 'tag': tag, 'status': 'ok',
                'seconds': seconds, 'model_forward_calls': len(gpu_times),
                'prefill_gpu_seconds': gpu_times[0], 'decode_gpu_seconds': sum(gpu_times[1:]),
                'effective_position_ids': positions,
                'timing_has_light_position_event_hooks': True}
        except Exception as exc:
            self.append({'kind': kind, 'phase': phase, 'tag': tag, 'status': 'error', 'error': repr(exc)})
            raise
        finally:
            self.current = None
        self.append(event)
        return output, event

    def append(self, event):
        self.events.append(event)
        with (self.folder/'calls.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(event)+'\n')


def main(run_name):
    import torch
    import transformers
    from models import CVCommunicator
    from chain_core import State, add_segment, cache_lengths, cache_hash
    from owned_prefix_wire import full_text_packet, full_packet, decode_packet, decode_text_packet
    settings, data, cfg = read(CONDITIONS), read(INPUTS), read(ROOT/'config.json')
    for name, digest in settings['source_hashes'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in settings['input_source_hashes'].items():
        assert sha(ROOT/name) == digest, name
    assert sha(INPUTS) == settings['input_sha256']
    assets = read(ROOT/'assets.json')
    for name in ['models.py', 'model_attn.py', 'eval.py', 'layer_importance.py']:
        assert sha(ROOT/'work/KVComm'/name) == assets['source_files'][name]['work_sha256']
    verified = read(ROOT/'checks/model7b_manifest.json')
    assert verified['revision'] == settings['model_revision']
    model_path = Path(settings['model_path'])
    assert sha(model_path/'config.json') == settings['model_config_sha256']
    assert all((model_path/name).stat().st_size == spec['bytes'] for name,spec in verified['files'].items())
    folder = ROOT/'runs'/run_name
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'code').mkdir()
    for name in SOURCE_FILES:
        (folder/'code'/name).write_bytes((ROOT/name).read_bytes())
    write(folder/'manifest.json', {**settings, 'run_name': run_name, 'conditions_sha256': sha(CONDITIONS),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'gpu_launch': read(ROOT/'receiver_endpoint_worker.json')['gpu'],
        'expected_counts': {'generate':219, 'sender_prefill':21},
        'network_transfer_measured': False})
    torch.manual_seed(42)
    model, load_seconds = load_model(torch, settings['model_path'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(settings['model_path'], local_files_only=True)
    cv = CVCommunicator(model, model, 0, model.config.num_hidden_layers-1,
        layers_list=list(range(model.config.num_hidden_layers)), shift_back=False)
    cv.generation_config = copy.deepcopy(model.generation_config)
    args = dict(max_new_tokens=48, do_sample=False, num_beams=1, use_cache=True,
        repetition_penalty=1.0, eos_token_id=copy.deepcopy(model.generation_config.eos_token_id),
        pad_token_id=model.generation_config.pad_token_id, return_dict_in_generate=True,
        output_scores=False, output_logits=False)
    manifest = read(folder/'manifest.json')
    manifest.update(model_load_seconds=load_seconds, generation_kwargs=args,
        generation_config=model.generation_config.to_dict(), identity_layer_map=cv.layer_map,
        serialization_requires_cpu_bytes=True)
    write(folder/'manifest.json', manifest)
    ledger = CallLedger(torch, model, folder)
    eos = args['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    revision = settings['model_revision']
    senders = {s['question_id']: s for s in data['senders']}
    rows = []

    def stage(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        return value, time.perf_counter()-start

    def question(s, cases, phase):
        qid = s['question_id']
        a = torch.tensor([s['ids']], dtype=torch.int64, device=model.device)
        text_state = State(a, add_segment([], qid, 'endpoint_source', 'A', 'prompt', a, 0))
        text_packet, text_encode = stage(lambda: full_text_packet(torch, text_state, revision, qid))
        out, prefill = ledger.call('sender_prefill', phase, qid+':A',
            lambda: model(input_ids=a, attention_mask=torch.ones_like(a), use_cache=True, return_dict=True))
        cache = out.past_key_values
        del out
        n = a.shape[-1]
        assert cache_lengths(cache) == [n]*model.config.num_hidden_layers
        assert prefill['effective_position_ids'] == [[list(range(n))]]
        state = State(a, text_state.segments, cache)
        source_hash = cache_hash(torch, cache)
        kv_packet, kv_encode = stage(lambda: full_packet(torch, state, revision, qid))
        packet_dir = folder/'packets'
        packet_dir.mkdir(exist_ok=True)
        for name, packet in [('token_full', text_packet), ('kv_full', kv_packet)]:
            (packet_dir/f'{phase}_{qid}_{name}.bin').write_bytes(packet)
        sender_meta = {'question_id': qid, 'phase': phase, 'sender_prefill': prefill,
            'A_ids': s['ids'], 'cache_source_hash': source_hash,
            'text_packet': {**packet_info(text_packet), 'encode_seconds': text_encode},
            'kv_packet': {**packet_info(kv_packet, True), 'encode_seconds': kv_encode},
            'cache_resident_tensor_bytes': packet_info(kv_packet, True)['kv_tensor_content_bytes']}
        write(folder/'sources'/f'{phase}_{qid}.json', sender_meta)
        for case in cases:
            b = torch.tensor([case['ids']], dtype=torch.int64, device=model.device)
            assert ids_hash(s['ids']+case['ids']) == case['joint_ids_sha256']
            row = {**case, 'phase': phase, 'A_ids_sha256': s['ids_sha256'],
                'cache_source_hash': source_hash, 'methods': {}}
            order = METHODS.copy()
            random.Random(hashlib.sha256(case['key'].encode()).hexdigest()).shuffle(order)
            row['method_order'] = order
            for method in order:
                restored, restore_time = None, 0.0
                prefix = 0
                if method == 'no_msg':
                    input_ids, target = b, model
                    payload = {'payload_bytes': 0, 'payload_content_bytes': 0, 'payload_header_bytes': 0,
                        'token_content_bytes': 0, 'kv_tensor_content_bytes': 0, 'network_transfer_measured': False}
                elif method == 'token_full':
                    restored, restore_time = stage(lambda: decode_text_packet(torch, text_packet, revision, qid, device=model.device))
                    assert torch.equal(restored.history, a)
                    input_ids, target = torch.cat([restored.history,b], -1), model
                    payload = packet_info(text_packet)
                else:
                    restored, restore_time = stage(lambda: decode_packet(torch, kv_packet, revision, qid, device=model.device))
                    assert torch.equal(restored.history, a)
                    assert cache_lengths(restored.cache) == [n]*model.config.num_hidden_layers
                    input_ids, target, prefix = b, cv, n
                    payload = packet_info(kv_packet, True)
                kwargs = dict(args, attention_mask=torch.ones_like(input_ids))
                if method == 'kv_full':
                    kwargs['out_A_past_key_values'] = restored.cache
                output, call = ledger.call('generate', phase, case['key']+':'+method,
                    lambda: target.generate(input_ids, **kwargs))
                assert torch.equal(output.sequences[:, :input_ids.shape[-1]], input_ids)
                output_ids = output.sequences[0,input_ids.shape[-1]:].cpu().tolist()
                assert output_ids and call['model_forward_calls'] == len(output_ids)
                total_prompt = input_ids.shape[-1] + prefix
                positions = call['effective_position_ids']
                assert positions[0] == [list(range(prefix, total_prompt))]
                assert all(p == [[total_prompt+i-1]] for i,p in enumerate(positions[1:],start=1))
                assert len(positions) == len(output_ids)
                ended = output_ids[-1] in eos
                encode = kv_encode if method == 'kv_full' else text_encode if method == 'token_full' else 0.0
                pref = prefill['seconds'] if method == 'kv_full' else 0.0
                result = {**payload, 'response': tokenizer.decode(output_ids, skip_special_tokens=True),
                    'output_ids': output_ids, 'generated_tokens': len(output_ids), 'eos_stopped': ended,
                    'actually_truncated': len(output_ids)>=48 and not ended,
                    'receiver_model_seconds': call['seconds'],
                    'receiver_prefill_gpu_seconds': call['prefill_gpu_seconds'],
                    'receiver_decode_gpu_seconds': call['decode_gpu_seconds'],
                    'model_forward_calls': call['model_forward_calls'], 'sender_prefill_seconds': pref,
                    'packet_encode_seconds': encode, 'packet_restore_seconds': restore_time,
                    'from_scratch_stage_sum_seconds': pref+encode+restore_time+call['seconds'],
                    'preexisting_cache_stage_sum_seconds': encode+restore_time+call['seconds'],
                    'timing_has_light_position_event_hooks': True,
                    'standalone_time_is_sum_of_measured_stages_not_an_independent_run': True,
                    'checked_effective_positions': True}
                row['methods'][method] = result
                del output, restored, kwargs
                gc.collect()
            assert cache_hash(torch, cache) == source_hash, 'source cache mutated'
            row['source_cache_unchanged'] = True
            name = hashlib.sha256(case['key'].encode()).hexdigest()[:20]
            write(folder/'items'/f'{phase}_{name}.json', row)
            if phase == 'main':
                rows.append(row)
            print(json.dumps({'phase': phase, 'key': case['key'], 'EOS': {m:r['eos_stopped'] for m,r in row['methods'].items()}}), flush=True)
        del cache, state, text_state, kv_packet, text_packet
        gc.collect()

    with torch.inference_mode():
        warm = next(x for x in data['receivers'] if x['key'] == settings['warmup_key'])
        question(senders[warm['question_id']], [warm], 'warmup')
        for qid,s in senders.items():
            question(s, [x for x in data['receivers'] if x['question_id']==qid], 'main')
    counts = {phase+'_'+kind: sum(e['phase']==phase and e['kind']==kind for e in ledger.events)
        for phase in ['main','warmup'] for kind in ['generate','sender_prefill']}
    assert counts == settings['expected_calls'] and all(e['status']=='ok' for e in ledger.events)
    assert len(rows)==72
    summary = {'status': 'completed', 'unique_questions': 20, 'main_input_rows':72,
        'counts':dict(Counter(e['kind'] for e in ledger.events)), 'phase_counts':counts,
        'generate':219, 'sender_prefill':21,
        'underlying_model_forward_calls': sum(e['model_forward_calls'] for e in ledger.events),
        'EOS_main_answers': sum(v['eos_stopped'] for r in rows for v in r['methods'].values()),
        'truncated_main_answers': sum(v['actually_truncated'] for r in rows for v in r['methods'].values()),
        'network_transfer_measured':False, 'timing_repeats':0,
        'quality_not_evaluated_by_runner':True, 'new_unique_evaluation_questions':0}
    write(folder/'summary.json', summary)
    write(folder/'COMPLETE.json', {'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'counts':counts})
    for handle in ledger.handles:
        handle.remove()
    print(json.dumps(summary),flush=True)


def launch():
    import fcntl
    settings, check = read(CONDITIONS), read(ROOT/'checks/receiver_endpoint_cpu.json')
    assert check['passed'] and check['script_sha256']==sha(__file__)
    assert check['conditions_sha256']==sha(CONDITIONS) and check['inputs_sha256']==sha(INPUTS)
    lock = open(ROOT/'worker.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    candidates = sorted([g for g in gpu_snapshot() if int(g['utilization'])<10],
        key=lambda g:int(g['free_mib']),reverse=True)
    assert candidates and int(candidates[0]['free_mib'])>=settings['capacity_required_mib'], 'no qualified GPU with sufficient capacity; no CPU fallback'
    gpu = candidates[0]
    name = 'receiver_endpoint_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    logdir = ROOT/'logs'
    logdir.mkdir(exist_ok=True)
    log = open(logdir/(name+'.log'),'ab',buffering=0)
    os.set_inheritable(lock.fileno(),True)
    status = {'run_name':name,'pid':None,'gpu':gpu,'conditions_sha256':sha(CONDITIONS)}
    write(ROOT/'receiver_endpoint_worker.json',status)
    worker = subprocess.Popen(['/home/liuxin/.conda/envs/kv-graph/bin/python','-u',str(Path(__file__).resolve()),'--run-name',name],
        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
    status['pid'] = worker.pid
    write(ROOT/'receiver_endpoint_worker.json',status)
    print(json.dumps(status),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--freeze',action='store_true')
    modes.add_argument('--launch',action='store_true')
    modes.add_argument('--run-name')
    args=parser.parse_args()
    if args.freeze:
        freeze()
    elif args.launch:
        launch()
    else:
        try:
            main(args.run_name)
        except Exception:
            failure=ROOT/'runs'/args.run_name/'FAILED.json'
            write(failure,{'error':traceback.format_exc(),'failed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
            raise
