"""Calibration-only prompt/cache diagnostic; leaves the baseline runner unchanged."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

from run import ROOT, gpu_snapshot, load_model, sha, write

METHODS = ['skyline_p105', 'concat_p105', 'concat_bscope_p105', 'kv_p105',
           'skyline_p1', 'concat_p1', 'kv_p1']
PAIRS = [('concat_p1', 'kv_p1'), ('concat_bscope_p105', 'kv_p105'),
         ('concat_p105', 'concat_bscope_p105'), ('skyline_p1', 'concat_p1'),
         ('skyline_p105', 'concat_p105')]


class BScopePenalty:
    """Use HF's exact processor, excluding only the native A-token prefix."""
    def __init__(self, prefix_length, penalty=1.05):
        from transformers import RepetitionPenaltyLogitsProcessor
        self.prefix_length = int(prefix_length)
        self.processor = RepetitionPenaltyLogitsProcessor(penalty)

    def __call__(self, input_ids, scores):
        return self.processor(input_ids[:, self.prefix_length:], scores)


def cpu_check():
    import torch
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM, LogitsProcessorList
    from transformers import RepetitionPenaltyLogitsProcessor
    from models import CVCommunicator
    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, eos_token_id=63, pad_token_id=0, bos_token_id=1)
    config._attn_implementation = 'sdpa'
    model = Qwen2ForCausalLM(config).eval()
    cv = CVCommunicator(model, model, 0, 3, layers_list=list(range(4)), shift_back=False)
    cv.generation_config = copy.deepcopy(model.generation_config)
    cases = []
    with torch.inference_mode():
        for source_len, receiver_len in [(5, 3), (10, 7), (7, 4)]:
            source = torch.randint(2, 62, (1, source_len))
            receiver = torch.randint(2, 62, (1, receiver_len))
            joint = torch.cat([source, receiver], dim=-1)
            for step in range(3):
                suffix = torch.randint(2, 62, (1, step))
                history = torch.cat([joint, suffix], dim=-1)
                b_history = torch.cat([receiver, suffix], dim=-1)
                scores = torch.randn(1, 64)
                actual = BScopePenalty(source_len)(history, scores.clone())
                expected = RepetitionPenaltyLogitsProcessor(1.05)(b_history, scores.clone())
                assert torch.equal(actual, expected), 'penalty history differs'
            for penalty in [1.0, 1.05]:
                native_args = {'repetition_penalty': 1.0}
                if penalty != 1.0:
                    native_args['logits_processor'] = LogitsProcessorList([BScopePenalty(source_len)])
                common = dict(max_new_tokens=6, do_sample=False, pad_token_id=0,
                    return_dict_in_generate=True, output_scores=True, output_logits=True)
                native = model.generate(joint, attention_mask=torch.ones_like(joint),
                    **common, **native_args)
                cache = model(source, use_cache=True).past_key_values
                cached = cv.generate(receiver, attention_mask=torch.ones_like(receiver),
                    out_A_past_key_values=cache, repetition_penalty=penalty, **common)
                assert native.sequences[0, joint.shape[-1]:].tolist() == cached.sequences[0, receiver_len:].tolist()
                raw_max = float((native.logits[0] - cached.logits[0]).abs().max())
                assert torch.allclose(native.logits[0], cached.logits[0], atol=1e-5, rtol=1e-5)
                for a, b in zip(native.scores, cached.scores):
                    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
                cases.append({'source_len': source_len, 'receiver_len': receiver_len,
                    'penalty': penalty, 'generated_tokens_identical': True,
                    'first_raw_logits_max_abs': raw_max})
    result = {'passed': True, 'script_sha256': sha(__file__), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'processor_exact_tests': 9,
        'tiny_cpu_generation_pairs': cases, 'real_7b_claim': False}
    write(ROOT/'checks/alignment_cpu_check.json', result)
    print(json.dumps(result), flush=True)


def launch():
    import fcntl
    check = json.loads((ROOT/'checks/alignment_cpu_check.json').read_text())
    assert check['passed'] and check['script_sha256'] == sha(__file__)
    lock = open(ROOT/'worker.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Capacity is checked independently of the user's <10% utilization rule.
    while True:
        candidates = [x for x in gpu_snapshot() if int(x['utilization']) < 10]
        candidates.sort(key=lambda x: int(x['free_mib']), reverse=True)
        if candidates and int(candidates[0]['free_mib']) >= 18000:
            gpu = candidates[0]
            break
        print(json.dumps({'waiting_for_7b_capacity': candidates}), flush=True)
        time.sleep(30)
    run_name = 'alignment7b_' + time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false')
    python = '/home/liuxin/.conda/envs/kv-graph/bin/python'
    (ROOT/'logs').mkdir(exist_ok=True)
    log = open(ROOT/'logs'/f'{run_name}.log', 'ab', buffering=0)
    os.set_inheritable(lock.fileno(), True)
    worker = subprocess.Popen([python, '-u', str(Path(__file__).resolve()), '--run-name', run_name],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        pass_fds=(lock.fileno(),))
    status = {'run_name': run_name, 'pid': worker.pid, 'gpu': gpu, 'python': python}
    write(ROOT/'alignment_worker.json', status)
    print(json.dumps(status), flush=True)


class Audit:
    def __init__(self, torch, model, run_dir):
        self.torch, self.run_dir = torch, run_dir
        self.events = []
        self.tag = None
        self.records = []
        self.handles = [model.register_forward_pre_hook(self.model_hook, with_kwargs=True),
            model.model.layers[0].self_attn.register_forward_pre_hook(self.attention_hook, with_kwargs=True),
            model.model.rotary_emb.register_forward_pre_hook(self.rotary_hook, with_kwargs=True)]

    def sync(self):
        self.torch.cuda.synchronize()

    def tensor_ids(self, value):
        return None if value is None else value.detach().cpu().tolist()

    def model_hook(self, module, args, kwargs):
        if self.tag is None:
            return
        ids = kwargs.get('input_ids', args[0] if args else None)
        cache = kwargs.get('past_key_values')
        mask = kwargs.get('attention_mask')
        self.records.append({'level': 'native_model_forward_input', 'tag': self.tag,
            'input_ids': self.tensor_ids(ids), 'past_length': 0 if cache is None else int(cache.get_seq_length()),
            'position_ids': self.tensor_ids(kwargs.get('position_ids')),
            'cache_position': self.tensor_ids(kwargs.get('cache_position')),
            'attention_mask_shape': None if mask is None else list(mask.shape)})

    def rotary_hook(self, module, args, kwargs):
        if self.tag is None:
            return
        positions = kwargs.get('position_ids', args[1] if len(args) > 1 else None)
        self.records.append({'level': 'effective_rotary_position_ids', 'tag': self.tag,
            'position_ids': self.tensor_ids(positions)})

    def attention_hook(self, module, args, kwargs):
        if self.tag is None:
            return
        mask = kwargs.get('attention_mask')
        cache = kwargs.get('past_key_value', kwargs.get('past_key_values'))
        self.records.append({'level': 'effective_first_layer_attention', 'tag': self.tag,
            'cache_position': self.tensor_ids(kwargs.get('cache_position')),
            'position_ids': self.tensor_ids(kwargs.get('position_ids')),
            'past_length': None if cache is None else int(cache.get_seq_length()),
            'attention_mask_shape': None if mask is None else list(mask.shape)})

    def call(self, kind, tag, function):
        self.tag, self.records = tag, []
        self.sync()
        start = time.perf_counter()
        try:
            value = function()
            self.sync()
        except Exception as exc:
            self.event({'kind': kind, 'tag': tag, 'status': 'error', 'error': repr(exc)})
            raise
        finally:
            self.tag = None
        record = {'kind': kind, 'tag': tag, 'status': 'ok', 'diagnostic_instrumented_seconds': time.perf_counter()-start,
            'backend_forward_trace': self.records}
        self.event({k: v for k, v in record.items() if k != 'backend_forward_trace'})
        return value, record

    def event(self, data):
        self.events.append(data)
        with (self.run_dir/'calls.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(data)+'\n')


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def vector_diff(torch, left, right):
    left, right = left.float(), right.float()
    finite = torch.isfinite(left) & torch.isfinite(right)
    delta = (left[finite]-right[finite]).abs()
    assert delta.numel(), 'no finite logits to compare'
    denominator = float(left[finite].abs().max())
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
        'relative_linf': float(delta.max())/max(denominator, 1e-12),
        'nonfinite_values_match': bool(torch.equal(left[~finite], right[~finite])),
        'top1_agreement': float((left.argmax(-1) == right.argmax(-1)).float().mean())}


def top_info(torch, values, tokenizer):
    values = values.float()[0]
    scores, indices = torch.topk(values, 5)
    ids = indices.detach().cpu().tolist()
    return {'ids': ids, 'tokens': [tokenizer.decode([x], skip_special_tokens=False) for x in ids],
        'values': scores.detach().cpu().tolist(), 'top1_margin': float(scores[0]-scores[1])}


def compare_outputs(torch, left, right):
    a, b = left['ids'], right['ids']
    index = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if index is None and len(a) != len(b):
        index = min(len(a), len(b))
    # Only compare histories before and at their first divergence.
    limit = min(len(a), len(b)) if index is None else min(index+1, len(a), len(b))
    steps = [{'step': i, 'raw': vector_diff(torch, left['output'].logits[i], right['output'].logits[i]),
        'processed': vector_diff(torch, left['output'].scores[i], right['output'].scores[i])} for i in range(limit)]
    return {'generated_tokens_identical': a == b, 'first_different_index': index,
        'left_tokens': len(a), 'right_tokens': len(b), 'same_history_steps': steps,
        'max_raw_error_on_same_history': max([s['raw']['max_abs'] for s in steps], default=0),
        'max_processed_error_on_same_history': max([s['processed']['max_abs'] for s in steps], default=0)}


def main(run_name):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, LogitsProcessorList
    from eval import SkylineEvaluator, CommunicationEvaluator
    from models import CVCommunicator
    from score_repro_outputs import hotpot_token_f1, normalize

    run_dir = ROOT/'runs'/run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT/'config.json').read_text())
    baseline = ROOT/'runs/main7b_20261001_045806'
    base_manifest = json.loads((baseline/'manifest.json').read_text())
    assert (baseline/'COMPLETE.json').exists()
    for name, field in [('run.py', 'run_sha256'), ('config.json', 'config_sha256'), ('assets.json', 'assets_sha256')]:
        assert sha(ROOT/name) == base_manifest[field], f'baseline {name} changed'
    assets = json.loads((ROOT/'assets.json').read_text())
    assert sha(ROOT/'data/calibration.json') == assets['split_files']['calibration']['sha256']
    calibration = json.loads((ROOT/'data/calibration.json').read_text())
    chosen, warmup_item = calibration[:10], calibration[10]
    assert [x['id'] for x in calibration] == assets['split_files']['calibration']['ids']
    assert len({x['id'] for x in chosen}) == 10
    core_hashes = {}
    for name in ['eval.py', 'models.py', 'model_attn.py', 'layer_importance.py']:
        core_hashes[name] = sha(ROOT/'work/KVComm'/name)
        assert core_hashes[name] == assets['source_files'][name]['work_sha256'], f'official work {name} changed'
        assert sha(ROOT/'upstream/KVComm'/name) == assets['source_files'][name]['pristine_sha256'], f'official snapshot {name} changed'
    assert assets['commit'] == cfg['source_commit']
    verified = json.loads((ROOT/'checks/model7b_manifest.json').read_text())
    assert verified['revision'] == cfg['model_7b_revision']
    assert all((Path(cfg['model_7b_path'])/name).stat().st_size == spec['bytes'] for name, spec in verified['files'].items())
    small_hashes = {name: sha(Path(cfg['model_7b_path'])/name) for name, spec in verified['files'].items()
        if not spec['official_lfs_verified']}
    assert all(value == verified['files'][name]['sha256'] for name, value in small_hashes.items())
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_7b_path'], local_files_only=True)
    manifest = {'purpose': 'calibration-only prompt, penalty-scope and full-cache diagnostic',
        'script_sha256': sha(__file__), 'calibration_sha256': sha(ROOT/'data/calibration.json'),
        'original_baseline_manifest_sha256': sha(baseline/'manifest.json'), 'assets_sha256': sha(ROOT/'assets.json'),
        'question_ids': [x['id'] for x in chosen], 'question_hashes': [x['question_hash'] for x in chosen],
        'warmup_id': warmup_item['id'], 'config': cfg, 'methods': METHODS, 'pairs': PAIRS,
        'official_core_hashes': core_hashes, 'model_manifest_sha256': sha(ROOT/'checks/model7b_manifest.json'),
        'fresh_model_config_and_tokenizer_sha256': small_hashes,
        'torch': torch.__version__, 'transformers': transformers.__version__, 'python': sys.version,
        'gpu_at_launch': gpu_snapshot(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'tf32': False, 'numerical_policy': 'report BF16/SDPA errors and token matches separately; no adaptive tolerance',
        'expected_counts': {'generate': 72, 'sender_prefill': 31, 'raw_concat_forward': 10, 'raw_cv_receiver_forward': 10},
        'evaluation_and_confirmation_used': False}
    write(run_dir/'manifest.json', manifest)
    model, load_seconds = load_model(torch, cfg['model_7b_path'])
    assert model.generation_config.repetition_penalty == 1.05
    assert model.generation_config.eos_token_id == base_manifest['generation_config']['eos_token_id']
    holder = SimpleNamespace(name='hotpotqa', max_tokens=48, truncate_input=False, multiple_answers=False)
    sky = SkylineEvaluator(holder, tokenizer, False, model.config.max_position_embeddings)
    comm = CommunicationEvaluator(holder, tokenizer, False, model.config.max_position_embeddings)
    cv = CVCommunicator(model, model, 0, model.config.num_hidden_layers-1,
        layers_list=list(range(model.config.num_hidden_layers)), shift_back=False)
    cv.generation_config = copy.deepcopy(model.generation_config)
    assert all(k == v for k, v in cv.layer_map.items())
    common = dict(sky.generate_args, eos_token_id=copy.deepcopy(model.generation_config.eos_token_id),
        return_dict_in_generate=True, output_scores=True, output_logits=True)
    manifest.update(model_load_seconds=load_seconds, native_generation_config=model.generation_config.to_dict(),
        generation_kwargs=common, cv_supports_cache_class=cv._supports_cache_class,
        full_layers=cv.layers_list, identity_layer_map=cv.layer_map)
    write(run_dir/'manifest.json', manifest)
    audit = Audit(torch, model, run_dir)

    def inputs(item):
        a, b = comm.prepare_input_ids(item, model, model)
        s = sky.prepare_input_ids(item, model)
        joint = torch.cat([a, b], dim=-1)
        assert joint.shape[-1]+48 <= model.config.max_position_embeddings
        return a, b, s, joint

    def generate(item, name, phase):
        a, b, s, joint = inputs(item)
        penalty = 1.0 if name.endswith('_p1') else 1.05
        args = dict(common, repetition_penalty=penalty)
        if name.startswith('kv_'):
            out_a, pref = audit.call('sender_prefill', f'{phase}:{item["id"]}:{name}:A',
                lambda: model(input_ids=a, use_cache=True, return_dict=True))
            cache = out_a.past_key_values
            lengths = [int(k.shape[-2]) for k in cache.key_cache]
            assert lengths == [int(a.shape[-1])]*model.config.num_hidden_layers
            output, record = audit.call('generate', f'{phase}:{item["id"]}:{name}',
                lambda: cv.generate(b, attention_mask=torch.ones_like(b), out_A_past_key_values=cache, **args))
            prefix = b.shape[-1]
            record['sender_prefill'] = pref
            record['cache_lengths_before_generation'] = lengths
        else:
            ids = s if name.startswith('skyline') else joint
            if name == 'concat_bscope_p105':
                args.update(repetition_penalty=1.0,
                    logits_processor=LogitsProcessorList([BScopePenalty(a.shape[-1])]))
            output, record = audit.call('generate', f'{phase}:{item["id"]}:{name}',
                lambda: model.generate(ids, attention_mask=torch.ones_like(ids), **args))
            prefix = ids.shape[-1]
        tail = output.sequences[0, prefix:].detach().cpu().tolist()
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        response = tokenizer.decode(tail, skip_special_tokens=True)
        record.update(method=name, phase=phase, output_ids=tail, response=response,
            generated_tokens=len(tail), eos_stopped=bool(tail and tail[-1] in eos),
            actually_truncated=len(tail) >= 48 and not (tail and tail[-1] in eos),
            exact_match=normalize(response) == normalize(item['answer']),
            standard_token_f1=hotpot_token_f1(response, item['answer']),
            per_step=[{'step': i, 'chosen_id': tail[i], 'raw': top_info(torch, raw, tokenizer),
                'processed': top_info(torch, score, tokenizer)}
                for i, (raw, score) in enumerate(zip(output.logits, output.scores))])
        return {'ids': tail, 'output': output, 'record': record}

    all_rows = []
    with torch.inference_mode():
        for name in ['concat_p1', 'kv_p1']:
            value = generate(warmup_item, name, 'warmup')
            write(run_dir/'warmup'/f'{name}.json', value['record'])
            del value
        for item in chosen:
            a, b, s, joint = inputs(item)
            row = {'id': item['id'], 'question_hash': item['question_hash'], 'gold': item['answer'],
                'question': item['prompt_B'], 'material': item['prompt_A'], 'inputs': {}, 'methods': {}}
            for name, ids in [('A', a), ('B', b), ('Skyline', s), ('AB', joint)]:
                row['inputs'][name] = {'ids': ids.detach().cpu().tolist(), 'length': int(ids.shape[-1]),
                    'sha256': tensor_hash(ids), 'decoded_with_special_tokens': tokenizer.decode(ids[0], skip_special_tokens=False)}
            raw_joint, jt = audit.call('raw_concat_forward', f'diagnostic:{item["id"]}:raw_concat',
                lambda: model(input_ids=joint, use_cache=False, return_dict=True))
            raw_a, at = audit.call('sender_prefill', f'diagnostic:{item["id"]}:raw_A',
                lambda: model(input_ids=a, use_cache=True, return_dict=True))
            raw_b, bt = audit.call('raw_cv_receiver_forward', f'diagnostic:{item["id"]}:raw_CV_B',
                lambda: cv(input_ids=b, attention_mask=torch.ones_like(b),
                    out_A_past_key_values=raw_a.past_key_values, use_cache=True, return_dict=True))
            row['raw_forward_comparison'] = {'all_B_positions': vector_diff(torch,
                raw_joint.logits[:, -b.shape[-1]:, :], raw_b.logits), 'traces': [jt, at, bt]}
            del raw_joint, raw_a, raw_b
            outputs = {}
            order = METHODS.copy()
            random.Random(int(hashlib.sha256(item['id'].encode()).hexdigest()[:16], 16)).shuffle(order)
            row['generation_order'] = order
            for name in order:
                outputs[name] = generate(item, name, 'diagnostic')
                row['methods'][name] = outputs[name]['record']
            row['comparisons'] = {f'{left}__vs__{right}': compare_outputs(torch, outputs[left], outputs[right])
                for left, right in PAIRS}
            vectors = {}
            for name, value in outputs.items():
                vectors[name+'__raw'] = value['output'].logits[0][0].detach().float().cpu().numpy()
                vectors[name+'__processed'] = value['output'].scores[0][0].detach().float().cpu().numpy()
            for left, right in PAIRS:
                diff = row['comparisons'][f'{left}__vs__{right}']['first_different_index']
                if diff is not None and diff < min(len(outputs[left]['ids']), len(outputs[right]['ids'])):
                    for name in [left, right]:
                        vectors[f'{left}__vs__{right}__divergence__{name}__raw'] = outputs[name]['output'].logits[diff][0].detach().float().cpu().numpy()
                        vectors[f'{left}__vs__{right}__divergence__{name}__processed'] = outputs[name]['output'].scores[diff][0].detach().float().cpu().numpy()
            vector_path = run_dir/'vectors'/f'{item["id"]}.npz'
            vector_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(vector_path, **vectors)
            row['first_step_and_divergence_vectors'] = {'file': str(vector_path.relative_to(run_dir)), 'sha256': sha(vector_path)}
            write(run_dir/'items'/f'{item["id"]}.json', row)
            all_rows.append(row)
            print(json.dumps({'id': item['id'], 'pairs': {key: val['generated_tokens_identical']
                for key, val in row['comparisons'].items()}}), flush=True)
            del outputs, vectors
            gc.collect()
    counts = {kind: sum(x['kind'] == kind for x in audit.events) for kind in manifest['expected_counts']}
    assert counts == manifest['expected_counts'], (counts, manifest['expected_counts'])
    assert len(all_rows) == 10 and all(x['status'] == 'ok' for x in audit.events)
    summary = {'run_name': run_name, 'questions': 10, 'diagnostic_answers': 70, 'warmup_answers': 2,
        'counts': counts, 'status_ok': True, 'score_is_not_a_new_benchmark_result': True, 'pairs': {}}
    for left, right in PAIRS:
        key = f'{left}__vs__{right}'
        pairs = [row['comparisons'][key] for row in all_rows]
        summary['pairs'][key] = {'exact_token_matches': sum(x['generated_tokens_identical'] for x in pairs),
            'questions': 10, 'max_raw_error_on_same_history': max(x['max_raw_error_on_same_history'] for x in pairs),
            'max_processed_error_on_same_history': max(x['max_processed_error_on_same_history'] for x in pairs),
            'different_ids': [row['id'] for row in all_rows if not row['comparisons'][key]['generated_tokens_identical']]}
    summary['raw_forward'] = {'worst_absolute_error': max(row['raw_forward_comparison']['all_B_positions']['max_abs'] for row in all_rows),
        'min_top1_agreement': min(row['raw_forward_comparison']['all_B_positions']['top1_agreement'] for row in all_rows)}
    write(run_dir/'summary.json', summary)
    write(run_dir/'COMPLETE.json', {'phase': 'alignment7b', 'utc_completed': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'counts': counts})
    print(json.dumps(summary), flush=True)
    for handle in audit.handles:
        handle.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--cpu-check', action='store_true')
    modes.add_argument('--launch', action='store_true')
    modes.add_argument('--run-name')
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check()
    elif args.launch:
        launch()
    else:
        main(args.run_name)
