"""Locate joint-recomputation versus cache-reuse differences on one frozen case."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

from run import ROOT, sha, write, gpu_snapshot, load_model
from chain_core import ROLES, ChainAudit, ChainEngine, prepare_prompt, edge_payload
from alignment_probe import compare_outputs, top_info

FILES = ['cache_path_probe.py', 'native_cache_control.py', 'chain_core.py']
REFERENCE = 'chain3_20261001_122157'


def freeze():
    previous = json.loads((ROOT/'runs'/REFERENCE/'manifest.json').read_text(encoding='utf-8'))
    conditions = previous['settings']
    assert sha(ROOT/'chain_core.py') == conditions['source_hashes']['chain_core.py']
    cases = json.loads((ROOT/'data/three_node_cases.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'data/three_node_cases.json') == conditions['cases_sha256']
    settings = {'purpose': 'same sender-cache native receiver/CV control; not an accuracy experiment',
        'reference_run': REFERENCE, 'reference_manifest_sha256': sha(ROOT/'runs'/REFERENCE/'manifest.json'),
        'case_id': cases[0]['item']['id'], 'cases_sha256': conditions['cases_sha256'],
        'model_path': conditions['model_path'], 'model_revision': conditions['model_revision'],
        'roles': ROLES, 'budgets': conditions['budgets'], 'dtype': 'bfloat16', 'attention': 'sdpa',
        'greedy': True, 'repetition_penalty': 1.0, 'shift_back': False, 'all_layers': True,
        'generate_calls': 4, 'cache_tail_forward_max': 1, 'warmup_calls': 0,
        'order': ['A_native_with_EOS_completion', 'B_joint_recomputation', 'B_native_same_sender_cache', 'B_CV_same_sender_cache'],
        'selection': 'first frozen calibration case; audit the recorded divergence, no answer-based selection',
        'source_hashes': {name: sha(ROOT/name) for name in FILES},
        'evaluation_and_confirmation_used': False, 'network_transfer_measured': False}
    write(ROOT/'cache_path_conditions.json', settings)
    print(json.dumps({'case_id': settings['case_id'], 'generate_calls': 4, 'cache_tail_forward_max': 1}))


def launch():
    import fcntl
    cfg = json.loads((ROOT/'cache_path_conditions.json').read_text(encoding='utf-8'))
    checked = json.loads((ROOT/'checks/native_cache_control_cpu.json').read_text(encoding='utf-8'))
    assert checked['passed'] and checked['script_sha256'] == sha(ROOT/'native_cache_control.py')
    assert checked['chain_core_sha256'] == sha(ROOT/'chain_core.py')
    for name, digest in cfg['source_hashes'].items():
        assert sha(ROOT/name) == digest
    lock = open(ROOT/'worker.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    candidates = [x for x in gpu_snapshot() if int(x['utilization']) < 10 and int(x['free_mib']) >= 18000]
    assert candidates, 'no sufficiently idle GPU; do not start'
    gpu = sorted(candidates, key=lambda x: int(x['free_mib']), reverse=True)[0]
    name = 'cachepath_'+time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    python = '/home/liuxin/.conda/envs/kv-graph/bin/python'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    log = open(ROOT/'logs'/f'{name}.log', 'ab', buffering=0)
    os.set_inheritable(lock.fileno(), True)
    worker = subprocess.Popen([python, '-u', str(Path(__file__).resolve()), '--run-name', name],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
    state = {'pid': worker.pid, 'run_name': name, 'gpu': gpu, 'python': python}
    write(ROOT/'cache_path_worker.json', state)
    print(json.dumps(state), flush=True)


def main(name):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer
    from models import CVCommunicator
    from native_cache_control import native_cached_node
    cfg = json.loads((ROOT/'cache_path_conditions.json').read_text(encoding='utf-8'))
    assets = json.loads((ROOT/'assets.json').read_text(encoding='utf-8'))
    for source, digest in cfg['source_hashes'].items():
        assert sha(ROOT/source) == digest
    assert sha(ROOT/'runs'/REFERENCE/'manifest.json') == cfg['reference_manifest_sha256']
    assert sha(ROOT/'data/three_node_cases.json') == cfg['cases_sha256']
    case = json.loads((ROOT/'data/three_node_cases.json').read_text(encoding='utf-8'))[0]
    assert cfg['case_id'] == case['item']['id']
    assert cfg['roles'] == ROLES
    for source in ['models.py', 'model_attn.py']:
        assert sha(ROOT/'work/KVComm'/source) == assets['source_files'][source]['work_sha256']
    verified = json.loads((ROOT/'checks/model7b_manifest.json').read_text(encoding='utf-8'))
    assert verified['revision'] == cfg['model_revision']
    for file, spec in verified['files'].items():
        path = Path(cfg['model_path'])/file
        assert path.stat().st_size == spec['bytes']
        if not spec['official_lfs_verified']:
            assert sha(path) == spec['sha256']
    folder = ROOT/'runs'/name
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'code').mkdir()
    for file in FILES+['cache_path_conditions.json']:
        (folder/'code'/file).write_bytes((ROOT/file).read_bytes())
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = {'settings': cfg, 'settings_sha256': sha(ROOT/'cache_path_conditions.json'),
        'cpu_check_sha256': sha(ROOT/'checks/native_cache_control_cpu.json'),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu_at_launch': gpu_snapshot()}
    write(folder/'manifest.json', manifest)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    model, seconds = load_model(torch, cfg['model_path'])
    cv = CVCommunicator(model, model, 0, model.config.num_hidden_layers-1,
        layers_list=list(range(model.config.num_hidden_layers)), shift_back=False)
    cv.generation_config = copy.deepcopy(model.generation_config)
    audit = ChainAudit(torch, model, folder)
    engine = ChainEngine(torch, model, cv, tokenizer, cfg['budgets'], audit)
    item, material = case['item'], case['materials']
    with torch.inference_mode():
        a = engine.node(item, 'cachepath', 'A', None, prepare_prompt(tokenizer, model, item, 'A', material), outgoing=True)
        previous = json.loads((ROOT/'runs'/REFERENCE/'items'/f'main_{item["id"]}.json').read_text(encoding='utf-8'))
        assert a['ids'] == previous['reference']['A']['output_ids'], 'sender did not reproduce; preserve failure'
        prompt = prepare_prompt(tokenizer, model, item, 'B', material)
        outputs = {'joint': engine.node(item, 'cachepath:joint', 'B', a['state'], prompt, outgoing=False),
            'native_cached': native_cached_node(torch, model, tokenizer, audit, a['state'], prompt,
                cfg['budgets']['B'], item['id']+':cachepath:native_cached:B'),
            'cv': engine.node(item, 'cachepath:cv', 'B', a['state'], prompt, use_kv=True, outgoing=False)}
        assert len({x['record']['input_history_sha256'] for x in outputs.values()}) == 1
        comparisons, vectors = {}, {}
        for left, right in [('joint', 'native_cached'), ('native_cached', 'cv'), ('joint', 'cv')]:
            key = left+'_vs_'+right
            comparisons[key] = compare_outputs(torch, outputs[left], outputs[right])
            diff = comparisons[key]['first_different_index']
            if diff is not None and diff < min(len(outputs[left]['ids']), len(outputs[right]['ids'])):
                comparisons[key]['divergence_top5'] = {label: top_info(torch, outputs[label]['output'].logits[diff], tokenizer) for label in [left, right]}
                for label in [left, right]:
                    vectors[key+'_'+label+'_raw'] = outputs[label]['output'].logits[diff][0].float().cpu().numpy()
        for label, value in outputs.items():
            vectors[label+'_first_raw'] = value['output'].logits[0][0].float().cpu().numpy()
        (folder/'vectors').mkdir()
        np.savez_compressed(folder/'vectors/steps.npz', **vectors)
        row = {'id': item['id'], 'sender': a['record'], 'receivers': {k: x['record'] for k, x in outputs.items()},
            'same_sender_cache': True, 'sender_reproduces_previous_output': True,
            'sender_kv_payload': edge_payload(torch, tokenizer, a['state'], 'kv', cfg['model_revision']),
            'sender_text_payload': edge_payload(torch, tokenizer, a['state'], 'text_tokens', cfg['model_revision']),
            'comparisons': comparisons, 'vectors_sha256': sha(folder/'vectors/steps.npz')}
        write(folder/'items/case.json', row)
    counts = {kind: sum(x['kind'] == kind for x in audit.events) for kind in ['generate', 'cache_tail_forward']}
    assert counts['generate'] == 4 and counts['cache_tail_forward'] <= 1
    assert all(x['status'] == 'ok' for x in audit.events)
    summary = {'run_name': name, 'counts': counts, 'same_sender_cache': True,
        'sender_reproduces_previous_output': True,
        'native_cached_vs_cv': comparisons['native_cached_vs_cv'],
        'joint_vs_native_cached': comparisons['joint_vs_native_cached'],
        'receiver_responses': {label: x['record']['response'] for label, x in outputs.items()},
        'all_receivers_EOS': all(x['record']['eos_stopped'] for x in outputs.values()),
        'any_truncated': any(x['record']['actually_truncated'] for x in outputs.values()),
        'not_a_benchmark': True, 'network_transfer_measured': False}
    write(folder/'summary.json', summary)
    write(folder/'COMPLETE.json', {'utc_completed': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'counts': counts})
    print(json.dumps({'run_name': name, 'counts': counts,
        'native_cached_vs_cv_matches': comparisons['native_cached_vs_cv']['generated_tokens_identical'],
        'joint_vs_native_cached_matches': comparisons['joint_vs_native_cached']['generated_tokens_identical']}), flush=True)
    for handle in audit.handles:
        handle.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--run-name')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.launch:
        launch()
    else:
        main(args.run_name)
