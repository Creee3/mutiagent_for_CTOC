"""Independent actual CV relay, with native cached controls on the same prefixes."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from run import ROOT, sha, write, gpu_snapshot, load_model
from chain_core import ROLES, split_materials

REFERENCE = 'chain3_20261001_122157'
FILES = ['free_chain_probe.py', 'free_chain_core.py', 'chain_core.py',
    'native_cache_control.py', 'run.py', 'alignment_probe.py', 'score_repro_outputs.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze():
    previous = read(ROOT/'runs'/REFERENCE/'manifest.json')
    cfg = previous['settings']
    assert sha(ROOT/'chain_core.py') == cfg['source_hashes']['chain_core.py']
    assert cfg['roles'] == ROLES
    assert sha(ROOT/'data/three_node_cases.json') == cfg['cases_sha256']
    cases = read(ROOT/'data/three_node_cases.json')
    refs = {case['item']['id']: sha(ROOT/'runs'/REFERENCE/'items'/
        f'{"main" if index < 3 else "warmup"}_{case["item"]["id"]}.json') for index, case in enumerate(cases)}
    settings = {'purpose': 'actual three-node relay with same-cache native implementation controls',
        'reference_run': REFERENCE, 'reference_manifest_sha256': sha(ROOT/'runs'/REFERENCE/'manifest.json'),
        'reference_summary_sha256': sha(ROOT/'runs'/REFERENCE/'summary.json'),
        'old_reference_gate_passed': read(ROOT/'runs'/REFERENCE/'summary.json')['gate_passed'],
        'reference_rows_sha256': refs, 'reference_gate_result_preserved': previous['settings']['gate'],
        'main_ids': cfg['main_ids'], 'warmup_id': cfg['warmup_id'], 'roles': ROLES,
        'cases_sha256': cfg['cases_sha256'], 'calibration_sha256': cfg['calibration_sha256'],
        'model_path': cfg['model_path'], 'model_revision': cfg['model_revision'],
        'material_policy': cfg['material_policy'], 'budgets': cfg['budgets'],
        'dtype': 'bfloat16', 'attention': 'sdpa', 'greedy': True, 'repetition_penalty': 1.0,
        'shift_back': False, 'all_layers': True, 'generate_calls': 20, 'cache_tail_forward_max': 8,
        'warmup_generate': 5, 'main_generate': 15, 'controls_do_not_enter_main_chain': True,
        'stop_conditions': 'structural/history/cache-position/source-mutation assertions stop execution',
        'implementation_control': 'native cached B/C reuse actual incoming CV-chain sender cache; exact output match and no truncation reported separately',
        'reference_comparison': 'reuse frozen joint reference outputs; history and output differences reported, not a timing comparison',
        'source_hashes': {name: sha(ROOT/name) for name in FILES},
        'assets_sha256': sha(ROOT/'assets.json'), 'master_config_sha256': sha(ROOT/'config.json'),
        'evaluation_and_confirmation_used': False, 'network_transfer_measured': False}
    write(ROOT/'free_chain_conditions.json', settings)
    print(json.dumps({'generate_calls': 20, 'cache_tail_forward_max': 8, 'main_ids': cfg['main_ids']}))


def validate(cfg):
    for name, digest in cfg['source_hashes'].items():
        assert sha(ROOT/name) == digest, name
    assert sha(ROOT/'assets.json') == cfg['assets_sha256']
    assert sha(ROOT/'config.json') == cfg['master_config_sha256']
    assert sha(ROOT/'data/three_node_cases.json') == cfg['cases_sha256']
    assert sha(ROOT/'data/calibration.json') == cfg['calibration_sha256']
    assert sha(ROOT/'runs'/REFERENCE/'manifest.json') == cfg['reference_manifest_sha256']
    assert sha(ROOT/'runs'/REFERENCE/'summary.json') == cfg['reference_summary_sha256']
    cases = read(ROOT/'data/three_node_cases.json')
    assert [c['item']['id'] for c in cases[:3]] == cfg['main_ids'] and cases[3]['item']['id'] == cfg['warmup_id']
    assert cfg['roles'] == ROLES
    for index, case in enumerate(cases):
        assert split_materials(case['item']) == case['materials']
        phase = 'main' if index < 3 else 'warmup'
        assert sha(ROOT/'runs'/REFERENCE/'items'/f'{phase}_{case["item"]["id"]}.json') == cfg['reference_rows_sha256'][case['item']['id']]
    return cases


def launch():
    import fcntl
    cfg = read(ROOT/'free_chain_conditions.json')
    validate(cfg)
    checked = read(ROOT/'checks/free_chain_cpu.json')
    assert checked['passed']
    assert checked['script_sha256'] == sha(ROOT/'free_chain_core.py')
    assert checked['chain_core_sha256'] == sha(ROOT/'chain_core.py')
    assert checked['native_helper_sha256'] == sha(ROOT/'native_cache_control.py')
    lock = open(ROOT/'worker.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    candidates = [x for x in gpu_snapshot() if int(x['utilization']) < 10 and int(x['free_mib']) >= 18000]
    assert candidates, 'no sufficiently idle GPU; no experiment started'
    gpu = sorted(candidates, key=lambda x: int(x['free_mib']), reverse=True)[0]
    name = 'freechain_'+time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    python = '/home/liuxin/.conda/envs/kv-graph/bin/python'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    log = open(ROOT/'logs'/f'{name}.log', 'ab', buffering=0)
    os.set_inheritable(lock.fileno(), True)
    worker = subprocess.Popen([python, '-u', str(Path(__file__).resolve()), '--run-name', name],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
    state = {'pid': worker.pid, 'run_name': name, 'gpu': gpu, 'python': python}
    write(ROOT/'free_chain_worker.json', state)
    print(json.dumps(state), flush=True)


def main(name):
    import gc
    import torch
    import transformers
    from transformers import AutoTokenizer
    from models import CVCommunicator
    from chain_core import ChainAudit, ChainEngine
    from free_chain_core import run_free_case
    from score_repro_outputs import normalize, hotpot_token_f1
    cfg = read(ROOT/'free_chain_conditions.json')
    cases = validate(cfg)
    assets = read(ROOT/'assets.json')
    for source in ['eval.py', 'models.py', 'model_attn.py']:
        assert sha(ROOT/'work/KVComm'/source) == assets['source_files'][source]['work_sha256']
    verified = read(ROOT/'checks/model7b_manifest.json')
    assert verified['revision'] == cfg['model_revision']
    for file, spec in verified['files'].items():
        path = Path(cfg['model_path'])/file
        assert path.stat().st_size == spec['bytes']
        if not spec['official_lfs_verified']:
            assert sha(path) == spec['sha256']
    folder = ROOT/'runs'/name
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'code').mkdir()
    for file in FILES+['free_chain_conditions.json']:
        (folder/'code'/file).write_bytes((ROOT/file).read_bytes())
    manifest = {'settings': cfg, 'settings_sha256': sha(ROOT/'free_chain_conditions.json'),
        'cpu_check_sha256': sha(ROOT/'checks/free_chain_cpu.json'), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_at_launch': gpu_snapshot(), 'shared_frozen_weights': True,
        'not_a_timing_benchmark': True}
    write(folder/'manifest.json', manifest)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    model, load_seconds = load_model(torch, cfg['model_path'])
    cv = CVCommunicator(model, model, 0, model.config.num_hidden_layers-1,
        layers_list=list(range(model.config.num_hidden_layers)), shift_back=False)
    cv.generation_config = copy.deepcopy(model.generation_config)
    assert cv.layer_map == {i: i for i in range(model.config.num_hidden_layers)}
    manifest.update(model_load_seconds=load_seconds, generation_config=model.generation_config.to_dict(),
        model_config=model.config.to_dict(), full_layers=cv.layers_list)
    write(folder/'manifest.json', manifest)
    audit = ChainAudit(torch, model, folder)
    engine = ChainEngine(torch, model, cv, tokenizer, cfg['budgets'], audit)
    rows = []
    with torch.inference_mode():
        for index in [3, 0, 1, 2]:
            case = cases[index]
            item = case['item']
            phase = 'warmup' if index == 3 else 'main'
            row = run_free_case(torch, model, tokenizer, engine, audit, item,
                case['materials'], phase, cfg['model_revision'])
            reference = read(ROOT/'runs'/REFERENCE/'items'/f'{phase}_{item["id"]}.json')
            for node in ['A', 'B', 'C']:
                assert row['nodes'][node]['local_prompt_ids'] == reference['reference'][node]['local_prompt_ids']
            row.update(id=item['id'], phase=phase, question=item['prompt_B'], question_hash=item['question_hash'],
                gold=item['answer'], reference_row_sha256=cfg['reference_rows_sha256'][item['id']],
                vs_joint_reference={n: {'same_input_history': row['nodes'][n]['input_history_sha256'] == reference['reference'][n]['input_history_sha256'],
                    'same_output_tokens': row['nodes'][n]['output_ids'] == reference['reference'][n]['output_ids'],
                    'reference_response': reference['reference'][n]['response']} for n in ['A', 'B', 'C']})
            answer = row['nodes']['C']['response']
            row['final_scores'] = {'free_kv': {'exact_match': normalize(answer) == normalize(item['answer']),
                'standard_token_f1': hotpot_token_f1(answer, item['answer'])},
                'joint_reference': reference['reference_final_score']}
            write(folder/'items'/f'{phase}_{item["id"]}.json', row)
            if phase == 'main':
                rows.append(row)
            print(json.dumps({'id': item['id'], 'phase': phase,
                'main_chain_EOS': {n: row['nodes'][n]['eos_stopped'] for n in ['A', 'B', 'C']},
                'controls_match': {n: row['comparisons'][n]['generated_tokens_identical'] for n in ['B', 'C']}}), flush=True)
            gc.collect()
    counts = dict(Counter(x['kind'] for x in audit.events))
    assert counts['generate'] == 20 and counts.get('cache_tail_forward', 0) <= 8
    assert all(e['status'] == 'ok' for e in audit.events)
    control_matches = {n: sum(r['comparisons'][n]['generated_tokens_identical'] for r in rows) for n in ['B', 'C']}
    free_EOS = all(r['nodes'][n]['eos_stopped'] for r in rows for n in ['A', 'B', 'C'])
    no_truncation = not any(x['actually_truncated'] for r in rows for group in ['nodes', 'controls'] for x in r[group].values())
    implementation_ok = all(v == 3 for v in control_matches.values()) and no_truncation
    summary = {'run_name': name, 'main_questions': 3, 'warmup_questions': 1, 'counts': counts,
        'structural_assertions_passed': True, 'same_cache_control_matches': control_matches,
        'all_main_free_nodes_EOS': free_EOS, 'no_main_truncation': no_truncation,
        'implementation_control_passed': implementation_ok,
        'old_strict_gate_result_unchanged': True, 'old_reference_gate_passed': cfg['old_reference_gate_passed'],
        'not_a_benchmark_accuracy_claim': True,
        'network_transfer_measured': False, 'cases': [
            {'id': r['id'], 'A_matches_reference': r['vs_joint_reference']['A'],
             'final_free_kv_response': r['nodes']['C']['response'], 'vs_joint_reference': r['vs_joint_reference'],
             'final_scores': r['final_scores'], 'repeated': r['repeated']} for r in rows]}
    write(folder/'summary.json', summary)
    write(folder/'COMPLETE.json', {'utc_completed': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'counts': counts, 'structural_assertions_passed': True, 'implementation_control_passed': implementation_ok})
    if not implementation_ok:
        write(folder/'IMPLEMENTATION_CONTROL_MISMATCH.json', {'reason': 'same-cache output mismatch or truncation; not a trusted redundancy-ablation baseline'})
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}), flush=True)
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
        try:
            main(args.run_name)
        except Exception as exc:
            folder = ROOT/'runs'/args.run_name
            if folder.exists():
                write(folder/'FAILED.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
            raise
