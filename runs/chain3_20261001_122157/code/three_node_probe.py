"""Small three-role history/relay check; does not alter the completed baseline."""
import argparse
import copy
from collections import Counter
import gc
import json
import os
from pathlib import Path
import subprocess
import time

from run import ROOT, sha, write, gpu_snapshot, load_model
from alignment_probe import compare_outputs
from chain_core import ROLES, split_materials, prepare_prompt, ChainAudit, ChainEngine
from chain_core import edge_payload, repeated_transmissions

NODES = ['A', 'B', 'C']
FILES = ['chain_core.py', 'three_node_probe.py', 'three_node_cpu.py']


def freeze():
    cfg = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    assets = json.loads((ROOT/'assets.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'data/calibration.json') == assets['split_files']['calibration']['sha256']
    items = json.loads((ROOT/'data/calibration.json').read_text(encoding='utf-8'))[:4]
    assert [x['id'] for x in items] == assets['split_files']['calibration']['ids'][:4]
    data = [{'item': item, 'materials': split_materials(item)} for item in items]
    write(ROOT/'data/three_node_cases.json', data)
    settings = {'purpose': 'three-node calibration-only protocol check', 'main_ids': [x['id'] for x in items[:3]],
        'warmup_id': items[3]['id'], 'calibration_sha256': sha(ROOT/'data/calibration.json'),
        'cases_sha256': sha(ROOT/'data/three_node_cases.json'), 'roles': ROLES,
        'material_policy': data[0]['materials']['policy'], 'model_revision': cfg['model_7b_revision'],
        'model_path': cfg['model_7b_path'], 'source_commit': cfg['source_commit'],
        'dtype': 'bfloat16', 'attention': 'sdpa', 'seed': 42, 'repetition_penalty': 1.0,
        'budgets': {'A': 256, 'B': 256, 'C': 48}, 'shift_back': False,
        'expected_generate_max': 29, 'expected_generate_if_gate_stops': 20,
        'cache_tail_forward_max': 14, 'expected_reference_generate': 12, 'expected_replay_generate': 8,
        'expected_free_kv_generate_if_gate_passes': 9,
        'gate': 'all six main replay outputs match reference token-for-token; no truncation in reference/replay',
        'exact_text_transport': 'int32 token-history packet; readable UTF-8 size also reported',
        'network_transfer_measured': False, 'source_hashes': {name: sha(ROOT/name) for name in FILES},
        'evaluation_and_confirmation_used': False}
    write(ROOT/'three_node_conditions.json', settings)
    print(json.dumps({'frozen_main_ids': settings['main_ids'], 'warmup_id': settings['warmup_id'],
        'generate_max': 29, 'cache_tail_forward_max': 14}), flush=True)


def launch():
    import fcntl
    settings = json.loads((ROOT/'three_node_conditions.json').read_text())
    checked = json.loads((ROOT/'checks/three_node_cpu_check.json').read_text())
    assert checked['passed']
    assert checked['chain_core_sha256'] == sha(ROOT/'chain_core.py')
    assert checked['script_sha256'] == sha(ROOT/'three_node_cpu.py')
    for name, value in settings['source_hashes'].items():
        assert sha(ROOT/name) == value, f'frozen {name} changed'
    assert settings['cases_sha256'] == sha(ROOT/'data/three_node_cases.json')
    lock = open(ROOT/'worker.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        candidates = [x for x in gpu_snapshot() if int(x['utilization']) < 10]
        candidates.sort(key=lambda x: int(x['free_mib']), reverse=True)
        if candidates and int(candidates[0]['free_mib']) >= 18000:
            gpu = candidates[0]
            break
        print(json.dumps({'waiting_for_7b_capacity': candidates}), flush=True)
        time.sleep(30)
    run_name = 'chain3_'+time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    python = '/home/liuxin/.conda/envs/kv-graph/bin/python'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    (ROOT/'logs').mkdir(exist_ok=True)
    log = open(ROOT/'logs'/f'{run_name}.log', 'ab', buffering=0)
    os.set_inheritable(lock.fileno(), True)
    worker = subprocess.Popen([python, '-u', str(Path(__file__).resolve()), '--run-name', run_name],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
    state = {'run_name': run_name, 'pid': worker.pid, 'gpu': gpu, 'python': python}
    write(ROOT/'three_node_worker.json', state)
    print(json.dumps(state), flush=True)


def main(run_name):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer
    from models import CVCommunicator
    from score_repro_outputs import hotpot_token_f1, normalize
    settings = json.loads((ROOT/'three_node_conditions.json').read_text())
    cfg = json.loads((ROOT/'config.json').read_text())
    assets = json.loads((ROOT/'assets.json').read_text())
    assert settings['roles'] == ROLES
    for name, value in settings['source_hashes'].items():
        assert sha(ROOT/name) == value
    assert sha(ROOT/'data/three_node_cases.json') == settings['cases_sha256']
    assert sha(ROOT/'data/calibration.json') == settings['calibration_sha256']
    cases = json.loads((ROOT/'data/three_node_cases.json').read_text())
    assert [x['item']['id'] for x in cases[:3]] == settings['main_ids'] and cases[3]['item']['id'] == settings['warmup_id']
    for case in cases:
        assert split_materials(case['item']) == case['materials']
    for name in ['eval.py', 'models.py', 'model_attn.py']:
        assert sha(ROOT/'work/KVComm'/name) == assets['source_files'][name]['work_sha256']
    verified = json.loads((ROOT/'checks/model7b_manifest.json').read_text())
    assert verified['revision'] == settings['model_revision'] == cfg['model_7b_revision']
    for name, spec in verified['files'].items():
        path = Path(settings['model_path'])/name
        assert path.stat().st_size == spec['bytes']
        if not spec['official_lfs_verified']:
            assert sha(path) == spec['sha256']
    folder = ROOT/'runs'/run_name
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    manifest = {'settings': settings, 'settings_sha256': sha(ROOT/'three_node_conditions.json'),
        'cpu_check_sha256': sha(ROOT/'checks/three_node_cpu_check.json'), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'gpu_at_launch': gpu_snapshot(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'shared_frozen_weights': True,
        'time_is_instrumented_diagnostic_not_speed_benchmark': True}
    write(folder/'manifest.json', manifest)
    (folder/'code').mkdir()
    for name in FILES+['three_node_conditions.json']:
        (folder/'code'/name).write_bytes((ROOT/name).read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(settings['model_path'], local_files_only=True)
    model, load_seconds = load_model(torch, settings['model_path'])
    cv = CVCommunicator(model, model, 0, model.config.num_hidden_layers-1,
        layers_list=list(range(model.config.num_hidden_layers)), shift_back=False)
    cv.generation_config = copy.deepcopy(model.generation_config)
    assert all(k == v for k, v in cv.layer_map.items())
    manifest.update(model_load_seconds=load_seconds, generation_config=model.generation_config.to_dict(),
        full_layers=cv.layers_list, identity_layer_map=cv.layer_map)
    write(folder/'manifest.json', manifest)
    audit = ChainAudit(torch, model, folder)
    engine = ChainEngine(torch, model, cv, tokenizer, settings['budgets'], audit)
    rows = []

    def run_chain(case, condition, use_kv):
        item, material = case['item'], case['materials']
        outputs, edges, state = {}, [], None
        for node in NODES:
            prompt = prepare_prompt(tokenizer, model, item, node, material)
            if state is not None:
                edges.append(edge_payload(torch, tokenizer, state, 'kv' if use_kv else 'text_tokens', settings['model_revision']))
            outputs[node] = engine.node(item, condition, node, state, prompt, use_kv=use_kv, outgoing=node != 'C')
            state = outputs[node]['state']
        return outputs, edges

    def reference_and_replay(case, phase):
        reference, edges = run_chain(case, phase+':text', False)
        replay, comparisons = {}, {}
        for node, predecessor in [('B', 'A'), ('C', 'B')]:
            prompt = prepare_prompt(tokenizer, model, case['item'], node, case['materials'])
            replay[node] = engine.node(case['item'], phase+':replay', node,
                reference[predecessor]['state'], prompt, use_kv=True, outgoing=False)
            comparisons[node] = compare_outputs(torch, reference[node], replay[node])
        return reference, replay, comparisons, edges

    def save_case(case, phase, reference, replay, comparisons, edges, free=None, free_edges=None):
        item = case['item']
        row = {'id': item['id'], 'phase': phase, 'question': item['prompt_B'], 'question_hash': item['question_hash'],
            'gold': item['answer'], 'materials': case['materials'],
            'reference': {n: x['record'] for n, x in reference.items()},
            'replay': {n: x['record'] for n, x in replay.items()}, 'comparisons': comparisons,
            'reference_edges': edges, 'reference_repeated_transmissions': repeated_transmissions(edges)}
        vectors = {}
        for label, outputs in [('reference', reference), ('replay', replay), ('free_kv', free or {})]:
            for node, x in outputs.items():
                vectors[f'{label}_{node}_first_raw'] = x['output'].logits[0][0].float().cpu().numpy()
                vectors[f'{label}_{node}_first_processed'] = x['output'].scores[0][0].float().cpu().numpy()
        for node, comparison in comparisons.items():
            diff = comparison['first_different_index']
            if diff is not None and diff < min(len(reference[node]['ids']), len(replay[node]['ids'])):
                for label, outputs in [('reference', reference), ('replay', replay)]:
                    vectors[f'{label}_{node}_divergence_raw'] = outputs[node]['output'].logits[diff][0].float().cpu().numpy()
                    vectors[f'{label}_{node}_divergence_processed'] = outputs[node]['output'].scores[diff][0].float().cpu().numpy()
        if free:
            row['free_kv'] = {n: x['record'] for n, x in free.items()}
            row['free_kv_edges'] = free_edges
            row['free_kv_repeated_transmissions'] = repeated_transmissions(free_edges)
            row['free_vs_reference'] = {n: {'same_input_history': free[n]['record']['input_history_sha256'] == reference[n]['record']['input_history_sha256'],
                'same_output_tokens': free[n]['ids'] == reference[n]['ids']} for n in NODES}
        for label in ['reference', 'free_kv']:
            if label in row:
                response = row[label]['C']['response']
                row[label+'_final_score'] = {'exact_match': normalize(response) == normalize(item['answer']),
                    'standard_token_f1': hotpot_token_f1(response, item['answer'])}
        vector_path = folder/'vectors'/f'{phase}_{item["id"]}.npz'
        vector_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(vector_path, **vectors)
        row['vectors'] = {'file': str(vector_path.relative_to(folder)), 'sha256': sha(vector_path)}
        write(folder/'items'/f'{phase}_{item["id"]}.json', row)
        return row

    def retire_gpu_outputs(*groups):
        # Replay has finished; free chains regenerate their own history. Preserve
        # vectors on CPU and release all reference/replay caches before the next case.
        for outputs in groups:
            for value in outputs.values():
                output = value['output']
                output.logits = tuple(x.cpu() for x in output.logits)
                output.scores = tuple(x.cpu() for x in output.scores)
                output.sequences = output.sequences.cpu()
                output.past_key_values = None
                value['state'].cache = None
                value['state'].history = value['state'].history.cpu()
        gc.collect()

    with torch.inference_mode():
        ref, rep, cmp, edges = reference_and_replay(cases[3], 'warmup')
        save_case(cases[3], 'warmup', ref, rep, cmp, edges)
        del ref, rep, cmp, edges
        gc.collect()
        gate_passed = True
        # Keep CPU records/vectors; each free chain constructs its own caches.
        prepared = []
        for case in cases[:3]:
            ref, rep, cmp, edges = reference_and_replay(case, 'main')
            passed = all(x['generated_tokens_identical'] for x in cmp.values())
            passed = passed and not any(x['record']['actually_truncated'] for group in [ref, rep] for x in group.values())
            gate_passed = gate_passed and passed
            prepared.append((case, ref, rep, cmp, edges))
            save_case(case, 'main', ref, rep, cmp, edges)
            retire_gpu_outputs(ref, rep)
            print(json.dumps({'id': case['item']['id'], 'replay_gate': passed}), flush=True)
        for case, ref, rep, cmp, edges in prepared:
            free, free_edges = run_chain(case, 'main:free_kv', True) if gate_passed else (None, None)
            row = save_case(case, 'main', ref, rep, cmp, edges, free, free_edges)
            rows.append(row)
            del free
            gc.collect()
    counts = dict(Counter(x['kind'] for x in audit.events))
    assert counts.get('generate') == (29 if gate_passed else 20)
    assert counts.get('cache_tail_forward', 0) <= 14 and all(x['status'] == 'ok' for x in audit.events)
    free_no_truncation = gate_passed and all(not r['free_kv'][n]['actually_truncated'] for r in rows for n in NODES)
    free_matches = {n: sum(r['free_vs_reference'][n]['same_input_history'] and
        r['free_vs_reference'][n]['same_output_tokens'] for r in rows) for n in NODES} if gate_passed else None
    free_chain_passed = bool(free_no_truncation and all(v == 3 for v in free_matches.values()))
    summary = {'run_name': run_name, 'main_questions': 3, 'warmup_questions': 1,
        'gate_passed': gate_passed, 'counts': counts,
        'replay_matches': {n: sum(r['comparisons'][n]['generated_tokens_identical'] for r in rows) for n in ['B', 'C']},
        'free_kv_chains': 3 if gate_passed else 0, 'free_chain_passed': free_chain_passed,
        'free_chain_no_truncation': bool(free_no_truncation), 'free_node_matches': free_matches,
        'not_a_benchmark_accuracy_claim': True,
        'message_and_repetition': []}
    for row in rows:
        record = {'id': row['id'], 'reference_repeated': row['reference_repeated_transmissions']}
        if gate_passed:
            record.update(free_kv_repeated=row['free_kv_repeated_transmissions'],
                free_vs_reference=row['free_vs_reference'], final_scores={k: row[k] for k in ['reference_final_score', 'free_kv_final_score']})
        summary['message_and_repetition'].append(record)
    write(folder/'summary.json', summary)
    write(folder/'COMPLETE.json', {'phase': 'chain3_protocol_check', 'utc_completed': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'gate_passed': gate_passed, 'free_chain_passed': free_chain_passed, 'counts': counts})
    if not gate_passed:
        write(folder/'GATE_MISMATCH.json', {'reason': 'same-history replay output diverged or hit generation budget; free KV expansion skipped',
            'question_ids': settings['main_ids']})
    print(json.dumps(summary), flush=True)
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
