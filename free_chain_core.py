"""Actual CV A->B->C chain with isolated native-cache receiver controls."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'work/KVComm'))


def _require(value, message):
    if not value:
        raise AssertionError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positions(record):
    return [row['position_ids'][0] for row in record['backend_forward_trace']
        if row['level'] == 'effective_rotary_position_ids']


def _ledger(state, node_count):
    from alignment_probe import tensor_hash

    parts = state.segments
    _require(len(parts) == 2*node_count and len({part['id'] for part in parts}) == len(parts),
        'free chain must include each prompt/output source event once')
    expected = [(node, kind) for node in ['A', 'B', 'C'][:node_count]
        for kind in ['prompt', 'output']]
    _require([(part['node'], part['kind']) for part in parts] == expected,
        'free chain source ledger order differs')
    cursor = 0
    for part in parts:
        _require(part['start'] == cursor and part['end'] > cursor,
            'free chain ledger is not contiguous')
        _require(part['tokens_sha256'] == tensor_hash(state.history[:, part['start']:part['end']]),
            'free chain source ledger token hash differs')
        cursor = part['end']
    _require(cursor == int(state.history.shape[-1]), 'free chain ledger omits history tokens')
    return {'event_count': len(parts), 'unique_event_count': len(parts),
        'global_history_tokens': cursor, 'contiguous': True, 'token_hashes_match': True}


def _capture_vectors(vectors, node, cv_value, control, comparison):
    steps = [('first', 0)]
    index = comparison['first_different_index']
    if index is not None and index < min(len(cv_value['ids']), len(control['ids'])):
        steps.append(('first_divergence', index))
    for label, step in steps:
        for side, value in [('cv', cv_value), ('native_cached', control)]:
            for kind, source in [('raw', value['output'].logits), ('processed', value['output'].scores)]:
                vectors[f'{node}__{side}__{label}__step{step}__{kind}'] = (
                    source[step][0].detach().float().cpu().numpy())


def run_free_case(torch, model, tokenizer, engine, audit, item, materials, phase, model_revision):
    """Run the actual CV chain; native controls never supply a downstream state.

    Return JSON-compatible node/control records and payload/provenance statistics.
    Raw/processed vectors are saved only at the first and first-divergence steps.
    Output differences are reported; structural input/cache invariants are checked.
    """
    import numpy as np
    from chain_core import State, cache_hash, cache_lengths, edge_payload
    from chain_core import prepare_prompt, repeated_transmissions
    from native_cache_control import native_cached_node
    from alignment_probe import compare_outputs, tensor_hash

    _require(engine.model is model and engine.audit is audit,
        'free chain engine must use the supplied model and audit')
    phase, qid = str(phase), str(item['id'])
    filename = f'{phase}_{qid}.npz'
    _require(Path(filename).name == filename and '/' not in filename and '\\' not in filename,
        'phase and question ID must be safe filename components')
    condition = phase+':free_kv'
    prompts = {node: prepare_prompt(tokenizer, model, item, node, materials)
        for node in ['A', 'B', 'C']}
    _require(all(prompt.shape[-1] >= 2 for prompt in prompts.values()),
        'official CV receiver prompts must have at least two tokens')
    actual, controls, comparisons, provenance, vectors = {}, {}, {}, {}, {}
    outgoing_hashes = {}
    event_start = len(audit.events)
    incoming = None
    with torch.inference_mode():
        for index, node in enumerate(['A', 'B', 'C'], start=1):
            sender = None if incoming is None else ['A', 'B'][index-2]
            source = incoming
            source_hash = None if source is None else cache_hash(torch, source.cache)
            source_history_hash = None if source is None else tensor_hash(source.history)
            if node == 'C':
                _require(source is actual['B']['state'], 'C must receive the actual CV-B state')
                _require(actual['B']['record']['route'] == 'kv', 'C sender must be CV-B')
                _require(source_hash == outgoing_hashes['B'], 'CV-B cache changed before C received it')
            actual[node] = engine.node(item, condition, node, source, prompts[node],
                use_kv=node != 'A', outgoing=node != 'C')
            value, record, state = actual[node], actual[node]['record'], actual[node]['state']
            _require(isinstance(state, State), 'main free-chain output must use the existing State type')
            record.update(phase=phase, model_revision=model_revision, free_chain=True,
                sender_node=sender, sender_history_sha256=source_history_hash,
                sender_cache_sha256=source_hash, ledger_check=_ledger(state, index),
                control_output_used_as_main_chain_input=False)
            if source is not None:
                _require(torch.equal(state.history[:, :source.history.shape[-1]], source.history),
                    'free chain receiver did not inherit its actual sender history exactly once')
                _require(state.segments[:len(source.segments)] == source.segments,
                    'free chain receiver altered or duplicated inherited source events')
                _require(record['sender_cache_hash_before_and_after'] == source_hash,
                    'CV record source cache hash differs from actual sender')
                _require(cache_hash(torch, source.cache) == source_hash,
                    'CV reception mutated the actual source cache')
                provenance[node] = {'sender_node': sender,
                    'sender_route': actual[sender]['record']['route'],
                    'sender_generation_tag': actual[sender]['record']['tag'],
                    'source_cache_sha256': source_hash, 'source_history_sha256': source_history_hash,
                    'source_cache_sha256_after_cv': cache_hash(torch, source.cache),
                    'main_sender_state_used': True, 'control_state_used': False}
            if node != 'C':
                _require(cache_lengths(state.cache) == [int(state.history.shape[-1])]*model.config.num_hidden_layers,
                    'actual outgoing cache must include the complete main-chain history')
                outgoing_hashes[node] = cache_hash(torch, state.cache)
                record['outgoing_completed_cache_sha256'] = outgoing_hashes[node]

            # The control receives exactly the state that this CV node received.
            # Its generated cache is neither completed nor inserted into actual.
            if source is not None:
                control = native_cached_node(torch, model, tokenizer, audit, source,
                    prompts[node], engine.budgets[node], f'{qid}:{phase}:free_kv_control:{node}')
                control['record'].update(node=node, phase=phase, model_revision=model_revision,
                    sender_node=sender, main_chain_member=False, outgoing_cache_completed=False,
                    control_output_used_as_main_chain_input=False)
                _require(control['record']['input_history_sha256'] == record['input_history_sha256'],
                    'CV and native control did not receive the same input history')
                _require(control['record']['local_prompt_ids'] == record['local_prompt_ids'],
                    'CV and native control did not receive the same local prompt')
                _require(control['record']['sender_cache_hash_before'] == source_hash and
                    control['record']['sender_cache_hash_after'] == source_hash,
                    'native control did not preserve the actual source cache')
                _require(cache_hash(torch, source.cache) == source_hash,
                    'control reception mutated the actual source cache')
                comparison = compare_outputs(torch, value, control)
                shared_steps = comparison['same_history_steps']
                comparison.update(same_input_history_sha256=True, same_local_prompt_ids=True,
                    same_sender_cache_sha256=True,
                    input_history_sha256=record['input_history_sha256'],
                    source_cache_sha256=source_hash,
                    source_cache_sha256_after_cv=provenance[node]['source_cache_sha256_after_cv'],
                    source_cache_sha256_after_native=cache_hash(torch, source.cache),
                    global_histories_identical=record['global_history_sha256'] ==
                        control['record']['global_history_sha256'],
                    effective_positions_on_shared_steps_identical=_positions(record)[:len(shared_steps)] ==
                        _positions(control['record'])[:len(shared_steps)],
                    logit_comparison_scope='identical generated history through first token divergence',
                    control_only=True, control_output_used_as_main_chain_input=False)
                comparisons[node] = comparison
                controls[node] = control['record']
                provenance[node]['source_cache_sha256_after_native'] = cache_hash(torch, source.cache)
                _capture_vectors(vectors, node, value, control, comparison)
                del control
            incoming = state

        # Final hashes cover both receivers and their independent controls.
        for sender in ['A', 'B']:
            _require(cache_hash(torch, actual[sender]['state'].cache) == outgoing_hashes[sender],
                'a later node or control changed a completed main-chain sender cache')
        payloads, repeated, edge_metadata = {}, {}, {}
        for kind in ['kv', 'text']:
            edges = []
            for sender, receiver in [('A', 'B'), ('B', 'C')]:
                state = actual[sender]['state']
                payload = edge_payload(torch, tokenizer, state, kind, model_revision)
                source_metadata = {'sender_node': sender, 'receiver_node': receiver,
                    'source_cache_sha256': outgoing_hashes[sender],
                    'source_history_sha256': tensor_hash(state.history),
                    'source_route': actual[sender]['record']['route'],
                    'source_generation_tag': actual[sender]['record']['tag']}
                payload['metadata'].update(source_metadata)
                payload['metadata_bytes'] = len(json.dumps(payload['metadata'], sort_keys=True,
                    separators=(',', ':')).encode('utf-8'))
                payload['logical_total_bytes'] = payload['content_bytes']+payload['metadata_bytes']
                payload.update(edge=f'{sender}_to_{receiver}', **source_metadata)
                edges.append(payload)
                edge_metadata[f'{sender}_to_{receiver}'] = source_metadata
            payloads[kind] = edges
            repeated[kind] = repeated_transmissions(edges)

    vector_path = Path(audit.run_dir)/'vectors'/filename
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(vector_path, **vectors)
    events = audit.events[event_start:]
    return {'id': item['id'], 'phase': phase, 'model_revision': model_revision,
        'chain_policy': 'A native; B actual CV; C receives completed actual CV-B cache',
        'nodes': {node: actual[node]['record'] for node in ['A', 'B', 'C']},
        'controls': controls, 'comparisons': comparisons,
        'edge_payloads': payloads, 'edge_metadata': edge_metadata, 'repeated': repeated,
        'source_cache_hashes': outgoing_hashes, 'cache_provenance': provenance,
        'text_payload_reference': 'int32 token packets of the actual free-chain histories',
        'first_step_and_divergence_vectors': {
            'file': str(vector_path.relative_to(Path(audit.run_dir))), 'sha256': _sha(vector_path),
            'keys': sorted(vectors), 'dtype': 'float32',
            'scope': 'B/C first step and any first divergence with identical prior generated tokens'},
        'counts': {'generate': sum(event['kind'] == 'generate' for event in events),
            'cache_tail_forward': sum(event['kind'] == 'cache_tail_forward' for event in events)},
        'control_outputs_entered_main_chain': False, 'network_transfer_measured': False}


class _DummyTokenizer:
    def __init__(self, torch):
        self.torch = torch

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        _require(add_generation_prompt and return_tensors == 'pt', 'CPU prompt template contract differs')
        content = messages[0]['content']
        prompts = {'A': [1, 3, 5, 7, 9], 'B': [1, 4, 6], 'C': [1, 8, 10, 12]}
        node = next(node for node in prompts if content.startswith('You are Agent '+node+','))
        return self.torch.tensor([prompts[node]], dtype=self.torch.long)

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        ids = ids.detach().cpu().tolist() if hasattr(ids, 'detach') else list(ids)
        return ' '.join(str(int(value)) for value in ids
            if not skip_special_tokens or int(value) not in [0, 1, 63])


def _cpu_cases(report):
    import numpy as np
    import torch
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from models import CVCommunicator
    from chain_core import ChainAudit, ChainEngine

    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, eos_token_id=63, pad_token_id=0, bos_token_id=1)
    config._attn_implementation = 'sdpa'
    model = Qwen2ForCausalLM(config).to(device='cpu', dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generation = model.generation_config
    generation.do_sample, generation.num_beams, generation.repetition_penalty = False, 1, 1.0
    generation.use_cache = True
    generation.forced_eos_token_id, generation.min_new_tokens = 63, 3
    cv = CVCommunicator(model, model, 0, 3, layers_list=list(range(4)),
        apply_attn_tracer=False, shift_back=False)
    cv.generation_config = copy.deepcopy(generation)
    _require(model.device.type == 'cpu' and model.dtype == torch.float32, 'CPU FP32 model required')
    _require(cv.layer_map == {index: index for index in range(4)}, 'CPU CV layer map is not identity')
    ROOT.joinpath('checks').mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='free_chain_cpu_', dir=str(ROOT/'checks')))
    tokenizer = _DummyTokenizer(torch)
    audit = ChainAudit(torch, model, run_dir)
    engine = ChainEngine(torch, model, cv, tokenizer, {'A': 4, 'B': 4, 'C': 4}, audit)
    report.update(torch=torch.__version__, transformers=transformers.__version__, python=sys.version,
        seed=42, device='cpu', dtype='float32', model_config=config.to_dict(),
        generation_config=generation.to_dict(), audit_run_dir=str(run_dir),
        official_models_sha256=_sha(ROOT/'work/KVComm/models.py'))
    try:
        result = run_free_case(torch, model, tokenizer, engine, audit,
            {'id': 'tiny_free_chain_cpu', 'prompt_B': 'What is the supported answer?'},
            {'A': 'alpha material', 'B': 'beta material'}, 'cpu', 'tiny-random-fp32-qwen')
        report['case'] = result
        _require(result['counts'] == {'generate': 5, 'cache_tail_forward': 2},
            'free CPU chain must generate five times and complete two sender EOS tails')
        _require(result['nodes']['A']['route'] == 'native' and
            all(result['nodes'][node]['route'] == 'kv' for node in ['B', 'C']),
            'free CPU chain routes differ from A native/B CV/C CV')
        source_b = result['source_cache_hashes']['B']
        _require(result['cache_provenance']['C']['sender_node'] == 'B' and
            result['cache_provenance']['C']['sender_route'] == 'kv' and
            result['cache_provenance']['C']['source_cache_sha256'] == source_b and
            result['nodes']['C']['sender_cache_hash_before_and_after'] == source_b,
            'C did not receive the completed actual CV-B cache')
        _require(not result['control_outputs_entered_main_chain'], 'control output entered the main chain')
        for node in ['A', 'B', 'C']:
            record = result['nodes'][node]
            _require(record['output_ids'][-1] == 63 and record['generated_tokens'] == 4 and
                record['eos_stopped'] and not record['actually_truncated'], 'main-chain forced EOS check failed')
            _require(record['ledger_check']['event_count'] == 2*(['A', 'B', 'C'].index(node)+1),
                'main-chain source events were not included exactly once')
            if node != 'C':
                completion = record['cache_completion']
                _require(completion['missing_tokens'] == 1 and completion['filled_ids'] == [[63]] and
                    completion['expected_length'] == len(record['global_history_ids'][0]),
                    'actual sender EOS tail was not completed')
        for node in ['B', 'C']:
            comparison = result['comparisons'][node]
            _require(comparison['generated_tokens_identical'] and comparison['global_histories_identical'] and
                comparison['effective_positions_on_shared_steps_identical'],
                'free CPU CV/native control outputs or positions differ')
            for step in comparison['same_history_steps']:
                for kind in ['raw', 'processed']:
                    _require(step[kind]['nonfinite_values_match'], 'CPU control nonfinite logits differ')
            control = result['controls'][node]
            _require(control['cache_completion'] is None and not control['outgoing_cache_completed'] and
                control['checked_suffix_only_forward'], 'native control unexpectedly entered the main chain')
            _require(control['sender_cache_hash_before'] == control['sender_cache_hash_after'] ==
                result['nodes'][node]['sender_cache_hash_before_and_after'],
                'CV/native controls did not use the same unchanged actual sender cache')

        # Reopen the saved independent vectors for a fixed FP32 allclose check.
        vector_path = run_dir/result['first_step_and_divergence_vectors']['file']
        _require(_sha(vector_path) == result['first_step_and_divergence_vectors']['sha256'],
            'saved audit vector hash differs')
        with np.load(vector_path, allow_pickle=False) as arrays:
            for node in ['B', 'C']:
                for kind in ['raw', 'processed']:
                    left = arrays[f'{node}__cv__first__step0__{kind}']
                    right = arrays[f'{node}__native_cached__first__step0__{kind}']
                    _require(np.allclose(left, right, atol=1e-5, rtol=1e-5),
                        'saved first-step FP32 vectors exceed fixed tolerance')
            # Metrics include every step while the histories remain identical.
            for node in ['B', 'C']:
                for step in result['comparisons'][node]['same_history_steps']:
                    for kind in ['raw', 'processed']:
                        _require(step[kind]['max_abs'] <= 1e-5,
                            'same-history FP32 logit error exceeds fixed absolute tolerance')

        payload_checks = {}
        for kind, edges in result['edge_payloads'].items():
            per_token = 4 if kind == 'text' else 512
            _require(edges[1]['parts'][:len(edges[0]['parts'])] == edges[0]['parts'],
                'B-to-C did not include A source event spans exactly once')
            for edge, sender in zip(edges, ['A', 'B']):
                history_len = len(result['nodes'][sender]['global_history_ids'][0])
                _require(edge['content_bytes'] == per_token*history_len,
                    'edge payload content does not match actual history token count')
                _require(sum(part['content_bytes'] for part in edge['parts']) == edge['content_bytes'] and
                    len({part['id'] for part in edge['parts']}) == len(edge['parts']),
                    'edge ledger duplicates events or miscounts their bytes')
                _require(edge['metadata']['sender_node'] == sender and
                    edge['metadata']['source_cache_sha256'] == result['source_cache_hashes'][sender],
                    'edge source-cache metadata differs from actual sender')
            repeat = result['repeated'][kind]
            _require(repeat['repeated_source_event_bytes'] == edges[0]['content_bytes'] ==
                repeat['A_history_bytes_in_B_to_C'],
                'repeated bytes must be precisely one additional forwarding of A history')
            payload_checks[kind] = {'passed': True, 'bytes_per_token': per_token,
                'source_history_included_once_per_edge': True,
                'repeated_bytes_equal_one_A_history': True}
        _require(all(event['status'] == 'ok' for event in audit.events), 'an audited CPU operation failed')
        report.update(passed=True, fixed_fp32_tolerance={'atol': 1e-5, 'rtol': 1e-5,
            'same_history_step_max_abs_limit': 1e-5}, payload_checks=payload_checks,
            c_uses_actual_cv_b_completed_cache=True, audit_events=copy.deepcopy(audit.events))
    finally:
        for handle in audit.handles:
            handle.remove()


def cpu_check():
    # Importing this module for an authorized GPU run preserves the caller's device.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    path = ROOT/'checks/free_chain_cpu.json'
    report = {'passed': False, 'tiny_random_cpu_only': True, 'real_7b_claim': False,
        'script_sha256': _sha(__file__), 'chain_core_sha256': _sha(ROOT/'chain_core.py'),
        'native_helper_sha256': _sha(ROOT/'native_cache_control.py'),
        'utc_started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        _cpu_cases(report)
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['utc_completed'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
        print(json.dumps({'free_chain_cpu_passed': report['passed'], 'report': str(path),
            'real_7b_claim': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', required=True)
    parser.parse_args()
    cpu_check()
