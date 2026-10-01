"""CPU-only checks of the three-node full-history adapter and official KV routing.

Run with the project's existing torch/transformers environment. This constructs a
tiny random Qwen in FP32; its outputs are diagnostics, never real 7B results.
"""
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
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
sys.path.insert(0, str(ROOT/'work/KVComm'))
REPORT = ROOT/'checks/three_node_cpu_check.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(data):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(REPORT.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(REPORT)


class DummyTokenizer:
    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        values = ids.detach().cpu().tolist() if hasattr(ids, 'detach') else list(ids)
        return ' '.join(str(int(value)) for value in values
            if not skip_special_tokens or int(value) not in [0, 1, 63])


def require(value, message):
    if not value:
        raise AssertionError(message)


def run_checks(report):
    import torch
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from models import CVCommunicator
    from chain_core import ChainAudit, ChainEngine, State, cache_hash, cache_lengths
    from chain_core import edge_payload, repeated_transmissions
    from alignment_probe import tensor_hash

    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, eos_token_id=63, pad_token_id=0, bos_token_id=1)
    config._attn_implementation = 'sdpa'
    model = Qwen2ForCausalLM(config).to(device='cpu', dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Three non-EOS steps make the forced last-step EOS and its tail fill observable.
    budgets = {'A': 4, 'B': 4, 'C': 4}
    generation = model.generation_config
    generation.do_sample = False
    generation.num_beams = 1
    generation.repetition_penalty = 1.0
    generation.forced_eos_token_id = 63
    generation.min_new_tokens = 3
    generation.eos_token_id = 63
    generation.pad_token_id = 0
    generation.use_cache = True
    cv = CVCommunicator(model, model, 0, 3, layers_list=list(range(4)),
        apply_attn_tracer=False, shift_back=False)
    cv.generation_config = copy.deepcopy(generation)
    require(cv.layers_list == list(range(4)), 'official CV must transfer every layer')
    require(cv.layer_map == {index: index for index in range(4)}, 'CV layer map is not identity')
    require(model.device.type == 'cpu' and model.dtype == torch.float32, 'CPU FP32 model required')
    tokenizer = DummyTokenizer()
    prompts = {'A': torch.tensor([[1, 3, 5, 7, 9]], dtype=torch.long),
        'B': torch.tensor([[1, 4, 6]], dtype=torch.long),
        'C': torch.tensor([[1, 8, 10, 12]], dtype=torch.long)}
    require(all(prompt.shape[-1] >= 2 for prompt in prompts.values()),
        'official CV first receiver prompt must have at least two tokens')
    item = {'id': 'tiny_three_node_cpu'}
    report.update(torch=torch.__version__, transformers=transformers.__version__,
        python=sys.version, seed=42, device='cpu', dtype='float32',
        official_models_sha256=sha(ROOT/'work/KVComm/models.py'),
        model_config=config.to_dict(), generation_config=generation.to_dict(),
        budgets=budgets, prompt_ids={key: value.tolist() for key, value in prompts.items()},
        full_layers=list(cv.layers_list), identity_layer_map=dict(cv.layer_map),
        cv_supports_cache_class=cv._supports_cache_class)
    cases = report['cases']
    forward_count = [0]

    def count_forward(module, args, kwargs):
        forward_count[0] += 1

    counter_handle = model.register_forward_pre_hook(count_forward, with_kwargs=True)

    def positions(record):
        values = [row['position_ids'] for row in record['backend_forward_trace']
            if row['level'] == 'effective_rotary_position_ids']
        require(all(value is not None and len(value) == 1 for value in values),
            'expected one batch row of effective positions')
        return [value[0] for value in values]

    def check_ledger(state, node_count):
        segments = state.segments
        require(len(segments) == 2*node_count, 'each node must insert prompt and output once')
        require(len({segment['id'] for segment in segments}) == len(segments),
            'duplicate source event within a history')
        expected_pairs = [(node, kind) for node in ['A', 'B', 'C'][:node_count]
            for kind in ['prompt', 'output']]
        require([(segment['node'], segment['kind']) for segment in segments] == expected_pairs,
            'history ledger is out of order')
        cursor = 0
        for segment in segments:
            require(segment['start'] == cursor and segment['end'] > cursor,
                'history ledger is not contiguous')
            require(segment['tokens_sha256'] == tensor_hash(
                state.history[:, segment['start']:segment['end']]),
                'ledger token hash differs from the actual history')
            cursor = segment['end']
        require(cursor == state.history.shape[-1], 'ledger does not cover the whole history')
        return {'event_count': len(segments), 'unique_event_count': len(segments),
            'global_history_tokens': cursor, 'contiguous': True, 'tokens_sha256_match': True,
            'events': copy.deepcopy(segments)}

    def node_details(value, prompt, prefix_len, kv, node_count):
        record, state = value['record'], value['state']
        input_len = prefix_len+int(prompt.shape[-1])
        actual = positions(record)
        expected = [list(range(prefix_len if kv else 0, input_len))]
        expected.extend([[input_len+step-1] for step in range(1, len(value['ids']))])
        require(actual == expected, 'effective positions differ from global-history positions')
        require(record['checked_effective_positions'], 'core position checks did not run')
        require(value['ids'][-1] == 63 and len(value['ids']) == budgets[record['node']],
            'forced EOS must close the complete generation budget')
        require(record['eos_stopped'] and not record['actually_truncated'], 'EOS bookkeeping differs')
        return {'output_ids': list(value['ids']), 'global_history_ids': state.history.tolist(),
            'global_history_sha256': tensor_hash(state.history),
            'effective_positions': actual, 'expected_effective_positions': expected,
            'cache_lengths_before_completion': record['cache_lengths_before_completion'],
            'completed_cache_lengths': None if state.cache is None else cache_lengths(state.cache),
            'cache_completion': copy.deepcopy(record['cache_completion']),
            'eos_stopped': record['eos_stopped'], 'ledger': check_ledger(state, node_count),
            'backend_forward_trace': copy.deepcopy(record['backend_forward_trace'])}

    def compare(native, cached, prefix_len):
        require(native['ids'] == cached['ids'], 'native and KV generated token IDs differ')
        require(torch.equal(native['state'].history, cached['state'].history),
            'native and KV global histories differ')
        native_positions = positions(native['record'])
        receiver_positions = [native_positions[0][prefix_len:]]+native_positions[1:]
        require(receiver_positions == positions(cached['record']),
            'native and KV positions differ over the same receiver tokens')
        raw_max = 0.0
        processed_max = 0.0
        require(len(native['output'].logits) == len(cached['output'].logits),
            'native and KV logit step counts differ')
        for left, right in zip(native['output'].logits, cached['output'].logits):
            raw_max = max(raw_max, float((left-right).abs().max()))
            require(torch.allclose(left, right, atol=1e-5, rtol=1e-5),
                'native and KV FP32 raw logits differ beyond fixed tolerance')
        for left, right in zip(native['output'].scores, cached['output'].scores):
            # The forced EOS processor creates -inf; compare finite values separately.
            finite_left, finite_right = torch.isfinite(left), torch.isfinite(right)
            require(torch.equal(finite_left, finite_right), 'processed score finite masks differ')
            require(torch.equal(left[~finite_left], right[~finite_right]),
                'processed nonfinite scores differ')
            if finite_left.any():
                processed_max = max(processed_max,
                    float((left[finite_left]-right[finite_right]).abs().max()))
            require(torch.allclose(left, right, atol=1e-5, rtol=1e-5),
                'native and KV FP32 processed scores differ beyond fixed tolerance')
        return {'generated_tokens_identical': True, 'global_history_identical': True,
            'receiver_global_positions_identical': True, 'raw_logits_max_abs': raw_max,
            'processed_scores_finite_max_abs': processed_max, 'atol': 1e-5, 'rtol': 1e-5}

    def fresh_state(source):
        fresh = State(source.history.clone(), copy.deepcopy(source.segments), copy.deepcopy(source.cache))
        require(cache_hash(torch, fresh.cache) == cache_hash(torch, source.cache),
            'fresh sender clone differs before replay')
        return fresh

    def check_eos_cache(value):
        state, completion = value['state'], value['record']['cache_completion']
        length = int(state.history.shape[-1])
        require(cache_lengths(state.cache) == [length]*4, 'outgoing cache omits global history')
        require(int(state.history[0, -1]) == 63, 'completed history does not end with EOS')
        require(completion['missing_tokens'] == 1, 'one-token EOS tail-fill case was not exercised')
        require(completion['filled_ids'] == [[63]], 'cache completion must forward the global EOS tail')
        require(positions(completion['forward']) == [[length-1]],
            'EOS tail fill did not use the final global position')
        require(all(tensor.dtype == torch.float32 for tensors in
            [state.cache.key_cache, state.cache.value_cache] for tensor in tensors),
            'cache tensors must remain FP32')
        return {'contains_complete_history_including_eos': True, 'length': length,
            'layer_lengths': cache_lengths(state.cache), 'eos_id': 63,
            'one_token_tail_fill_exercised': True, 'tail_position': length-1,
            'cache_sha256': cache_hash(torch, state.cache)}

    try:
        with tempfile.TemporaryDirectory(prefix='three_node_cpu_') as temporary:
            audit = ChainAudit(torch, model, Path(temporary))
            engine = ChainEngine(torch, model, cv, tokenizer, budgets, audit)
            try:
                with torch.inference_mode():
                    text = {}
                    cases['text_three_node_chain'] = {'nodes': {}}
                    incoming = None
                    for index, node in enumerate(['A', 'B', 'C'], start=1):
                        prefix_len = 0 if incoming is None else int(incoming.history.shape[-1])
                        text[node] = engine.node(item, 'text', node, incoming, prompts[node],
                            use_kv=False, outgoing=node != 'C')
                        cases['text_three_node_chain']['nodes'][node] = node_details(
                            text[node], prompts[node], prefix_len, False, index)
                        if incoming is not None:
                            require(text[node]['state'].segments[:len(incoming.segments)] == incoming.segments,
                                'text relay changed or reinserted inherited source events')
                        incoming = text[node]['state']
                    cases['text_three_node_chain']['passed'] = True
                    cases['text_sender_eos_cache_completion'] = {
                        node: check_eos_cache(text[node]) for node in ['A', 'B']}
                    original_text_hashes = {node: cache_hash(torch, text[node]['state'].cache)
                        for node in ['A', 'B']}

                    # B and C receive independent clones of native full-history caches.
                    cases['native_cache_kv_replay'] = {}
                    replay = {}
                    for index, (node, sender) in enumerate([('B', 'A'), ('C', 'B')], start=2):
                        source = fresh_state(text[sender]['state'])
                        source_hash = cache_hash(torch, source.cache)
                        replay[node] = engine.node(item, 'native_cache_replay', node, source,
                            prompts[node], use_kv=True, outgoing=node == 'B')
                        require(cache_hash(torch, source.cache) == source_hash,
                            'replay mutated the source cache clone')
                        require(cache_hash(torch, text[sender]['state'].cache) == original_text_hashes[sender],
                            'replay mutated the original text sender cache')
                        details = node_details(replay[node], prompts[node], source.history.shape[-1], True, index)
                        details.update(comparison_to_text=compare(text[node], replay[node], source.history.shape[-1]),
                            source_cache_sha256_before=source_hash, source_cache_sha256_after=source_hash,
                            original_sender_cache_unchanged=True, fresh_clone_used=True, passed=True)
                        cases['native_cache_kv_replay'][node] = details

                    free = {}
                    cases['free_kv_three_node_chain'] = {'nodes': {}}
                    free_hashes = {}
                    incoming = None
                    for index, node in enumerate(['A', 'B', 'C'], start=1):
                        prefix_len = 0 if incoming is None else int(incoming.history.shape[-1])
                        source_hash = None if incoming is None else cache_hash(torch, incoming.cache)
                        free[node] = engine.node(item, 'free_kv', node, incoming, prompts[node],
                            use_kv=node != 'A', outgoing=node != 'C')
                        if incoming is not None:
                            require(cache_hash(torch, incoming.cache) == source_hash,
                                'free KV relay mutated its sender cache')
                            require(free[node]['state'].segments[:len(incoming.segments)] == incoming.segments,
                                'free KV relay changed or duplicated inherited source events')
                        details = node_details(free[node], prompts[node], prefix_len, node != 'A', index)
                        details.update(comparison_to_text=compare(text[node], free[node], prefix_len),
                            sender_cache_sha256_before=source_hash, sender_cache_sha256_after=source_hash)
                        cases['free_kv_three_node_chain']['nodes'][node] = details
                        incoming = free[node]['state']
                        if node != 'C':
                            free_hashes[node] = cache_hash(torch, incoming.cache)
                    for node in ['A', 'B']:
                        require(cache_hash(torch, free[node]['state'].cache) == free_hashes[node],
                            'later free KV stages changed a previously completed sender cache')
                        require(cache_hash(torch, text[node]['state'].cache) == original_text_hashes[node],
                            'later stages changed an original text sender cache')
                    cases['free_kv_three_node_chain']['passed'] = True
                    cases['free_kv_sender_eos_cache_completion'] = {
                        node: check_eos_cache(free[node]) for node in ['A', 'B']}

                    cases['complete_cache_noop'] = {}
                    for node in ['A', 'B']:
                        source = text[node]['state']
                        source_hash = cache_hash(torch, source.cache)
                        calls_before, events_before = forward_count[0], len(audit.events)
                        completed, completion = engine.complete_cache(source.cache, source.history,
                            'cpu:no_op:'+node)
                        require(completed is source.cache and completion['missing_tokens'] == 0,
                            'completed cache should be returned directly with zero missing tokens')
                        require(forward_count[0] == calls_before and len(audit.events) == events_before,
                            'complete_cache unexpectedly forwarded an already completed history')
                        require(cache_hash(torch, completed) == source_hash, 'no-op completion changed cache')
                        cases['complete_cache_noop'][node] = {'passed': True, 'new_model_forwards': 0,
                            'new_audit_events': 0, 'cache_unchanged': True, 'completion': completion}

                    # A two-token gap is invalid; the adapter must reject it before any forward.
                    source = text['B']['state']
                    cropped = copy.deepcopy(source.cache)
                    cropped.crop(int(source.history.shape[-1])-2)
                    require(cache_lengths(cropped) == [int(source.history.shape[-1])-2]*4,
                        'negative case did not crop every layer by exactly two tokens')
                    cropped_hash = cache_hash(torch, cropped)
                    calls_before, events_before = forward_count[0], len(audit.events)
                    try:
                        engine.complete_cache(cropped, source.history, 'cpu:reject_missing_two')
                    except AssertionError as exc:
                        require('unexpected unmaterialized history length: 2' in str(exc),
                            'two-token gap failed for an unrelated reason')
                        rejection = str(exc)
                    else:
                        raise AssertionError('complete_cache accepted an invalid two-token gap')
                    require(forward_count[0] == calls_before and len(audit.events) == events_before,
                        'invalid two-token gap triggered a model forward')
                    require(cache_hash(torch, cropped) == cropped_hash, 'rejection mutated cropped cache')
                    require(cache_hash(torch, source.cache) == original_text_hashes['B'],
                        'cropping a clone changed the original source cache')
                    cases['complete_cache_reject_missing_two'] = {'passed': True,
                        'missing_tokens': 2, 'error': rejection, 'new_model_forwards': 0,
                        'new_audit_events': 0, 'cropped_cache_unchanged': True,
                        'original_source_cache_unchanged': True}

                    cases['two_edge_payload_and_repeated_transmission'] = {}
                    for kind, chain in [('text', text), ('kv', free)]:
                        edges = [edge_payload(torch, tokenizer, chain[node]['state'], kind,
                            'tiny-random-qwen-cpu') for node in ['A', 'B']]
                        transmission_counts = {}
                        per_token = 4 if kind == 'text' else (
                            2*config.num_hidden_layers*config.num_key_value_heads*
                            (config.hidden_size//config.num_attention_heads)*4)
                        for edge, node in zip(edges, ['A', 'B']):
                            require(edge['content_bytes'] == int(chain[node]['state'].history.shape[-1])*per_token,
                                'payload content bytes do not cover the exact full history')
                            require(sum(part['content_bytes'] for part in edge['parts']) == edge['content_bytes'],
                                'payload part bytes do not sum to content bytes')
                            require(len({part['id'] for part in edge['parts']}) == len(edge['parts']),
                                'payload duplicated an event within one edge')
                            for part in edge['parts']:
                                transmission_counts[part['id']] = transmission_counts.get(part['id'], 0)+1
                                require(part['content_bytes'] == (part['end']-part['start'])*per_token,
                                    'source event byte count differs from token span')
                        repeated = repeated_transmissions(edges)
                        require(repeated['repeated_source_event_bytes'] == edges[0]['content_bytes'],
                            'two-edge repeated bytes must equal the A history forwarded one extra time')
                        require(repeated['A_history_bytes_in_B_to_C'] == edges[0]['content_bytes'],
                            'B-to-C payload does not contain A history exactly once')
                        require(all(transmission_counts[part['id']] == 2 for part in edges[0]['parts']),
                            'A source events were not transmitted exactly twice across both edges')
                        require(all(transmission_counts[part['id']] == 1 for part in edges[1]['parts']
                            if part['node'] == 'B'), 'B source events were transmitted more than once')
                        cases['two_edge_payload_and_repeated_transmission'][kind] = {'passed': True,
                            'content_bytes_per_token': per_token, 'edges': edges,
                            'repeated_transmissions': repeated, 'source_event_transmission_counts': transmission_counts,
                            'repeated_bytes_equal_one_A_history': True,
                            'within_edge_ledger_duplicate_count': 0,
                            'text_int32_four_bytes_per_token': kind == 'text'}
                    require(all(event['status'] == 'ok' for event in audit.events),
                        'an audited operation failed')
                    report['counts'] = {'model_forwards': forward_count[0],
                        'generate': sum(event['kind'] == 'generate' for event in audit.events),
                        'cache_tail_forward': sum(event['kind'] == 'cache_tail_forward' for event in audit.events)}
                    require(report['counts']['generate'] == 8, 'unexpected generate call count')
                    require(report['counts']['cache_tail_forward'] == 5, 'unexpected EOS tail-fill count')
                    report['audit_events'] = copy.deepcopy(audit.events)
                    report['passed'] = True
            finally:
                for handle in audit.handles:
                    handle.remove()
    finally:
        counter_handle.remove()


def main():
    report = {'passed': False, 'purpose': 'tiny random FP32 CPU three-node cache diagnostic',
        'tiny_random_cpu_only': True, 'real_7b_claim': False, 'network_transfer_measured': False,
        'script_sha256': sha(__file__), 'chain_core_sha256': sha(ROOT/'chain_core.py'),
        'utc_started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'cases': {}}
    try:
        run_checks(report)
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['utc_completed'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write_report(report)
        print(json.dumps({'three_node_cpu_check_passed': report['passed'],
            'report': str(REPORT), 'real_7b_claim': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
