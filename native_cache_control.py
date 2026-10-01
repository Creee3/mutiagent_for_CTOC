"""Native HF cache-relay control; --cpu runs only a tiny random FP32 check."""
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


def _positions(record):
    return [row['position_ids'][0] for row in record['backend_forward_trace']
        if row['level'] == 'effective_rotary_position_ids']


def native_cached_node(torch, model, tokenizer, audit, sender_state, prompt, budget, tag):
    """Give generate the global history; HF must forward only its uncached suffix.

    The sender cache already contains every sender-history token, including EOS.
    This bypasses CVCommunicator while preserving the same prefix cache and greedy
    generation policy. Returned sequences retain the full global input prefix.
    """
    from chain_core import cache_hash, cache_lengths
    from alignment_probe import tensor_hash

    _require(prompt.ndim == 2 and prompt.shape[0] == 1 and prompt.shape[-1] > 0,
        'native cached control requires one nonempty receiver prompt')
    _require(sender_state.history.ndim == 2 and sender_state.history.shape[0] == 1,
        'native cached control requires one sender history')
    prefix_len = int(sender_state.history.shape[-1])
    _require(cache_lengths(sender_state.cache) == [prefix_len]*model.config.num_hidden_layers,
        'sender cache must materialize the entire global sender history')
    _require(sender_state.segments and sender_state.segments[-1]['end'] == prefix_len,
        'sender ledger must end at the global prefix length')
    budget = int(budget)
    _require(budget > 0, 'generation budget must be positive')
    full_input = torch.cat([sender_state.history, prompt], dim=-1)
    input_len = int(full_input.shape[-1])
    _require(input_len+budget <= model.config.max_position_embeddings,
        'native cached control would exceed the context limit')
    sender_hash = cache_hash(torch, sender_state.cache)
    fresh = copy.deepcopy(sender_state.cache)
    _require(cache_hash(torch, fresh) == sender_hash, 'cache clone differs before native relay')
    args = dict(max_new_tokens=budget, do_sample=False, num_beams=1,
        repetition_penalty=1.0, use_cache=True,
        eos_token_id=copy.deepcopy(model.generation_config.eos_token_id),
        pad_token_id=model.generation_config.pad_token_id,
        return_dict_in_generate=True, output_scores=True, output_logits=True)
    with torch.inference_mode():
        output, record = audit.call('generate', tag,
            lambda: model.generate(input_ids=full_input, past_key_values=fresh,
                attention_mask=torch.ones_like(full_input), **args))
    after_hash = cache_hash(torch, sender_state.cache)
    _require(after_hash == sender_hash, 'native cache relay mutated the sender cache')
    _require(torch.equal(output.sequences[:, :input_len], full_input),
        'native cached sequences lost the global input prefix')
    tail = output.sequences[:, input_len:]
    ids = tail[0].detach().cpu().tolist()
    _require(ids and len(output.logits) == len(output.scores) == len(ids),
        'native cache relay returned inconsistent generation steps')

    forwards = [row for row in record['backend_forward_trace']
        if row['level'] == 'native_model_forward_input']
    expected_positions = [list(range(prefix_len, input_len))]
    expected_positions.extend([[input_len+step-1] for step in range(1, len(ids))])
    _require(len(forwards) == len(ids), 'native cache relay forward count differs from generation steps')
    _require(forwards[0]['input_ids'] == prompt.detach().cpu().tolist(),
        'HF prepare_inputs_for_generation did not forward exactly the uncached receiver prompt')
    _require(forwards[0]['past_length'] == prefix_len,
        'native cache relay first forward did not use the complete sender prefix')
    for step, (forward, positions) in enumerate(zip(forwards, expected_positions)):
        expected_ids = prompt.detach().cpu().tolist() if step == 0 else [[ids[step-1]]]
        _require(forward['input_ids'] == expected_ids,
            'native cache relay re-forwarded global history or skipped a generated token')
        _require(forward['past_length'] == (prefix_len if step == 0 else input_len+step-1),
            'native cache relay past length differs from global history')
        _require(forward['cache_position'] == positions,
            'native cache positions differ from uncached global positions')
        _require(forward['position_ids'] == [positions],
            'native generation position IDs differ from global positions')
        _require(forward['attention_mask_shape'] == [1, input_len+step],
            'native cached attention mask does not cover the full global history')
    _require(_positions(record) == expected_positions,
        'native effective RoPE positions differ from global positions')
    attention = [row for row in record['backend_forward_trace']
        if row['level'] == 'effective_first_layer_attention']
    _require(len(attention) == len(ids), 'native first-layer attention trace is incomplete')
    _require(all(row['cache_position'] == positions and row['position_ids'] == [positions]
        for row, positions in zip(attention, expected_positions)),
        'native effective attention cache/RoPE positions differ from global positions')

    history = torch.cat([full_input, tail], dim=-1)
    lengths = cache_lengths(output.past_key_values)
    _require(lengths[0] in [int(history.shape[-1])-1, int(history.shape[-1])],
        'native generated cache length differs from global history')
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    record.update(route='native_cached', input_prefix_tokens=prefix_len,
        local_prompt_ids=prompt.detach().cpu().tolist(), input_tokens=input_len,
        input_history_sha256=tensor_hash(full_input),
        global_history_ids=history.detach().cpu().tolist(), global_history_sha256=tensor_hash(history),
        sender_cache_hash_before=sender_hash, sender_cache_hash_after=after_hash,
        sender_cache_hash_before_and_after=sender_hash, sender_cache_unchanged=True,
        cache_lengths_before_generation=[prefix_len]*model.config.num_hidden_layers,
        cache_lengths_before_completion=lengths, cache_completion=None,
        output_ids=ids, generated_tokens=len(ids),
        response=tokenizer.decode(ids, skip_special_tokens=True), eos_stopped=ids[-1] in eos,
        actually_truncated=len(ids) >= budget and ids[-1] not in eos,
        reached_token_budget=len(ids) >= budget, max_new_tokens=budget,
        checked_effective_positions=True, checked_suffix_only_forward=True,
        expected_effective_positions=expected_positions,
        sequences_include_global_prefix=True, attention_mask_scope='full_global_history')
    return {'ids': ids, 'output': output, 'record': record}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_report(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class _DummyTokenizer:
    def decode(self, ids, skip_special_tokens=False, **kwargs):
        ids = ids.detach().cpu().tolist() if hasattr(ids, 'detach') else list(ids)
        return ' '.join(str(int(value)) for value in ids
            if not skip_special_tokens or int(value) not in [0, 1, 63])


def _cpu_cases(report):
    import torch
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from models import CVCommunicator
    from chain_core import ChainAudit, ChainEngine, cache_hash, cache_lengths

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
    _require(cv.layer_map == {index: index for index in range(4)}, 'CV layer map is not identity')
    _require(model.device.type == 'cpu' and model.dtype == torch.float32, 'CPU FP32 model required')
    prompt_a = torch.tensor([[1, 3, 5, 7, 9]], dtype=torch.long)
    prompt_b = torch.tensor([[1, 4, 6]], dtype=torch.long)
    tokenizer = _DummyTokenizer()
    report.update(torch=torch.__version__, transformers=transformers.__version__,
        python=sys.version, seed=42, device='cpu', dtype='float32',
        model_config=config.to_dict(), generation_config=generation.to_dict(),
        official_models_sha256=_sha(ROOT/'work/KVComm/models.py'))
    with tempfile.TemporaryDirectory(prefix='native_cache_control_cpu_') as temporary:
        audit = ChainAudit(torch, model, Path(temporary))
        engine = ChainEngine(torch, model, cv, tokenizer, {'A': 4, 'B': 4, 'C': 4}, audit)
        try:
            with torch.inference_mode():
                item = {'id': 'native_cache_control_cpu'}
                sender = engine.node(item, 'sender', 'A', None, prompt_a, outgoing=True)
                state = sender['state']
                _require(sender['ids'][-1] == 63 and sender['record']['cache_completion']['missing_tokens'] == 1,
                    'sender must exercise the one-token EOS cache completion')
                _require(sender['record']['cache_completion']['filled_ids'] == [[63]],
                    'sender cache completion must include EOS')
                source_hash = cache_hash(torch, state.cache)
                report['sender'] = sender['record']
                native = native_cached_node(torch, model, tokenizer, audit, state, prompt_b,
                    4, 'cpu:native_cached:B')
                wrapped = engine.node(item, 'official_cv', 'B', state, prompt_b, use_kv=True)
                report['native_cached'] = native['record']
                report['official_cv'] = wrapped['record']
                _require(native['ids'] == wrapped['ids'], 'native cached and official CV output IDs differ')
                _require(native['record']['global_history_ids'] == wrapped['record']['global_history_ids'],
                    'native cached and official CV global histories differ')
                _require(_positions(native['record']) == _positions(wrapped['record']),
                    'native cached and official CV effective positions differ')
                _require(len(native['output'].logits) == len(wrapped['output'].logits),
                    'native cached and official CV logit counts differ')
                raw_max, processed_max = 0.0, 0.0
                for left, right in zip(native['output'].logits, wrapped['output'].logits):
                    raw_max = max(raw_max, float((left-right).abs().max()))
                    _require(torch.allclose(left, right, atol=1e-5, rtol=1e-5),
                        'native cached and official CV raw logits exceed fixed FP32 tolerance')
                for left, right in zip(native['output'].scores, wrapped['output'].scores):
                    finite_left, finite_right = torch.isfinite(left), torch.isfinite(right)
                    _require(torch.equal(finite_left, finite_right), 'processed finite masks differ')
                    _require(torch.equal(left[~finite_left], right[~finite_right]),
                        'processed nonfinite values differ')
                    if finite_left.any():
                        processed_max = max(processed_max,
                            float((left[finite_left]-right[finite_right]).abs().max()))
                    _require(torch.allclose(left, right, atol=1e-5, rtol=1e-5),
                        'native cached and official CV processed scores exceed fixed FP32 tolerance')
                _require(cache_hash(torch, state.cache) == source_hash,
                    'receiver controls mutated the completed sender cache')
                _require(cache_lengths(state.cache) == [int(state.history.shape[-1])]*4,
                    'sender cache length changed after receiver controls')
                report['comparison'] = {'generated_tokens_identical': True,
                    'global_histories_identical': True, 'effective_positions_identical': True,
                    'raw_logits_max_abs': raw_max, 'processed_scores_finite_max_abs': processed_max,
                    'atol': 1e-5, 'rtol': 1e-5,
                    'sender_cache_sha256_before_and_after': source_hash, 'sender_cache_unchanged': True}
                report['audit_events'] = copy.deepcopy(audit.events)
                report['passed'] = True
        finally:
            for handle in audit.handles:
                handle.remove()


def cpu_check():
    # Set this only for the CPU CLI; importing the reusable helper must preserve
    # the caller's device environment for the separately authorized BF16 control.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    path = ROOT/'checks/native_cache_control_cpu.json'
    report = {'passed': False, 'tiny_random_cpu_only': True, 'real_7b_claim': False,
        'purpose': 'native HF cached suffix versus official CV receiver control',
        'script_sha256': _sha(__file__), 'chain_core_sha256': _sha(ROOT/'chain_core.py'),
        'utc_started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        _cpu_cases(report)
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['utc_completed'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        _write_report(path, report)
        print(json.dumps({'native_cache_control_cpu_passed': report['passed'],
            'report': str(path), 'real_7b_claim': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', required=True)
    parser.parse_args()
    cpu_check()
