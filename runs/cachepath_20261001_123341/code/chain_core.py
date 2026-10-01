"""Full-history three-role adapter built on the official CVCommunicator."""
import copy
import hashlib
import json
from dataclasses import dataclass

from alignment_probe import Audit, tensor_hash

ROLES = {
    'A': 'You are Agent A, an evidence reader. Using only your supplied material, extract the entities and facts relevant to the question. State what the evidence supports in at most three short sentences. Do not invent missing facts.',
    'B': 'You are Agent B, an evidence integrator. The preceding history contains Agent A\'s evidence. Combine it with your own material to identify the link needed to answer the question. Give a concise evidence chain and a candidate answer in at most three short sentences. Do not invent missing facts.',
    'C': 'You are Agent C, an answer verifier. Check the evidence and analyses in the preceding history. Directly answer the question, with no explanation. Do not invent unsupported facts.'}


def split_materials(item):
    """Keep official support sentences, grouping by document without consulting gold."""
    facts = item['supporting_facts']
    assert len(item['support']) == len(facts['title']) == len(facts['sent_id'])
    groups = {}
    for title, sent_id, sentence in zip(facts['title'], facts['sent_id'], item['support']):
        groups.setdefault(title, []).append({'sent_id': int(sent_id), 'sentence': sentence})
    assert len(groups) >= 2, 'pilot requires two evidence documents; do not select by answers'
    titles = list(groups)
    def render(names):
        return '\n'.join(f'[{title}]\n'+'\n'.join(x['sentence'] for x in groups[title]) for title in names)
    return {'A': render(titles[:1]), 'B': render(titles[1:]), 'titles': titles,
        'groups': groups, 'policy': 'first supporting-fact title to A; remaining titles to B'}


def prepare_prompt(tokenizer, model, item, node, materials):
    message = ROLES[node]+'\nQuestion: '+item['prompt_B']
    if node in ['A', 'B']:
        message += '\nYour material:\n'+materials[node]
    return tokenizer.apply_chat_template([{'role': 'user', 'content': message}],
        add_generation_prompt=True, return_tensors='pt').to(model.device)


@dataclass
class State:
    history: object
    segments: list
    cache: object = None


def cache_lengths(cache):
    assert cache is not None and len(cache.key_cache) == len(cache.value_cache)
    lengths = [int(k.shape[-2]) for k in cache.key_cache]
    assert lengths and all(int(v.shape[-2]) == n for v, n in zip(cache.value_cache, lengths))
    assert len(set(lengths)) == 1, 'full-history layers must all have the same length'
    return lengths


def cache_hash(torch, cache):
    digest = hashlib.sha256()
    digest.update(str(cache_lengths(cache)).encode())
    for tensors in [cache.key_cache, cache.value_cache]:
        for value in tensors:
            digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def add_segment(segments, qid, condition, node, kind, tokens, start):
    result = list(segments)+[{'id': f'{qid}:{condition}:{node}:{kind}', 'node': node, 'kind': kind,
        'start': int(start), 'end': int(start+tokens.shape[-1]), 'tokens_sha256': tensor_hash(tokens)}]
    assert len({s['id'] for s in result}) == len(result), 'source event inserted twice'
    end = 0
    for part in result:
        assert part['start'] == end, 'history must be contiguous and included once'
        end = part['end']
    return result


class ChainAudit(Audit):
    def __init__(self, torch, model, run_dir):
        self.on_cuda = model.device.type == 'cuda'
        super().__init__(torch, model, run_dir)

    def sync(self):
        if self.on_cuda:
            self.torch.cuda.synchronize()


class ChainEngine:
    def __init__(self, torch, model, cv, tokenizer, budgets, audit):
        self.torch, self.model, self.cv = torch, model, cv
        self.tokenizer, self.budgets, self.audit = tokenizer, budgets, audit

    def complete_cache(self, cache, full_history, tag):
        lengths = cache_lengths(cache)
        assert len(lengths) == self.model.config.num_hidden_layers
        missing = int(full_history.shape[-1])-lengths[0]
        assert missing in [0, 1], f'unexpected unmaterialized history length: {missing}'
        record = {'length_before': lengths[0], 'expected_length': int(full_history.shape[-1]), 'missing_tokens': missing}
        if missing:
            # Index the global history, never CV's shorter, node-local sequences.
            suffix = full_history[:, lengths[0]:]
            out, trace = self.audit.call('cache_tail_forward', tag,
                lambda: self.model(input_ids=suffix, past_key_values=cache, use_cache=True, return_dict=True))
            cache = out.past_key_values
            record.update(filled_ids=suffix.detach().cpu().tolist(), forward=trace)
            positions = [x['position_ids'] for x in trace['backend_forward_trace']
                if x['level'] == 'effective_rotary_position_ids']
            assert positions == [[[lengths[0]]]], 'tail fill must use the next global position'
        assert cache_lengths(cache) == [int(full_history.shape[-1])]*self.model.config.num_hidden_layers
        return cache, record

    def node(self, item, condition, node, incoming, prompt, use_kv=False, outgoing=False):
        torch = self.torch
        prefix_len = 0 if incoming is None else int(incoming.history.shape[-1])
        prior = prompt[:, :0] if incoming is None else incoming.history
        full_input = torch.cat([prior, prompt], dim=-1)
        assert full_input.shape[-1]+self.budgets[node] <= self.model.config.max_position_embeddings
        assert incoming is None or incoming.segments[-1]['end'] == prefix_len
        args = dict(max_new_tokens=self.budgets[node], do_sample=False, num_beams=1, use_cache=True,
            repetition_penalty=1.0, eos_token_id=copy.deepcopy(self.model.generation_config.eos_token_id),
            pad_token_id=self.model.generation_config.pad_token_id,
            return_dict_in_generate=True, output_scores=True, output_logits=True)
        tag = f'{item["id"]}:{condition}:{node}'
        parent_hash = None
        if use_kv and incoming is not None:
            assert cache_lengths(incoming.cache) == [prefix_len]*self.model.config.num_hidden_layers
            parent_hash = cache_hash(torch, incoming.cache)
            fresh = copy.deepcopy(incoming.cache)
            assert cache_hash(torch, fresh) == parent_hash
            output, record = self.audit.call('generate', tag,
                lambda: self.cv.generate(prompt, attention_mask=torch.ones_like(prompt),
                    out_A_past_key_values=fresh, **args))
            assert cache_hash(torch, incoming.cache) == parent_hash, 'sender cache was mutated'
            local_input = prompt
        else:
            output, record = self.audit.call('generate', tag,
                lambda: self.model.generate(full_input, attention_mask=torch.ones_like(full_input), **args))
            local_input = full_input
        assert torch.equal(output.sequences[:, :local_input.shape[-1]], local_input)
        tail = output.sequences[:, local_input.shape[-1]:]
        ids = tail[0].detach().cpu().tolist()
        assert ids and len(output.logits) == len(output.scores) == len(ids)
        history = torch.cat([full_input, tail], dim=-1)
        segments = [] if incoming is None else incoming.segments
        segments = add_segment(segments, item['id'], condition, node, 'prompt', prompt, prefix_len)
        segments = add_segment(segments, item['id'], condition, node, 'output', tail, full_input.shape[-1])
        assert segments[-1]['end'] == history.shape[-1]
        positions = [x['position_ids'][0] for x in record['backend_forward_trace']
            if x['level'] == 'effective_rotary_position_ids']
        expected_first = list(range(prefix_len if use_kv and incoming is not None else 0, int(full_input.shape[-1])))
        assert len(positions) == len(ids) and positions[0] == expected_first
        assert all(p == [int(full_input.shape[-1])+i-1] for i, p in enumerate(positions[1:], start=1))
        before = cache_lengths(output.past_key_values)
        assert before[0] in [int(history.shape[-1])-1, int(history.shape[-1])]
        cache, completion = (None, None)
        if outgoing:
            cache, completion = self.complete_cache(output.past_key_values, history, tag+':complete')
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        record.update(node=node, condition=condition, route='kv' if use_kv and incoming is not None else 'native',
            input_prefix_tokens=prefix_len, local_prompt_ids=prompt.detach().cpu().tolist(),
            input_history_sha256=tensor_hash(full_input), global_history_ids=history.detach().cpu().tolist(),
            global_history_sha256=tensor_hash(history), segments=segments,
            sender_cache_hash_before_and_after=parent_hash, cache_lengths_before_completion=before,
            cache_completion=completion, output_ids=ids, generated_tokens=len(ids),
            response=self.tokenizer.decode(ids, skip_special_tokens=True), eos_stopped=ids[-1] in eos,
            actually_truncated=len(ids) >= self.budgets[node] and ids[-1] not in eos,
            checked_effective_positions=True)
        return {'ids': ids, 'output': output, 'record': record, 'state': State(history, segments, cache)}


def edge_payload(torch, tokenizer, state, kind, revision):
    n = int(state.history.shape[-1])
    assert state.segments[-1]['end'] == n
    metadata = {'kind': kind, 'length': n, 'segments': state.segments, 'revision': revision,
        'history_sha256': tensor_hash(state.history), 'position_base': 0}
    if kind == 'kv':
        lengths = cache_lengths(state.cache)
        assert lengths == [n]*len(lengths)
        metadata.update(dtype=str(state.cache.key_cache[0].dtype), lengths=lengths)
        total = sum(t.numel()*t.element_size() for tensors in [state.cache.key_cache, state.cache.value_cache] for t in tensors)
        assert total % n == 0
        per_token = total//n
    else:
        # Exact input baseline uses int32 token packets, not re-tokenized joined strings.
        metadata.update(dtype='int32_little_endian', tokenizer_revision=revision)
        total, per_token = n*4, 4
    meta_bytes = len(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    parts = [{**s, 'content_bytes': (s['end']-s['start'])*per_token} for s in state.segments]
    readable = tokenizer.decode(state.history[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {'kind': kind, 'content_bytes': total, 'metadata_bytes': meta_bytes,
        'logical_total_bytes': total+meta_bytes, 'readable_utf8_bytes': len(readable.encode('utf-8')),
        'parts': parts, 'metadata': metadata, 'network_transfer_measured': False}


def repeated_transmissions(edges):
    seen, repeated = set(), 0
    for edge in edges:
        for part in edge['parts']:
            if part['id'] in seen:
                repeated += part['content_bytes']
            seen.add(part['id'])
    total = sum(edge['content_bytes'] for edge in edges)
    inherited = sum(s['content_bytes'] for s in edges[1]['parts'] if s['node'] == 'A')
    return {'repeated_source_event_bytes': repeated, 'total_content_bytes': total,
        'repeated_fraction_across_edges': repeated/total, 'A_history_bytes_in_B_to_C': inherited,
        'A_fraction_of_B_to_C_content': inherited/edges[1]['content_bytes'],
        'interpretation': 'Repeated forwarding across edges; not proven removable semantic redundancy.'}
