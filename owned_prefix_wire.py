"""Real byte packets for exact, source-bound owned-prefix reuse.

Revision/qid are caller-attested identities. Hashes bind bytes and source ledgers;
they do not authenticate the caller or prove which model produced a cache.
Only complete histories starting at global position zero are supported. The
caller must supply an audited full-history State; these hashes do not prove
actual RoPE positions, which are verified by the separate ChainEngine audit.
Importing this module preserves the caller's device environment.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
FORMAT = 'owned-prefix-wire-v1'
MAX_HEADER = 1024*1024
MAX_BODY = 2*1024**3
MAX_TOKENS = 2**31-1
MAX_LAYERS = 512
DTYPE_BYTES = {'float16': 2, 'bfloat16': 2, 'float32': 4, 'float64': 8}


def _must(value, message):
    if not value:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _integer(value, lower, upper, label):
    _must(type(value) is int and lower <= value <= upper, 'invalid/overflow '+label)
    return value


def _digest(value, label):
    _must(isinstance(value, str) and len(value) == 64 and
        all(character in '0123456789abcdef' for character in value), 'invalid '+label)


def _identity(revision, qid):
    _must(sys.byteorder == 'little', 'wire protocol requires a little-endian host')
    for name, value in [('revision', revision), ('qid', qid)]:
        _must(isinstance(value, str) and 0 < len(value.encode('utf-8')) <= 4096,
            'invalid '+name)


def _tensor_bytes(torch, tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _history(torch, state):
    value = state.history
    _must(value.dtype == torch.int64 and value.ndim == 2 and value.shape[0] == 1,
        'history must be batch-one int64 tokens')
    n = _integer(int(value.shape[-1]), 1, MAX_TOKENS, 'history length')
    _must(n*4 <= MAX_BODY, 'token byte overflow')
    _must(int(value.min()) >= 0 and int(value.max()) <= MAX_TOKENS, 'int32 token overflow')
    return n


def _validate_ledger(parts, n, qid, history=None):
    from alignment_probe import tensor_hash

    _must(isinstance(parts, list) and 0 < len(parts) <= 100000, 'invalid source ledger')
    cursor, ids = 0, set()
    required = {'id', 'node', 'kind', 'start', 'end', 'tokens_sha256'}
    for part in parts:
        _must(isinstance(part, dict) and required <= set(part), 'invalid ledger event')
        _must(isinstance(part['id'], str) and part['id'].startswith(qid+':') and
            part['id'] not in ids, 'wrong qid or duplicate source event')
        _must(all(isinstance(part[key], str) and part[key] for key in ['node', 'kind']),
            'invalid source event identity')
        _must(part['id'].endswith(':'+part['node']+':'+part['kind']),
            'source event id/node/kind mismatch')
        start = _integer(part['start'], 0, n, 'ledger start')
        end = _integer(part['end'], 1, n, 'ledger end')
        _must(start == cursor and end > start, 'bad ledger boundaries')
        _digest(part['tokens_sha256'], 'source token hash')
        if history is not None:
            _must(tensor_hash(history[:, start:end]) == part['tokens_sha256'],
                'source token hash mismatch')
        cursor = end
        ids.add(part['id'])
    _must(cursor == n, 'ledger does not cover target history')
    _json(parts)  # Additional source metadata is preserved and included in hashes.


def _cache_info(torch, state, n):
    from chain_core import cache_hash, cache_lengths

    try:
        lengths = cache_lengths(state.cache)
    except (AssertionError, AttributeError, TypeError) as exc:
        raise ValueError('invalid full cache') from exc
    _must(0 < len(lengths) <= MAX_LAYERS and lengths == [n]*len(lengths),
        'cache/history length mismatch')
    descriptors, raws = [], []
    total = 0
    for layer, (key, value) in enumerate(zip(state.cache.key_cache, state.cache.value_cache)):
        _must(key.shape == value.shape, 'K/V shape mismatch')
        entries = {}
        for name, tensor in [('key', key), ('value', value)]:
            dtype = str(tensor.dtype).removeprefix('torch.')
            shape = list(tensor.shape)
            _must(dtype in DTYPE_BYTES and len(shape) == 4 and shape[0] == 1 and
                shape[2] == n and all(type(dim) is int and dim > 0 for dim in shape),
                'unsupported cache dtype/shape')
            size = math.prod(shape)*DTYPE_BYTES[dtype]
            total += size
            _must(total <= MAX_BODY, 'cache byte overflow')
            raw = _tensor_bytes(torch, tensor)
            _must(len(raw) == size, 'cache byte size mismatch')
            entries[name] = {'shape': shape, 'dtype': dtype, 'nbytes': size}
            raws.append(raw)
        descriptors.append(entries)
    descriptor_bytes = _json(descriptors)
    byte_hash = hashlib.sha256()
    enhanced = hashlib.sha256(descriptor_bytes+b'\x00')
    for raw in raws:
        byte_hash.update(raw)
        enhanced.update(raw)
    return {'cache_descriptors': descriptors, 'cache_byte_sha256': byte_hash.hexdigest(),
        'cache_descriptor_sha256': _sha(descriptor_bytes), 'cache_sha256': enhanced.hexdigest(),
        'chain_cache_sha256': cache_hash(torch, state.cache)}


def _snapshot(torch, state, revision, qid, kv):
    from alignment_probe import tensor_hash

    _identity(revision, qid)
    n = _history(torch, state)
    parts = copy.deepcopy(state.segments)
    _validate_ledger(parts, n, qid, state.history)
    result = {'length': n, 'position_base': 0, 'position_extent': n,
        'history_sha256': tensor_hash(state.history),
        'segments': parts, 'ledger_sha256': _sha(_json(parts)),
        'origin': _sha(_json({'revision': revision, 'qid': qid, 'root_event': parts[0]}))}
    if kv:
        result.update(_cache_info(torch, state, n))
    return result


def _owner_reference(snapshot, kv):
    fields = ['length', 'position_base', 'position_extent', 'history_sha256', 'ledger_sha256', 'origin']
    if kv:
        fields += ['cache_byte_sha256', 'cache_descriptor_sha256', 'cache_sha256', 'chain_cache_sha256']
    return {field: snapshot[field] for field in fields}


def _same_source_prefix(torch, target, owned, target_info, owner_info, kv):
    n = owner_info['length']
    _must(n <= target_info['length'], 'owned prefix exceeds target history')
    _must(target_info['origin'] == owner_info['origin'], 'wrong source origin')
    _must(torch.equal(target.history[:, :n], owned.history), 'owned token prefix mismatch')
    _must(target_info['segments'][:len(owner_info['segments'])] == owner_info['segments'],
        'owned ledger/source prefix mismatch')
    _must(target_info['segments'][len(owner_info['segments'])-1]['end'] == n,
        'owner is not at a source event boundary')
    if kv:
        left, right = target_info['cache_descriptors'], owner_info['cache_descriptors']
        _must(len(left) == len(right), 'owner layer count mismatch')
        for layer in range(len(left)):
            for name, field in [('key', 'key_cache'), ('value', 'value_cache')]:
                a, b = left[layer][name], right[layer][name]
                _must(a['dtype'] == b['dtype'] and
                    [a['shape'][index] for index in [0, 1, 3]] ==
                    [b['shape'][index] for index in [0, 1, 3]], 'owner cache shape/dtype mismatch')
                prefix = getattr(target.cache, field)[layer][:, :, :n, :]
                owned_tensor = getattr(owned.cache, field)[layer]
                _must(torch.equal(prefix, owned_tensor) and
                    _tensor_bytes(torch, prefix) == _tensor_bytes(torch, owned_tensor),
                    'owned KV prefix byte mismatch')


def _encode(torch, state, owned_state, revision, qid, kv):
    target = _snapshot(torch, state, revision, qid, kv)
    owner = None if owned_state is None else _snapshot(torch, owned_state, revision, qid, kv)
    if owner is not None:
        _same_source_prefix(torch, state, owned_state, target, owner, kv)
    base = 0 if owner is None else owner['length']
    count = target['length']-base
    token_raw = _tensor_bytes(torch, state.history[:, base:].to(dtype=torch.int32))
    blocks, cursor = [token_raw], len(token_raw)
    header = {'format': FORMAT, 'kind': 'kv' if kv else 'text',
        'mode': 'full' if owner is None else 'delta', 'revision': revision, 'qid': qid,
        **{key: target[key] for key in ['length', 'position_base', 'position_extent',
            'history_sha256', 'segments', 'ledger_sha256', 'origin']},
        'owner': None if owner is None else _owner_reference(owner, kv),
        'tokens': {'dtype': 'int32_le', 'count': count, 'offset': 0, 'nbytes': cursor}}
    if kv:
        header.update({key: target[key] for key in ['cache_byte_sha256',
            'cache_descriptor_sha256', 'cache_sha256', 'chain_cache_sha256']})
        header['cache'] = []
        for layer, descriptors in enumerate(target['cache_descriptors']):
            entries = {}
            for name, field in [('key', 'key_cache'), ('value', 'value_cache')]:
                descriptor = descriptors[name]
                raw = _tensor_bytes(torch, getattr(state.cache, field)[layer][:, :, base:, :])
                payload_shape = list(descriptor['shape'])
                payload_shape[2] = count
                entries[name] = {**descriptor, 'payload_shape': payload_shape,
                    'offset': cursor, 'payload_nbytes': len(raw)}
                blocks.append(raw)
                cursor += len(raw)
            header['cache'].append(entries)
    _must(cursor <= MAX_BODY, 'packet payload overflow')
    header['body_nbytes'] = cursor
    raw_header = _json(header)
    _must(len(raw_header) <= MAX_HEADER, 'header overflow')
    return struct.pack('<I', len(raw_header))+raw_header+b''.join(blocks)


def full_packet(torch, state, revision, qid):
    return _encode(torch, state, None, revision, qid, True)


def delta_packet(torch, state, owned_state, revision, qid):
    _must(owned_state is not None, 'missing owner')
    return _encode(torch, state, owned_state, revision, qid, True)


def full_text_packet(torch, state, revision, qid):
    return _encode(torch, state, None, revision, qid, False)


def delta_text_packet(torch, state, owned_state, revision, qid):
    _must(owned_state is not None, 'missing owner')
    return _encode(torch, state, owned_state, revision, qid, False)


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        _must(key not in result, 'duplicate JSON header key')
        result[key] = value
    return result


def _read(packet, revision, qid, kv):
    _identity(revision, qid)
    _must(isinstance(packet, bytes) and 4 <= len(packet) <= MAX_BODY+MAX_HEADER+4,
        'invalid/overflow packet size')
    size = struct.unpack('<I', packet[:4])[0]
    _must(0 < size <= MAX_HEADER and size <= len(packet)-4, 'invalid/overflow header length')
    raw_header = packet[4:4+size]
    try:
        header = json.loads(raw_header.decode('utf-8'), object_pairs_hook=_no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('invalid JSON constant')))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('invalid UTF8 JSON header') from exc
    common = {'format', 'kind', 'mode', 'revision', 'qid', 'length', 'position_base',
        'position_extent', 'history_sha256',
        'segments', 'ledger_sha256', 'origin', 'owner', 'tokens', 'body_nbytes'}
    expected = common | ({'cache', 'cache_byte_sha256', 'cache_descriptor_sha256',
        'cache_sha256', 'chain_cache_sha256'} if kv else set())
    _must(isinstance(header, dict) and set(header) == expected, 'invalid header fields')
    _must(_json(header) == raw_header, 'noncanonical header bytes')
    _must(header['format'] == FORMAT and header['kind'] == ('kv' if kv else 'text'),
        'wrong packet format/kind')
    _must(header['revision'] == revision and header['qid'] == qid, 'wrong revision/qid')
    _must(header['mode'] in ['full', 'delta'], 'wrong packet mode')
    n = _integer(header['length'], 1, MAX_TOKENS, 'target length')
    _integer(header['position_base'], 0, 0, 'position_base (only zero supported)')
    _must(_integer(header['position_extent'], 1, MAX_TOKENS, 'position_extent') == n,
        'position extent must equal complete history length')
    _must(n*4 <= MAX_BODY, 'target token overflow')
    _validate_ledger(header['segments'], n, qid)
    for key in ['history_sha256', 'ledger_sha256', 'origin']:
        _digest(header[key], key)
    _must(header['ledger_sha256'] == _sha(_json(header['segments'])), 'ledger hash mismatch')
    _must(header['origin'] == _sha(_json({'revision': revision, 'qid': qid,
        'root_event': header['segments'][0]})), 'source origin hash mismatch')
    body = memoryview(packet)[4+size:]
    body_nbytes = _integer(header['body_nbytes'], 0, MAX_BODY, 'body length')
    _must(len(body) == body_nbytes, 'truncated payload or trailing bytes')
    return header, body


def _plan(header, body, kv):
    n, owner = header['length'], header['owner']
    if header['mode'] == 'full':
        _must(owner is None, 'full packet cannot reference an owner')
        base = 0
    else:
        fields = {'length', 'position_base', 'position_extent', 'history_sha256', 'ledger_sha256', 'origin'}
        if kv:
            fields |= {'cache_byte_sha256', 'cache_descriptor_sha256', 'cache_sha256', 'chain_cache_sha256'}
        _must(isinstance(owner, dict) and set(owner) == fields, 'invalid/missing owner reference')
        base = _integer(owner['length'], 1, n, 'owner length')
        _integer(owner['position_base'], 0, 0, 'owner position_base (only zero supported)')
        _must(_integer(owner['position_extent'], 1, n, 'owner position_extent') == base,
            'owner position extent must equal owned history length')
        for key in fields-{'length', 'position_base', 'position_extent'}:
            _digest(owner[key], 'owner '+key)
        _must(owner['origin'] == header['origin'], 'wrong owner origin')
        _must(base in [part['end'] for part in header['segments']], 'bad owner event boundary')
    count = n-base
    tokens = header['tokens']
    _must(isinstance(tokens, dict) and set(tokens) == {'dtype', 'count', 'offset', 'nbytes'},
        'invalid token payload descriptor')
    _must(tokens['dtype'] == 'int32_le', 'wrong token wire dtype')
    _must(_integer(tokens['count'], 0, n, 'token count') == count and
        _integer(tokens['offset'], 0, MAX_BODY, 'token offset') == 0 and
        _integer(tokens['nbytes'], 0, MAX_BODY, 'token bytes') == count*4,
        'bad token payload boundaries')
    cursor, descriptors = count*4, []
    if kv:
        cache = header['cache']
        _must(isinstance(cache, list) and 0 < len(cache) <= MAX_LAYERS, 'invalid cache layer count')
        target_bytes = 0
        for entries in cache:
            _must(isinstance(entries, dict) and set(entries) == {'key', 'value'}, 'invalid layer descriptor')
            desc = {}
            for name in ['key', 'value']:
                entry = entries[name]
                _must(isinstance(entry, dict) and set(entry) == {'shape', 'dtype', 'nbytes',
                    'payload_shape', 'offset', 'payload_nbytes'}, 'invalid tensor descriptor')
                dtype, shape = entry['dtype'], entry['shape']
                _must(isinstance(dtype, str) and dtype in DTYPE_BYTES and
                    isinstance(shape, list) and len(shape) == 4, 'unsupported cache dtype/shape')
                for dim in shape:
                    _integer(dim, 1, MAX_BODY, 'cache dimension')
                _must(shape[0] == 1 and shape[2] == n, 'bad target cache boundaries')
                size = math.prod(shape)*DTYPE_BYTES[dtype]
                target_bytes += size
                _must(target_bytes <= MAX_BODY and
                    _integer(entry['nbytes'], 0, MAX_BODY, 'target cache bytes') == size,
                    'target cache byte overflow/mismatch')
                payload_shape = list(shape)
                payload_shape[2] = count
                _must(isinstance(entry['payload_shape'], list) and
                    all(type(dim) is int for dim in entry['payload_shape']) and
                    entry['payload_shape'] == payload_shape, 'bad cache suffix shape')
                payload_size = math.prod(payload_shape)*DTYPE_BYTES[dtype]
                _must(_integer(entry['offset'], 0, MAX_BODY, 'cache offset') == cursor and
                    _integer(entry['payload_nbytes'], 0, MAX_BODY, 'cache payload bytes') == payload_size,
                    'bad cache payload boundaries')
                cursor += payload_size
                _must(cursor <= len(body), 'cache payload overflow/truncation')
                desc[name] = {'shape': shape, 'dtype': dtype, 'nbytes': size}
            _must(desc['key']['shape'] == desc['value']['shape'], 'K/V target shape mismatch')
            descriptors.append(desc)
        for key in ['cache_byte_sha256', 'cache_descriptor_sha256', 'cache_sha256', 'chain_cache_sha256']:
            _digest(header[key], key)
        _must(_sha(_json(descriptors)) == header['cache_descriptor_sha256'],
            'cache descriptor hash mismatch')
    _must(cursor == len(body) == header['body_nbytes'], 'trailing bytes or incomplete payload boundaries')
    return base, count


def _decode(torch, packet, revision, qid, owned_state, device, kv):
    from chain_core import State

    header, body = _read(packet, revision, qid, kv)
    base, count = _plan(header, body, kv)  # Validate every size/interval before allocating tensors.
    owner_info = None
    if header['mode'] == 'delta':
        _must(owned_state is not None, 'missing owner state')
        owner_info = _snapshot(torch, owned_state, revision, qid, kv)
        _must(_owner_reference(owner_info, kv) == header['owner'], 'owner hash/identity mismatch')
        _must(header['segments'][:len(owner_info['segments'])] == owner_info['segments'],
            'owner ledger/source prefix mismatch')
        if kv:
            _must(len(header['cache']) == len(owner_info['cache_descriptors']), 'owner layer mismatch')
            for target, owner in zip(header['cache'], owner_info['cache_descriptors']):
                for name in ['key', 'value']:
                    _must(target[name]['dtype'] == owner[name]['dtype'] and
                        [target[name]['shape'][i] for i in [0, 1, 3]] ==
                        [owner[name]['shape'][i] for i in [0, 1, 3]], 'owner cache metadata mismatch')
    if count:
        suffix = torch.frombuffer(bytearray(body[:count*4]), dtype=torch.int32).clone()
        _must(int(suffix.min()) >= 0, 'negative wire token/overflow')
        suffix = suffix.to(device=device, dtype=torch.int64).reshape(1, count)
    else:
        suffix = torch.empty((1, 0), dtype=torch.int64, device=device)
    history = suffix if base == 0 else torch.cat([owned_state.history.to(device=device), suffix], dim=-1)
    cache = None
    if kv:
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for layer, entries in enumerate(header['cache']):
            tensors = {}
            for name, field in [('key', 'key_cache'), ('value', 'value_cache')]:
                entry = entries[name]
                start, size = entry['offset'], entry['payload_nbytes']
                dtype = getattr(torch, entry['dtype'])
                if size:
                    tail = torch.frombuffer(bytearray(body[start:start+size]), dtype=torch.uint8).clone()
                    tail = tail.view(dtype).reshape(entry['payload_shape']).to(device=device)
                else:
                    tail = torch.empty(entry['payload_shape'], dtype=dtype, device=device)
                tensors[name] = tail if base == 0 else torch.cat([
                    getattr(owned_state.cache, field)[layer].to(device=device), tail], dim=-2)
            cache.update(tensors['key'], tensors['value'], layer)
    state = State(history, copy.deepcopy(header['segments']), cache)
    restored = _snapshot(torch, state, revision, qid, kv)
    fields = ['length', 'position_base', 'position_extent', 'history_sha256', 'ledger_sha256', 'origin', 'segments']
    if kv:
        fields += ['cache_byte_sha256', 'cache_descriptor_sha256', 'cache_sha256', 'chain_cache_sha256']
    _must(all(restored[key] == header[key] for key in fields), 'target history/cache hash mismatch')
    return state


def decode_packet(torch, packet, revision, qid, owned_state=None, device='cpu'):
    return _decode(torch, packet, revision, qid, owned_state, device, True)


def decode_text_packet(torch, packet, revision, qid, owned_state=None, device='cpu'):
    return _decode(torch, packet, revision, qid, owned_state, device, False)


def _rewrite_header(packet, change):
    size = struct.unpack('<I', packet[:4])[0]
    header = json.loads(packet[4:4+size].decode('utf-8'))
    change(header)
    raw = _json(header)
    return struct.pack('<I', len(raw))+raw+packet[4+size:]


def _cpu_cases(report):
    import torch
    import transformers
    from transformers.cache_utils import DynamicCache
    from chain_core import State, add_segment, cache_hash

    report.update(torch=torch.__version__, transformers=transformers.__version__, python=sys.version,
        device='cpu', model_inference_performed=False)
    revision, qid = 'synthetic-cache-revision', 'wire_cpu'
    history_a = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    history_ab = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    segments_a = add_segment([], qid, 'synthetic', 'A', 'prompt', history_a, 0)
    segments_ab = add_segment(segments_a, qid, 'synthetic', 'B', 'prompt', history_ab[:, 3:], 3)
    cases, rejected, accounting = [], [], {}
    for dtype in [torch.bfloat16, torch.float32]:
        a, ab = DynamicCache(), DynamicCache()
        for layer in range(2):
            key = (torch.arange(20, dtype=torch.float32).reshape(1, 2, 5, 2)+layer*32).to(dtype)
            value = (key.float()+64).to(dtype)
            a.update(key[:, :, :3, :].clone(), value[:, :, :3, :].clone(), layer)
            ab.update(key.clone(), value.clone(), layer)
        owner, target = State(history_a.clone(), copy.deepcopy(segments_a), a), State(
            history_ab.clone(), copy.deepcopy(segments_ab), ab)
        source_hash = cache_hash(torch, owner.cache)
        full_a = full_packet(torch, owner, revision, qid)
        full = full_packet(torch, target, revision, qid)
        delta = delta_packet(torch, target, owner, revision, qid)
        received_owner = decode_packet(torch, full_a, revision, qid)
        restored_full = decode_packet(torch, full, revision, qid)
        restored_delta = decode_packet(torch, delta, revision, qid, received_owner)
        _must(full_packet(torch, restored_full, revision, qid) == full ==
            full_packet(torch, restored_delta, revision, qid), 'KV roundtrip packet bytes differ')
        _must(cache_hash(torch, owner.cache) == source_hash, 'wire codec mutated owner cache')
        full_a_text = full_text_packet(torch, owner, revision, qid)
        full_text = full_text_packet(torch, target, revision, qid)
        delta_text = delta_text_packet(torch, target, owner, revision, qid)
        text_full = decode_text_packet(torch, full_text, revision, qid)
        text_owner = decode_text_packet(torch, full_a_text, revision, qid)
        text_delta = decode_text_packet(torch, delta_text, revision, qid, text_owner)
        _must(full_text_packet(torch, text_full, revision, qid) == full_text ==
            full_text_packet(torch, text_delta, revision, qid), 'text roundtrip packet bytes differ')
        _must(torch.equal(restored_delta.history, text_delta.history), 'KV/text target histories differ')

        zero_kv = delta_packet(torch, owner, owner, revision, qid)
        zero_text = delta_text_packet(torch, owner, text_owner, revision, qid)
        zero_restored = decode_packet(torch, zero_kv, revision, qid, received_owner)
        zero_text_restored = decode_text_packet(torch, zero_text, revision, qid, text_owner)
        _must(full_packet(torch, zero_restored, revision, qid) == full_a and
            full_text_packet(torch, zero_text_restored, revision, qid) == full_a_text,
            'zero suffix full/delta roundtrip bytes differ')
        _must(zero_restored.history.data_ptr() != received_owner.history.data_ptr() and
            zero_text_restored.history.data_ptr() != text_owner.history.data_ptr(),
            'zero suffix decode aliases the owner history')
        _must(all(restored.data_ptr() != received.data_ptr()
            for field in ['key_cache', 'value_cache']
            for restored, received in zip(getattr(zero_restored.cache, field),
                getattr(received_owner.cache, field))), 'zero suffix decode aliases the owner cache')
        sizes = {}
        for name, packet in [('kv_full_A', full_a), ('kv_full', full), ('kv_delta', delta),
            ('text_full_A', full_a_text), ('text_full', full_text), ('text_delta', delta_text),
            ('kv_zero_suffix_delta', zero_kv), ('text_zero_suffix_delta', zero_text)]:
            header_bytes = struct.unpack('<I', packet[:4])[0]
            header = json.loads(packet[4:4+header_bytes].decode('utf-8'))
            _must(len(packet) == 4+header_bytes+header['body_nbytes'], 'wire byte accounting mismatch')
            sizes[name] = {'packet_bytes': len(packet), 'header_prefix_bytes': 4,
                'utf8_json_header_bytes': header_bytes, 'payload_bytes': header['body_nbytes'],
                'token_payload_bytes': header['tokens']['nbytes']}
        _must(sizes['text_full']['payload_bytes'] == 5*4 and
            sizes['text_delta']['payload_bytes'] == 2*4, 'text token packet byte count differs')
        _must(sizes['kv_full']['payload_bytes'] == 5*4+2*2*1*2*5*2*DTYPE_BYTES[str(dtype)[6:]] and
            sizes['kv_delta']['payload_bytes'] == 2*4+2*2*1*2*2*2*DTYPE_BYTES[str(dtype)[6:]],
            'KV payload byte count differs')
        _must(sizes['kv_zero_suffix_delta']['payload_bytes'] ==
            sizes['text_zero_suffix_delta']['payload_bytes'] == 0, 'zero suffix payload must be empty')
        three_edge_totals = {}
        for kind, packet_a, packet_ab, packet_delta in [('kv', full_a, full, delta),
            ('text', full_a_text, full_text, delta_text)]:
            linear_full = len(packet_a)+len(packet_ab)
            triangle_full = 2*len(packet_a)+len(packet_ab)
            triangle_delta = 2*len(packet_a)+len(packet_delta)
            three_edge_totals[kind] = {'A_full_packet_bytes': len(packet_a),
                'AB_full_packet_bytes': len(packet_ab), 'AB_delta_packet_bytes': len(packet_delta),
                'linear_full_packet_bytes': linear_full, 'triangle_full_packet_bytes': triangle_full,
                'triangle_delta_packet_bytes': triangle_delta,
                'triangle_delta_minus_triangle_full_bytes': triangle_delta-triangle_full,
                'triangle_delta_minus_linear_full_bytes': triangle_delta-linear_full,
                'net_savings_vs_triangle_full_bytes': triangle_full-triangle_delta,
                'net_savings_vs_linear_full_bytes': linear_full-triangle_delta,
                'includes_all_header_reference_bytes': True}
        owner_token_bytes = owner.history.numel()*owner.history.element_size()
        owner_kv_bytes = sum(tensor.numel()*tensor.element_size()
            for field in ['key_cache', 'value_cache'] for tensor in getattr(owner.cache, field))
        accounting[str(dtype)] = {'packets': sizes, 'topology_totals': three_edge_totals,
            'owner_raw_resident_tensor_bytes': {'int64_history': owner_token_bytes,
                'kv': owner_kv_bytes, 'total': owner_token_bytes+owner_kv_bytes,
                'is_total_process_or_allocator_memory': False}}

        wrong = copy.deepcopy(owner)
        wrong.cache.key_cache[0][0, 0, 0, 0] += 1
        changed_target = copy.deepcopy(target)
        changed_target.cache.value_cache[1][0, 0, 1, 0] += 1
        changed_ledger = copy.deepcopy(owner)
        changed_ledger.segments[0]['id'] = qid+':other:A:prompt'
        invalid_token = copy.deepcopy(target)
        invalid_token.history[0, 4] = 2**31
        invalid_token.segments[-1]['tokens_sha256'] = '0'*64
        attempts = [
            ('missing_owner', lambda: decode_packet(torch, delta, revision, qid)),
            ('wrong_owner_cache', lambda: decode_packet(torch, delta, revision, qid, wrong)),
            ('changed_prefix_encoder', lambda: delta_packet(torch, changed_target, owner, revision, qid)),
            ('wrong_owner_origin', lambda: decode_packet(torch, delta, revision, qid, changed_ledger)),
            ('wrong_revision', lambda: decode_packet(torch, full, 'wrong', qid)),
            ('wrong_qid', lambda: decode_packet(torch, full, revision, 'wrong')),
            ('trailing_byte', lambda: decode_packet(torch, full+b'\x00', revision, qid)),
            ('truncated_payload', lambda: decode_packet(torch, full[:-1], revision, qid)),
            ('bad_token_boundary', lambda: decode_packet(torch, _rewrite_header(full,
                lambda h: h['tokens'].update(offset=1)), revision, qid)),
            ('bad_cache_boundary', lambda: decode_packet(torch, _rewrite_header(full,
                lambda h: h['cache'][0]['key'].update(offset=0)), revision, qid)),
            ('target_hash', lambda: decode_packet(torch, _rewrite_header(full,
                lambda h: h.update(cache_sha256='0'*64)), revision, qid)),
            ('shape_overflow', lambda: decode_packet(torch, _rewrite_header(full,
                lambda h: h['cache'][0]['key'].update(shape=[1, 2**63, 5, 2])), revision, qid)),
            ('int32_token_overflow', lambda: full_packet(torch, invalid_token, revision, qid)),
            ('text_missing_owner', lambda: decode_text_packet(torch, delta_text, revision, qid)),
            ('text_wrong_origin', lambda: decode_text_packet(torch, delta_text, revision, qid, changed_ledger)),
            ('text_trailing_byte', lambda: decode_text_packet(torch, full_text+b'x', revision, qid)),
            ('zero_suffix_trailing_byte', lambda: decode_packet(torch, zero_kv+b'x', revision, qid, owner)),
            ('wrong_position_base', lambda: decode_packet(torch, _rewrite_header(full,
                lambda h: h.update(position_base=1)), revision, qid)),
        ]
        for name, attempt in attempts:
            try:
                attempt()
            except ValueError as exc:
                rejected.append({'dtype': str(dtype), 'case': name, 'rejected': True, 'error': str(exc)})
            else:
                raise AssertionError('wire accepted invalid case: '+name)
        _must(cache_hash(torch, owner.cache) == source_hash, 'rejected packet mutated owner')
        cases.append({'dtype': str(dtype), 'kv_full_delta_restored_packet_bytes_identical': True,
            'text_full_delta_restored_packet_bytes_identical': True, 'owner_cache_unchanged': True,
            'zero_suffix_delta_restored_bytes_identical': True, 'zero_suffix_owner_storage_not_aliased': True,
            'target_chain_cache_sha256': cache_hash(torch, target.cache), 'byte_accounting': sizes})
    report.update(passed=True, cases=cases, rejection_cases=rejected,
        byte_accounting=accounting,
        topology_formulas={'linear_full': 'len(A_full)+len(AB_full)',
            'triangle_full': '2*len(A_full)+len(AB_full)',
            'triangle_delta': '2*len(A_full)+len(AB_delta)'},
        metadata_is_in_packet=True, synthetic_cache_only=True, real_7b_claim=False)


def cpu_check():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    path = ROOT/'checks/owned_prefix_wire_cpu.json'
    report = {'passed': False, 'script_sha256': _sha(Path(__file__).read_bytes()),
        'chain_core_sha256': _sha((ROOT/'chain_core.py').read_bytes()),
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
        print(json.dumps({'owned_prefix_wire_cpu_passed': report['passed'], 'report': str(path)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', required=True)
    parser.parse_args()
    cpu_check()
