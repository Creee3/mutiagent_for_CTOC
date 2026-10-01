"""Prepare calibration-only receiver inputs with the existing offline tokenizer.

No models, cache generation, selection training, or GPU calls. Gold labels and
oracle sentence membership are written separately from selector input views.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def build(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    from transformers import AutoTokenizer, __version__ as transformers_version
    # These imports only define existing helpers. They do not load a model.
    from chain_core import ROLES, split_materials
    from run import sample_text, write

    config_path = ROOT / 'config.json'
    calibration_path = ROOT / 'data/calibration.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    items = json.loads(calibration_path.read_text(encoding='utf-8'))
    assert len(items) == 20 and len({x['id'] for x in items}) == 20
    assert len({x['question_hash'] for x in items}) == 20
    output = (ROOT / args.output_dir).resolve()
    assert output.is_relative_to(ROOT.resolve()), 'output must stay in this experiment'
    assert not output.exists(), 'preserve existing preparation; use a new directory'
    tokenizer = AutoTokenizer.from_pretrained(config['model_7b_path'], local_files_only=True)
    model_path = Path(config['model_7b_path'])
    source_hash = sha(calibration_path)
    material = {item['id']: split_materials(item) for item in items}
    receiver_role = ROLES['C'].replace('Agent C', 'Agent B')

    def encode_chat(content):
        return tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
            add_generation_prompt=True, tokenize=True)

    def receiver_content(item, extra):
        return (receiver_role + '\nQuestion: ' + item['prompt_B']
            + '\nYour material:\n' + material[item['id']]['B']
            + '\nAdditional material:\n' + extra)

    def render_units(title, units):
        return f'[{title}]\n' + '\n'.join(u['sentence'] for u in units)

    def unit_id(item, title, unit):
        return f'{item["id"]}:{title}:{unit["sent_id"]}'

    def control_for(item, target_tokens):
        target = material[item['id']]
        titles = set(target['titles'])
        sentences = set(item['support'])
        donors = [other for other in items
            if other['id'] != item['id'] and other['question_hash'] != item['question_hash']
            and not titles.intersection(material[other['id']]['titles'])
            and not sentences.intersection(other['support'])]
        donors.sort(key=lambda x: hashlib.sha256(
            f'{item["id"]}|{x["id"]}'.encode()).hexdigest())
        assert donors, 'no calibration-only donor candidate'
        donor_text = '\n'.join(other['prompt_A'] for other in donors)
        donor_ids = tokenizer.encode(donor_text, add_special_tokens=False)
        base_tokens = len(encode_chat(receiver_content(item, '')))
        max_prefix = min(len(donor_ids), max(1, target_tokens - base_tokens + 64))
        nearest = None
        for size in range(1, max_prefix + 1):
            text = tokenizer.decode(donor_ids[:size], skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            ids = encode_chat(receiver_content(item, text))
            difference = abs(len(ids) - target_tokens)
            if nearest is None or difference < nearest['difference']:
                nearest = {'text': text, 'ids': ids, 'difference': difference,
                    'donor_prefix_tokens': size}
            if difference == 0:
                break
        nearest.update(donor_order=[x['id'] for x in donors],
            donor_pool_text_sha256=hashlib.sha256(donor_text.encode()).hexdigest(),
            semantic_irrelevance_verified=False,
            can_end_inside_sentence=True)
        return nearest

    senders, records, diagnostics, scoring = [], [], [], []
    partial_undefined = []
    control_matches = 0
    full_prompt_pairs = []
    for item in items:
        groups = material[item['id']]
        title = groups['titles'][0]
        units = groups['groups'][title]
        sender_content = (ROLES['A'] + '\nQuestion: ' + item['prompt_B']
            + '\nYour material:\n' + groups['A'])
        sender_ids = encode_chat(sender_content)
        source_tokens = tokenizer.encode(groups['A'], add_special_tokens=False)
        budget = max(1, math.floor(len(source_tokens) * 0.5))
        sampled = sample_text(groups['A'], tokenizer, budget, config['seed'])
        sampled_ids = tokenizer.encode(sampled, add_special_tokens=False)
        assert 0 < len(sampled_ids) <= budget < len(source_tokens)
        senders.append({'question_id': item['id'], 'question': item['prompt_B'],
            'sender_prompt': sender_content, 'sender_prompt_ids_sha256': ids_hash(sender_ids),
            'sender_prompt_tokens': len(sender_ids), 'sender_material': groups['A'],
            'material_tokens': len(source_tokens), 'output_cache_status': 'not_generated',
            'text_sampling_control': {'budget_tokens': budget, 'sampled_text': sampled,
                'sampled_tokens': len(sampled_ids), 'unchanged': False,
                'scope': 'existing text sampling baseline, not an oracle or new algorithm'}})
        all_ids = [unit_id(item, title, u) for u in units]
        states = [('missing', [], '')]
        if len(units) >= 2:
            held = units[:len(units) // 2]
            states.append(('partial', [unit_id(item, title, u) for u in held],
                render_units(title, held)))
        else:
            partial_undefined.append(item['id'])
        states.append(('full', all_ids, groups['A']))
        for condition, held_ids, extra in states:
            key = f'{item["id"]}:{condition}'
            body = receiver_content(item, extra)
            ids = encode_chat(body)
            records.append({'key': key, 'question_id': item['id'], 'condition': condition,
                'question': item['prompt_B'], 'receiver_prompt': body,
                'receiver_prompt_tokens': len(ids), 'receiver_prompt_ids_sha256': ids_hash(ids),
                'sender_prompt_ids_sha256': ids_hash(sender_ids),
                'inherited_history_provided': False, 'received_message_status': 'not_generated'})
            diagnostics.append({'key': key, 'held_sentence_ids': held_ids,
                'sender_sentence_ids': all_ids, 'held_sentence_fraction': len(held_ids) / len(all_ids),
                'held_material_tokens': len(tokenizer.encode(extra, add_special_tokens=False)),
                'oracle_annotation': True, 'selector_input': False})
            if condition == 'missing':
                continue
            control = control_for(item, len(ids))
            control_key = key + ':other_document_control'
            is_match = control['difference'] == 0
            control_matches += int(is_match)
            records.append({'key': control_key, 'question_id': item['id'],
                'condition': condition + '_other_document_control', 'question': item['prompt_B'],
                'receiver_prompt': receiver_content(item, control['text']),
                'receiver_prompt_tokens': len(control['ids']),
                'receiver_prompt_ids_sha256': ids_hash(control['ids']),
                'sender_prompt_ids_sha256': ids_hash(sender_ids),
                'inherited_history_provided': False, 'received_message_status': 'not_generated'})
            diagnostics.append({'key': control_key, 'paired_key': key,
                'held_sentence_ids': [], 'sender_sentence_ids': all_ids,
                'donor_order': control['donor_order'],
                'donor_pool_text_sha256': control['donor_pool_text_sha256'],
                'donor_prefix_tokens': control['donor_prefix_tokens'],
                'complete_receiver_prompt_token_difference': len(control['ids']) - len(ids),
                'exact_length_match': is_match, 'semantic_irrelevance_verified': False,
                'can_end_inside_sentence': True, 'oracle_annotation': True,
                'selector_input': False})
            full_prompt_pairs.append((key, control_key))
        scoring.append({'question_id': item['id'], 'answer': item['answer']})

    assert len(records) == 72 and len({x['key'] for x in records}) == 72
    assert len(partial_undefined) == 14
    assert all('answer' not in x and 'held_sentence_ids' not in x for x in records)
    assert all('answer' not in x and 'supporting_facts' not in x for x in senders)
    counts = {name: sum(x['condition'] == name for x in records)
        for name in sorted({x['condition'] for x in records})}
    provenance = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'input preparation only; no inference protocol or GPU conditions frozen',
        'dataset': 'frozen HotpotQA distractor validation calibration support sentences',
        'dataset_revision': '1908d6afbbead072334abe2965f91bd2709910ab',
        'calibration_sha256': source_hash, 'ordered_question_ids': [x['id'] for x in items],
        'model_revision_for_tokenizer_only': config['model_7b_revision'],
        'tokenizer_files_sha256': {name: sha(model_path / name)
            for name in ['tokenizer.json', 'tokenizer_config.json']},
        'transformers_version': transformers_version, 'config_sha256': sha(config_path),
        'code_sha256': {name: sha(ROOT / name) for name in
            ['prepare_receiver_evidence.py', 'chain_core.py', 'run.py', 'alignment_probe.py']},
        'unique_questions': len(items), 'receiver_input_rows': len(records),
        'condition_counts': counts, 'partial_undefined_single_sentence_ids': partial_undefined,
        'partial_eligible_questions': 6, 'three_state_comparison_unique_questions': 6,
        'exact_full_prompt_length_control_pairs': control_matches,
        'total_full_prompt_length_control_pairs': len(full_prompt_pairs),
        'material_sampling_really_shortened': sum(not x['text_sampling_control']['unchanged'] for x in senders),
        'no_model_or_generate_calls': True, 'model_calls': 0,
        'future_generation_budget_and_call_count': 'not_frozen',
        'limits': ['support-sentence oracle input, not retrieval or long-context benchmark',
            'held means explicitly visible, not understood or absent from model weights',
            'partial comparison must use the same six questions in all three states',
            'other-document control is exactly token matched if flagged, not proved semantically irrelevant',
            'control may truncate a donor sentence; separate intervention, not public benchmark score',
            'oracle metadata and scoring labels must never be supplied to deployed selectors',
            'full-history route would reveal all sender material and bypass message selection',
            'sampling scope is source material; generated node-output text still required as a baseline']}
    output.mkdir(parents=True)
    write(output / 'selector_input_views.json', {'senders': senders, 'receivers': records})
    write(output / 'diagnostic_annotations.json', {'records': diagnostics,
        'paired_control_keys': full_prompt_pairs, 'partial_undefined': partial_undefined})
    write(output / 'scoring_labels.json', scoring)
    provenance['outputs_sha256'] = {name: sha(output / name) for name in
        ['selector_input_views.json', 'diagnostic_annotations.json', 'scoring_labels.json']}
    write(output / 'manifest.json', provenance)
    print(json.dumps({'output': str(output), 'unique_questions': 20,
        'receiver_input_rows': 72, 'partial_eligible': 6,
        'exact_length_control_pairs': control_matches,
        'control_pairs': len(full_prompt_pairs), 'model_calls': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='data/receiver_evidence_20261002')
    build(parser.parse_args())
