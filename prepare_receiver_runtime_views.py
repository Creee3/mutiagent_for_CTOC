"""Prepare safe selector payloads and a body-only sampling control. No models."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    from transformers import AutoTokenizer
    from run import sample_text, write

    source = ROOT / 'data/receiver_evidence_20261002/selector_input_views.json'
    views = json.loads(source.read_text(encoding='utf-8'))
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(config['model_7b_path'], local_files_only=True)
    destination = ROOT / 'data/receiver_runtime_views_20261002'
    assert not destination.exists(), 'do not overwrite a frozen preparation'
    senders = {row['question_id']: row for row in views['senders']}
    payloads, scheduler, sampling = [], [], []
    for row in views['receivers']:
        # Only the payload, not its opaque scheduling ID or any metadata, goes
        # to the selector. Never pass the original full/partial record object.
        opaque_id = hashlib.sha256(('receiver-view:' + row['key']).encode()).hexdigest()
        payloads.append({'lookup_id': opaque_id, 'payload': {
            'question': row['question'],
            'sender_material': senders[row['question_id']]['sender_material'],
            'receiver_visible_text': row['receiver_prompt']}})
        scheduler.append({'lookup_id': opaque_id, 'source_key': row['key'],
            'question_id': row['question_id'], 'condition': row['condition'],
            'selector_input': False})

    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    for sender in views['senders']:
        header, body = sender['sender_material'].split('\n', 1)
        header += '\n'
        budget = sender['text_sampling_control']['budget_tokens']
        body_budget = budget - len(encode(header))
        heading_included = True
        while body_budget > 0:
            sampled_body = sample_text(body, tokenizer, body_budget, config['seed'])
            sampled_text = header + sampled_body
            if sampled_body.strip() and len(encode(sampled_text)) <= budget:
                break
            body_budget -= 1
        if body_budget <= 0:
            # A long title can consume the whole half-budget. Preserve body
            # content in this random control instead of sending only a title.
            heading_included = False
            sampled_body = sample_text(body, tokenizer, budget, config['seed'])
            sampled_text = sampled_body
        assert sampled_body.strip()
        assert 0 < len(encode(sampled_text)) <= budget < sender['material_tokens']
        original = sender['text_sampling_control']['sampled_text']
        sampling.append({'question_id': sender['question_id'], 'budget_tokens': budget,
            'sampled_text': sampled_text, 'payload_tokens': len(encode(sampled_text)),
            'sampled_body_tokens': len(encode(sampled_body)), 'heading_tokens': len(encode(header)),
            'heading_included_in_payload': heading_included,
            'original_heading_only': original.strip() == header.strip(),
            'sampler': 'existing run.sample_text applied to body; heading included only when budget permits nonempty body',
            'may_cut_sentence': True, 'scope': 'random sampling control, not a strong semantic selector'})

    destination.mkdir()
    write(destination / 'runtime_payloads.json', payloads)
    write(destination / 'scheduler_metadata.json', scheduler)
    write(destination / 'text_sampling_body_controls.json', sampling)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha(source), 'code_sha256': sha(Path(__file__)),
        'existing_sampler_sha256': sha(ROOT / 'run.py'),
        'runtime_payload_count': len(payloads), 'unique_questions': len(senders),
        'original_heading_only_count': sum(x['original_heading_only'] for x in sampling),
        'nonempty_body_and_within_budget_count': len(sampling),
        'model_calls': 0, 'gpu_used': False,
        'selector_whitelist': ['question', 'sender_material', 'receiver_visible_text'],
        'limits': ['only payload fields are selector inputs; lookup IDs and scheduler metadata are not',
            'no sender output/cache or legal receiver KV ingestion has been prepared',
            'sampling may cut sentences and does not prove preserved useful information',
            'full text and original node-output text baselines are still required'],
        'outputs_sha256': {name: sha(destination / name) for name in
            ['runtime_payloads.json', 'scheduler_metadata.json', 'text_sampling_body_controls.json']}}
    write(destination / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['runtime_payload_count', 'unique_questions',
        'original_heading_only_count', 'nonempty_body_and_within_budget_count', 'model_calls']}))


if __name__ == '__main__':
    main()
