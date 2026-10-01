"""Freeze official sources and public data. CPU only; no model download."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'latent_kv_reproduction_20260929'
UPSTREAM = ROOT.parent / 'kv_redundancy_20260930/upstream/KVComm'
REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'
COMMIT = '1fa086edb17d6e00d80953cb06f391bc0a85268c'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')

def main():
    for name in ['upstream', 'work']:
        dst = ROOT / name / 'KVComm'
        if not dst.exists():
            shutil.copytree(UPSTREAM, dst, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    work = ROOT / 'work/KVComm'
    # Scoring behavior unchanged. Avoid mandatory downloads on every import.
    metric = work / 'utils/metric_tools.py'
    source = (ROOT / 'upstream/KVComm/utils/metric_tools.py').read_text()
    for resource, location in [('averaged_perceptron_tagger','taggers/averaged_perceptron_tagger'),
        ('punkt','tokenizers/punkt'),('wordnet','corpora/wordnet.zip'),
        ('omw-1.4','corpora/omw-1.4.zip'),('punkt_tab','tokenizers/punkt_tab'),
        ('averaged_perceptron_tagger_eng','taggers/averaged_perceptron_tagger_eng')]:
        source = source.replace(f"nltk.download('{resource}', quiet=True)", f"nltk.data.find('{location}')")
    metric.write_text(source, encoding='utf8')
    shutil.copyfile(OLD / 'kvcomm_src/score_repro_outputs.py', ROOT / 'score_repro_outputs.py')
    sys.path.insert(0, str(work))
    os.environ['HF_HOME'] = str(OLD / 'kvcomm_cache/hf')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['NLTK_DATA'] = str(OLD / 'kvcomm_cache/nltk')
    from datasets import load_dataset
    from dataloader.hotpotqa import construct_support
    data = load_dataset('hotpotqa/hotpot_qa','distractor',revision=REVISION,split='validation')
    shuffled = data.shuffle(seed=42)
    used_ids, used_questions, rows = set(), set(), []
    for item in shuffled:
        question_hash = hashlib.sha256(' '.join(item['question'].lower().split()).encode()).hexdigest()
        if item['id'] in used_ids or question_hash in used_questions:
            continue
        used_ids.add(item['id']); used_questions.add(question_hash)
        support = construct_support(item)
        assert support and all(isinstance(x,str) for x in support)
        rows.append({'id':item['id'],'question_hash':question_hash,'answer':item['answer'],
            'prompt_A':'\n'.join(support),'prompt_B':item['question'],'support':support,
            'supporting_facts':item['supporting_facts'],'type':item['type'],'level':item['level']})
        if len(rows) == 620:
            break
    splits = {'calibration':rows[:20],'evaluation':rows[20:120],'reserved_confirmation':rows[120:620]}
    known = {'5add1d575542992c1e3a2540','5ac55ea55542993e66e82377'}
    assert known <= {x['id'] for x in splits['calibration']}
    for name, items in splits.items():
        write(ROOT / f'data/{name}.json', items)
    pristine = ROOT / 'upstream/KVComm'
    files = ['eval.py','models.py','model_attn.py','layer_importance.py','dataloader/hotpotqa.py','utils/metric_tools.py']
    manifest = {'upstream_url':'https://github.com/Zephyroam/KVComm.git','commit':COMMIT,
        'dataset':'hotpotqa/hotpot_qa','dataset_revision':REVISION,'config':'distractor',
        'split':'validation','seed':42,'dataset_rows':len(data),'dataset_fingerprint':data._fingerprint,
        'support_mode':'official gold supporting sentences; no retrieval',
        'split_files':{name:{'count':len(items),'sha256':sha(ROOT/f'data/{name}.json'),
            'ids':[x['id'] for x in items]} for name,items in splits.items()},
        'source_files':{name:{'pristine_sha256':sha(pristine/name),'work_sha256':sha(work/name)} for name in files},
        'patches':['NLTK imports verify cached resources instead of unconditional downloads'],
        'scoring_helper_source':str(OLD/'kvcomm_src/score_repro_outputs.py'),
        'scoring_helper_sha256':sha(ROOT/'score_repro_outputs.py')}
    write(ROOT/'assets.json',manifest)
    print(json.dumps({'splits':{k:len(v) for k,v in splits.items()},'assets_sha256':sha(ROOT/'assets.json')}),flush=True)

if __name__ == '__main__':
    main()
