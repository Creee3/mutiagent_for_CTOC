"""Download one pinned 7B snapshot to the server, then verify official LFS hashes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'config.json').read_text())
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='90'
os.environ['HF_HUB_ETAG_TIMEOUT']='30'
from huggingface_hub import HfApi,snapshot_download

repo='Qwen/Qwen2.5-7B-Instruct'; revision=cfg['model_7b_revision']; target=Path(cfg['model_7b_path'])
api=HfApi(token=False)
info=api.model_info(repo,revision=revision,files_metadata=True)
assert info.sha==revision
names=['config.json','generation_config.json','merges.txt','model.safetensors.index.json',
    'tokenizer.json','tokenizer_config.json','vocab.json']
names += [s.rfilename for s in info.siblings if s.rfilename.endswith('.safetensors')]
expected={}
for sibling in info.siblings:
    if sibling.rfilename in names:
        lfs=sibling.lfs
        expected[sibling.rfilename]={'bytes':sibling.size,'lfs_sha256':None if lfs is None else lfs.sha256}
total=sum(x['bytes'] or 0 for x in expected.values())
target.mkdir(parents=True,exist_ok=True)
present=sum((target/name).stat().st_size for name in names if (target/name).exists())
assert shutil.disk_usage(target).free > max(0,total-present)+5*1024**3,'insufficient disk margin'
ROOT.joinpath('checks').mkdir(exist_ok=True)
(ROOT/'checks/model7b_expected.json').write_text(json.dumps({'repo':repo,'revision':revision,
    'expected_files':expected,'expected_total_bytes':total},indent=2))
print(json.dumps({'event':'download_start','revision':revision,'bytes':total,'free_bytes':shutil.disk_usage(target).free}),flush=True)
start=time.perf_counter()
snapshot_download(repo,revision=revision,local_dir=target,allow_patterns=names,token=False,max_workers=2)
verified={}
for name,spec in expected.items():
    path=target/name; digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024**2),b''): digest.update(chunk)
    result=digest.hexdigest()
    assert path.stat().st_size==spec['bytes'], f'size mismatch: {name}'
    if spec['lfs_sha256']: assert result==spec['lfs_sha256'], f'official hash mismatch: {name}'
    verified[name]={'bytes':path.stat().st_size,'sha256':result,'official_lfs_verified':bool(spec['lfs_sha256'])}
ready_path=ROOT/'checks/model7b_manifest.json'
ready_temp=ready_path.with_suffix('.json.tmp')
ready_temp.write_text(json.dumps({'repo':repo,'revision':revision,
    'model_path':str(target),'files':verified,'download_and_verify_seconds':time.perf_counter()-start},indent=2))
ready_temp.replace(ready_path)
print(json.dumps({'event':'download_verified','files':len(verified),'seconds':time.perf_counter()-start}),flush=True)
