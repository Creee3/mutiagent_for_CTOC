"""Fresh shared-server GPU selection and detached launch; no environment mutation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--phase',choices=['preflight3b','main7b'],required=True)
args=parser.parse_args()
ROOT.joinpath('logs').mkdir(exist_ok=True)
lock_path=ROOT/'worker.lock'
lock=open(lock_path,'a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
config=json.loads((ROOT/'config.json').read_text())
if args.phase=='main7b':
    model=Path(config['model_7b_path'])
    assert (ROOT/'checks/model7b_manifest.json').exists(),'7B weights have not been verified'
    verified=json.loads((ROOT/'checks/model7b_manifest.json').read_text())
    assert verified['revision']==config['model_7b_revision'] and Path(verified['model_path'])==model
    assert all((model/name).stat().st_size==spec['bytes'] for name,spec in verified['files'].items())
    assert (model/'model.safetensors.index.json').exists(),'7B not available'
    current_source=(ROOT/'run.py').read_bytes()
    current_sha=hashlib.sha256(current_source).hexdigest()
    # Documented non-inference fixes: ledger phase, and NumPy layer list -> JSON list in calibration metadata.
    before_logging_fix=current_source.replace(b",'phase':self.row['phase']",b'')
    before_logging_fix=before_logging_fix.replace(b'    settings.layers_list=selected\n',b'')
    compatible_sha={current_sha,hashlib.sha256(before_logging_fix).hexdigest()}
    compatible=[]
    for path in (ROOT/'runs').glob('*/preflight_result.json'):
        result=json.loads(path.read_text()); manifest=json.loads((path.parent/'manifest.json').read_text())
        if result.get('passed') and result['comparison_count']==70 and manifest['config']==config and manifest['run_sha256'] in compatible_sha:
            compatible.append(path)
    assert compatible,'no deployment check for the current inference settings/code'
else:
    assert json.loads((ROOT/'checks/cpu_check.json').read_text())['passed']
while True:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.free',
        '--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in raw.strip().splitlines():
        index,uuid,util,free=[x.strip() for x in line.split(',')]
        rows.append({'index':int(index),'uuid':uuid,'utilization':int(util),'free_mib':int(free)})
    eligible=[x for x in rows if x['utilization']<10]
    eligible.sort(key=lambda x:x['free_mib'],reverse=True)
    # Capacity check is separate from eligibility. 2x3B / 1x7B need these startup margins.
    minimum=15000 if args.phase=='preflight3b' else 18000
    if eligible and eligible[0]['free_mib']>=minimum:
        gpu=eligible[0]; break
    print(json.dumps({'waiting_for_capacity':minimum,'gpus':rows}),flush=True); time.sleep(30)
run_name=args.phase+'_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())
env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gpu['uuid']; env['PYTHONUNBUFFERED']='1'
env['OMP_NUM_THREADS']='4'; env['TOKENIZERS_PARALLELISM']='false'
env['NLTK_DATA']=str(ROOT.parent/'latent_kv_reproduction_20260929/kvcomm_cache/nltk')
python='/home/liuxin/.conda/envs/kv-graph/bin/python'
log=open(ROOT/'logs'/f'{run_name}.log','ab',buffering=0)
# Worker inherits an exclusive file lock; launch cannot duplicate while the child is alive.
os.set_inheritable(lock.fileno(),True)
proc=subprocess.Popen([python,'-u',str(ROOT/'run.py'),'--phase',args.phase,'--run-name',run_name],
    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
status={'pid':proc.pid,'run_name':run_name,'phase':args.phase,'gpu':gpu,'python':python}
(ROOT/'worker.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status),flush=True)
