"""Run the already authorized 7B study only after model and deployment checks finish."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from run import ROOT,write

parser=argparse.ArgumentParser()
parser.add_argument('--preflight-run',required=True)
parser.add_argument('--download-pid',type=int,required=True)
args=parser.parse_args()
python='/home/liuxin/.conda/envs/kv-graph/bin/python'
run=ROOT/'runs'/args.preflight_run

def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists(): return False
    return p.read_text().split(') ')[1][0] != 'Z'

def update(stage,**kwargs):
    data={'stage':stage,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs}
    write(ROOT/'pipeline_status.json',data); print(json.dumps(data),flush=True)

update('waiting_for_download_and_preflight',preflight_run=args.preflight_run)
started=time.perf_counter()
while True:
    result=run/'preflight_result.json'
    if result.exists() and not json.loads(result.read_text())['passed']:
        update('failed',reason='shared/dual outputs differ'); raise SystemExit(1)
    preflight_pid=json.loads((ROOT/'worker.json').read_text())['pid']
    if not (run/'COMPLETE.json').exists() and not alive(preflight_pid):
        update('failed',reason='3B check exited without completion'); raise SystemExit(1)
    downloaded=(ROOT/'checks/model7b_manifest.json').exists()
    if not downloaded and not alive(args.download_pid):
        update('failed',reason='download exited without hash verification'); raise SystemExit(1)
    if (run/'COMPLETE.json').exists() and result.exists() and downloaded: break
    if time.perf_counter()-started>4*3600:
        update('failed',reason='prerequisites not ready within four hours; no inference launched'); raise SystemExit(1)
    time.sleep(30)
subprocess.run([python,str(ROOT/'analyze.py'),'--run-name',args.preflight_run],check=True)
update('starting_main7b')
# Allow the completed preflight process to release its exclusive lock.
while alive(json.loads((ROOT/'worker.json').read_text())['pid']): time.sleep(2)
subprocess.run([python,str(ROOT/'launch.py'),'--phase','main7b'],check=True)
worker=json.loads((ROOT/'worker.json').read_text()); main=ROOT/'runs'/worker['run_name']
update('running_main7b',**worker)
while alive(worker['pid']): time.sleep(30)
if not (main/'COMPLETE.json').exists():
    update('failed',reason='7B run exited without completion',run_name=worker['run_name']); raise SystemExit(1)
subprocess.run([python,str(ROOT/'analyze.py'),'--run-name',worker['run_name']],check=True)
update('complete',run_name=worker['run_name'],summary=str(main/'summary.json'))
