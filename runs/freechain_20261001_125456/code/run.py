"""Independent paired QA runner, using the official KVComm inference methods."""
import argparse
import copy
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'latent_kv_reproduction_20260929'
os.environ.setdefault('NLTK_DATA',str(OLD/'kvcomm_cache/nltk'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
sys.path.insert(0,str(ROOT/'work/KVComm'))

def write(path,data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    temp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def gpu_snapshot():
    output = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.free,memory.used',
        '--format=csv,noheader,nounits'],text=True)
    return [dict(zip(['index','uuid','utilization','free_mib','used_mib'],[x.strip() for x in line.split(',')]))
        for line in output.strip().splitlines()]

# Reused from kv_redundancy_20260930/cloud/graph_adapter.py:sample_text.
def sample_text(text,tokenizer,budget,seed):
    parts = [p for p in re.split(r'(?<=[.!?])\s+|\n+',text) if p.strip()]
    order = list(range(len(parts)))
    random.Random(seed).shuffle(order)
    selected = []
    for index in order:
        proposal = ' '.join(parts[i] for i in sorted(selected+[index]))
        if len(tokenizer.encode(proposal,add_special_tokens=False))<=budget:
            selected.append(index)
    if selected:
        return ' '.join(parts[i] for i in sorted(selected))
    ids = tokenizer.encode(text,add_special_tokens=False)
    start = random.Random(seed).randrange(max(1,len(ids)-budget+1))
    value = tokenizer.decode(ids[start:start+budget],skip_special_tokens=True)
    while len(tokenizer.encode(value,add_special_tokens=False))>budget:
        value = value[:-1]
    return value

class Trace:
    """Per generate stage, not per decoding token; preserves generate signatures."""
    def __init__(self,torch,tokenizer,run_dir):
        self.torch,self.tokenizer,self.run_dir = torch,tokenizer,run_dir
        self.wrapped = set(); self.depth = 0; self.hooks=[]
        self.row = None; self.ordinal = 0
    def sync(self):
        if self.torch.cuda.is_available(): self.torch.cuda.synchronize()
    def begin(self,row):
        self.row = row; self.ordinal = 0; self.overhead=0.; row['stages'] = []
    def append(self,stage):
        self.row['stages'].append(stage)
        if self.run_dir:
            event={'id':self.row['id'],'method':self.row['method'],'repeat':self.row['repeat'],
                'deployment':self.row['deployment'],'phase':self.row['phase'],**stage}
            with (self.run_dir/'calls.jsonl').open('a',encoding='utf8') as handle:
                handle.write(json.dumps(event,ensure_ascii=False)+'\n'); handle.flush()
    def wrap(self,model,label):
        if id(model) in self.wrapped: return
        self.wrapped.add(id(model))
        original = model.generate
        def generate(*args,**kwargs):
            ids=kwargs.get('input_ids',args[0] if args else None)
            prefix=ids.shape[-1]
            cache=kwargs.get('out_A_past_key_values')
            cache_prefix=0 if cache is None else cache.get_seq_length()
            budget=int(kwargs.get('max_new_tokens',48))
            assert prefix+cache_prefix+budget <= model.config.max_position_embeddings, 'context limit exceeded; no truncation'
            self.ordinal += 1; ordinal=self.ordinal
            self.sync(); start=time.perf_counter(); self.depth += 1
            try:
                output=original(*args,**kwargs)
            except Exception as exc:
                self.sync(); self.append({'kind':'generate','label':label,'ordinal':ordinal,
                    'status':'error','error':repr(exc),'seconds':time.perf_counter()-start})
                raise
            finally:
                self.depth -= 1
            self.sync(); elapsed=time.perf_counter()-start
            bookkeeping_start=time.perf_counter()
            tail=output[0,prefix:].detach().cpu().tolist()
            eos=model.generation_config.eos_token_id
            eos=[] if eos is None else eos if isinstance(eos,list) else [eos]
            stopped=bool(tail and tail[-1] in eos)
            self.append({'kind':'generate','label':label,'ordinal':ordinal,'status':'ok',
                'input_tokens':prefix,'cache_prefix_tokens_layer0':cache_prefix,
                'input_ids_sha256':hashlib.sha256(str(ids.detach().cpu().tolist()).encode()).hexdigest(),
                'budget':budget,'generated_tokens':len(tail),'output_ids':tail,
                'text':self.tokenizer.decode(tail,skip_special_tokens=True),'eos_stopped':stopped,
                'actually_truncated':len(tail)>=budget and not stopped,'seconds':elapsed})
            self.overhead += time.perf_counter()-bookkeeping_start
            return output
        model.generate=generate
    def prefill_hooks(self,model):
        starts=[]
        def before(module,args,kwargs):
            if self.depth or self.row is None:
                starts.append(None); return
            ids=kwargs.get('input_ids',args[0] if args else None)
            self.sync(); starts.append((time.perf_counter(),int(ids.shape[-1])))
        def after(module,args,kwargs,output):
            stamp=starts.pop()
            if stamp is None: return
            self.sync()
            elapsed=time.perf_counter()-stamp[0]; bookkeeping_start=time.perf_counter()
            self.append({'kind':'sender_prefill','status':'ok','input_tokens':stamp[1],
                'seconds':elapsed, 'generated_tokens':0})
            self.overhead += time.perf_counter()-bookkeeping_start
        self.hooks += [model.register_forward_pre_hook(before,with_kwargs=True),
            model.register_forward_hook(after,with_kwargs=True)]
    def payload_hook(self,cv):
        original=cv.prepare_key_cache
        def prepare(cache):
            selected=original(cache)
            tensor_bytes=sum(x.numel()*x.element_size() for xs in [selected.key_cache,selected.value_cache] for x in xs)
            metadata={'selected_layers':[int(x) for x in cv.layers_list],
                'effective_full_layers':sorted({0,*[int(x) for x in cv.layers_list]}),
                'layer_map':cv.layer_map,'dtype':str(selected.key_cache[0].dtype),
                'source_id':self.row['id'],'lengths':[int(x.shape[-2]) for x in selected.key_cache],
                'shift_back':cv.shift_back,'source_model_revision':self.row['model_revision']}
            metadata_bytes=len(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode())
            self.row['payload']={'kind':'kv','tensor_bytes':tensor_bytes,'metadata_bytes':metadata_bytes,
                'logical_total_bytes':tensor_bytes+metadata_bytes,'metadata':metadata,
                'network_transfer_measured':False}
            return selected
        cv.prepare_key_cache=prepare

class Engine:
    def __init__(self,torch,tokenizer,A,B,cfg,trace,tracer=False):
        from eval import BaselineEvaluator,SkylineEvaluator,CommunicationEvaluator,NLDEvaluator
        from models import CVCommunicator
        self.torch,self.tokenizer,self.A,self.B,self.cfg,self.trace=torch,tokenizer,A,B,cfg,trace
        holder=SimpleNamespace(name='hotpotqa',max_tokens=cfg['answer_max_new_tokens'],
            truncate_input=False,multiple_answers=False)
        limit=int(A.config.max_position_embeddings)
        self.baseline=BaselineEvaluator(holder,tokenizer,False,limit)
        self.skyline=SkylineEvaluator(holder,tokenizer,False,limit)
        self.comm=CommunicationEvaluator(holder,tokenizer,False,limit)
        self.nld=NLDEvaluator(holder,tokenizer,False,limit,cfg['message_max_new_tokens'],sender_aware=False)
        # Match all methods to the model's released generation config, including EOS and repetition penalty.
        args={'eos_token_id':copy.deepcopy(B.generation_config.eos_token_id),
            'repetition_penalty':B.generation_config.repetition_penalty}
        for evaluator in [self.baseline,self.skyline,self.comm,self.nld]:
            evaluator.generate_args.update(args)
        self.cv=CVCommunicator(A,B,0,A.config.num_hidden_layers-1,layers_list=[-1],
            apply_attn_tracer=tracer,shift_back=cfg['shift_back'])
        self.cv.generation_config=copy.deepcopy(B.generation_config)
        for model,label in [(A,'native_A'),(B,'native_B'),(self.cv,'kv_receiver')]: trace.wrap(model,label)
        # Shared weight copy is hooked once. B forwards inside cv.generate have depth > 0.
        trace.prefill_hooks(A)
        trace.payload_hook(self.cv)
    def set_layers(self,layers): self.cv.layers_list=[int(x) for x in layers]
    def text_payload(self,text):
        meta={'source_id':self.trace.row['id'],'encoding':'utf-8'}
        n=len(json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
        self.trace.row['payload']={'kind':'text','text':text,'utf8_bytes':len(text.encode()),
            'metadata_bytes':n,'logical_total_bytes':len(text.encode())+n,'network_transfer_measured':False}
    def infer(self,method,item,layers):
        from eval import COMMUNICATION_QA_MSG_TEMPLATE_A,SENDER_QA_INSTRUCTION,apply_chat_template
        if method=='receiver_only': return self.baseline.inference(self.B,item)
        if method=='raw_context':
            self.text_payload(item['prompt_A']); return self.skyline.inference(self.B,item)
        if method=='nld_original':
            answer=self.nld.inference(self.A,self.B,item)
            self.text_payload(self.trace.row['stages'][0]['text']); return answer
        if method=='summary_oneway':
            msg=COMMUNICATION_QA_MSG_TEMPLATE_A.format(instruction=SENDER_QA_INSTRUCTION,context=item['prompt_A'])
            ids=apply_chat_template(self.skyline.evaluator,self.tokenizer,msg,self.A)
            kwargs={**self.skyline.generate_args,'max_new_tokens':self.cfg['message_max_new_tokens']}
            out=self.A.generate(ids,attention_mask=self.torch.ones_like(ids),**kwargs)[0]
            text=self.tokenizer.decode(out[ids.shape[-1]:],skip_special_tokens=True)
            # Generation cap includes EOS. Tokenizer roundtrip cannot silently exceed the agreed message budget.
            assert len(self.tokenizer.encode(text,add_special_tokens=False))<=self.cfg['message_max_new_tokens']
            self.text_payload(text)
            return self.skyline.inference(self.B,{**item,'prompt_A':text})
        if method=='sampled_context':
            text=sample_text(item['prompt_A'],self.tokenizer,self.cfg['message_max_new_tokens'],self.cfg['seed'])
            self.text_payload(text)
            return self.skyline.inference(self.B,{**item,'prompt_A':text})
        if method in ['kv_full','kv_top50']:
            self.set_layers(range(self.A.config.num_hidden_layers) if method=='kv_full' else layers)
            return self.comm.inference(self.A,self.cv,item)
        raise ValueError(method)

def load_model(torch,path):
    from transformers import AutoModelForCausalLM
    start=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda').eval()
    model.name='Qwen/'+Path(path).name if '7B' in str(path) else 'Qwen/Qwen2.5-3B-Instruct'
    for p in model.parameters(): p.requires_grad_(False)
    torch.cuda.synchronize()
    return model,time.perf_counter()-start

def run_item(engine,item,method,layers,run_dir,deployment,revision,repeat=0,phase='evaluation'):
    from score_repro_outputs import hotpot_token_f1,normalize
    from utils import f1_match
    row={'id':item['id'],'question_hash':item['question_hash'],'gold':item['answer'],
        'method':method,'deployment':deployment,'model_revision':revision,'repeat':repeat,'phase':phase,
        'utc_started':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu_before':gpu_snapshot(),
        'payload':{'kind':'none','logical_total_bytes':0}}
    engine.trace.begin(row); engine.torch.cuda.reset_peak_memory_stats(); engine.trace.sync()
    start=time.perf_counter()
    try:
        with engine.torch.inference_mode(): response=engine.infer(method,item,layers)
        engine.trace.sync(); row['instrumented_seconds']=time.perf_counter()-start
        row['logging_overhead_seconds']=engine.trace.overhead
        row['seconds']=row['instrumented_seconds']-row['logging_overhead_seconds']
        row.update(status='ok',response=response,exact_match=normalize(response)==normalize(item['answer']),
            standard_token_f1=hotpot_token_f1(response,item['answer']),
            upstream_f1_gt_half_pass=bool(f1_match(item['answer'],response)))
    except Exception as exc:
        engine.trace.sync(); row.update(status='error',error=repr(exc),traceback=traceback.format_exc(),
            seconds=time.perf_counter()-start,exact_match=False,standard_token_f1=0.,upstream_f1_gt_half_pass=False)
    row['peak_allocated_bytes']=engine.torch.cuda.max_memory_allocated()
    row['peak_reserved_bytes']=engine.torch.cuda.max_memory_reserved()
    row['gpu_after']=gpu_snapshot()
    write(run_dir/'items'/f'{phase}_{deployment}_{item["id"]}_{method}_r{repeat}.json',row)
    engine.trace.row=None
    print(json.dumps({'phase':phase,'deployment':deployment,'id':item['id'],'method':method,'repeat':repeat,
        'status':row['status'],'seconds':round(row['seconds'],2),'f1':row['standard_token_f1']}),flush=True)
    if row['status']=='error': raise RuntimeError(row['error'])
    return row

def calibrate(engine,items,cfg,run_dir,revision):
    from layer_importance import calc_layer_importance,get_layer_ranking,get_top_layers
    total=defaultdict(list); rows=[]
    for item in items:
        rows.append(run_item(engine,item,'kv_full',[],run_dir,'shared',revision,phase='calibration'))
        with engine.torch.inference_mode():
            engine.cv.calc_attn_weights_from_qk()
            total=calc_layer_importance(engine.cv.B_attn_weights,engine.A.name,total)
        engine.cv.B_attn_weights.clear()
        for block in engine.B.model.layers: block.self_attn.attn_inputs=None
        gc.collect()
    settings=SimpleNamespace(**{k:cfg[k] for k in ['alpha','mu','sigma','top_layers']})
    ranking=[int(x) for x in get_layer_ranking(total,settings)]
    selected=[int(x) for x in get_top_layers(total,settings).layers_list]
    settings.layers_list=selected
    write(run_dir/'calibration.json',{'question_ids':[x['id'] for x in items],'raw_importance':dict(total),
        'ranking':ranking,'selected_layers':selected,'effective_full_layers':sorted({0,*selected}),
        'configuration':vars(settings),'assets_sha256':sha(ROOT/'assets.json'),'model_revision':revision})
    # Restore ordinary modules; tracer extra work/references must not bias final timing.
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    for block in engine.B.model.layers:
        old=block.self_attn
        new=Qwen2Attention(old.config,old.layer_idx).to(engine.B.device,engine.B.dtype)
        new.load_state_dict(old.state_dict(),strict=True); new.eval()
        for p in new.parameters(): p.requires_grad_(False)
        block.self_attn=new
    engine.cv.apply_attn_tracer=False
    engine.torch.cuda.empty_cache()
    return selected

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['preflight3b','main7b'],required=True)
    parser.add_argument('--run-name',required=True)
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text())
    run_dir=ROOT/'runs'/args.run_name
    run_dir.mkdir(parents=True,exist_ok=True)
    if (run_dir/'manifest.json').exists(): raise RuntimeError('run already exists; choose a new run name')
    import torch
    import transformers
    from transformers import AutoTokenizer
    assert torch.cuda.is_available(),'GPU required; no CPU fallback'
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    path=cfg['model_3b_path'] if args.phase=='preflight3b' else cfg['model_7b_path']
    revision=cfg['model_3b_revision'] if args.phase=='preflight3b' else cfg['model_7b_revision']
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    manifest={'phase':args.phase,'config':cfg,'run_sha256':sha(__file__),'config_sha256':sha(ROOT/'config.json'),
        'assets_sha256':sha(ROOT/'assets.json'),'model_path':path,'revision':revision,
        'torch':torch.__version__,'transformers':transformers.__version__,
        'python':sys.version,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_at_launch':gpu_snapshot(),'old_project_modified':False}
    write(run_dir/'manifest.json',manifest)
    A,seconds=load_model(torch,path)
    manifest['model_load_seconds']=seconds; manifest['generation_config']=A.generation_config.to_dict()
    write(run_dir/'manifest.json',manifest)
    calibration=json.loads((ROOT/'data/calibration.json').read_text())
    trace=Trace(torch,tokenizer,run_dir)
    engine=Engine(torch,tokenizer,A,A,cfg,trace,tracer=args.phase=='main7b')
    if args.phase=='preflight3b':
        L=A.config.num_hidden_layers
        fixed=list(range((L-int(.5*L))//2,(L-int(.5*L))//2+int(.5*L)))
        shared={}
        # A single extra unscored KV warmup, logged separately.
        run_item(engine,calibration[10],'kv_full',fixed,run_dir,'shared',revision,phase='warmup')
        for item in calibration[:10]:
            for method in cfg['methods']:
                shared[(item['id'],method)]=run_item(engine,item,method,fixed,run_dir,'shared',revision,phase='preflight')
        # Unwrap and remove only our hooks before constructing dual deployment.
        for hook in trace.hooks: hook.remove()
        # Reload A as well to avoid nested generate instrumentation left by the first deployment.
        del engine,trace,A; gc.collect(); torch.cuda.empty_cache()
        A,a_sec=load_model(torch,path); B,b_sec=load_model(torch,path)
        trace=Trace(torch,tokenizer,run_dir); engine=Engine(torch,tokenizer,A,B,cfg,trace)
        run_item(engine,calibration[10],'kv_full',fixed,run_dir,'dual',revision,phase='warmup')
        comparisons=[]
        for item in calibration[:10]:
            for method in cfg['methods']:
                row=run_item(engine,item,method,fixed,run_dir,'dual',revision,phase='preflight')
                other=shared[(item['id'],method)]
                equal=[x['output_ids'] for x in row['stages'] if x['kind']=='generate']==[
                    x['output_ids'] for x in other['stages'] if x['kind']=='generate']
                comparisons.append({'id':item['id'],'method':method,'all_generate_tokens_identical':equal,
                    'final_text_identical':row['response']==other['response']})
        passed=all(x['all_generate_tokens_identical'] and x['final_text_identical'] for x in comparisons)
        write(run_dir/'preflight_result.json',{'passed':passed,'comparison_count':len(comparisons),
            'comparisons':comparisons,'fixed_layers':fixed,'warning':'3B engineering check; not paper model scores'})
        if not passed: raise RuntimeError('shared and dual outputs differ; do not run main7b')
    else:
        preflights=list((ROOT/'runs').glob('*/preflight_result.json'))
        assert any(json.loads(p.read_text()).get('passed') for p in preflights),'3B deployment equivalence not verified'
        start=time.perf_counter(); layers=calibrate(engine,calibration,cfg,run_dir,revision)
        manifest['calibration_seconds']=time.perf_counter()-start; write(run_dir/'manifest.json',manifest)
        # Warm each condition with a calibration question only; never tune on evaluation answers.
        for method in cfg['methods']:
            run_item(engine,calibration[10],method,layers,run_dir,'shared',revision,phase='warmup')
        items=json.loads((ROOT/'data/evaluation.json').read_text())
        for repeat in range(1+cfg['timing_extra_passes']):
            chosen=items if repeat==0 else items[:cfg['timing_repeat_questions']]
            for item in chosen:
                order=cfg['methods'].copy()
                seed=int(hashlib.sha256(f'{cfg["seed"]}:{item["id"]}:{repeat}'.encode()).hexdigest()[:16],16)
                random.Random(seed).shuffle(order)
                for method in order:
                    run_item(engine,item,method,layers,run_dir,'shared',revision,repeat=repeat)
    write(run_dir/'COMPLETE.json',{'phase':args.phase,'utc_completed':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    print('COMPLETE',flush=True)

if __name__=='__main__':
    main()
