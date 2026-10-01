"""Tiny random Qwen: real official cache routing, generation, and scoring checks."""
import copy
from collections import defaultdict
import json
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
from run import ROOT,write
from models import CVCommunicator
from layer_importance import calc_layer_importance,get_top_layers
from types import SimpleNamespace
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from score_repro_outputs import normalize,hotpot_token_f1

torch.manual_seed(42)
cfg=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
    num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
    eos_token_id=63,pad_token_id=0,bos_token_id=1)
cfg._attn_implementation='sdpa'
A=Qwen2ForCausalLM(cfg).eval(); B=Qwen2ForCausalLM(cfg).eval()
B.load_state_dict(A.state_dict())
source=torch.tensor([[1,3,5,7,9]])
question=torch.tensor([[1,4,6]])
results=[]
for layers in [list(range(4)),[1,2]]:
    outputs=[]
    for receiver in [A,B]:
        cv=CVCommunicator(A,receiver,0,3,layers_list=layers,shift_back=False)
        cv.generation_config=copy.deepcopy(receiver.generation_config)
        with torch.inference_mode():
            cache=A(source,use_cache=True).past_key_values
            selected=cv.prepare_key_cache(cache)
            lengths=[int(x.shape[-2]) for x in selected.key_cache]
            assert lengths==[5 if i==0 or i in layers else 1 for i in range(4)]
            output=cv.generate(question,attention_mask=torch.ones_like(question),out_A_past_key_values=cache,
                max_new_tokens=6,do_sample=False,pad_token_id=0)
        outputs.append(output.tolist())
    assert outputs[0]==outputs[1]
    results.append({'layers':layers,'lengths':lengths,'shared_dual_identical':True})
# Exercise the shared-weight attention tracer and restoration used by the 7B calibration.
original_weights={k:v.clone() for k,v in A.state_dict().items()}
cv=CVCommunicator(A,A,0,3,layers_list=[-1],apply_attn_tracer=True,shift_back=False)
cv.generation_config=copy.deepcopy(A.generation_config)
with torch.inference_mode():
    cache=A(source,use_cache=True).past_key_values
    output=cv.generate(question,attention_mask=torch.ones_like(question),out_A_past_key_values=cache,
        max_new_tokens=6,do_sample=False,pad_token_id=0)
    cv.calc_attn_weights_from_qk()
    importance=calc_layer_importance(cv.B_attn_weights,'Qwen/tiny',defaultdict(list))
settings=get_top_layers(importance,SimpleNamespace(alpha=.8,mu=.5,sigma=10,top_layers=.5))
assert len(settings.layers_list)==2
for block in A.model.layers:
    old=block.self_attn; new=Qwen2Attention(old.config,old.layer_idx)
    new.load_state_dict(old.state_dict(),strict=True); new.eval(); block.self_attn=new
assert all(torch.equal(v,original_weights[k]) for k,v in A.state_dict().items())
assert all(type(block.self_attn) is Qwen2Attention for block in A.model.layers)
with torch.inference_mode():
    joint=torch.cat([source,question],dim=-1)
    cache=A(source,use_cache=True).past_key_values
    cached_logits=A(question,past_key_values=cache).logits
    joint_logits=A(joint).logits[:,-question.shape[-1]:,:]
    max_difference=float((cached_logits-joint_logits).abs().max())
    assert torch.allclose(cached_logits,joint_logits,atol=1e-5,rtol=1e-5)
    cache=A(source,use_cache=True).past_key_values
    cached_output=cv.generate(question,attention_mask=torch.ones_like(question),out_A_past_key_values=cache,
        max_new_tokens=6,do_sample=False,pad_token_id=0,repetition_penalty=1.0)
    joint_output=A.generate(joint,attention_mask=torch.ones_like(joint),max_new_tokens=6,
        do_sample=False,pad_token_id=0,repetition_penalty=1.0)
    assert cached_output[0,question.shape[-1]:].tolist()==joint_output[0,joint.shape[-1]:].tolist()
assert normalize('The Louvre!')=='louvre'
assert hotpot_token_f1('yes','no')==0
assert hotpot_token_f1('New York','New York City')==.8
write(ROOT/'checks/cpu_check.json',{'passed':True,'cache_and_generation':results,'metric_checks':3,
    'shared_calibration_tracer_and_restore':True,'calibration_layers':[int(x) for x in settings.layers_list],
    'full_cache_vs_same_token_concat':{'tiny_random_cpu_only':True,'repetition_penalty':1.0,
        'max_logit_absolute_difference':max_difference,'generated_tokens_identical':True,
        'cv_supports_cache_class':cv._supports_cache_class}})
print(json.dumps({'cpu_check_passed':True,'cache_checks':len(results)}))
