"""Audit completeness and summarize paired first-pass answers; repeats are timing only."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
from run import ROOT,write

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run-name',required=True); args=parser.parse_args()
    run=ROOT/'runs'/args.run_name
    manifest=json.loads((run/'manifest.json').read_text())
    cfg=manifest['config']
    rows=[json.loads(p.read_text()) for p in sorted((run/'items').glob('*.json'))]
    events=[json.loads(x) for x in (run/'calls.jsonl').read_text().splitlines() if x.strip()]
    # First 3B run did not repeat phase in the stage ledger; its item records identify it uniquely.
    phases=defaultdict(set)
    for row in rows:
        phases[(row['id'],row['method'],row['repeat'],row['deployment'])].add(row['phase'])
    for event in events:
        if 'phase' not in event:
            choices=phases[(event['id'],event['method'],event['repeat'],event['deployment'])]
            assert len(choices)==1,'ambiguous legacy stage phase'
            event['phase']=next(iter(choices))
    counts=dict(Counter((r['phase']+':'+r['kind']) for r in events))
    report={'run_name':args.run_name,'complete_marker':(run/'COMPLETE.json').exists(),
        'local_model_inference_not_external_api':True,'recorded_stage_counts':counts,
        'recorded_item_rows':len(rows),'quality_samples':'first-pass evaluation only; 100 unique questions expected',
        'timing_note':'seconds subtracts separately recorded logging/decode overhead; single-GPU compute, not network latency'}
    if manifest['phase']=='preflight3b':
        report['quality_samples']='none; 3B deployment and interface check only'
        report['preflight']=json.loads((run/'preflight_result.json').read_text()) if (run/'preflight_result.json').exists() else None
        write(run/'summary.json',report); print(json.dumps({k:v for k,v in report.items() if k!='preflight'})); return
    evaluation=json.loads((ROOT/'data/evaluation.json').read_text())
    expected={(x['id'],method,repeat) for repeat in range(1+cfg['timing_extra_passes'])
        for x in (evaluation if repeat==0 else evaluation[:cfg['timing_repeat_questions']]) for method in cfg['methods']}
    eval_rows=[r for r in rows if r['phase']=='evaluation']
    found={(r['id'],r['method'],r['repeat']) for r in eval_rows}
    duplicates=len(eval_rows)-len(found)
    report['coverage']={'expected_evaluation_rows':len(expected),'found':len(found),'duplicates':duplicates,
        'missing':sorted(expected-found),'unexpected':sorted(found-expected)}
    expected_stages={'calibration:generate':20,'calibration:sender_prefill':20,
        'warmup:generate':10,'warmup:sender_prefill':2,
        'evaluation:generate':1200,'evaluation:sender_prefill':240}
    report['stage_count_audit']={'expected':expected_stages,'recorded':counts,
        'passed':counts==expected_stages}
    grouped=defaultdict(list)
    for row in eval_rows:
        if row['repeat']==0: grouped[row['method']].append(row)
    summaries={}
    for method,group in grouped.items():
        good=[r for r in group if r['status']=='ok']
        mean=lambda key:sum(r.get(key,0) for r in group)/len(group)
        final_stages=[next((s for s in reversed(r['stages']) if s['kind']=='generate'),{}) for r in group]
        summaries[method]={'questions':len(group),'completed':len(good),
            'exact_match':mean('exact_match'),'standard_token_f1':mean('standard_token_f1'),
            'upstream_f1_gt_half_pass_rate':mean('upstream_f1_gt_half_pass'),
            'mean_total_seconds':mean('seconds'),'median_total_seconds':statistics.median(r['seconds'] for r in group),
            'mean_payload_bytes':sum(r['payload']['logical_total_bytes'] for r in group)/len(group),
            'mean_peak_allocated_bytes':mean('peak_allocated_bytes'),
            'final_answer_truncated':sum(s.get('actually_truncated',False) for s in final_stages),
            'all_stage_truncated':sum(s.get('actually_truncated',False) for r in group for s in r['stages'])}
    report['methods']=summaries
    paired=[]
    bykey={(r['id'],r['method']):r for r in eval_rows if r['repeat']==0}
    for kv in ['kv_full','kv_top50']:
        for text in ['raw_context','nld_original','summary_oneway','sampled_context']:
            pairs=[(bykey[(x['id'],kv)],bykey[(x['id'],text)]) for x in evaluation
                if (x['id'],kv) in bykey and (x['id'],text) in bykey]
            if not pairs: continue
            paired.append({'kv':kv,'text':text,'paired_count':len(pairs),
                'mean_f1_difference_kv_minus_text':statistics.mean(a['standard_token_f1']-b['standard_token_f1'] for a,b in pairs),
                'mean_seconds_difference_kv_minus_text':statistics.mean(a['seconds']-b['seconds'] for a,b in pairs),
                'kv_f1_wins':sum(a['standard_token_f1']>b['standard_token_f1'] for a,b in pairs),
                'kv_f1_losses':sum(a['standard_token_f1']<b['standard_token_f1'] for a,b in pairs)})
    report['paired_exploratory']=paired
    repeat_groups=defaultdict(list)
    for r in eval_rows:
        if r['id'] in {x['id'] for x in evaluation[:cfg['timing_repeat_questions']]}:
            repeat_groups[(r['method'],r['repeat'])].append(r)
    report['timing_repeats']={f'{method}/pass{repeat}':{'n':len(group),
        'mean_seconds':statistics.mean(r['seconds'] for r in group),
        'median_seconds':statistics.median(r['seconds'] for r in group)}
        for (method,repeat),group in repeat_groups.items()}
    report['sampled_message_diagnostics']={'unchanged_source_fact_count':sum(
        r['payload'].get('text','').split()==next(x['prompt_A'] for x in evaluation if x['id']==r['id']).split()
        for r in grouped.get('sampled_context',[])),'total':len(grouped.get('sampled_context',[]))}
    write(run/'summary.json',report)
    lines=['# 两节点探索对照结果','',f'运行：{args.run_name}。质量只计首遍，额外两遍只检查耗时。',
        '',f'评测记录 {len(found)}/{len(expected)}，重复键 {duplicates}。完整标记：{report["complete_marker"]}。',
        '', '| 方法 | 完成/题数 | EM | 标准F1 | 作者通过率 | 平均秒/题 | 消息平均字节 | 答案触顶 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method in cfg['methods']:
        if method not in summaries: continue
        s=summaries[method]
        lines.append(f'| {method} | {s["completed"]}/{s["questions"]} | {s["exact_match"]:.3f} | {s["standard_token_f1"]:.3f} | {s["upstream_f1_gt_half_pass_rate"]:.3f} | {s["mean_total_seconds"]:.3f} | {s["mean_payload_bytes"]:.0f} | {s["final_answer_truncated"]} |')
    lines += ['', '这是100题探索结果；原版材料是标注支持句。同卡逻辑KV载荷不能当真实网络成本，共享权重的显存不能当原论文双模型内存。',
        '', '完整逐题配对、分阶段调用数、重复计时与采样是否真正删减材料，见summary.json。']
    (run/'结果.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'coverage':report['coverage'],'methods':summaries},ensure_ascii=False))
    assert found==expected and duplicates==0 and all(r['status']=='ok' for r in eval_rows),'incomplete or invalid evaluation'
    assert report['stage_count_audit']['passed'],'unexpected actual generate/prefill counts'

if __name__=='__main__': main()
