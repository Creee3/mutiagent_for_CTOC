"""Paired exploratory summaries; never train, load models, or tune on answers."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from plot_results import ROOT,METHODS,LABELS,load_verified

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run-name',required=True); args=parser.parse_args()
    run,manifest,summary,ids,rows,sources=load_verified(args.run_name)
    indices=np.random.default_rng(42).integers(0,100,size=(10000,100))
    paired=[]
    for first,second in [('kv_full','raw_context'),('kv_full','sampled_context'),
        ('kv_full','summary_oneway'),('kv_full','nld_original'),('kv_top50','kv_full')]:
        out={'first':first,'second':second,'questions':100}
        for key,label,multiplier in [('standard_token_f1','f1_points',100),
            ('exact_match','em_percentage_points',100),('seconds','seconds',1)]:
            delta=np.array([float(rows[(q,first,0)][key])-float(rows[(q,second,0)][key]) for q in ids])*multiplier
            ci=np.percentile(delta[indices].mean(axis=1),[2.5,97.5])
            out[label]={'mean_difference_first_minus_second':float(delta.mean()),
                'paired_percentile_ci95':[float(x) for x in ci]}
        paired.append(out)
    methods=summary['methods']
    ratio=methods['kv_full']['mean_payload_bytes']/methods['raw_context']['mean_payload_bytes']
    reduction=1-methods['kv_top50']['mean_payload_bytes']/methods['kv_full']['mean_payload_bytes']
    findings={'paired_comparisons':paired,'bootstrap':{'resamples':10000,'seed':42,'unit':'question',
        'interval':'95% percentile; exploratory, unadjusted; not an equivalence test'},
        'full_kv_to_raw_text_payload_ratio':ratio,'top50_vs_full_payload_reduction':reduction,
        'sampling':summary['sampled_message_diagnostics'],
        'source_summary_sha256':hashlib.sha256((run/'summary.json').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (run/'interpretation.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 本轮结果：KV有初步质量信号，通信量很大','',
        '2026-10-01，学校服务器Qwen2.5-7B-Instruct，两节点HotpotQA公开问答。100道评测题全部跑完：700个首遍最终答案，另140个回答仅用于重复测时。20题校准与预热单独记，840条评测记录、1230次生成、262次发送方prefill全部对账通过。没有运行失败，GPU worker已退出；500道确认题未使用。','',
        '| 方法 | 标准F1（0–100） | 完全匹配题数/100 | 平均计算秒/题 | 平均逻辑消息 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for method in METHODS:
        s=methods[method]; size=s['mean_payload_bytes']
        payload=f'{size:.0f} B' if size<1000 else f'{size/1000000:.2f} MB'
        lines.append(f'| {LABELS[method]} | {100*s["standard_token_f1"]:.1f} | {round(100*s["exact_match"])} | {s["mean_total_seconds"]:.3f} | {payload} |')
    f1=paired[0]['f1_points']; time=paired[0]['seconds']; low,high=f1['paired_percentile_ci95']
    lines += ['',f'全量KV比全文文本的F1均值高{f1["mean_difference_first_minus_second"]:.2f}分，100题配对bootstrap区间是[{low:.2f}, {high:.2f}]分，包含零。完全匹配56题对51题；这是初步信号，尚不能确认稳定优势。两者平均计算约0.21秒/题；差值{time["mean_difference_first_minus_second"]:.4f}秒，不声称已证实加速。',
        '',f'全量KV消息约8.13 MB/题，全文文本约422 B，差约{ratio:.0f}倍。这里是同卡逻辑载荷，不是真实网络传输耗时。',
        '',f'选层实际完整保留15/28层，其余留首token；消息减少{100*reduction:.1f}%，F1却从75.8降到52.5，平均回答耗时也增加。不能把这一设置当成保质量压缩成功。',
        '', '原文采样在99/100题保留了与原文相同的词序；材料短于256预算，主要变化是换行/空白格式。本轮没有充分检验有实质删减的采样压缩。',
        '', '原版NLD要生成A消息、B初答、B修订，计算更贵。全量KV对它的收益很明显；全文与采样是更强的文字参考，应一起比较。',
        '', '还需排除格式影响：官方全文文本把材料和问题放在一个chat里，KV分别套A/B两个chat；虽然repetition penalty数值统一为1.05，KV生成处理器只看B输入，全文文本则看材料和问题。当前结果比较完整工作流，不能把质量差异全归因于KV这种表示。CPU随机小模型在惩罚为1时，同token拼接和全KV的前向/生成等价检查已通过；这不是7B实测等价证明。',
        '', '下一步建议：先在预先固定的校准小子集做同提示格式与解码范围的等价检查，分清缓存接入和prompt效果，再设计三节点多跳检查。此次100题与所有方法保持原样，不根据评测答案挑层或改prompt。',
        '', '![质量、计算与逻辑载荷](figures/discussion_3panel.png)',
        '', '图中质量和计算区间按100道题共同重采样：10000次、seed42、95%百分位bootstrap；额外测时重复未并入主均值。原版材料是标注支持句，未评测检索；模型加载、一次性校准、额外计账开销和网络传输不计入表中的计算秒数。',
        '', '详细数据：summary.json、interpretation.json、items、calls.jsonl；图表CSV和来源记录在figures。']
    (run/'report.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'paired':paired,'payload_ratio':ratio,'payload_reduction':reduction}))

if __name__=='__main__':main()
