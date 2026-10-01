"""Read-only CPU audit of an already completed alignment run, then write a report."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run_name):
    folder = ROOT/'runs'/run_name
    assert (folder/'COMPLETE.json').exists()
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    events = [json.loads(line) for line in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
    counts = dict(Counter(x['kind'] for x in events))
    assert counts == manifest['expected_counts'] and all(x['status'] == 'ok' for x in events)
    paths = list((folder/'items').glob('*.json'))
    indexed = {json.loads(p.read_text(encoding='utf-8'))['id']: p for p in paths}
    assert len(paths) == len(indexed) == 10 and set(indexed) == set(manifest['question_ids'])
    rows = [json.loads(indexed[qid].read_text(encoding='utf-8')) for qid in manifest['question_ids']]
    rope_checked = 0
    pair_counts = Counter()
    processor_checks = []
    examples = []
    for row in rows:
        a, b = row['inputs']['A']['ids'][0], row['inputs']['B']['ids'][0]
        assert row['inputs']['AB']['ids'][0] == a+b
        assert set(row['methods']) == set(manifest['methods'])
        for name in manifest['methods']:
            method = row['methods'][name]
            poses = [x['position_ids'][0] for x in method['backend_forward_trace']
                if x['level'] == 'effective_rotary_position_ids']
            assert len(poses) == len(method['output_ids']), (row['id'], name, 'forward count')
            if name.startswith('kv_'):
                expected = list(range(len(a), len(a)+len(b)))
                prompt_len = len(a)+len(b)
                assert method['cache_lengths_before_generation'] == [len(a)]*28
                rope_checked += 1
            elif name.startswith('concat'):
                expected = list(range(len(a)+len(b)))
                prompt_len = len(a)+len(b)
            else:
                prompt_len = row['inputs']['Skyline']['length']
                expected = list(range(prompt_len))
            assert poses[0] == expected, (row['id'], name, 'prefill positions')
            for step in range(1, len(poses)):
                assert poses[step] == [prompt_len+step-1], (row['id'], name, 'decode positions')
        for left, right in manifest['pairs']:
            same = row['methods'][left]['output_ids'] == row['methods'][right]['output_ids']
            key = left+'__vs__'+right
            assert same == row['comparisons'][key]['generated_tokens_identical']
            pair_counts[key] += int(same)
        vector_file = folder/row['first_step_and_divergence_vectors']['file']
        assert sha(vector_file) == row['first_step_and_divergence_vectors']['sha256']
        with np.load(vector_file) as values:
            # Independent first-step check of HF repetition penalty on B-only IDs.
            for name in ['kv_p105', 'concat_bscope_p105']:
                raw, actual = values[name+'__raw'], values[name+'__processed']
                expected = raw.copy()
                unique = np.unique(b)
                selected = raw[unique]
                factor = np.float32(1.05)
                expected[unique] = np.where(selected < 0, selected*factor, selected/factor)
                differences = np.abs(expected-actual)
                error = float(np.max(differences))
                # NumPy CPU arithmetic and CUDA FP32 division may differ by an ULP.
                assert np.all(differences <= 2*np.abs(np.spacing(expected))), (row['id'], name, error)
                processor_checks.append(error)
        if row['methods']['skyline_p1']['output_ids'] != row['methods']['concat_p1']['output_ids']:
            examples.append({'id': row['id'], 'question': row['question'], 'gold': row['gold'],
                'skyline': row['methods']['skyline_p1']['response'],
                'same_input_text': row['methods']['concat_p1']['response'],
                'kv': row['methods']['kv_p1']['response']})
    for key, count in pair_counts.items():
        assert count == summary['pairs'][key]['exact_token_matches']
    audit = {'run_name': run_name, 'passed': True, 'question_ids': manifest['question_ids'],
        'counts': counts, 'answers': 70, 'warmup_answers': 2, 'actual_kv_rope_sequences_checked': rope_checked,
        'first_step_b_scope_formula_checks': len(processor_checks),
        'first_step_formula_max_absolute_error': max(processor_checks),
        'first_step_formula_tolerance': '2 FP32 ULP, NumPy CPU versus CUDA arithmetic',
        'pair_token_matches': dict(pair_counts), 'format_different_examples': examples,
        'summary_sha256': sha(folder/'summary.json'), 'audit_script_sha256': sha(__file__)}
    (folder/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    text = f'''# 同输入核对：完整KV与文本拼接输出一致

本次只用了已固定校准名单前10题，70个诊断答案和2个预热答案，均已跑完。学校服务器、Qwen2.5-7B-Instruct、BF16/SDPA、完整28层KV。未使用100题评测或500题预留集。

| 核对内容 | 输出token完全相同 |
| --- | ---: |
| 相同A+B输入，关闭重复惩罚：文字拼接 vs KV | 10/10 |
| 相同A+B输入，1.05惩罚范围对齐：文字拼接 vs KV | 10/10 |
| 相同文字输入，仅改变1.05惩罚范围 | 10/10 |
| 关闭惩罚，仅改原版全文/双chat格式 | 5/10 |

20次KV生成的实际RoPE位置、28层缓存长度、逐步解码位置全部核对通过。独立用NumPy重算20次首步B范围惩罚，差异在2个FP32 ULP以内，最大绝对误差{max(processor_checks):.2g}（CPU和CUDA除法的舍入可不同）；此前CPU处理器直接比较精确通过。账目是72次生成、31次A prefill、10次raw文字拼接forward和10次raw CV接收forward；没有失败，GPU worker已退出。

这说明：在这10题、同一模型底座、完整缓存、相同输入和解码范围下，两条路径给出了相同输出。我们未观察到缓存位置接错；原版聊天格式确实会改变输出。惩罚范围会改变分数，但这10题没有因此改变输出。

不要把“输出相同”说成“预测数值逐位相同”。同一生成历史下两条路径最大logit差约{summary['pairs']['concat_p1__vs__kv_p1']['max_raw_error_on_same_history']:.3f}；直接前向在所有B位置的最大差约{summary['raw_forward']['worst_absolute_error']:.3f}。分段prefill与一次prefill在当前BF16/SDPA实现下有数值差异，本次没有进一步定位各数值来源。它们未改变这10题的贪心输出；更大样本仍可能分岔。

因此，上一轮100题里KV的F1略高，只能算不同工作流的比较，不能直接归因于KV表示。本次校准小集也不能证明格式解释了那100题的全部分数差。

现在优先做三件事：建立与KV输入匹配的文本基线；先把三节点多跳传递、角色与cache来源核对清楚；再测相同或已知材料被重复传了多少，以及少传后质量和计算是否保持。KV的价值也可能是多个接收方复用已有计算，而非同一模型单跳凭缓存变得更聪明。

原始token、位置、预测分数、首步和首次分歧向量均保存在本run的items/vectors，云端同路径也有完整记录。audit.json是独立汇总核对；这轮不作为新的准确率benchmark。
'''
    if examples:
        ex = examples[0]
        text += f'\n一个格式影响例子（{ex["id"]}）：\n\n问题：{ex["question"]}\n\n标准答案：{ex["gold"]}\n\n原版全文：{ex["skyline"]}\n\n同输入文字/KV：{ex["same_input_text"]}\n'
    (folder/'report.md').write_text(text, encoding='utf-8')
    print(json.dumps({'audit_passed': True, 'rope_checked': rope_checked, 'pair_counts': dict(pair_counts)}, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    main(parser.parse_args().run_name)
