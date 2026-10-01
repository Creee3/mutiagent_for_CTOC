# 同输入核对：完整KV与文本拼接输出一致

本次只用了已固定校准名单前10题，70个诊断答案和2个预热答案，均已跑完。学校服务器、Qwen2.5-7B-Instruct、BF16/SDPA、完整28层KV。未使用100题评测或500题预留集。

| 核对内容 | 输出token完全相同 |
| --- | ---: |
| 相同A+B输入，关闭重复惩罚：文字拼接 vs KV | 10/10 |
| 相同A+B输入，1.05惩罚范围对齐：文字拼接 vs KV | 10/10 |
| 相同文字输入，仅改变1.05惩罚范围 | 10/10 |
| 关闭惩罚，仅改原版全文/双chat格式 | 5/10 |

20次KV生成的实际RoPE位置、28层缓存长度、逐步解码位置全部核对通过。独立用NumPy重算20次首步B范围惩罚，差异在2个FP32 ULP以内，最大绝对误差1.9e-06（CPU和CUDA除法的舍入可不同）；此前CPU处理器直接比较精确通过。账目是72次生成、31次A prefill、10次raw文字拼接forward和10次raw CV接收forward；没有失败，GPU worker已退出。

这说明：在这10题、同一模型底座、完整缓存、相同输入和解码范围下，两条路径给出了相同输出。我们未观察到缓存位置接错；原版聊天格式确实会改变输出。惩罚范围会改变分数，但这10题没有因此改变输出。

不要把“输出相同”说成“预测数值逐位相同”。同一生成历史下两条路径最大logit差约1.016；直接前向在所有B位置的最大差约3.266。分段prefill与一次prefill在当前BF16/SDPA实现下有数值差异，本次没有进一步定位各数值来源。它们未改变这10题的贪心输出；更大样本仍可能分岔。

因此，上一轮100题里KV的F1略高，只能算不同工作流的比较，不能直接归因于KV表示。本次校准小集也不能证明格式解释了那100题的全部分数差。

现在优先做三件事：建立与KV输入匹配的文本基线；先把三节点多跳传递、角色与cache来源核对清楚；再测相同或已知材料被重复传了多少，以及少传后质量和计算是否保持。KV的价值也可能是多个接收方复用已有计算，而非同一模型单跳凭缓存变得更聪明。

原始token、位置、预测分数、首步和首次分歧向量均保存在本run的items/vectors，云端同路径也有完整记录。audit.json是独立汇总核对；这轮不作为新的准确率benchmark。

一个格式影响例子（5ac1a4745542991316484b82）：

问题：Para Hills West, South Australia lies within a city with what estimated population?

标准答案：138,535

原版全文：The context does not provide the estimated population of the City of Salisbury. Therefore, based on the given information, it is not possible to determine the estimated population of the City of Salisbury.

同输入文字/KV：The context does not provide the estimated population of the City of Salisbury. Therefore, I cannot answer the question based on the given information.
