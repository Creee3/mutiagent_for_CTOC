该方案可以准备，建议将本轮明确限定为原同组 6 题的答案格式核查和完整前缀控制，暂不据此宣称模型“语义已知”、最小有效消息或三角色协作成立。

freeze 前需要明确以下约束：

1. 所有路线共用同一短答案 system、问题和解码规则。B user 必须自带完整问题，no_msg 仍保留自身及 held 材料。A user 只是标记清楚的 Source A 问题和材料数据，不包含三句提取等执行要求；A 只做 prefill。
2. 以 tokenizer 编码完整 `[system, A user, B user]` 得到 canonical IDs；单独编码 `[system, A user]`、不开 assistant header 得到 A IDs。每题每状态都断言 canonical 的前缀与 A IDs 完全相同，B suffix 直接取剩余 IDs，并断言 `A IDs + B suffix == canonical IDs`。suffix 不能另加 system，也不能先解码字符串再重编码。A 在同题各状态中保持不变。
3. KV 路线使用新的 A 完整 cache，位置为 `[0,nA)`；B suffix 从 `nA` 接续。no_msg 和 strong_text 各从自己的完整输入位置 0 开始。不要引用旧 reader-role cache：system、角色或消息边界改变后，来源表示也已改变。
4. canonical token_full 与 KV 是输入逐 token 相同的表示/复用控制；strong_text 是一个共有 system 下的直接材料工作流。strong_text 应使用同一 B 问题、角色、材料顺序和 held 材料，再加入完整 A。它与 canonical 的消息轮次、问题重复及材料位置可能不同，质量或成本差异只能按工作流解释，不能全归因于文本/KV 表示。
5. 在该构造中 held 证据是精确重复的文本，不等于模型理解或语义已知。完整消息路线应保留这种重复。strong_text 可以再准备一个明确标记的可部署文本去重对照，只按实际可见文本判断，不能读取 gold、held/full 标签或支持事实 mask；不能默默只对 strong_text 去重后称所有方法信息设置相同。不同上下文的相似句子也不代表可删除对应原始 KV。
6. KV 计入新 A prefill、完整前缀 token/张量和包头、恢复、B 接续与解码；strong_text 计入实际文本/ID 消息、拼装或去重、完整 receiver prefill 与解码。strong_text 无需执行 A 模型调用，这是其工作流机会，不能人为给它加一次 A prefill。接收方没有私有 cache 时，共有 system 的预处理也不能当免费命中。

先检查能否稳定输出题目所需的短答案，并保留原始输出、严格 EM、token F1、EOS/触顶及实际删减量。不能事后抽取 gold 对应词来制造格式通过，也不要把短答案 system 在这 6 道已使用校准题上的改善当成新评测成绩。若后续要研究信息增量，应另冻结比较协议，并保留完整文本、合理文本去重及原版节点输出文本阶段；本次准备不完成这些结论。

本复核只提出设计约束；未改冻结数据、旧代码或计划，未运行模型。
