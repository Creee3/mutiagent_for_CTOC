只读复核当前 `receiver_answer_format_probe.py` 与已冻结的 run conditions（SHA256 `6065b653e402b8d10b41a5db2414e8756077ce77364ce6045fc0cbfc8a44fad6`）。未发现可静态确认的执行阻断；没有运行 GPU、SSH、模型或重做既有输入/旧 run 审计。

- `freeze_run` 与 `verify` 已绑定独立 AUDIT 的 inputs/准备条件 SHA；AUDIT 文件 SHA、主审计脚本 SHA、准备 CPU 报告 SHA、builder/依赖和输入来源均与当前文件匹配。先前 stale passed 审计可能被接纳的缺项已修正。
- 四路读取冻结 `method_inputs` IDs。运行时只用 tokenizer 解码输出，不重新编码 suffix 或材料字符串。token_full 的包恢复 A IDs＋冻结 B suffix 必须等于冻结 canonical IDs；KV 使用同题新来源完整 cache 的独立包恢复。metadata、condition、gold 和审计内容没有作为模型输入传入。
- `material_text` 分支仅校验收到的 UTF-8 JSON material 与冻结 strong_text_user 对应，执行冻结完整输入；没有额外 A forward，也不使用 A cache。其 sender_prefill_seconds 为 0。公共 source 设置仍会为 KV 对照执行一次 A prefill，所以这是正确的分路线成本归属，不是强文本单独端到端运行的观测。
- A prefill 的有效位置检查为 `[0,nA)`、全 28 层长度均为 nA。KV B 从 nA 连续接续；其他路线从自身输入位置 0 开始。逐次恢复新 cache，每条 case 后核对原 source cache hash 未变；无 receiver 私有 cache 合并或 B 尾补 forward。
- 18 个正式 case×4 方法为 72 次 generate；已有校准 case 预热另 4 次。正式 6 次 A prefill、预热 1 次，共候选 76 generate／7 prefill。阶段账本和 COMPLETE 均保留分阶段计数，底层 forward 另记。当前仅 freeze，不代表这些调用已执行。
- launch 在 verify 后持有 worker lock，新鲜检查 GPU 利用率与容量，固定 UUID；worker 描述在 Popen 前落盘，main 再核对 run_name、条件 SHA 和设备。异常保留 FAILED，不覆盖已有 run。该 launch 针对学校 Linux 环境；本复核没有执行它。

资产冻结仍有范围限制：AUDIT 记录的 ByteLevel 辅助解码脚本 SHA 当前匹配，但 verify 不单独校验该辅助脚本；`assets.json` 与 `checks/model7b_manifest.json` 没有纳入 run conditions 的 SHA。main 依赖这两份清单核对官方工作代码与既有权重尺寸，因此不能称所有执行资产都由本次 freeze 独立绑定。建议备份保留这两份清单与审计辅助脚本；这是 provenance 限制，尚未发现当前文件错误或需要推翻现有 freeze 的阻断。权重校验沿用既有验证和尺寸检查，不是本轮重做完整权重 SHA。

计时从冻结 token 数组准备完成后开始；UTF-8 接收后的在线 tokenization 不在本轮计时中，各阶段之和也不是独立网络运行。只能报告这一成本范围。system 的 unknown 规则、A 数据表示和 chat 布局共同变化，本轮仍是六题格式机制检查，不能称旧格式问题的单因素因果实验或语义已知验证。
