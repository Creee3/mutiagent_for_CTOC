# 这轮分数为什么不能直接解释成“消息没用”

只读核查 `receiver_endpoint_20261001_204127` 的72份主结果、analysis及冻结prompt。没有改回答、评分、金标或代码，也没有追加模型调用。

**当前首先暴露的是回答格式混杂。完整文本和完整KV都倾向于整句回答，标准EM/F1因此下降；这不等于新增信息没有价值。** 同时也有真实错误，不能把所有下降都归因于格式。

## 逐题看到的现象

人工按“实体/数字/日期等答案短语”与“含主谓陈述的整句”检查missing/full各20题。该分类只描述输出形式，不是新的正确率，不使用金标抽取答案。

| 接收状态 | 方法 | 答案短语 | 整句/长说明 | 其中明确缺材料说明 |
| --- | --- | ---: | ---: | ---: |
| missing | 无消息 | 8 | 12 | 2 |
| missing | 完整token文本 | 0 | 20 | 0 |
| missing | 完整KV | 0 | 20 | 0 |
| full | 无消息 | 11 | 9 | 0 |
| full | 完整token文本 | 0 | 20 | 0 |
| full | 完整KV | 0 | 20 | 0 |

无消息组missing的8个短语对应人口、Sinclair创始人、Afroman专辑、Oldham县、Rensselaer县、出生日期、出生年份、Nassau县题；其中专辑答案本身错误。full的11个短语另包括国籍、Charlie Murphy、Dudman题。形式简短不能当正确。

具体例子如下，均来自保留的原始输出：

| 题目/状态 | 无消息 | 完整消息实际回答 | 能说明什么 |
| --- | --- | --- | --- |
| 人口，missing | `138,535` | 两路均为“Para Hills West…estimated population of 138,535 people.” | 答案数值相同，但整句降低EM/F1。 |
| Axl Rose，missing/full | “The band's lead singer is Axl Rose.” | 两路均为“The band's lead singer of November Rain is Axl Rose.” | 内容相同，不能把多出来的文字当消息失败。 |
| Royal Flash国籍，missing | “…was British.” | 两路明确“Otto von Bismarck…Prussian, thus German.” | 新材料修正了具体事实，但完整消息EM仍为0。 |
| Dudman，missing | “specific name…is not provided…” | 两路明确“Bill Dudman was Industrial Chaplain…” | 新材料补出姓名；标签是`Dudman`，整句/姓名形式影响匹配。 |
| Holliday日期，missing | 缺材料说明并触顶 | 两路明确`October 25, 1881`，包在整句里 | 新材料补出题目所问日期；不能用EM=0说完全无效。 |
| Afroman专辑，missing | `Because I Got High` | 两路均给`Starvation Is` | 这是实际选错“前两个词”；不是加一句包装造成的。 |
| Afroman专辑，full | `The Hungry Hustlerz` | 两路明确前两个词是`The Hungry`，包在整句里 | 形式与内容变化并存；不能只看主分数下降。 |
| 电影年份，missing/full | “No…”后带解释 | 两路同样“No…”后带解释 | 既有Hotpot评分对yes/no要求规范后的整个回答完全匹配；正确判断包在长句里也可能得到F1=0。 |

另有原公开标签形式本身较长或存在姓名变体。例如Tenerife比较题的金标是一长段事故描述，输出却是“Tenerife airport disaster occurred first…”；Weill题金标为`Kurt Julian Weill`，输出是`Kurt Weill`。这些应保留并注明，不私改公开金标、补别名或自动用gold-span判对。

## 为什么仍不能定性为冗余效果

B的新指令已统一为直接答、无解释，旧“preceding analyses”误指令已经移除。A仍带着“提取事实、至多三句”的角色指令，完整消息也带A的角色与题目。这可能影响回答风格，**尚未用干预确认原因**。

实际输入是两个独立chat模板编码后接续；A为结束完整user turn，B有自己的模板及回答提示。完整文本/KV用的是同一份A+B token序列，所以共同的低分不能归咎于KV截片或位置故障。它们66/72份逐token相同，只能作为路线描述。

full状态的B已显式看见A材料，额外前缀还增加角色、题目、系统模板和材料重复。因此此端点测的是**整个A前缀的边际作用**，不是单独的新增事实价值，也不是已实现的消息删减算法。B也没有先形成独立私有cache。

原始分数继续保留：missing无消息F1约43.70，完整文本23.16、KV22.92；full无消息56.20，完整文本25.04、KV23.57。不能据此宣布“KV无用”“删掉冗余更准”或“接收方不需要信息”。本核查不生成替代正确率。

## 成本也受到输出长度影响

missing平均生成长度为无消息18.35、完整文本25.60、KV25.85 tokens；full为16.35、24.85、26.20。接收总耗时不同同时包含解码长度差，不能只归因于cache表示。

实际包均值约：完整token文本1,334.6字节，完整KV8,433,248.4字节，均含头；KV包还含用于来源恢复/校验的A token IDs，不能描述成只发送裸tensor。missing模型接收阶段均值约0.529秒/0.511秒；加编码恢复及独立A prefill后，文本约0.531秒、KV约0.611秒。这里只是单次记录，没有真实网络和稳健计时。

主实验实际复用20份A来源prefill，合计约0.520秒；72个独立请求各重做A的分配估算约1.882秒。二者是不同使用场景，不能混成同一个观测成本。已有cache场景也不是免费创建cache。

## 下一步先修什么

先冻结统一的答案格式，明确“只输出所问名字、数值、日期或yes/no，不复述题目、不加完整陈述句”。同一规则用于所有方法和接收状态，不放金标或题目特定正确示例。考虑把A待执行的角色指令作为来源记录而非接收方要服从的任务，并用规范的单份chat模板构造共享A/B token边界；改变前缀时需要新来源cache和独立版本。

先做少量校准检查确认输出真的遵守格式，再决定后续规模；运行前说明新条件和调用数。原结果保留。不要用金标抽取、事后裁剪到gold-span或换评分，把这一轮“修成”更高正确率。

来源：同run的`items/*.json`、`analysis.json`、`manifest.json`和`data/receiver_endpoint_tokens_20261002.json`。这是20道短支持句校准机制核查，无A生成，原版“仅传节点输出”的文本对照仍未运行。
