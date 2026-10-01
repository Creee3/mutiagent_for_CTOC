# 进度

2026-10-01，用户同意先讨论原版公开问答的文本/KV收益对照，询问想法。应用research-ideation的研究问题与支持/否定条件规划，沿用planning-with-files记录。

核对全文模型对、官方sender prefill/NLD/选层/评分源码和旧两题检查；只读查询服务器模型目录与GPU资源。发现所查位置未有Qwen7B，已有3B只能用于接口检查。未下载模型、安装依赖、加载真实模型或启动GPU。

提出七组对照，包括原NLD、高效单向摘要与之前要求的原文采样；主模型拟用论文Qwen7B，相同模型共享冻结权重以适应单卡，需接口核对。20校准/100探索/另500确认、质量/完整计算/载荷分别记录。写研究问题卡和《两节点文本KV对照_讨论稿.md》。旧五角色工程、配置、checkpoint未改；当时停在设置讨论。

2026-10-01用户授权执行。云端已有kv-graph conda（torch2.6.0/transformers4.53.3/datasets3.6.0），未安装依赖或改系统CUDA。bootstrap复制独立upstream/work；eval、models、model_attn、layer_importance和HotpotQA支持句处理hash与本地官方仓库一致，工作副本仅修改NLTK资源导入下载检查，评分不改。从固定revision离线缓存冻结20/100/500题，ID和题干hash无重叠，旧2题在校准侧；assets和名单已回传。

8081被Wailmote占用、7890未监听；用户告知12000，实际确认iKuuuVPNCore监听且系统代理启用，Edge有24条到此端口连接。Hugging Face本地/云端反向隧道均200。只停止本任务旧18083隧道，新18084转本机12000，PID见checks/proxy_tunnel.json。不能从连接列表确认单条YouTube流量归属。

CPU随机Qwen官方选层与生成检查通过。3B预检查run=preflight3b_20261001_041752，PID3306174，GPU0 UUID GPU-0a075077-3745-799a-aff3-f0d34a17b6ac；已出现COMPLETE，逐方法10题共享/双实例共200段生成+40次sender prefill，另2次KV预热单列。是否一致以preflight_result.json为准，质量不作正式结果。

7B revision a09a35458c702b33eeacc393d103063234e8bc28来自官方公共元数据；下载PID3306587，目标/home/liuxin/models/Qwen2.5-7B-Instruct，11文件共15242788168 bytes，启动时49.45GB可用。local_dir避免双份权重，下载后核对官方LFS SHA256。未核验完整权重前不启动7B。

条件与调用数已告知并写《运行条件.md》。pipeline PID3308994：检查通过且下载完整后才启动7B；新鲜选卡，利用率<10%并单独检查容量。7B含20校准、100题七组及固定10题额外两次测时，500题不运行。原生EOS统一、答案48、消息256、greedy、BF16、SDPA、共享冻结权重。答案/中间生成/EOS触顶/载荷/阶段时间/GPU信息落盘，日志开销单列，网络时延未测。pipeline_status.json记录阶段，COMPLETE和summary覆盖核对是完成证据。

3B已核对通过：70/70方法-题目配对的全部生成token和最终文字完全一致。调用账目实测200次生成+40次sender prefill，预热另2+2，142条最终答案记录。首次ledger缺phase冗余字段，逐题JSON有phase且键唯一；analyze由逐题键恢复计账、汇总成功。后续run.py添加ledger phase字段，未改变推理或题目，未重跑3B。该代码hash变化仅是日志修复，记录在主run manifest，不掩盖版本。

用户再次授权继续。下载前两个权重分片完成、后两个继续，没有网络错误。独立静态复核确认官方tracer只取B多token prefill、校准后普通attention恢复与20/100题隔离正确；每题10次generate/2次sender prefill、7B全阶段1230/262计数正确。analyze新增实际阶段数审计，评测记录须840条（700主对照+140重复测时）、calibration20/20、warmup10/2、evaluation1200/240全部对账后才认为完整。运行参数和推理方法未改变。

在主实验启动前发现并修复校准结果JSON保存问题：官方get_top_layers将settings.layers_list设置为numpy.ndarray，vars(settings)原样写入会报不可JSON序列化。已有selected为Python int列表，将settings.layers_list=selected后保存；排序、选层和3B路径不变，无需重跑3B。launch源码兼容性校验明确允许这项序列化修复及前面的ledger phase修复，其余源码须与已验证3B SHA一致。静态复核其他numpy/torch输出均转为Python标量/list，无其他已知序列化遗漏。当前下载进程未重启；download_model.py未来运行改用原子就绪manifest写入，当前运行器若真的撞到极小读写竞争，则核实完整manifest后只重启pipeline。

2026-10-01下载约31分钟时第3权重分片HTTP连接中断（ChunkedEncodingError），下载器和守护流程均退出，7B未加载。第1/2/4分片完整保留，第3分片已写2380267520 bytes，剩约1.485GB；代理仍返回200。从相同snapshot/local_dir断点重启下载，PID3325834，日志download7b_resume01.log，未删除数据、未重跑3B。重新启动顺序流程，日志pipeline_resume01.log，明确绑定新下载PID，后续仍须完整LFS验证。此前失败日志保留。

断点续传317秒完成并核验11文件，model7b_manifest已回传本地。只停止本任务SSH隧道PID2328（先核对18084→12000命令），保留用户VPN与其他连接。云端已自动启动main7b_20261001_045806（Asia/Shanghai 2026-10-01 12:58:06），worker PID3328740，pipeline PID3326057，GPU0，同一独立conda。20题校准与普通attention恢复成功，开始100题交错评测，无运行异常。校准排名前14层为5/22/14/4/8/26/11/16/24/17/12/23/6/9，额外第0层后实际15/28层全量，其余保留首token。阶段和真实字节数已记录，不将名义50%当实际50%载荷。

已应用scientific-visualization技能，绘图辅助plot_results.py在本地默认Python已有matplotlib/numpy运行，不安装依赖；只接受完整main7b和840条评测记录，均值/置信区间仅基于首遍100题，另存数据与provenance。等待真实结果后才生成并视检，不能把未生成图或3B检查当7B结论。

main7b_20261001_045806已完成，COMPLETE时间2026-10-01T05:10:40Z（北京时间13:10:40）；流程随后完成汇总，GPU与守护进程退出。100个唯一题目、700条首遍答案、140条重复计时、20校准、7预热全部对账，1230次生成与262次额外sender prefill无失败。逐题原始输出和账目完整回传。独立只读重算700条标准EM/F1，与逐条记录和summary完全一致，重复计时没有混入评分。

全KV F1=75.79、EM=56/100，全文70.92、51/100；F1配对差4.86分的95% bootstrap区间[-1.01,10.88]包含零。两者平均0.210/0.212秒，不能称已证实加速。全KV8.13 MB对全文422 B，约1.9万倍；当前未测网络。选层降载荷46.1%但F1掉至52.47。采样99/100题保持源词序，不构成充分压缩检验。NLD无问题输入的原版流程有自行生成问题、丢掉答案事实的样例，不能只用它判断KV是否胜过文本。

报告report.md、interpretation.json、三面板PNG/PDF与CSV已生成，PNG已视检，provenance记录视检和hash。运行时run.py/config.json/assets.json与manifest一致，复制到本轮code目录保存。官方全文文本与KV提示格式、重复惩罚作用范围不同，本轮比较完整流程，不能把分数差都归因于通信表示。下一步建议先固定校准子集检查同格式和解码范围，再讨论三节点；此次未追加7B调用，500题仍未使用。

用户授权继续同格式诊断。已重新查询服务器GPU，0–4号均0%利用率、约24GB空闲；实际启动仍新鲜选卡。独立审查并行检查方案。准备读取exports/upstream_sources.tar.gz时发现不存在，改读现有coding_base/KVComm-official。此时尚未启动新的模型调用。

新诊断条件已先告知并保存prompt_alignment_conditions.md：校准名单前10题×7条件=70答案，第11题2个预热；72次generate、31次显式A prefill、10次raw拼接forward、10次raw CV接收forward。独立alignment_probe.py不修改run.py/config/官方源码。raw/processed logits、实际模型输入与首层cache_position和mask、首步和首次分歧完整向量均保存；只比较生成分歧前相同历史。CPU9次惩罚范围逐分数精确检查通过，3种长度×2种惩罚共6组拼接/CV生成token完全一致，首步logit最大差4.47e-8。CPU结果不是7B结论。

第一次启动alignment7b_20261001_054607在加载模型前失败：独立upstream目录为无.git的文件快照，git rev-parse检查不适用。worker PID3357455已退出，无GPU生成。保留失败目录和日志。改用既有assets中的pristine/work SHA验证官方源文件；同时补充实际RoPE位置hook、配置/tokenizer重新hash核对、显式FP32向量存储。CPU自检随脚本hash重新验证后再新run启动；实验条件和问题名单不变。

alignment7b_20261001_054917已完成，worker PID3359577退出；10题70诊断答案+2预热，72/31/10/10调用全部成功。独立复核20组KV、150个生成forward的实际RoPE/cache位置正确；无惩罚及1.05对齐范围的两对控制各10/10生成token完全一致。原版Skyline与拼接有5/10输出改变，其中3题改变事实/实体、2题主要是形式。生成阶段logits差非零（最大1.015625），全部B位置前向最大3.265625；不称逐位数值等价，不把全部误差归因于BF16。完整结果和51MB档案已回传。首次NumPy公式复算要求与CUDA逐位相等失败（差4.77e-7）；改为记录误差并核对固定2个FP32 ULP的跨算术实现舍入界限，GPU实验不变、不重跑。

用户授权Git提交与30分钟自动推进。工作区及本实验目录目前没有Git仓库，将只为独立实验目录建仓库，保留代码、固定数据、汇总和报告；完整原始items/vectors/账目仍保存在本地与云端。复用同线程已暂停的automation-2，已更新为“KV多智能体实验推进”、每30分钟ACTIVE，避免另建重复定时器。自动化先审计再准备三节点，不未经授权使用500题，不推送Git。

Git已在本实验目录初始化并提交c84e2d0，包含代码、805KB固定公开数据和经审计报告，不含模型、下载源码副本、原始向量、日志或瞬时进程状态。自动化本地配置再次核对ACTIVE、间隔30分钟、目标当前线程。补充.gitattributes禁止换行转换，防止Windows checkout改变冻结源码和数据的SHA；提交后核对Git存储字节与原运行manifest。现阶段下一步见next_steps.md：三节点角色/材料分工、完整历史文本与KV公平对照、cache末token补齐及逐边继承/新增字节。

三节点适配器chain_core.py、three_node_probe.py、three_node_cpu.py已构建，复用官方CVCommunicator，不改旧工程。三角色按支持文档分工，使用校准前3题与第4预热，完整历史每段有唯一来源账本。区分CV局部sequences与全局cache，缺0/1token才允许补尾，其余拒绝；检查源cache不被接收方修改。最多29次generate和14次补尾，接入门禁失败则20次并跳过自由链。先保存并把参考向量移到CPU释放cache，避免累积显存；门禁和自由链结果分别记。此刻尚未运行本轮CPU/7B检查；服务器新鲜查询0–4卡均0%和24081MiB空闲，未有本测试进程。

学校服务器已有conda的CPU随机FP32 Qwen检查通过：8次生成、5次EOS尾补，参考/重放/自由链均相同，缺2token被拒绝且未前向，已完成缓存再次补全不触发前向；源缓存不变，历史与两条边来源账本正确。不是7B实验结论。GPU条件已在对话先告知，启动独立chain3_20261001_122157，PID3553892，GPU0（0%/24081MiB空闲），最多29次生成/14补尾，只用3校准+1预热。新run源代码与配置快照保存，不安装环境、不改旧实验、不用500题。

chain3_20261001_122157完成并回传，进程退出/GPU0释放。20次generate+8次EOS尾补、20节点712生成forward均对账；B重放2/3、C3/3相同，所有EOS正常无触顶，门禁失败所以自由链0次。首题B第47token的means/corresponds排序翻转，末句释义同为German，尚未定位具体数值来源。原门禁未改。analyze_three_node.py独立重算历史、来源、位置、字节及向量hash；首次把CV可省略的原生cache_position入参误当实际位置，改为检查attention实际cache_position/RoPE，原始数据和GPU结果不变。另有独立只读审查得出相同结论。报告已写；后续同缓存原生接收控制正在准备。

新增独立native_cache_control.py/cache_path_probe.py，不修改原三节点脚本/条件/结果。CPU原生cached与CV检查通过，条件在对话先告知（同一首题4次生成/至多1补尾）；启动cachepath_20261001_123341，PID3560168，GPU4 UUID GPU-89523765-dcd4-46ea-1f2d-4ff7d509138c，fresh 0%/24081MiB。4次生成+1EOS尾补完成，197生成forward核对正确，A复现旧记录；原生cached与CV52token相同、logits最大差0.0625，joint与原生cached同样第47token分岔。所有EOS正常。实际A缓存内容11,411,456 bytes，text int32内容796 bytes；未测网络。不能精确归因为BF16/SDPA，也不能说已完成自由链。原始结果回传，审计脚本和独立只读审查一致；首次Windows本地freeze默认GBK读UTF-8失败，显式UTF-8修复，发生在GPU启动前。工程说明_简单版.md供用户阅读，下一步单独冻结自由链/原生缓存实现控制，不倒改原门禁。

Git35fe885已提交前述结果，工作树原先干净，bundle已备份云端。新heartbeat核对服务器无本任务GPU进程；0–3卡有其他任务，4卡0%/24081MiB，实际启动仍新鲜选择。正在新增free-chain独立运行器，只用原3校准+1预热、20生成/至多8补尾；原角色/材料/模型/预算一致，C读取实际CV-B缓存，另两次原生接收控制只做检查。独立设计审查通过，要求分开结构/EOS/控制匹配/全文差异，旧严格门禁失败不改。尚未启动本轮GPU。

新free_chain_core.py的CPU检查通过：实际A native→B CV→C CV，B/C控制只读取同一发送cache；5generate/2EOS尾补、来源账本/位置/固定FP32误差/实际边字节核对通过。已在对话说明正式条件与20/≤8账目，启动freechain_20261001_125456，PID3571380，GPU4（fresh0%/24081MiB），沿用既有conda。旧参考manifest/逐题输出/summary与数据均冻结hash，旧gate不改；本轮所有依赖源码和条件另有快照，新增FAILED.json便于保留断言失败。尚待本轮结果。

freechain_20261001_125456已完成并全部回传，进程退出/4卡释放。20generate/8补尾、主三题B/C控制各3/3、9主节点EOS，A和最终C分别复现旧参考；首题错误保留。analyze_free_chain.py独立核对714生成forward、实际CV-B→C来源、历史/分段/输入/旧参考hash、真实KV字节和评分，均通过；独立只读审查相同。边账本KV57,344bytes/token、跨边重复约31%，不称接收方已知冗余。CPU小检查及其报告保留。

用户要求整体审查及更新定时任务。发现自动化提示仍在准备三节点、research-question-card.md/next_steps.md也有过时下一步。已整体更新到接收方信息缺口：基础诊断收住，不继续无新故障的浮点/逐字检查；先准备C确已收到A的来源去重对照并给文本相同机会，语义增量另做且先保证多源接入。暂不扩G-Design图/训练/大评测，500仍需授权。通过app工具更新原automation-2并回读核对ACTIVE、30分钟、同一thread，未新建或暂停。此后未启动新GPU实验，当前补齐报告/代码/文档提交。
