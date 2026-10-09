# 面向不平衡数据集的多目标优化集成学习（研究准备）

本项目与 `C:\git` 根目录既有 Athena/AWS 脚本隔离；不导入、修改或执行它们。仅按 JSON 配置只读访问 `inputs.data_dir` 与 `inputs.reference_pdf`，不下载数据、依赖或资料。实现是 **spMODE-II-inspired** 的可审计原型，不是对未提供原文算法的严格复现。

## 命令
在项目根目录执行：

```powershell
python -m src.imbalance_research.cli check-env --config configs/smoke.json
python -m src.imbalance_research.cli audit --config configs/smoke.json
python -m unittest discover -s tests -v
python -m src.imbalance_research.cli run --config configs/smoke.json
python -m src.imbalance_research.cli review --config configs/smoke.json
```

每一条命令均创建 `results/<run_id>/`，其中保存 command、status、解析后的配置快照和相应报告；绝不覆盖既有运行。`run` 仅处理 `datasets` 中具有显式 `label_column` 与 `positive_label` 的数据集。

`configs/smoke.json` 是唯一可执行的缩小冒烟配置，强制为 N=K=4、2 个进化代，且**不能代表正式实验或论文结论**。`configs/formal.template.json` 固定正式边界 N=K=20、100 代，但标记 `requires_user_label_semantics_confirmation=true` 且 CLI 明确拒绝运行；必须先由用户确认所有 14 个数据集的标签业务语义、正类映射及适用性，再创建可运行的 `formal.json`。仓库不保留空数据集的 `default.json`。

## 无泄漏预处理、阈值与指标
原始记录先以固定种子做 7:2:1 分层训练/验证/测试划分，三组索引互斥。`preprocessing` 必须为全部原始特征显式指定且只指定一次：`drop_columns`、`identifier_columns`（均在拟合前删除）、`numeric_columns` 或 `categorical_columns`。配置列名会与实际数据集特征核验；保留列未分类、未知列或重复分类都会拒绝运行。

数值列仅以训练集非缺失值中位数填补并标准化；训练全缺失列安全删除。类别列仅从训练集建立 one-hot 类别集合，缺失值为训练内 `__MISSING__` 类别，验证/测试未知类别编码为全零；训练全缺失类别列安全删除。训练常量的编码输出也会删除。结果 JSON 的 `preprocessing` 字段记录列策略、删除理由、训练拟合中位数、类别集合、未知类别计数和标准化统计量，配置快照保留原始决策。

`optimization.threshold` 必须在 `[0,1]`，统一用于基线测试预测、验证 FPR/FNR 目标及 PROMETHEE 后一次独立测试。实际阈值记录于顶层结果、指标、优化清单和运行状态。优化接口只接收训练/验证数据；测试集不传入候选评价或 PROMETHEE，决策后才按选中清单独立评估一次。指标为混淆矩阵、FPR、FNR、precision、recall、F1、G-mean，以及条件满足时的 ROC-AUC/PR-AUC。

## 原型语义
统一染色体为 `x=(f,i,m,w)`，f/i/m 均以 0.5 解码。空段确定性保留 argmax（并列取低索引）；i 再补足每个训练类别的最小索引样本。若一个模型槽位不可训练则标为 unavailable，选中它的候选无效，绝不静默跳过。每个槽位的 `f/i`、种子、SVM 参数和训练修复被不可变 `GenerationPoolManifest` 保存。m/w 只能选择同一 `pool_uid` 的 K 槽位；最终重训接收候选和该清单，逐槽位按清单的不同 f/i 重建。

初始化只评估一个 parent pool。随后每个 `g=1..generations` 只产生并评估一个 N=K 的 trial pool；结果保存逐池 manifest，并记录 `pool_count=1+generations`、`initialization_pool_count=1` 和 `pools_per_evolution_generation=1`。验证目标均最小化：FPR、FNR 与 `C=.35|f|/D+.35|i|/T+.20|m|/K+.10(1-H(w)/ln|m|)`；单模型时最后项为 1。PROMETHEE 使用最小化 V-shape 偏好，q/p/权重均由配置记录。

## 限制
标签语义、标签派生字段、近重复、组/时间泄漏和 spMODE-II 原文细节仍须用户或原始资料核实。若依赖缺失或本机 Python 二进制包不兼容，报告会记录固定版本建议，不会自动安装。论文材料只提供结构和方法草稿，不会制造实验结论。
