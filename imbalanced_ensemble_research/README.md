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

每一条命令均创建 `results/<run_id>/`，其中保存 command、status、解析后的配置快照和相应报告；绝不覆盖既有运行。`run` 仅处理 `datasets` 中具有显式 `label_column` 与 `positive_label` 的数据集。`default.json` 的正式参数为 N=K=20、100 进化代、CR=0.2、F=0.5；`smoke.json` 明确缩小为 N=K=4、2 代，不能替代正式实验。

## 无泄漏边界与指标
先以固定种子做 7:2:1 分层训练/验证/测试划分。填补、编码、标准化只在训练集拟合；优化只读取验证集；PROMETHEE 决策后才对独立测试集评估一次。基线包括 RBF-SVM、class_weight='balanced' RBF-SVM、RandomForest；只有检测到 `imbalanced-learn` 才运行 RUSBoost。指标为混淆矩阵、FPR、FNR、precision、recall、F1、G-mean，以及条件满足时的 ROC-AUC/PR-AUC。

## 原型语义
统一染色体为 `x=(f,i,m,w)`，f/i/m 均以 0.5 解码。空段确定性保留 argmax（并列取低索引）；i 再补足每个训练类别的最小索引样本。若一个模型槽位不可训练则标为 unavailable，选中它的候选无效，绝不静默跳过。每个槽位的 `f/i`、种子、SVM 参数和训练修复被不可变 `GenerationPoolManifest` 保存。m/w 只能选择同一 `pool_uid` 的 K 槽位；最终重训接收候选和该清单，逐槽位按清单的不同 f/i 重建。

初始化只评估一个 parent pool。随后每一个 g=1..generations 仅产生并评估一个 N=K 的 trial pool；保留的父解缓存其最后验证目标和绑定清单。运行清单记录 `initialization_pool_count=1`、`evolution_generation_count` 和 `pools_per_evolution_generation=1`。

验证目标均最小化：FPR、FNR 与 `C=.35|f|/D+.35|i|/T+.20|m|/K+.10(1-H(w)/ln|m|)`；单模型时最后项为 1。系数来自配置。PROMETHEE 使用最小化 V-shape 偏好：q=0.01、p=0.05（均可配置），权重默认 0.45/0.45/0.10，以净流降序选择。

## 限制
标签语义、标签派生字段、组/时间泄漏和 spMODE-II 原文细节仍须用户/原文核实。若依赖缺失或本机 Python 二进制包不兼容，报告会记录固定版本建议，不会自动安装。论文材料只提供结构和方法草稿，不会制造实验结论。
