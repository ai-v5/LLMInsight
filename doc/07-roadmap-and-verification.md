# 07 · 路线图与验证

> 分阶段落地（含当前状态）、单卡优先 / 多卡专项、以及数值 + 渲染双重验证。

## 7.1 分阶段落地

| 阶段 | 内容 | 状态 |
|---|---|---|
| **P0** | 数据解析 + 统一数据模型（单卡），跑通本数据集 | ✅ 已完成 |
| **P1** | 核心可视化：总览 / 算子热点 / 算子效率（MFU·MBU·Roofline）/ 通信 / 时间线 / 隐性开销 / 显存 + 回放骨架 | ✅ 已完成 |
| **P2** | 分析引擎：规则引擎 + 模型结构归因 + 理论上界·What-if + 并行策略顾问 + 采集体检 | ✅ 已完成 |
| **P3** | LLM 洞察层：Provider 抽象 + 「摘要 → LLM」构造 + 诊断面板（先渲染规则洞察） | ✅ 已落地（调用默认关闭） |
| **P4** | 全程化 & 对比：多 step 全训练回放接入 + 跨实验 A/B + 回归告警 / CI | ⏳ 待多 step 采集 |
| **P5** | 多卡专项（M1–M5）：慢卡 / 负载均衡 / 通信矩阵 / 通信对齐 / 并行维度 | ⏳ 待多卡数据 |

> P3 与原计划「调用先 no-op 占位」相比已**前进一步**：Provider 抽象 + 摘要构造 + 真实 OpenAI 兼容 / Anthropic 客户端均已实现，仅**默认关闭**（分发安全）。对话 / 知识库 / Timeline 自动导读仍属后续增强。

## 7.2 单卡优先、架构预留多卡

- 当前样例为**单卡 / 单 step**；主要功能先面向单卡做透。
- 数据模型、`meta`、视图按 `rank × step` 设计，rank 维度先留空接口：`communication_matrix` 为空 `{}` 时空跑不报错。
- 多卡数据接入后，rank 维度填充即解锁 M1–M5（见 [01 §1.6](01-overview-and-goals.md)）。

## 7.3 验证

端到端目标：用现有这份数据跑通「解析 8 文件 → 派生指标 → 各视图渲染 → 洞察命中」。两套脚本守护：

### 7.3.1 后端数值基线（`scripts/verify.py`）

断言关键派生数字与回归基线一致：

| 断言 | 期望值 |
|---|---|
| Computing | 1,728,916.559 us |
| Communication (Not Overlapped) | 817,426.625 us |
| Stage（step 总时长） | 3,126,771.5 us |
| 计算-通信重叠率 | ≈ 12.62 % |
| 有效计算占比 | ≈ 55.29 % |
| 空闲 Free 占比 | ≈ 18.56 % |
| `HcclLaunchAicpuKernel` 占比 | 28.091 % |
| `aclnnMaskedSelect` 单次 max | 166,285.31 us |
| `matmul_mfu` | ≤ 100 %（样例 ≈ 0.9313） |
| **每个 per-type MFU** | **≤ 100 %**（守护 FlashAttention causal 因子） |
| 峰值校准 | observed > assumed → calibrated=true |
| 理论 What-if 场景数 | == 3 |
| 诊断卡片数 | 12 |

> per-type MFU ≤100% 这条专门守护融合注意力的 causal 0.5 因子：若有人误删它，FA 的 MFU 会冲破 100%，断言立即失败。

### 7.3.2 前端渲染校验（`scripts/verify_web.js`）

无头 Node 谐波：注入 stubbed DOM / ECharts，加载 `util.js` + `views.js`，对运行中的服务器（默认 :8765）拉取真实数据，**渲染全部 10 个视图**，并扫描 `undefined` 泄漏。芯片切换后 `CHIP_VIEWS` 重渲染同样校验（910B 与 950DT 两种状态均需通过）。

### 7.3.3 洞察校验

方案应自动命中 [06](06-insights-and-llm.md) 的 12 条洞察，尤其：通信未掩盖 / AI_CPU 下发 / 同步阻塞 / 隐性开销总账 / 理论上界差距。

### 7.3.4 多卡预留校验

数据模型在 rank 维度可空跑（当前 `communication_matrix` 为空不报错）。

## 7.4 如何运行验证

```bash
# 1) 启动服务器（后端 + 静态前端）
python -m llminsight.server --no-browser --port 8765
#    或用 ./restart_insight.sh（自动停旧起新 + 轮询就绪）

# 2) 后端数值基线
python scripts/verify.py

# 3) 前端渲染校验（需服务器在 :8765 运行）
node scripts/verify_web.js
```

---

上一篇：[06 · 洞察与 LLM](06-insights-and-llm.md) ｜ 返回：[文档入口](README.md)
