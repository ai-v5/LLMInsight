# 06 · 洞察与 LLM

> 规则引擎 12 条诊断卡片（事实依据）+ 可插拔、默认关闭的 LLM 叙述层（增强）。

## 6.1 设计原则

- **规则引擎是事实底座**：命中项 + 昇腾最佳实践，让建议可执行、降低 LLM 幻觉。
- **LLM 是可选增强**：未配置任何 LLM 时，规则引擎 + 全部可视化照常工作（优雅降级）。
- **隐私优先**：只把**结构化指标摘要**（KB 级 JSON）发给 LLM，**绝不上传原始 trace**。
- **默认关闭、独立分发**：网络调用默认 no-op；软件可在任何机器独立运行，不依赖任何特定环境。

## 6.2 规则引擎：诊断卡片

`rules/engine.py::run_rules` 读 metrics 字典 + **从 profiling 反推的配置**（`parser/derive.py::derive_config` 由 kernel shape / 算子计数 / 通信结构反推模型结构与训练·采集状态，**不依赖启动脚本**——脚本与 profiling 不保证一一对应）。单卡单 step 无法确定的字段（EP world size、总层数、总专家数、global batch）以 `未知(猜X)` 标注，不臆造确定值。输出卡片，结构：

```python
{ "id", "severity"(high/medium/low/info), "category", "title",
  "root_cause", "suggestion", "expected_gain", "confidence", "evidence" }
```
卡片按 `(severity, confidence)` 降序排序。

### 12 条洞察清单

| # | id | 严重度 | 类别 | 触发条件（要点） | 置信 |
|---|---|---|---|---|---|
| 1 | `comm_not_overlapped` | high | 通信 | 未掩盖通信 ≥10% 或 重叠率 <40% | 0.9 |
| 1b | `comm_zero_transit` | medium | 通信 | 零载荷（Transit≈0）collective 占通信 ≥5% 或 step ≥3%——并行组内 barrier/相位对齐占位 | 0.6 |
| 2 | `aicpu_dispatch` | high | 通信 | `HcclLaunchAicpuKernel` 占比 ≥10%（AICPU 驱动的集合通信执行，≈通信本身、非下发延迟） | 0.85 |
| 3/11 | `capture_blocking` | high | 采集体检 | 反推 blocking=True（Synchronize/launch 次数比 >0.5）——异步下发时**不命中** | 1.0 |
| 3b | `host_sync_stall` | medium | 等待·同步 | 非 blocking 但 `aclrtSynchronizeStream` 占 host API ≥20%（根因：动态 shape 的 D2H 同步，如 `aclnnMaskedSelect`） | 0.8 |
| 4 | `dynamic_shape` | medium | 动态shape | MaskedSelect/NonZero host 累计 >50ms | 0.7 |
| 5a | `low_efficiency_ops` | medium | 算子效率 | 存在 `wasted_us>0` 的 Top 候选 | 0.6 |
| 5b | `peak_underestimated` | info | 算子效率 | 实测 matmul 峰值 > 假设峰值（已校准） | 0.8 |
| 6 | `free_bubble` | medium | 空泡 | `free_pct ≥10%` | 0.65 |
| 7 | `recompute_full` | low | 显存-时间 | 反推 recompute=full（FA 前向/反向次数比 ≈2） | 0.6 |
| 8 | `hidden_overhead_ledger` | medium | 隐性开销 | hidden_overhead 可用 | 0.7 |
| 9 | `theoretical_whatif` | info | 理论上界 | theoretical 可用（取最优 What-if） | 0.7 |
| 10 | `parallelism_advisor` | medium | 并行策略 | 存在 alltoallv（EP 开启，world size 为猜测）且未掩盖通信 ≥15% | 0.55 |
| 12 | `memory_capture` | info | 显存 | 无 memory-level 采集 | 0.9 |

卡片集合根据本次采集的配置、覆盖率和阈值动态命中，不使用固定模型或内部采集结果
作为公开基线。规则证据显示在本地 UI 中；LLM 只接收经过有限脱敏的摘要。

### 洞察的解读边界

- 通信暴露、重叠不足和 AICPU 通信展开可能描述同一时间段，不能重复累计优化收益。
- Host 同步与动态 shape 的关联是排查线索，不能仅凭同步耗时认定采集开启 blocking。
- 芯片分母来自使用者选定的参考配置，实际 SKU 和采集条件需要独立核对。

## 6.3 LLM 洞察层

### 6.3.1 调用门控（默认关闭、显式触发）

```mermaid
flowchart TD
  A[打开 LLM 洞察面板] --> B[GET /api/insights]
  B --> C[规则卡片 + 摘要<br/>narrative = null]
  C --> D{点「运行 LLM 分析」?}
  D -- 否 --> C
  D -- 是 --> E[POST /api/llm]
  E --> F{provider.available?<br/>enabled 且 有 key}
  F -- 否 --> G[返回清晰原因<br/>仍显示卡片]
  F -- 是 --> H[发送 KB 级摘要<br/>返回中文叙述]
```

- `GET /api/insights` 用 `call_llm=False`：**仅打开面板从不消耗 LLM 请求**。
- `POST /api/llm` 是**唯一**触发；任何失败（网络 / 鉴权 / 解析）降级为「仅卡片 + 错误原因」。

### 6.3.2 可插拔 Provider（`insight/provider.py`）

纯 stdlib `urllib`（无三方 SDK），以 **OpenAI 兼容**为统一抽象，外加 **Anthropic** 适配器。预置后端：

| provider | base_url | 默认 model | 协议 |
|---|---|---|---|
| `glm`（默认） | open.bigmodel.cn | glm-4-flash | openai |
| `deepseek` | api.deepseek.com | deepseek-chat | openai |
| `openai` | api.openai.com | gpt-4o-mini | openai |
| `siliconflow` | api.siliconflow.cn | Qwen2.5-7B-Instruct | openai |
| `ollama`（本地，免 key） | localhost:11434 | qwen2.5 | openai |
| `anthropic` | api.anthropic.com | claude-3-5-haiku | anthropic |

一处配置即可切换，不锁死任何一家；本地 Ollama / vLLM / MindIE 可离线、数据不出内网。

### 6.3.3 环境变量开关

| 变量 | 作用 | 默认 |
|---|---|---|
| `LLMINSIGHT_LLM_ENABLED` | `1` 开启调用 | **关（0）** |
| `LLMINSIGHT_LLM_PROVIDER` | glm / deepseek / openai / siliconflow / ollama / anthropic | glm |
| `LLMINSIGHT_LLM_MODEL` | 覆盖模型 id | 预置 |
| `LLMINSIGHT_LLM_BASE_URL` | 覆盖端点 | 预置 |
| `LLMINSIGHT_LLM_API_KEY` | key（否则回落 `secret/api_key.txt`） | — |

> **分发约定**：代码层 `enabled` **默认 False**，重启脚本也默认关闭网络调用。需要先设置 `LLMINSIGHT_LLM_ENABLED=1`，核对接收服务和摘要，再显式运行分析。API key 不进入 Git、摘要或日志。

### 6.3.4 隐私安全摘要（`insight/summarizer.py`）

`build_summary` 产出**就是 LLM 看到的确切 payload**（服务端暴露给用户审计）：
- 不发送 raw trace、张量数据或完整环境。Top 算子实例以 `operator_1` 等本地序号表示；含可识别路径、URL、邮箱、工号样式标识或 IP 的字符串字段整段移除。
- 这是有限脱敏，不能保证匿名或允许公开：模型结构、算子类别及性能指标本身仍可能敏感。运行前应展开面板中的摘要，核对接收服务与发送内容。
- 采集配置经**白名单**过滤：env 仅 `ASCEND_LAUNCH_BLOCKING / HCCL_BUFFSIZE / …`，flags 仅性能相关开关。
- system prompt 把 LLM 定位为「昇腾 NPU 训练调优专家」，要求**只用给定数据、不臆造数字**，建议落地到具体开关 / 参数，并对被 blocking 放大的 host 指标标注采集干扰。

### 6.3.5 后续增强（设计到位、待接线）

对话式追问（「为什么 alltoall 这么贵」）、昇腾最佳实践知识库、Timeline 自动导读（在时间线上自动标注「此处 alltoall 卡顿」「此处重计算开始」）—— 接口已预留，按 [07](07-roadmap-and-verification.md) 的 P3/P4 推进。

---

上一篇：[05 · 展示视图](05-views.md) ｜ 下一篇：[07 · 路线图与验证](07-roadmap-and-verification.md)
