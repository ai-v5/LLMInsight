# 04 · 指标层

> 从 ProfileData 派生的全部分析段：口径、公式与设计取舍。编排见 `metrics/build.py::compute_all`，落点 `core.py` / `efficiency.py` / `timeline.py`。

每个段返回 `{"available": bool, ...}`；不可用时前端优雅显示占位。

---

## 4.1 overview · Step 时间构成与关键比率

读 `step_trace_time.csv` 中首个 device 的完整 step 窗口，给出时间构成与比率。单 step 时窗口即该 step；多 step 时 `us` 为窗口总量、`avg_us` 为单步平均，界面同时显示两种口径：

- **时间构成**（堆叠条）：Computing / Communication(Not Overlapped) / Free，分母为 `Stage`。
- **关键比率**：
  - `effective_compute_pct = Computing / Stage`（有效计算占比 ≈ **55.29%**）
  - `comm_not_overlapped_pct = Communication(Not Overlapped) / Stage`（≈ 26%）
  - `free_pct = Free / Stage`（≈ **18.56%**）
  - `overlap_rate_pct = Overlapped / Communication`（计算-通信重叠率 ≈ **12.62%** —— 偏低，重点洞察）
  - `step_time_s = avg(Stage) / 1e6`（单步平均）
  - `window_time_s = sum(Stage) / 1e6`（完整窗口；MFU/HFU 的 kernel 分子也覆盖同一窗口）

## 4.2 hotspots · 算子热点

读 `op_statistic.csv`：
- **Top 榜**：按 `Total Time(us)` 降序（HcclLaunchAicpuKernel ~28%、GroupedMatmul、MatMulV3、FlashAttention 前后向、GemmV3…）。
- **按 Core 类型聚合**：AI_CORE / AI_VECTOR_CORE / MIX_AIC / **AI_CPU**，一眼看出 AI_CPU（通信下发）占比异常。

## 4.3 efficiency · MFU / MBU / Roofline（H5 核心）

`efficiency.py` 对每个 device kernel 由 shape·dtype·duration 估算达成算力 / 带宽，对照芯片 Roofline 排出优化余量。**这是与现有工具拉开差距的关键模块**。

### 4.3.1 FLOPs 估算

**Matmul 家族**（`MatMulV3 / MatMul / GroupedMatmul / GemmV3 / BatchMatMul…`）：
```
FLOPs = 2 · M · N · K · batch
```
- batch 只取**激活侧**。`GroupedMatmul` 的权重是 3-D `[E,K,N]`，但每个 token 只访问一个专家，token 维已经统计了总工作量，**乘 E 会重复计数** → 故不乘。

**融合注意力**（`FlashAttentionScore / FlashAttentionScoreGrad / PromptFlashAttention…`）—— 因为 QK^T + softmax·V 都在 cube 上，其有效 FLOPs 应计入 MFU：
```
FLOPs_fwd = 2 · B · N · S² · (d_qk + d_v) · causal_factor
FLOPs_grad = FLOPs_fwd · 2.5
```
- **布局无关**：B, N, S 从 softmax max/sum 张量 `[B,N,S,k]`（trailing 维很小，fwd 的输出 / grad 的输入）读出；head 维由 `numel / (B·N·S)` 反推，于是 MLA 的 `d_qk(192)=128+64` 与 `d_v(128)` 被自动捕获。对真实 **SBH** 布局同样成立。
- **`causal_factor = 0.5`（物理必需，非便利）**：DeepSeek-V3 注意力是因果的，FlashAttention 跳过上三角被 mask 的块，只做约一半 QK^T+PV。**去掉这个 0.5，达成算力会超过硅片峰值** —— 这就是它必需的证明。
- **反向 ≈ 2.5× 前向** 的 matmul FLOPs。

### 4.3.2 峰值校准（避免非物理的 MFU > 100%）

`config.ChipSpec` 的峰值是**假设值**（910B 保守取 376 TFLOPS）。配置值不等于采集设备身份；若同一采集中由完整 shape/dtype 支持的算子结果明显超过所选假设峰值，应先检查参考配置、时钟、数据口径及模型公式。

```
observed_peak = max over clean matmul of (FLOPs / duration)
calibrated    = observed_peak > configured_peak
effective_peak = min(observed_peak · 1.02, configured_peak · 2.0)   # 仅当 calibrated
```
- 把有效峰值抬到实测上界（+2% 余量，使顶点 kernel 读到 ~98% 而非可疑的平 100%）；上限 2× 防止 shape 误解析伪装成巨大峰值。
- **assumed / observed / effective 三个峰值都在 UI 暴露**；在 `ChipSpec` 填真实 SKU 峰值即可用精确值替代校准。此过程是采集内估算，不是厂商规格校准。

### 4.3.3 bound 判定、浪费时间与优化排行

对每个 kernel：
```
achieved_flops = FLOPs / dur ; mfu = achieved_flops / effective_peak
achieved_bw    = bytes / dur ; mbu = achieved_bw / hbm_bandwidth
t_compute = FLOPs / effective_peak ; t_mem = bytes / hbm_bandwidth
ideal_us  = max(t_compute, t_mem) ; wasted_us = max(0, dur - ideal_us)
bound = "compute" if t_compute >= t_mem else "memory"
```
- 只有「有 FLOP 模型（matmul / 融合注意力）」**或**「向量核访存算子」才建模理想耗时；其它无 FLOP 模型的 cube/MIX 算子在纯访存 Roofline 下会显得 100% 浪费，故**留空不评分**（`_bound_from_ratios` 用流水线占比给个定性 bound）。
- KDA、gated-delta、GLA、causal-conv，以及 MC2/MegaMoE 融合模型计算族进入覆盖分母并列入 `unmodeled_flop_types`。仅在同采集公式锚与计数器足够时，用实际 MAC/Vector 活跃时间（缺失时退回周期比例）校准工作量，标为 `PROXY MFU(est)`，保留 ±25% 未建模贡献敏感性区间；缺锚、缺计数器或峰值不一致时保持不可用，未知算子不生成 Roofline 收益。时间覆盖率不等于 FLOP 完整性。
- `top_optimization`：按 `wasted_us`（实测 − 理想）降序，自动圈出「耗时大但 MFU/MBU 低」的 kernel。
- 通信下发"kernel"（AI_CPU / HCCL）无可建模的计算 / 访存足迹，**从效率排行剔除**（归通信视图与隐性开销）。

### 4.3.4 口径：headline matmul_mfu 保持纯 GEMM

`matmul_mfu` 这个**头条数字仅统计纯 GEMM**（MATMUL_TYPES），是峰值校准的锚点；融合注意力有自己的 per-type MFU 行，不并入头条以保持其含义。`kernels_with_flops` 与 Roofline 散点则包含注意力。

> 示例数值由合成测试覆盖；不在公开文档中保留未经外部来源核验的真实采集 MFU。

## 4.4 communication · 通信分析

读归一化后的集合通信：
- **按类型聚合** elapse / wait / sync / transit，并算**等待占比**。等待占比用「每算子 wait/elapse 比率的均值」（`overall_wait_pct`），**避免** ∑wait / ∑elapse 产生的 >100% 假象。
- 单卡现象：集合通信几乎全为 Wait/Synchronization，Transit≈0、带宽≈0 → 通信时间以「等待对端」为主，真实链路带宽需多卡数据。

## 4.5 hidden_overhead · 隐性开销分桶（H6）

把分散开销归类为桶，每桶给占比 + 来源 + 减负建议：

| 桶 | 域 | 来源 | 典型现象 |
|---|---|---|---|
| AICPU 集合通信执行（同段，不计入合计） | device | op_statistic (AI_CPU) | `HcclLaunchAicpuKernel` 占 28%，≈ Communication 的 72% |
| 通信未掩盖 | device | step_trace + comm + kernel | 直接计入 step |
| 空泡 / Free | device | step_trace | ≈18.56% |
| 格式转换 / 内存初始化 | device | op_statistic | Cast/ZerosLike/TensorMove |
| Host Kernel Launch | host | api_statistic | launch 类累计 |
| Host 同步阻塞 | host | api_statistic | `aclrtSynchronize*`（被 blocking 放大） |
| 动态 shape 抖动 | host | api_statistic | MaskedSelect/NonZero，max 166ms |
| 重计算 | config | 训练脚本 | `--recompute-granularity full` |

> **口径警示**：host 与 device 时间**不可直接相加**（部分并发 / 被 blocking 放大）。device 桶与 step 同口径可比；host 桶反映下发 / 同步压力。用于「相对量级与归因」。
>
> **避免重复计入**：`HcclLaunchAicpuKernel` 是 AICPU 展开模式下驱动集合通信的 AI_CPU 算子。其设备时长可能覆盖 collective 执行或等待，不应直接当作内核启动延迟。它与通信分解可能是同一时段的两种视角，标记为 `additive:false`，不并入 Device 合计；具体等待与传输比例由当前采集核对。

## 4.6 attribution · 模型结构归因（H3）

按算子命名启发式把 device 计算时间归类到模型结构：

| 模块 | 代表算子 |
|---|---|
| MoE-Experts | GroupedMatmul, SwiGlu, ScatterAdd… |
| MoE-Router | TopKV2, Sigmoid, Sort, Cumsum… |
| Attention-MLA | FlashAttentionScore(Grad), RotaryPositionEmbedding… |
| Norm | RmsNorm(Grad) |
| Optimizer | ApplyAdamW(V2) |
| Embedding/Loss | GatherV2, EmbeddingDenseGrad, Exp, Log |
| GEMM/Projections（共用） | MatMulV3, GemmV3, BatchMatMul |

- **排除通信下发算子**（AI_CPU / HCCL 前缀），否则 676k us 会污染 Elementwise。
- **通信单独列出**（MoE-Dispatch alltoallv vs TP/SP/DP），为 wall-clock（多为等待），**不并入计算环**以免混口径。
- **MoE 专项**：专家计算（GroupedMatmul）vs 分发通信（alltoallv）占比。

## 4.7 memory · 显存与内存-时间权衡（H8）

未记录 memory-level 文件（如 `memory_record.csv` / `npu_module_mem.csv`）时，HBM 峰值时间线会返回 `available:false` 和原因；不由频率计数器推断内存占用。

当前提供**配置驱动的内存-时间权衡顾问**：
- `--recompute-granularity full`：省激活显存，代价是反向重跑前向。
- `--swap-optimizer`：优化器状态 HBM↔Host 换入换出，省 HBM、代价 H2D/D2H 拷贝与同步。

芯片 HBM 容量（910B=64GB / 950DT=96GB，来自 `ChipSpec.hbm_capacity_gb`）作为 OOM 余量上下文展示。完整分解待 memory-level 采集接入。

## 4.8 theoretical · 理论上界 + What-if（H7）

由 step 时间构成推导优化上界（用于**排序**，非精确预测）：

| What-if 场景 | new_step | 节省 |
|---|---|---|
| 通信完全掩盖（未掩盖→0） | Computing + Free | comm_not_overlapped |
| 消除空泡（Free→0） | Computing + comm_no | Free |
| 通信掩盖 + 空泡减半 | Computing + Free·0.5 | comm_no + Free·0.5 |

并结合 `matmul_mfu` 给 **compute-bound** 上界：理想 matmul 耗时 = `Computing · matmul_mfu`，余量 = `Computing − ideal`。若 MFU>1 则报「峰值假设偏低」而非负余量。混合模型的重计算由三层证据联合判断：显式 `recompute_*` kernel、前向/反向调用数超额、MatMul 方向/次数相位拆分；输出主判断、候选算子、耗时/FLOPs 占比及置信区间，不再因自定义模型族而放弃结论。

## 4.9 timeline · 时间线（流式 + 缓存）

`timeline.py` 两遍流式扫描 `trace_view.json`：
- **泳道占用**：Device / Communication / Host Runtime(CANN) / Framework(Python) 四泳道，每个时间桶（默认 600 桶）内事件覆盖比例归一化到 [0,1]。
- **Overlap 段**：取自 profiler 的「Overlap Analysis」泳道（Computing / Communication / Free 轨道）。
- **Top 切片**：Device 泳道 >1.5ms 的事件 Top 60。
- **AI Core 频率**：Counter 事件（Die 0），降采样到 ~240 点。

结果 ~100KB，按 trace 文件签名落盘缓存。

---

## 4.10 框架阶段与融合 MoE 的核算边界

有 `ascend_pytorch_profiler_*.db` 时，只读关联框架注解、CANN launch 与设备任务。
优化器阶段按同进程 `Optimizer.step` 注解窗口中的 launch 判定，支持工作线程下发；
这是窗口归属推断，不等同于完整调用树。模型 FLOPs、重计算和模型效率排行排除这些
任务，完整 step 分母及矩阵算力视图仍保留优化器工作。

MegaMoE 以计算/通信融合窗口单列。框架输入 shape 可补充隐藏维、路由宽度与专家中间维，
但不从权重切片首维外推总专家数。缺少实际路由 token 数与内部阶段记录时，通信时长、
通信字节和纯计算时长保持未知；不得用融合时长减 MAC 活跃时间充当通信时间。
无公式的已识别模型族仅在同采集公式锚和计数器齐备、峰值一致时参与周期代理估算。
代理结果标为 `PROXY`；贡献敏感性区间不是统计置信区间或真机校准误差。

矩阵算力视图同时展示 MegaMoE 前后向：计数和耗时包含完整融合窗口，
`PROXY` FLOPs 仅取同采集校准的 MAC/CUBE 贡献，不把 Vector 工作计为矩阵 FLOPs。
与模型效率页的 MAC+Vector 代理口径不同，矩阵 MFU 仍以融合窗口作分母，
不代表纯计算 MFU。无 CSV dtype 时可使用明确的 kernel 精度标记；缺少精度、
校准或 MAC 计数器时 MFU 保持未知，MBU 也保持未知。

Compressor/Grad、LightningIndexerV2 与 SparseLightningIndexerKLLossGrad 作为独立的
计数器代理族核算，不直接复用旧 Indexer 的 shape 公式。首页 Computing/Stage 是设备
占用口径，包含优化器和访存算子；矩阵算力页也保留优化器，因此二者不能直接相乘
得到模型 MFU。融合 MoE 的详细工作量核算放在算子效率页。

未配对 GEMM 不自动认定为重计算。重计算候选时间是累计上界，尚未扣除通信或其它
算子重叠，不能当作已验证的关键路径收益。通信等待按实际区间跨桶分摊，
`NOTIFY_RECORD` 不计作等待。P2P 可能来自优化器或权重交换，不能据此推断 PP。

上一篇：[03 · 数据与解析](03-data-and-parsing.md) ｜ 下一篇：[05 · 展示视图](05-views.md)
