# 05 · 展示视图

> 前端 10 个视图（`web/js/views.js` 的 `LI.views[*]`，导航定义在 `app.js` 的 `NAV`）。每个视图首访拉取对应 `/api/*` 段并用 ECharts 渲染，结果缓存。

## 5.0 视图总览

| # | id | 标题 | 数据源 | 现状 |
|---|---|---|---|---|
| 0 | `replay` | 全训练回放 ★ | step 粒度（多 step 待采集） | 骨架 + 联动机制 |
| 1 | `overview` | 总览 | `/api/overview` | ✅ 完整 |
| 2 | `hotspots` | 算子热点 | `/api/hotspots` | ✅ 完整 |
| 3 | `efficiency` | 算子效率 | `/api/efficiency` | ✅ 完整（MFU/MBU/Roofline） |
| 4 | `communication` | 通信分析 | `/api/communication` | ✅ 完整（单卡口径） |
| 5 | `hidden_overhead` | 隐性开销 | `/api/hidden_overhead` | ✅ 完整 |
| 6 | `attribution` | 结构归因 | `/api/attribution` | ✅ 完整 |
| 7 | `memory` | 显存洞察 | `/api/memory` + `/api/meta` | ⏳ 权衡顾问（峰值待采集） |
| 8 | `timeline` | 时间线 | `/api/timeline` | ✅ 泳道占用 / 频率 / 切片 |
| 9 | `insights` | LLM 洞察 ★ | `/api/insights`（+ POST `/api/llm`） | ✅ 卡片（LLM 默认关闭） |

> ★ = 旗舰差异化视图。导航顺序与上表一致；`overview` 为默认首页。

## 5.1 总览（Overview）
时间构成堆叠条 / 甜甜圈（计算 / 未掩盖通信 / Free）+ 关键比率卡片（有效计算 55%、未掩盖通信 26%、空闲 18.6%、通信掩盖率 12.6%）。若补芯片峰值 + 理论 FLOPs 可给 MFU。

## 5.2 算子热点（Op Hotspots）
排序条形 Top 榜 + **按 Core 类型聚合**（AI_CORE / AI_VECTOR_CORE / MIX_AIC / AI_CPU），一眼看出 AI_CPU 通信下发占比异常。下钻预留（点 OP Type → kernel 实例分布）。

## 5.3 算子效率（Compute Efficiency · H5）
- **Roofline 散点**：点 = kernel，x = 算术强度（FLOP/Byte），y = 达成算力（TFLOPS），对照 ridge point；自动圈出「耗时大但 MFU/MBU 低」。
- **按算子类型 MFU / MBU / 浪费表**：每个 type 一行，含融合注意力（FlashAttentionScore/Grad 现已计 MFU）。
- **优化余量排行**：按 `wasted_us`（vs Roofline 理想）排序，区分 compute-bound / memory-bound。
- 顶部横幅显示芯片名、假设 / 实测 / 有效峰值、HBM 带宽与**容量**（96/64 GB），并提示芯片为假设值、可切换或在 `config.ChipSpec` 校正。

## 5.4 通信分析（Communication）
按通信类型（allGather / alltoallv / allReduce / reduceScatter）聚合耗时与**等待占比**、带宽。单卡现象：几乎全为 Wait/Synchronization、Transit≈0。`communication_matrix` 热力图为**多卡预留**。

## 5.5 隐性开销（Hidden Overhead · H6）
device / host / config 三域分桶堆叠 + 每桶占 step 比例、Top 来源算子、减负建议。明确标注 host 与 device 不可直接相加、`ASCEND_LAUNCH_BLOCKING=1` 为采集干扰项。

## 5.6 结构归因（Attribution · H3）
模型结构耗时占比环（Embedding / MLA / MoE / Norm / Optimizer / Loss / 共用 GEMM）+ 通信单列（MoE-Dispatch vs TP/SP/DP）+ MoE 专项（专家计算 vs 分发通信）。

## 5.7 显存洞察（Memory · H8）
HBM 容量卡片（读 `/api/meta` 的 `chip.hbm_capacity_gb`）+ 内存-时间权衡顾问（recompute / swap）。峰值时间线 + 构成分解待 memory-level 采集；当前明确给出「待采集」原因。属 `CHIP_VIEWS`，切芯片刷新容量。

## 5.8 时间线（Timeline / Trace）
多泳道占用热力（Device / Communication / Host Runtime / Framework）+ Overlap 段 + AI Core 频率曲线 + Top 切片。104MB / 34 万事件经**后端流式切片**，前端只收 ~100KB。

## 5.9 LLM 洞察（Insight · H1）
- **诊断卡片**（规则引擎，始终渲染）：现象 / 根因 / 建议 / 预计收益 / 置信度，按严重度排序。
- **隐私安全摘要**：暴露「**将要发送给 LLM 的确切内容**」（透明可审计）。
- **「运行 LLM 分析」按钮** → `POST /api/llm`：仅在启用 + 有 key 时触发；默认关闭时显示规则引擎结果，优雅降级。
- 详见 [06](06-insights-and-llm.md)。

## 5.0′ 全训练回放（Training Replay · H2）旗舰
顶部贯穿训练过程的可拖动时间轴，叠加趋势线（耗时构成 / 未掩盖通信 / 显存峰值 / loss·吞吐·grad-norm）。拖动 / 播放即「回放」训练，联动刷新下方面板至该 step 快照，自动打标异常点，支持任意两 step Δ 对比。**数据现状**：当前仅 step5 单帧 —— 先以 step 粒度搭好回放骨架与联动机制，多 step 全程指标随后续采集（profiler 多 step + 训练日志）接入。

## 5.A 前端外壳与约定

- **`LI` 全局命名空间**（`util.js`）：`api`（带客户端缓存的 GET）、`apiPost`、`clearCache`、`fmt`、`esc`、`el`（DOM 构造）、ECharts 封装与 `resizeAll`。
- **路由**：hash 路由（`#overview` 等），首访渲染并缓存，芯片切换清相关缓存重渲染。
- **顶栏 badges**：模型名、并行（TP/PP/EP/CP）、**芯片下拉**（切换重算 MFU/MBU/Roofline/理论上界）、加载耗时。
- **侧栏**：LLM 状态 pill（开 / 关）、芯片峰值提示。
- **样式**：单一暗色主题 `css/app.css`，已做密度收紧使每个视图单屏可展示完毕。

---

上一篇：[04 · 指标层](04-metrics.md) ｜ 下一篇：[06 · 洞察与 LLM](06-insights-and-llm.md)
