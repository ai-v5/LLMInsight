# 03 · 数据与解析

> 8 个 profiler 文件 → 能力映射、统一数据模型 ProfileData、解析细节、维度预留、校验基线。

## 3.1 数据 → 能力映射

一份 `ASCEND_PROFILER_OUTPUT` 目录含 8 个文件，每个解锁不同分析能力：

| 文件 | 内容 | 解锁的展示 / 洞察 |
|---|---|---|
| `step_trace_time.csv` | step 级耗时拆解（计算 / 未掩盖通信 / Overlapped / Free / Bubble / Preparing） | 总览、计算-通信重叠率、空闲占比、有效计算占比 |
| `op_statistic.csv` | 按算子类型聚合（Count / 总耗时 / 占比 / Core 类型） | 算子热点榜、按 Core 类型聚合、结构归因 |
| `api_statistic.csv` | host 侧 API（acl / aclnn / 通信）耗时 + 方差 | host-bound 判定、同步点、动态 shape 抖动 |
| `kernel_details.csv` | 每个 device kernel 的流水线指标（mac/mte2/mte3/vec 占比、cube 利用率、shape、dtype） | 计算 vs 访存 bound、MFU/MBU、Roofline |
| `operator_details.csv` | 框架层算子（aten / aclnn）host+device 耗时、调用栈 | host/device 自耗时对比（预留细化） |
| `communication.json` | 每个 HCCL 集合通信的 时间 / 等待 / 同步 / 带宽 | 通信瀑布、等待占比、带宽利用率 |
| `communication_matrix.json` | 跨 rank 通信矩阵（单卡为空 `{}`） | **多卡预留**：通信热力图、链路瓶颈 |
| `trace_view.json` | Chrome/Perfetto 时间线，34 万事件、多泳道 | 可缩放时间线、泳道占用、重叠分析 |

## 3.2 统一数据模型 ProfileData

`parser/profile.py` 的 `load_profile(data_dir)` 把 8 文件载入一个 `ProfileData`（概念上按 `(rank, step)` 索引；本样例为 rank 0 / step5，rank 维度预留多卡）：

```python
@dataclass
class ProfileData:
    data_dir: str
    step_trace: pd.DataFrame          # step_trace_time.csv
    op_statistic: pd.DataFrame        # op_statistic.csv
    api_statistic: pd.DataFrame       # api_statistic.csv
    kernel_details: pd.DataFrame      # kernel_details.csv
    operator_details: pd.DataFrame    # operator_details.csv
    communication: List[Dict]         # communication.json 归一化后的集合通信列表
    communication_raw: Dict           # 原始 JSON
    communication_matrix: Dict        # communication_matrix.json（单卡空）
    trace_path: Optional[str]         # trace_view.json 路径（只存路径，不读入）
    meta: Dict                        # rank/device/step/文件大小/计数等
    def iter_trace_events(self): ...  # 按需流式产出 trace 事件
```

`meta` 包含 `rank`（预留）、`device_id`、`step`、`multi_card`（由 `communication_matrix` 是否有 collective 判定）、各表行数 `counts`、各文件 `file_sizes`。

## 3.3 解析细节

### CSV 脏数据清洗
昇腾 CSV 常带尾随制表符 / 引号。`num(series)` 统一 `str → strip → strip('"') → to_numeric(errors="coerce")`，把脏列稳健转为 float。`_read_csv` 缺失文件返回空 DataFrame、解析异常回退 `engine="python"`，**任何单文件缺失都不致命**。

### communication.json 归一化
`_normalize_communication` 把嵌套结构（step → collective/p2p → op → Time/Bandwidth Info）拍平为统一记录，并从算子名 `hcom_<type>_...` 正则抽出通信类型（allGather / alltoallv / allReduce / reduceScatter…）。每条含 `elapse/transit/wait/sync/idle_ms`、`wait_ratio`、`sync_ratio` 与各 link 的 `transit_mb / bandwidth_gbps`。

### trace_view.json 流式（关键工程取向）
104MB / 34 万事件**从不整体读入内存**。`parser/trace.py` 的 `iter_events` 逐事件产出（生成器），`timeline.py` 用两遍流式扫描完成泳道占用聚合，结果（~100KB）落盘缓存。内存占用与文件大小**解耦**。

### shape / dtype 解析
`parser/shapes.py` 把 `"Input Shapes"` / `"Input Data Types"` 字符串解析为 `List[List[int]]` / `List[str]`，并提供 `numel`。这是 MFU/MBU 估算的输入（见 [04](04-metrics.md)）。

## 3.4 rank × step 维度预留

数据模型、`meta`、视图均按 `rank × step` 设计：
- `meta.rank = 0`（预留），`communication_matrix` 为空 `{}` 时**空跑不报错**。
- 同一设备包含多个完整 step 时，overview 汇总整个 step 窗口，并同时返回 `steps`、`step_count`、窗口总量 `us` 与单步平均 `avg_us`；MFU/HFU 分子和分母必须使用同一窗口。
- 多卡数据接入后，rank 维度填充即解锁 M1–M5（见 [07](07-roadmap-and-verification.md)）。

## 3.5 校验基线（来自本数据，可作单测断言）

| 指标 | 值 | 来源 |
|---|---|---|
| step 总时长 Stage | **3,126,771.5 us ≈ 3.13 s** | step_trace |
| Computing | 1,728,916.559 us | step_trace |
| Communication (Not Overlapped) | 817,426.625 us | step_trace |
| Free（空闲） | ≈ 18.56 % | 派生 |
| 有效计算占比 | ≈ 55.29 % | 派生 |
| 计算-通信重叠率 | ≈ 12.62 %（118,079 / 935,506） | 派生 |
| `HcclLaunchAicpuKernel` 占 device | **28.091 %** | op_statistic |
| `aclnnMaskedSelect` 单次 max | 166,285.31 us | api_statistic |

> 这些数字同时是 `scripts/verify.py` 的断言基线（见 [07](07-roadmap-and-verification.md)），用于回归保护。

---

上一篇：[02 · 架构与技术取向](02-architecture.md) ｜ 下一篇：[04 · 指标层](04-metrics.md)
