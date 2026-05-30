# LLMInsight

**昇腾（Ascend）NPU 大模型训练 Profiling 与性能洞察平台**
*Profiling & Performance Analysis Platform*

把一份 `ASCEND_PROFILER_OUTPUT`（torch_npu / MindSpeed-LLM 采集）目录，**零配置**解析为统一数据模型 → 派生指标 → 规则引擎诊断 → 无构建 Web 仪表盘，产出「**现象 → 根因 → 建议 → 预计收益 → 置信度**」式的可执行洞察。

> 当前实现对象：**DeepSeek-V3 结构 MoE 模型（MLA + 128 专家、EP64）** 的单卡单 step 训练采集。单卡做透、架构预留多卡。

## 为什么需要它

昇腾 MindStudio Insight / msprof-analyze / TensorBoard / Perfetto 能看时间线与算子，但：① 只是一次性快照，看不出训练演化；② 只给图不给结论，要专家手工解读；③ 停在底层算子层，不懂 MLA / MoE 模型结构；④ 不报理论上界与「改了能省多少」；⑤ 采集被 `ASCEND_LAUNCH_BLOCKING=1` 等扭曲却无人提示。LLMInsight 针对这些盲区而生。

## 核心亮点

- **🤖 LLM 自动诊断**：内置性能专家经验，自动产出诊断卡片（可选自然语言叙述）。
- **🎬 全训练回放**：可拖动时间轴回放训练，定位「第几步开始变慢 / 显存爬升 / 通信抖动」。
- **🧩 训练语义化归因**：算子 / 通信归因到 Embedding / MLA / MoE / Norm / Optimizer / Loss。
- **🎯 MFU / MBU + Roofline**：逐算子达成算力 / 带宽对照硬件天花板，排出优化余量（含 FlashAttention 等融合算子）。
- **🫥 隐性开销专项**：下发 / 等待·同步 / 重计算 / 格式转换 / 动态 shape / 空泡 总账量化。
- **📈 理论上界 + What-if**：算 step 时间下界并模拟「修复某瓶颈后能省多少」。
- **🔌 开箱即用**：纯 Python stdlib 后端 + 无构建前端，指向目录即可跑，可独立分发。

## 快速开始

### 方式一 · 从 Release 下载 whl（推荐，开箱即用）

到 [**Releases**](https://github.com/ai-v5/LLMInsight/releases/latest) 下载 `llminsight-0.1.0-py3-none-any.whl`，安装即可运行（whl 已内置 Web 前端与芯片配置，离线可用，无需克隆仓库）：

```bash
pip install llminsight-0.1.0-py3-none-any.whl   # 自动装好 pandas / numpy / pyyaml

python -m llminsight.server                      # 默认 http://127.0.0.1:8000，自动开浏览器
python -m llminsight.server --port 8765 --no-browser
# 安装后也可直接用控制台命令： llminsight  （等价于 python -m llminsight.server）
```

### 方式二 · 从源码运行（开发）

```bash
pip install -r requirements.txt          # 运行依赖仅 pandas + numpy（前端零依赖）

python -m llminsight.server              # 默认 http://127.0.0.1:8000，自动开浏览器
python -m llminsight.server --port 8765 --no-browser
./restart_insight.sh                     # 本地开发：自动停旧起新 + 轮询就绪
```

- 数据目录默认指向仓库内置样例，可用环境变量 `LLMINSIGHT_DATA_DIR` 覆盖。
- 首次启动解析 104MB trace（socket 在解析完成后开放）；之后磁盘缓存使重启 ~0.09s。

## 技术栈

| 层 | 选型 | 取向 |
|---|---|---|
| 后端 | Python 3 stdlib `http.server`（无 FastAPI） | 零三方框架，可独立分发 |
| 解析 | pandas（CSV）+ stdlib json 流式（trace） | 104MB trace 永不入内存 |
| 前端 | 无构建 vanilla JS（全局 `LI` 命名空间） | 无 npm / 打包，静态文件直出 |
| 图表 | ECharts 5.5.1（本地 vendored） | 离线可用、无 CDN 依赖 |
| LLM | 可插拔（OpenAI 兼容 + Anthropic），**默认关闭** | 只发 KB 级摘要，绝不上传 trace |

## 设计文档

完整设计见 **[`doc/`](doc/README.md)**：

- [01 · 总览与目标](doc/01-overview-and-goals.md) — 动机、盲区、亮点 H1–H10 / 多卡 M1–M5
- [02 · 架构与技术取向](doc/02-architecture.md) — 分层、模块、API、芯片切换
- [03 · 数据与解析](doc/03-data-and-parsing.md) — 8 文件映射、ProfileData、校验基线
- [04 · 指标层](doc/04-metrics.md) — MFU/MBU/Roofline、FlashAttention FLOP 模型、What-if
- [05 · 展示视图](doc/05-views.md) — 前端 10 视图
- [06 · 洞察与 LLM](doc/06-insights-and-llm.md) — 规则引擎 12 卡片 + LLM 后端
- [07 · 路线图与验证](doc/07-roadmap-and-verification.md) — P0–P5 阶段 + 验证基线

## 验证

```bash
python scripts/verify.py                 # 后端数值基线断言（12 卡片、MFU≤100%、校准等）
node   scripts/verify_web.js             # 无头渲染校验 10 视图（需服务器在 :8765）
```

## 关键约束

- **单卡优先**：数据模型按 `rank × step` 预留多卡；当前 `communication_matrix` 为空可空跑。
- **LLM 默认关闭**：分发给无 key 用户时纯规则引擎可用、零网络调用。
- **隐私优先**：只把结构化摘要发给 LLM（UI 可审计），原始 trace 不出本机。
- **芯片峰值为假设值**：MFU/MBU 随峰值缩放，可在 UI 右上角切换（910B / 950DT）或在 `config.ChipSpec` 校正。

## 许可证

见 [LICENSE](LICENSE)。
