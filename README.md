# LLMInsight

[项目网站](https://ai-v5.github.io/LLMInsight/) · [设计与数据契约](doc/README.md)

昇腾（Ascend）NPU 大模型训练 Profiling 分析工具。它读取
`ASCEND_PROFILER_OUTPUT` 或 MindStudio `msprof-export` 目录，生成统一指标、
时间线、规则诊断和可分享报告。

当前重点支持 DeepSeek-V3 结构的 MoE 训练 Profiling；多卡数据模型已预留，
具体指标以实际采集内容为准。

## 开源免责声明

LLMInsight 是独立开源项目，由维护者与贡献者按开源协作方式维护和运作，
项目内容供技术研究、学习和工程交流参考。项目中的代码、文档、示例、分析结果
及讨论，不代表任何个人或组织（包括维护者、贡献者及其雇主、所属机构或关联方）
的观点、立场、政策或承诺。

本项目不是任何厂商或机构的官方产品，不构成其认证、推荐、背书或技术支持承诺。
文中提及的个人、组织、厂商、产品、模型和商标，仅用于说明兼容对象或技术背景，
不表示隶属、代言、合作或授权关系。

除非另有书面约定，本项目按“现状”和“可用状态”提供，不对准确性、完整性、
适用性、稳定性或特定性能收益作保证，也不承诺持续维护、响应时限或商业支持。
性能指标、诊断建议和推演结果可能受采集数据、配置及实现假设影响，使用者应
独立核验，并自行评估使用、修改及部署风险。

在适用法律允许的范围内，除非另有书面约定，维护者、贡献者及其他权利人不对因
使用或无法使用本项目产生的损失承担责任，包括数据丢失、业务中断或其他损失。
依法不得排除或限制的责任不受本说明影响。

本说明不增加 AGPL-3.0-only 许可证之外的使用限制；具体授权、无担保与责任限制
以 [`LICENSE`](LICENSE) 为准。

## 快速开始

### Windows（推荐）

在仓库根目录打开 PowerShell：

```powershell
pip install -r requirements.txt
.\restart_insight.bat
```

脚本会停止旧进程并启动服务。服务固定绑定 `127.0.0.1`，不支持远程监听。打开
<http://127.0.0.1:8765/>，选择要分析的 Profiling 目录。

### Linux/macOS

```bash
pip install -r requirements.txt
./restart_insight.sh
```

也可以直接运行：

```bash
python -m llminsight.server --port 8765 --no-browser
```

如果没有手动选择目录，可用 `LLMINSIGHT_DATA_DIR` 指定数据目录。重启脚本默认
关闭 LLM 网络调用；确认数据可以发送给配置的模型服务后，再显式设置
`LLMINSIGHT_LLM_ENABLED=1`。

## 能做什么

- 解析 Ascend profiler 和 MindStudio 导出的 CSV/JSON/DB 数据。
- 支持多 rank 采集根目录（每 rank 一个 `*_ascend_pt/ASCEND_PROFILER_OUTPUT`）：
  自动选 step 中位数的代表 rank 做单卡分析，并汇总全部 rank 的 step 分解、
  慢卡/离群 rank 与按节点聚合的集群概览。
- 展示 MFU、MBU、Roofline、通信、显存、时间线和隐藏开销。
- 将算子和通信归因到模型结构，并给出规则诊断与优化建议。
- 导出自包含 HTML 报告，支持芯片参考值切换和 What-if 分析。
- LLM 洞察为可选能力；默认只使用本地规则引擎，不上传原始 trace。

## 验证

```bash
python scripts/verify.py
node scripts/verify_web.js
```

第二条命令需要先启动服务，并默认访问 `http://127.0.0.1:8765`。

## 文档

设计与数据契约见 [`doc/`](doc/README.md)，包括架构、解析、指标、视图、LLM
洞察和验证说明。

## 许可证

项目使用 AGPL-3.0-only，见 [`LICENSE`](LICENSE)。第三方组件和依赖见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
