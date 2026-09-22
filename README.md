# LLMInsight

昇腾（Ascend）NPU 大模型训练 Profiling 分析工具。它读取
`ASCEND_PROFILER_OUTPUT` 或 MindStudio `msprof-export` 目录，生成统一指标、
时间线、规则诊断和可分享报告。

当前重点支持 DeepSeek-V3 结构的 MoE 训练 Profiling；多卡数据模型已预留，
具体指标以实际采集内容为准。

## 开源免责声明

本项目是开发者个人的开源技术实践，仅代表开发者个人观点和实现，不代表任何
厂商、组织、雇主或其关联机构的立场、观点、承诺或官方支持。本项目与文中提及
的厂商、组织、产品或模型不存在隶属、代言、授权或合作关系；相关名称仅用于说明
兼容对象和技术背景。项目按现状提供，使用者应自行评估其适用性、准确性和风险。

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

项目使用 GPL-3.0-only，见 [`LICENSE`](LICENSE)。第三方组件和依赖见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
