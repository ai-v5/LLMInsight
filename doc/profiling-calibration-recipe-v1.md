# Profiling → Calibration Recipe Producer v1 合同

## 定位与声明边界

Producer v1 把 LLMInsight 已解析的 profiling 中、语义可无歧义重建的普通 GEMM 汇总成供下游校准链消费的候选 recipe。输出固定为：

- `schema=llm.profiling-calibration-recipe`
- `version=v1`
- `readiness=PROFILE_DERIVED_RECIPE`
- `evidence_role=DERIVED_FROM_PROFILING`

允许声明仅限“从 profiling 派生出 ordinary GEMM 候选及其来源完整性、频次、相对优先级和映射覆盖率”。禁止把 recipe 声明为 `CALIBRATED`、隔离单算子校准结果、真机校准曲线或 product 已绑定结果。observed kernel duration 只参与相对优先级和交叉检查，不是校准时延，也不写入 recipe。

recipe 不接受 `target`、`product_ref` 或其它 configured product 字段。Ascend 950PR、950DT 等产品身份只能由 LLMSim 执行阶段的 trusted registry 绑定；profile 中的 family、format 或 capture scope 不能升级为产品权威。portable `recipe_sha256` 因此不含目标产品。

## 输入与复用边界

CLI 必须指向现有 LLMInsight loader 能读取的 profile 目录。producer 调用 `llminsight.parser.load_profile`，复用统一 `ProfileData`、`parse_shapes`、`parse_dtypes` 和 `_matmul_mnk`，不建立第二套 profile parser。

lineage 只保存逻辑 source role 与内容 SHA-256，不保存绝对路径、用户名、真实 basename、raw row、文件大小、DB 路径或 trace 路径：

| `profile_layout` | `selected_sources.role` |
| --- | --- |
| `TORCH_NPU_ASCEND_PROFILER_OUTPUT` | `KERNEL_DETAILS_CSV` |
| `HYBRID_TORCH_NPU_MINDSTUDIO_DB` | `KERNEL_DETAILS_CSV`, `MINDSTUDIO_DB` |
| `MINDSTUDIO_DB` | `MINDSTUDIO_DB` |
| `MSPROF_OP_SUMMARY` | `MSPROF_OP_SUMMARY_CSV` |

`producer_revision` 是调用方显式提供的 40 位小写 Git SHA。producer 不从 profile 推断该值。`source_manifest_sha256` 绑定 canonical `selected_sources` 数组；`capture_scope` 只接受 parser metadata，缺失时必须为 `UNKNOWN + UNAVAILABLE + PARSER_SCOPE_NOT_RECORDED`。

## v1 映射规则

v1 只接受大小写不敏感的 exact op type：`MatMul`、`MatMulV3`、`Gemm`、`GemmV3`。一个 accepted row 必须同时满足：

1. 恰有两个 rank-2 输入矩阵和一个 rank-2 输出矩阵；
2. 输出 shape 能使 `M/N/K` 唯一成立；
3. 按真实 A/B 维度推导出的 `transpose.a/b` 组合唯一；方阵等多解情况不能猜；
4. 两个输入和输出 dtype 都存在、可规范化且相同；v1 只接受 `BF16`、`FP16`、`FP32`、`FP8_E4M3`；
5. `_matmul_mnk` 的结果与上述唯一解释一致。

accepted rows 按 `canonical_op + shape + dtype + transpose + implementation_hint` 聚合。`case_id` 是该 semantic payload 的 canonical SHA-256。`source_evidence.digest_sha256` 绑定 path-free 的规范化 row evidence multiset；它提供内容完整性，不提供外部身份认证。

BatchMatMul、GroupedMatMul、Attention 和 fused MatMul/GEMM 在 v1 一律进入 `unmapped`，分别给出 `*_SEMANTICS_INCOMPLETE`。普通 GEMM 的缺 shape、shape 冲突、transpose 多解、缺 dtype 或 dtype 冲突也分别计数。每个 candidate row 只计一个首要原因；非候选 kernel 不进入 `unmapped`。

## Coverage 与优先级

`coverage` 同时报告全部 kernel rows、candidate rows、mapped rows、unmapped candidate rows 和聚合 case 数。存在 candidate 时，`mapping.mapped_ppm` 是 `mapped / candidate` 的整数 ppm；完全没有 candidate 时必须输出：

```json
{"status":"UNAVAILABLE","reason":"NO_CANDIDATE_ROWS"}
```

不得以 `0` 冒充可用覆盖率。

只有全部 mapped rows 的 duration 都是有限正数时，所有 case 才统一使用 `OBSERVED_TOTAL_DURATION` 排优先级；任何 mapped duration 不可用时，整份 recipe 统一回退到 `FREQUENCY`。`score_ppm` 总和为 1,000,000，稳定 tie-break 使用 `case_id`。recipe 不包含 duration、latency、mean、percentile 或最大/最小时延统计。

## Canonical JSON 与 fail-closed 验证

canonical bytes 定义为：

```python
json.dumps(value, sort_keys=True, separators=(",", ":"),
           ensure_ascii=False, allow_nan=False).encode("utf-8")
```

文件末尾恰有一个换行。`recipe_sha256` 等于移除 `recipe_sha256` 字段后，整个其余文档 canonical bytes 的 SHA-256；因此 lineage、coverage、cases、unmapped 和所有声明字段都在完整性链内。

JSON Schema 见 `doc/contracts/llm.profiling-calibration-recipe.v1.schema.json`。它冻结字段、类型、枚举和 `additionalProperties=false`；`llminsight.calibration_recipe.validate_recipe` 另外强制以下跨字段规则：

- Python `bool` 不得冒充 integer；只有 transpose flags 接受 bool；
- 拒绝 float、负数、NaN/Infinity、duplicate JSON key 和 path-like string；
- source role 必须与 layout 一致且排序、去重；
- coverage、frequency、unmapped count、ppm 与 case count 必须闭合；
- cases/unmapped 必须排序、去重；priority rank 必须连续，score 总和必须闭合；
- 重新计算 source manifest、case ID 和 recipe digest，任何不一致都拒绝。

SHA-256 是内容完整性摘要，不是防止有意重签的数字签名；需要外部 authority 的消费者必须另行验证 producer commit/PR 与 trusted registry。

## CLI 与消费者步骤

```powershell
python -m llminsight.calibration_recipe `
  --profile <existing-profile-dir> `
  --output <non-git-temp-dir>\recipe.json `
  --producer-revision <40-lowercase-git-sha>
```

成功输出只包含 path-free 的 case/mapped/unmapped 聚合计数；失败只输出异常类型，不回显输入路径或 raw row。消费者应先调用 `load_and_validate_recipe`，再依据 trusted registry 在执行阶段绑定目标产品。fixture `tests/fixtures/profiling_recipe/expected_recipe.json` 是 v1 的精确、synthetic consumer fixture。

`PROFILE_OBSERVATION`（独立的时延统计层）是后续工作包；v1 recipe 不内嵌 observation，也不允许由本 producer 直接产出 calibrated latency。
