# 调度接口规格

[返回接口目录](../README.md)

<a name="api-tl-program-id"></a>
<a name="api-tla-program-id"></a>

## tla.program_id

计划签名：`tla.program_id(axis)`

返回：指定轴的程序实例编号，为 int32 标量。

语义沿用 tl.program_id。新入口在 [triton-ascend PR #3268](https://github.com/triton-lang/triton-ascend/pull/3268) 已验证轴 0 / 1 / 2、非法轴以及三轴网格读回，主线进度见 [SCHEDULE-001](../tracking/schedule.md#schedule-001)。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `axis` | launch 网格轴 | 编译期 int | 0 / 1 / 2 | TLA adapter 映射到对应 grid 参数。 | 必填 |

计划调用示例

```python
@tla.jit
def example():
    block = tla.program_id(0)
```

新入口用例路径为 tla_ut/scope/program_id.py。已有 tl.program_id 用例：[tla_ut/vector/fused_weight.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/fused_weight.py#L35)。

<a name="api-tla-annotate-buffer-versions"></a>
<a name="api-tla-annotate-multi-buffer"></a>

## tla.annotate_multi_buffer

计划签名：`tla.annotate_multi_buffer(buf, versions)`

返回：None。

多缓冲版本数从 [tla.alloc](memory.md#api-tla-alloc) 的分配签名移到显式标注。当前实现入口为 `tla.annotate_buffer_versions`，计划沿用现有参数与行为，见 [SCHEDULE-002](../tracking/schedule.md#schedule-002)。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `buf` | 要标注的缓冲区 | tla.buffer | 可添加 annotation.mark 的 buffer | — | 必填 |
| `versions` | 缓冲区版本数 | 编译期 int | 1 至 2³¹−1（现有版本标记的 i32 表示范围） | 1 不添加标记，大于 1 添加 hivm.multi_buffer。实际可分配版本数还受目标内存规划约束。 | 必填 |

计划调用示例

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tl.float32)
    tla.annotate_multi_buffer(tile, 2)
```

## 范围说明

以下条目在当前整理表中没有可定位的 TLA 接口名。

| 原序号 | 范围结论 | 涉及场景 |
| --- | --- | --- |
| 179 | 不涉及 | T.PerCoreTask（开发者模式） |
| 180 | 不涉及 | T.Persistent（开发者模式） |
| 181 | 不涉及 | T.Pipelined（开发者模式） |
| 182 | 不涉及 | T.Stage（开发者模式） |
| 183 | 不涉及 | T.Task（开发者模式） |
| 185 | 暂不支持 | T.annotate_manual_multi_buffer |
| 186 | 暂不支持 | T.annotate_unlimit_memory |
| 187 | 暂不支持 | T.assume_conflict |
| 188 | 暂不支持 | T.assume_no_conflict |
| 189 | 暂不支持 | T.AscendBaseTileScheduler |
| 190 | 暂不支持 | T.AscendTileScheduler |
| 191 | 暂不支持 | T.AscendBatchedTileScheduler |
| 192 | 暂不支持 | T.AscendMGroupedTileScheduler |
| 193 | 暂不支持 | T.AscendKGroupedTileScheduler |
