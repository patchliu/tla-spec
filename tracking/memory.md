# 内存评审跟踪

[返回接口目录](../README.md)

<a name="memory-001"></a>

## MEMORY-001 · tla.alloc

[查看规格](../spec/memory.md#api-tla-alloc)

- 原表序号 2 要求分配独占并禁止 planmem retry 复用。当前物理内存是否复用由内存规划决定，需确认该场景的要求。

结论：待决。移除分配参数和物理内存能否复用是两个问题。现有实现仍由后端内存规划决定复用，已确认的签名变更记录在 [MEMORY-002](#memory-002)。

<a name="memory-002"></a>

## MEMORY-002 · tla.alloc 签名调整

[查看规格](../spec/memory.md#api-tla-alloc)

- 目标签名为 `tla.alloc(shape, dtype, space=None)`。删除 `scope` 关键字，调用方用 `space` 指定地址空间。
- 移除 `versions`。需要多缓冲时改用 [tla.annotate_multi_buffer](../spec/schedule.md#api-tla-annotate-multi-buffer)。

结论：设计已确认，待实现。当前签名仍接收这两个参数。实现更新后核对旧调用的迁移、新签名和默认 UB 行为。

<a name="memory-003"></a>

## MEMORY-003 · tla.fill 的 Ascend 950PR 设备模板

[查看规格](../spec/memory.md#api-tla-fill)

- 950PR 上，bfloat16、int32、float32 的一维和二维 fill 设备用例在链接阶段缺少对应的标量广播模板。
- 前端与 Adapter IR 用例覆盖参数校验、偏移拒绝和标量 hivm.hir.vbrc。
- 待目标侧补齐 lowering 或模板后，使用专用用例验证设备编译与数值，并重复验证零偏移的合法 reshape。

结论：待决。保留前端校验和 Adapter IR 用例。
