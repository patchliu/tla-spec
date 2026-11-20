# DMA评审跟踪

[返回接口目录](../README.md)

<a name="dma-refresh-baseline"></a>

GM↔UB 依据库上已有接口，以及 2026-11-18 的 `tla-programming-api.md` §4.1.2、§4.2.2、§4.3.1 及 `affinity-ir-contract.md` §2.6、§2.12 修订。后端核对范围为 Ascend950、NPUIR `65e061b2404c8808ead47d9a6be95e102c28c3c0`。

<a name="dma-001"></a>

## DMA-001 · tla.load_gm_to_l1

[查看规格](../spec/dma.md#api-tla-load-gm-to-l1)

结论：FP32→FP32 纳入待转测范围；逻辑 M 按 16、N 按 8 对齐，目的 NZ 内层为 `(16, 8)`。

依据：AscendNPU-IR `0e7d8bb2d0d6a39b83996e909eee1b973685e164` 的 `Nd2NzCapabilityProfile.td` 中 C310 同类型能力，以及 `bishengir/lib/Template/lib/RegBase/Cube/compat/DMA/Cbuf/nd2nz.cpp` 的 `REGISTE_ND2NZ(gm, cbuf, 2, 4, float)`。模板以 `32 / sizeof(T)` 计算 C0，因此 FP32 为 8。核对范围为类型能力、目标模板及 TLA builder 路径；本次规格更新未新增设备精度验证。

补充验证：Ascend950DT_9582 和 AscendNPU-IR `1389120d` 的 Process 路径上，(32,24) FP32 ND 输入使用 32 元素行距与 8 元素偏移，回读逐元素精确一致。FP16 缓存回归中，默认路径和显式模式 4 在目标 LLVM IR 分别传递 0 和 4，实机均通过。需在接口表保留 [PR #3267](https://github.com/triton-lang/triton-ascend/pull/3267) 的 MTE2 原生缓存控制契约。

<a name="dma-002"></a>

## DMA-002 · tla.load_gm_to_ub

[查看规格](../spec/dma.md#api-tla-load-gm-to-ub)

- 执行域检查：wrapper 尚无统一 core/VF guard，须与 Vector 核级、VF 外契约一致；早期 IR 接受不作为合法目标用法。
- 缓存策略范围：NORMAL_FV/NORMAL_LV 桥接 HIVM 的 EvictFirst/EvictLast，普通 1D/2D DMA 编码为 0/1；被测 scalar fallback 使用相同 LD_IO 序列，不表达该 DMA 策略；3D 转置 NDDMA 样本固定编码 0，NORMAL_LV 未保留。完整 L2 配置与其他路由的透传范围待确认。
- dtype 与布局：int1 / float8e4m3fn / float8e5m2 及其他 dtype、布局组合的目标支持边界待确认。

结论：公开规格及示例已刷新，移除显式 padding 和旧缓存入口，仅保留 `l2_cache_ctrl`。上述 3 项待决；固有 32B 补齐是已确认的硬件使用约束，容量、别名和同步须覆盖完整物理写入范围。

<a name="dma-003"></a>

## DMA-003 · tla.store_ub_to_gm

[查看规格](../spec/dma.md#api-tla-store-ub-to-gm)

- 执行域检查：wrapper 尚无统一 core/VF guard，须与 Vector 核级、VF 外契约一致；默认核内同步不代替显式核间协议。
- dtype 与布局：各 dtype、stride、offset、fallback 及原子写回的组合约束待确认。

结论：签名仅接收 src、dst，无配置对象或随路原子属性；copy 的 UB→GM 分发仅接受 `params=None`。原子模式由独立 [set_atomic](scalar.md#scalar-001) 控制，写回完成后显式 NONE 复位。上述 2 项待决。

<a name="dma-004"></a>

## DMA-004 · tla.copy_l1_to_l0

[查看规格](../spec/dma.md#api-tla-copy-l1-to-l0)

- 对齐 RegBase 模板库的单 batch 搬运形式，以源、目的 buffer 视图表达分块；公开配置仅保留 `is_transpose`。

结论：接口设计已确认。每次调用仅处理单 batch，不提供批量搬运配置；`k_part*` 和 `m` 不作为公开参数。

实现要求：前端根据目的视图推导 K 和 M/N，根据源视图及 is_transpose 检查搬运范围。源视图负责选择分块起点，stride 和 offset 保留在 memref 描述符中。底层操作需要的分区字段由前端内部构造，不重新暴露给调用方，也不引入批量参数。

实现按此规格对齐并进入待转测范围；当前实现缺口不作为删除规格或免测的依据。L0 DMA 输出由显式 MMAD L0 消费，相关入口基于 [Triton Ascend PR #2982](https://github.com/triton-lang/triton-ascend/pull/2982)。后端接口依据 AscendNPU-IR c521aba2 的 RegBase 单 batch 搬运模板；底层保留分区字段不影响公开接口只提供 is_transpose。

补充结论：源、目的的 stride 均以元素计，内层固定 `(256,16,1)`，仅最外层允许按 fractal 对齐的额外间距。该要求已写入规格，合法的外层 pitch、offset 和子视图仍属于待转测范围。已知静态 offset 的对齐诊断应穿透 `subview` 和 `reshape`；动态 offset 仍须满足同一约束。

补充依据：TA main `f1997518` 合入 PR #3267（`c8a02948`）与 PR #2982（`99373b0d`），并按规格对齐 L1→L0 前端；实际使用 AscendNPU-IR `c521aba2`。检查覆盖 FP16/BF16、L0A/L0B、转置、外层间距、静态/动态 offset 及容量边界，不代表任意视图穷举。

<a name="dma-005"></a>

## DMA-005 · tla.make_partition_view

[查看规格](../spec/dma.md#api-tla-make-partition-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-006"></a>

## DMA-006 · tla.make_strided_view

[查看规格](../spec/dma.md#api-tla-make-strided-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-007"></a>

## DMA-007 · tla.make_gather_scatter_view

[查看规格](../spec/dma.md#api-tla-make-gather-scatter-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-008"></a>

## DMA-008 · tla.load_view

[查看规格](../spec/dma.md#api-tla-load-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-009"></a>

## DMA-009 · tla.store_view

[查看规格](../spec/dma.md#api-tla-store-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-010"></a>

## DMA-010 · tla.set_pad_value

[查看规格](../spec/dma.md#api-tla-set-pad-value)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要。当前 GM→UB 不提供显式 padding，不以旧版 `LoadGM2UBParams.pad_value` 作为实现依据；待补齐实现后另行评审。

<a name="dma-011"></a>

## DMA-011 · tla.set_post_copy

[查看规格](../spec/dma.md#api-tla-set-post-copy)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-012"></a>

## DMA-012 · tla.stride_store

[查看规格](../spec/dma.md#api-tla-stride-store)

原评审问题：是否纳入本轮规格并补齐实现。

结论：已关闭。确认不需要此接口，退出规格范围。

<a name="dma-013"></a>

## DMA-013 · tla.stride_load

[查看规格](../spec/dma.md#api-tla-stride-load)

原评审问题：是否纳入本轮规格并补齐实现。

结论：已关闭。确认不需要此接口，退出规格范围。
