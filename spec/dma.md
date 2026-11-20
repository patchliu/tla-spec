# DMA接口规格

[返回接口目录](../README.md)

以下 DMA、L0 搬运和填充约束面向 Ascend950（RegBase）。

<a name="api-tla-load-gm-to-l1"></a>

## tla.load_gm_to_l1

签名：`tla.load_gm_to_l1(src, dst, params=None)`

返回：None。将 GM 中的矩阵装载到 L1，供后续矩阵运算使用。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源矩阵 | tla.buffer | GM；float16 / bfloat16 / float32 | 二维矩阵，末维连续；行间距不小于列数，访问范围不得越界。 | 必填 |
| `dst` | 目标矩阵 | tla.buffer | L1；与 src 类型相同 | 连续分块存放；以 `C = 32 / sizeof(dtype)` 计，逻辑 `(M,N)` 对应形状 `(N/C,M/16,16,C)`。 | 必填 |
| `params` | 搬运配置 | tla.LoadGM2L1Params / None | 下列字段 | None 使用默认配置。 | None |
| `params.l2_cache_ctrl` | MTE2 缓存模式（兼容条款草稿） | 编译期 int 或 None | 目标支持的原生模式；包括 0、1、2、3、4 | None 省略属性，后端使用 0；显式模式原样传递。 | None |
| `params.enable_nd2nz` | 启用矩阵装载所需的数据重排 | 编译期 bool | True | 不提供关闭转换的路径；False 抛出 NotImplementedError，非 bool 抛出 TypeError。 | True |

**约束**

- 长度：M 为 16 的正整数倍，N 为 C 的正整数倍；M、N 均不超过 65535 个元素。因此 M 最大为 65520；N 最大为 `floor(65535/C) × C`。
- 分块：基本块为 `(16,C)`；例如 float16 的 C=16，float32 的 C=8。类型不同时不能直接沿用相同的分块形状。
- 容量：目标总字节数 `M × N × sizeof(dtype)` 不得超过实际分配及可用 L1 空间；维度上限不代表最大矩阵可整体放入 L1。
- 地址：源偏移、行间距与目标分块须描述同一逻辑矩阵，访问不得越界。对于 float32，源、目的有效起始地址、源行间距和每行搬运宽度须按 32B 对齐。
- 转换：不改变元素类型，不提供反向转换、其他重排格式或非对齐尾块填充。
- 同步：装载完成后才能由矩阵运算读取；复用 L1 缓冲区前须保证先前消费者完成。

**调用示例**

```python
@tla.jit
def example(src, dst):
    with tla.scope.cube():
        tla.load_gm_to_l1(src, dst)
```

仓库测试：[tla_ut/dma/nd2nz.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/nd2nz.py#L22)。

<a name="api-tla-load-gm-to-ub"></a>

## tla.load_gm_to_ub

签名：`tla.load_gm_to_ub(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.load。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：GM<br>数据类型：与 dst 相同（不含 `float64`） | 源地址需要按照对应数据类型所占字节数对齐，与 dst 的 rank、逻辑 shape 相同。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：与 src 相同 | 目的地址需要32字节对齐。 | 必填 |
| `params` | L2 缓存配置 | tla.LoadGM2UBParams 或 None | None / tla.LoadGM2UBParams 实例 | None 等价于默认配置。 | None |
| `params.l2_cache_ctrl` | GM→UB 的 L2 缓存策略 | tla.L2CachePolicy | `NORMAL_FV` 表示普通模式优先淘汰<br>`NORMAL_LV` 表示普通模式、优先保留 | 无 | `None`（使用 `NORMAL_FV`） |

本接口在`tla.scope.vector()`内、SIMD/SIMT VF 外区域调用。

硬件可能在 UB 中写入补齐字节。

<a name="gm-ub-capacity"></a>

GM↔UB 搬运须按 UB 端视图的布局预留空间。

- 一维数据连续或二维各行紧密相接时，从 UB 视图起始地址算起，可用空间至少为有效数据总字节数按 32 字节向上对齐后的大小。补齐空间位于整块末尾。
- 二维每行连续且行间留有空隙时，补齐空间位于各行末尾。UB 行距须为 32 字节的整数倍，且能容纳一行有效数据。每行均按该行距分配空间，包括最后一行。

将仅覆盖有效数据的视图作为 `dst`。逐行预留时，用 `tla.subview` 取各行的有效区域。

**调用示例**

前提：`x` 指向连续的 60 个 float32 元素。UB 每行分配 32 个元素，前 30 个为有效数据，末尾 8 字节留给补齐。

```python
@tla.jit
def example(x):
    with tla.scope.vector():
        tile = tla.alloc((2, 32), tl.float32)
        dst = tla.subview(tile, offsets=(0, 0), sizes=(2, 30))
        p: tl.constexpr = tla.LoadGM2UBParams(l2_cache_ctrl=tla.L2CachePolicy.NORMAL_LV)
        tla.load_gm_to_ub(tla.view(x, (2, 30)), dst, p)
```

仓库测试：`tla_ut/dma/load_store.py`。

<a name="api-tla-store-ub-to-gm"></a>

## tla.store_ub_to_gm

签名：`tla.store_ub_to_gm(src, dst)`

返回：None。

编译表示：hivm.hir.store。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：与 dst 相同（不含 `float64`） | 源地址需要32字节对齐。 | 必填 |
| `dst` | 接收数据的目的缓冲区 | tla.buffer | 地址空间：GM<br>数据类型：与 src 相同 | 目的地址需要按照对应数据类型所占字节数对齐, 与 src 的 rank、逻辑 shape 相同。 | 必填 |

原子写回使用独立的 [tla.set_atomic](scalar.md#api-tla-set-atomic)，受控写回完成后用 NONE 显式复位。
本接口在`tla.scope.vector()`内、SIMD/SIMT VF 外区域调用。

以 `src` 的起始地址和布局为准，按 [UB 容量和行距规则](#gm-ub-capacity) 预留空间。

**调用示例**

前提：`x`、`y` 均指向连续的 64 个 float32 元素，使用 `(2, 32)` 布局。

```python
@tla.jit
def example(x, y):
    with tla.scope.vector():
        tile = tla.alloc((2, 32), tl.float32)
        tla.load_gm_to_ub(tla.view(x, (2, 32)), tile)
        tla.store_ub_to_gm(tile, tla.view(y, (2, 32)))
```

仓库测试：`tla_ut/dma/load_store.py`。

<a name="api-tla-copy-l1-to-l0"></a>

## tla.copy_l1_to_l0

签名：`tla.copy_l1_to_l0(src, dst, params=None)`

返回：None。将 L1 中的矩阵分块复制到矩阵乘法输入缓冲区。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源缓冲区 | tla.buffer | L1；float16 / bfloat16 | 有效地址及 offset 按 512B 对齐；完整连续分块，见下方通用约束。 | 必填 |
| `dst` | 目标缓冲区 | tla.buffer | L0A / L0B；与 src 类型相同 | 有效地址及 offset 按 512B 对齐；单次写入及访问跨度不得超过目标实际分配，L0A、L0B 各 64KiB。 | 必填 |
| `params` | 搬运配置 | tla.CopyL1ToL0Params / None | 下列字段 | None 使用默认配置。 | None |
| `params.is_transpose` | 是否转置源矩阵 | 编译期 bool | False / True | False 保持矩阵索引，True 交换行列；源视图须匹配选定模式，转置后的范围须与目的相容。 | False |

**约束**

- 分块：令 `C = 32 / sizeof(dtype)`，基本块为 `(16,C)`，每块 512B；对当前 float16/bfloat16，C=16。
- 步长：四维分块视图的 `strides[1:] = (16*C,C,1)`；`stride[0]` 为 `16*C` 的倍数且不小于 `shape[1]*16*C`。仅最外层允许额外间距。
- 长度：单次有效数据不超过 64KiB，即当前两种类型最多 32768 个元素；存在偏移、行间距或共享分配时，可用长度相应减小。
- 地址：源、目的完整访问跨度均不得越界，不提供非完整分块的自动填充。
- 同步：调用方须保证装载、搬运、矩阵运算以及缓冲区复用之间的依赖。

仓库测试：[tla_ut/dma/copy.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/copy.py#L42)。

<a name="api-tla-make-partition-view"></a>

## tla.make_partition_view

**暂不开放。** 以下为原始提案，不作为可调用接口。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`make_partition_view( base: tla.buffer (ptr), shape: list[int], strides: list[int], tile_shape: list[int], dim_map: list[int], padding_value: enum<zero,nan,inf,-inf> )-> tla.tensor_view`

<a name="api-tla-make-strided-view"></a>

## tla.make_strided_view

**暂不开放。** 以下为原始提案，不作为可调用接口。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`make_strided_view( base: tla.buffer (ptr), shape: list[int], strides: list[int], tile_shape: list[int], dim_map: list[int], padding_value: enum<zero,nan,inf,-inf>, traversal_strides: list[int] )-> tla.tensor_view`

<a name="api-tla-make-gather-scatter-view"></a>

## tla.make_gather_scatter_view

**暂不开放。** 以下为原始提案，不作为可调用接口。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`make_gather_scatter_view( base: tla.buffer (ptr), shape: list[int], strides: list[int], tile_shape: list[int], sparse_dim: list[int], padding_value: enum<zero,nan,inf,-inf> )-> tla.tensor_view`

<a name="api-tla-load-view"></a>

## tla.load_view

**暂不开放。** 以下为原始提案，不作为可调用接口。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`load_view( src: tla.tensor_view, dst: tla.buffer, tile: list[list[int]] ) 只用于vector kernel`

<a name="api-tla-store-view"></a>

## tla.store_view

**暂不开放。** 以下为原始提案，不作为可调用接口。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`store_view( src: tla.buffer, dst: tla.tensor_view, tile: list[list[int]] ) 只用于vector kernel`

<a name="api-tla-set-pad-value"></a>

## tla.set_pad_value

**暂不开放。** 以下为原始提案，不作为可调用接口。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`set_pad_value()`

<a name="api-tla-set-post-copy"></a>

## tla.set_post_copy

**暂不开放。** 以下为原始提案，不作为可调用接口。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`set_post_copy()`

<a name="api-tla-stride-store"></a>

## tla.stride_store

范围结论：不需要。保留原提案供追溯。

原始提案：`【主推｜结构化带步长访存，不暴露指针】 stride_store(src: tla.buffer, dst: tla.buffer, offset: tla.expr, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None stride_load(src: tla.buffer, dst: tla.buffer, offset: tla.expr, other: tla.scalar, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None # rank<=3；numels 是有效元素数（越界预测），非重复次数；仅 AIV 【SIMD 级｜VF 内逐寄存器写】 simd.store_with_stride(dst: tla.buffer, indices, stride: int, mask: tla.mask, val: tla.vreg) -> None # stride 必须编译期常量且为 32B 块整数倍；无 post-update 【流式推进｜用循环携带地址，不提供可变指针对象】 for addr in tla.carry(init_addr): # 下沉 scf.for iter_args buf = tla.make_tensor(addr, ...) addr = addr + step`

<a name="api-tla-stride-load"></a>

## tla.stride_load

范围结论：不需要。保留原提案供追溯。

原始提案：`【主推｜结构化带步长访存，不暴露指针】 stride_store(src: tla.buffer, dst: tla.buffer, offset: tla.expr, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None stride_load(src: tla.buffer, dst: tla.buffer, offset: tla.expr, other: tla.scalar, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None # rank<=3；numels 是有效元素数（越界预测），非重复次数；仅 AIV 【SIMD 级｜VF 内逐寄存器写】 simd.store_with_stride(dst: tla.buffer, indices, stride: int, mask: tla.mask, val: tla.vreg) -> None # stride 必须编译期常量且为 32B 块整数倍；无 post-update 【流式推进｜用循环携带地址，不提供可变指针对象】 for addr in tla.carry(init_addr): # 下沉 scf.for iter_args buf = tla.make_tensor(addr, ...) addr = addr + step`
