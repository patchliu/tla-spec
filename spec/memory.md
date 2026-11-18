# 内存接口规格

[返回接口目录](../README.md)

以下 DMA、L0 搬运和填充约束面向 Ascend950（RegBase）。

<a name="api-tla-alloc"></a>

## tla.alloc

计划签名：`tla.alloc(shape, dtype, space=None)`

返回：tla.buffer。

分配由 `space` 指定地址空间。多缓冲版本数通过 [tla.annotate_multi_buffer](schedule.md#api-tla-annotate-multi-buffer) 单独标注。当前实现仍接收 `versions` 和 `scope`，签名变更见 [MEMORY-002](../tracking/memory.md#memory-002)。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `shape` | 逻辑形状 | tuple[int, ...] / list[int] | 非空正整数组 | 按元素计。 | 必填 |
| `dtype` | 元素类型 | 编译期 tl.dtype | 可物化为 memref 元素的整数或浮点 dtype | 分配类型须同时满足该空间的内存规划和后续操作的 dtype 约束。 | 必填 |
| `space` | 指定地址空间 | 编译期 tla.AddressSpace / str / None | 地址空间：UB / L1 / L0C（已确认分配路径） | 省略或传 None 时为 UB。GM 禁止 alloc，L0A_SF/L0B_SF 无映射。其他有空间映射的名字不自动获得分配支持。 | None |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.vector():
        tile = tla.alloc((16, 16), tl.float16, space='ub')
```

仓库测试：[tla_ut/scope/pointer_abi.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/pointer_abi.py#L15)。

<a name="api-tla-reshape"></a>

## tla.reshape

签名：`tla.reshape(src, shape)`

返回：tla.buffer。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.buffer | 地址空间及数据类型继承源缓冲区 | 源的静态 strides 必须等于其形状对应的 row-major strides。允许非零或动态 offset，结果继承 offset。 | 必填 |
| `shape` | 逻辑形状 | tuple[int, ...] / list[int] | 非空正整数组 | reshape 后元素总数相同。 | 必填 |

**调用示例**

x_ub 为连续的 16×16 UB 缓冲区。

```python
@tla.jit
def example(x_ub, y):
    flat = tla.reshape(x_ub, (256,))
    tla.store_ub_to_gm(flat, tla.view(y, (256,)))
```

仓库测试：[tla_ut/vector/subview.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/subview.py#L64)。

<a name="api-tla-view"></a>

## tla.view

签名：`tla.view(src, shape, strides=None, offsets=None, space='gm')`

返回：tla.buffer。

GM 指针的偏移按有符号 64 位元素索引计算，保留 kernel 参数的指针来源。片上指针走地址绑定，形成 hivm.hir.pointer_cast 和 memref.reinterpret_cast。buffer 源生成 memref.subview。指针源的访问范围由调用者保证。

用例路径为 tla_ut/scope/view_address_space.py。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | 标量 Triton 指针 / tla.buffer | 指针源继承指针元素类型；GM 元素必须为整字节宽的整数或浮点 | buffer 源保持 dtype 和地址空间，输出 stride 为源 stride 与相对步长的乘积。 | 必填 |
| `shape` | 逻辑形状 | tuple[int, ...] / list[int] | 非空正整数组 | 按元素计。 | 必填 |
| `strides` | 各轴步长 | tuple[int, ...] / list[int] / None | 非空正整数组 / None | 与对应视图秩相同。省略时指针源用 row-major，buffer 源用单位相对步长。 | None |
| `offsets` | 基于源地址或视图的元素偏移 | tuple[int / 标量整数 tl.tensor, ...] / None | 按元素计的整数偏移 / None | 维数须匹配。静态子视图越界拒绝，动态偏移由调用者保证。 | None |
| `space` | 指定地址空间 | 编译期 str / tla.AddressSpace | 地址空间：GM / UB / L1 / L0A / L0B / L0C / SSBUF / BT / FB | 指针源的字符串使用小写，亦可传对应的 tla.AddressSpace。buffer 源保留默认参数。 | 'gm' |

**调用示例**

```python
@tla.jit
def example(x, dst_ub):
    tile = tla.view(x, (16, 16), offsets=(0, 0))
    tla.load_gm_to_ub(tile, dst_ub)
```

仓库测试：[tla_ut/scope/pointer_abi.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/pointer_abi.py#L16)。

<a name="api-tla-subview"></a>

## tla.subview

签名：`tla.subview(src, offsets, sizes, strides=None)`

返回：tla.buffer。

编译表示：memref.subview。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.buffer | 地址空间及数据类型继承源缓冲区 | 同秩切片，结果与源共享存储。 | 必填 |
| `offsets` | 相对于源视图的各轴逻辑元素偏移 | tuple[int / 标量整数 tl.tensor, ...] | 按元素计的整数偏移 | 维数须匹配。静态子视图越界拒绝，动态偏移由调用者保证。 | 必填 |
| `sizes` | 各轴要截取的元素个数 | tuple[int, ...] / list[int] | 非空正整数组 | 按元素计。 | 必填 |
| `strides` | 各轴步长 | tuple[int, ...] / list[int] / None | 非空正整数组 / None | 与对应视图秩相同。省略时各轴相对步长为 1。结果 stride 为源 stride 与相对步长的乘积。 | None |

**调用示例**

x_ub 为至少 16×16 个元素的二维缓冲区，目的视图与子视图类型一致。

```python
@tla.jit
def example(x_ub, y):
    tile = tla.subview(x_ub, (0, 0), (16, 16))
    tla.store_ub_to_gm(tile, tla.view(y, (16, 16)))
```

仓库测试：[tla_ut/simd/predicate_views.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/predicate_views.py#L39)。

<a name="api-tla-fill"></a>

## tla.fill

签名：`tla.fill(dst, value)`

返回：None。将标量广播到整个目标缓冲区，编译表示为标量 `hivm.hir.vbrc`。

Ascend 950PR 本轮只验收前端校验和 Adapter IR。使用 CANN 9.1 和 2aa6cc727 wheel 验证时，连续 UB 的 bfloat16、int32、float32 在一维 (64,) 和二维 (8, 8) 均在链接时报缺少对应的标量广播模板符号。设备正向范围见 [MEMORY-003](../tracking/memory.md#memory-003)。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `dst` | 待填充的缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：float32 / bfloat16 / int32<br>形状：1D / 2D | 连续 row-major，offset 必须已知且为 0。元素数为正且不超过 UB 容量，允许非对齐尾部。 | 必填 |
| `value` | 广播标量 | Python int / float / 秩为 0 的数值 tl.tensor | int32 目标：整数值，范围为 −2³¹ 至 2³¹−1。<br>float32 / bfloat16 目标：整数或浮点标量，按目标 dtype 转换。 | Python bool 和块 tensor 会被前端拒绝。int32 目标拒绝浮点值。 | 必填 |

当前前端拒绝 subview 结果，包括静态零偏移的子视图。调用方须确保生产、消费及复用缓冲区的依赖有正确同步。

用例路径为 tla_ut/memory/fill.py 和 tla_ut/memory/fill_frontend_contract.py。
