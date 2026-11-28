# 标量接口规格

[返回接口目录](../README.md)

数值输入仅支持标量。以下示例中的标量计算放在 SIMD VF 外。

<a name="api-tla-set-atomic"></a>

## tla.set_atomic

签名：`tla.set_atomic(kind, dtype)`

返回：None。

编译表示：hivm.hir.set_atomic。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `kind` | 原子更新操作 | 编译期 tla.AtomicKind | NONE / ADD / MAX / MIN | 必须使用枚举对象，不接受字符串或其余 AtomicKind 成员。NONE 恢复普通存储模式。 | 必填 |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B8：int8<br>B16：int16 / float16 / bfloat16<br>B32：int32 / float32<br>B64：不支持 | 须与实际原子写回的元素类型一致，不执行类型转换。NONE 仍须传入合法 dtype。 | 必填 |

该调用设置执行核后续相关 GM 写回的控制状态，不绑定某个目的 Buffer，也不只作用于下一次 store。模式须显式修改或用 NONE 复位，不会随 Python scope 自动恢复。ADD、MAX、MIN 分别将写入值与 GM 目标的原值相加、取较大值、取较小值；调用者须提供有效的目标初值。

约束：本接口不执行搬运、不返回旧值，也不承担 DMA 完成等待或核间同步。[store_ub_to_gm](dma.md#api-tla-store-ub-to-gm) 仅接收源和目标，原子模式由本接口独立控制。切换模式前须保证相关写回完成，并在原子写回完成后显式用 NONE 复位。

**调用示例**

前提：`x`、`y` 均指向连续的 64 个 float32 元素，`y` 已初始化；使用显式 PIPE.ALL 同步。

```python
@tla.jit
def example(x, y):
    ub = tla.alloc((64,), tl.float32)
    with tla.scope.vector():
        tla.load_gm_to_ub(tla.view(x, (64,)), ub)
        tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.set_atomic(tla.AtomicKind.ADD, tl.float32)
        tla.store_ub_to_gm(ub, tla.view(y, (64,)))
        tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.set_atomic(tla.AtomicKind.NONE, tl.float32)
```

仓库测试：[tla_ut/dma/set_atomic.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/dma/set_atomic.py#L35)。

<a name="api-tla-load-scalar"></a>

## tla.load_scalar

签名：`tla.load_scalar(buf)`

返回：从 `buf` 读取的单个标量，dtype 与 `buf.dtype` 一致。

编译表示：memref.load。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `buf` | 源缓冲区 | tla.buffer | 地址空间：GM / UB<br>数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 / float64 | 单元素 Buffer，各维尺寸须在编译期为 1。 | 必填 |

在 SIMD VF 外调用。用 [tla.view](memory.md#api-tla-view) 从 GM 指针构造视图，或用 [tla.subview](memory.md#api-tla-subview) 从已有 Buffer 选取一个元素。偏移可为运行时整数，调用者须保证访问在范围内。

**调用示例**

```python
@tla.jit
def example(src, index):
    with tla.scope.vector():
        value = tla.load_scalar(tla.view(src, (1,), offsets=(index,)))
```

仓库测试：[tla_ut/dma/scalar_load.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/dma/scalar_load.py#L211)。

<a name="api-tla-store-scalar"></a>

## tla.store_scalar

签名：`tla.store_scalar(buf, value)`

返回：None。

编译表示：memref.store。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `buf` | 目标缓冲区 | tla.buffer | 地址空间：GM / UB<br>数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 / float64 | 单元素 Buffer，各维尺寸须在编译期为 1。 | 必填 |
| `value` | 写入值 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 / float64 | Python 标量按 Triton 规则确定 dtype。同类型直接写入，包括 bfloat16。类型不同时，按 [tla.cast](#api-tla-cast) 的数值转换规则转为目标类型。 | 必填 |

在 SIMD VF 外调用。目标元素的选取和边界约束同 [tla.load_scalar](#api-tla-load-scalar)。

**调用示例**

前提：`src`、`dst` 均为 float32 GM 指针，且 `index` 在二者的有效范围内。

```python
@tla.jit
def example(src, dst, index):
    with tla.scope.vector():
        ub = tla.alloc((4,), tl.float32)
        slot = tla.subview(ub, offsets=(2,), sizes=(1,))
        tla.store_scalar(slot, tla.load_scalar(tla.view(src, (1,), offsets=(index,))))
        tla.store_scalar(tla.view(dst, (1,), offsets=(index,)), tla.load_scalar(slot))
```

仓库测试：[tla_ut/dma/scalar_load.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/dma/scalar_load.py#L211)。

<a name="api-tla-select"></a>

## tla.select

签名：`tla.select(condition, true_value, false_value)`

返回：按条件选出的数值标量，类型由两个分支按 Triton 规则提升后决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `condition` | 选择条件 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 输入。 | 必填 |
| `true_value` | 条件为真时选取的值 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 输入。 | 必填 |
| `false_value` | 条件为假时选取的值 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 输入。 | 必填 |

三个参数的类型范围均不含 float64。两个分支表达式都会在选择前求值，未选中的分支也须满足类型约束。不能用 `select` 保护越界访存或副作用；需要条件执行时使用显式控制流。

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.select(x > 0, x, 0)
```

仓库测试：[tla_ut/scalar/select.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/scalar/select.py#L35)。

<a name="api-tla-abs"></a>

## tla.abs

签名：`tla.abs(x)`

返回：绝对值标量。运行时输入保持原 dtype；Python 标量按 Triton 规则转换为标量值。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 不接受 bfloat16、float64、float8、Block、Buffer、Buffer 访存位置或 SIMD 寄存器。 | 必填 |

bool 和无符号整数保持不变。有符号最小整数的结果仍为该固定宽度的最小值，不饱和也不扩宽。浮点绝对值清除符号位，包括负零、无穷和 NaN；不保证保留 NaN payload。该接口不隐式读取 Buffer、分配 Buffer 或广播到 SIMD 寄存器。

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.abs(x)
```

仓库测试：[tla_ut/scalar/abs.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/scalar/abs.py#L36)。

<a name="api-xor"></a>

## ^

签名：`lhs ^ rhs`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `lhs` | 左操作数 | 整数标量 | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 对整数位模式做 XOR，类型提升遵循 Triton。 | 必填 |
| `rhs` | 右操作数 | 整数标量 | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 对整数位模式做 XOR，类型提升遵循 Triton。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x, y):
    value = x ^ y
```

<a name="api-tla-cast"></a>

## tla.cast

签名：`tla.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)`

返回：目标 `dtype` 的标量。`bitcast=False` 作数值转换，`bitcast=True` 重解释等宽位模式。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `input` | 待转换的标量 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 / float64 | 不接受 bfloat16。float64 不支持数值窄化到更窄的浮点类型。Ascend950 上，运行时 float64 转整数仅支持等宽 bitcast。 | 必填 |
| `dtype` | 目标类型 | 编译期标量 tl.dtype | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 不接受 bfloat16。Ascend950 前端拒绝 float64 目标，包括数值转换和等宽 bitcast。 | 必填 |
| `fp_downcast_rounding` | 浮点窄化舍入模式 | 编译期 str 或 None | None / rtne / rtz | 仅用于 Triton 支持的浮点窄化类型对。 | None |
| `bitcast` | 是否按位重解释 | 编译期 bool | False / True | True 需要等位宽。 | False |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.cast(x, tl.int32)
```

仓库测试：[tla_ut/scalar/cast.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/scalar/cast.py#L41)。

<a name="api-tla-cdiv"></a>

## tla.cdiv

签名：`tla.cdiv(x, div)`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | 整数标量 | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 面向非负被除数、正除数的常规用法，中间计算不能溢出。 | 必填 |
| `div` | 除数 | 整数标量 | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 面向非负被除数、正除数的常规用法，中间计算不能溢出。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.cdiv(x, 16)
```

仓库测试：[tla_ut/scalar/cdiv.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/scalar/cdiv.py#L42)。

<a name="api-tla-clamp"></a>

## tla.clamp

签名：`tla.clamp(x, min, max, propagate_nan=tl.PropagateNan.NONE)`

返回：限制在 `[min, max]` 内的标量，类型为三个输入共同提升后的浮点类型。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 输入。 | 必填 |
| `min` | 下界 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 输入。不得为 NaN，且须满足 min ≤ max。 | 必填 |
| `max` | 上界 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 输入。不得为 NaN，且须满足 min ≤ max。 | 必填 |
| `propagate_nan` | 选择 NaN 传播规则 | 编译期 tl.PropagateNan | NONE / ALL | 仅作用于 x。NONE 下 NaN x 暂不支持；ALL 传播 x 的 NaN。 | tl.PropagateNan.NONE |

约束：三个输入须共同形成浮点计算，全部为整数或 bool 的组合不支持。整数或 bool 可与本表允许的浮点输入混合，按 Triton 规则提升；每个原始输入均拒绝 float16 和 bfloat16，Ascend950 前端还拒绝 float64。

任一上下界为 NaN 或 `min > max` 时行为未定义，ALL 不改变该约束。调用者须保证运行时输入合法，接口不插入运行时 NaN 或边界检查。

**调用示例**

```python
@tla.jit
def example(x: tl.float32):
    value = tla.clamp(x, 0.0, 255.0)
```

仓库测试：[tla_ut/scalar/clamp.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/scalar/clamp.py#L44)。

<a name="api-tl-constexpr"></a>

## tl.constexpr

范围结论：不需要。保留原提案供追溯。

签名：`tl.constexpr`

返回：用于表示编译期值的类型，可用于内核参数标注。

沿用 Triton 入口，适用场景仍受 TLA 编译路径和目标约束。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `value` | 编译期值 | Python 对象 | 具体范围由使用该值的表达式决定 | 不作为设备运行时 SSA 参数。 | 必填 |

**调用示例**

```python
@tla.jit
def example(size: tl.constexpr):
    tile = tla.alloc((size,), tl.float32)
```

仓库测试：[tla_ut/scope/gm2l1.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/scope/gm2l1.py#L22)。

<a name="api-tla-infinity"></a>

## tla.infinity

签名：`tla.infinity(dtype)`

返回：指定 `dtype` 的正无穷标量常量。负无穷可用一元负号构造。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B8：不支持<br>B16：float16 / bfloat16<br>B32：float32<br>B64：不支持 | — | 必填 |

**调用示例**

```python
@tla.jit
def example():
    value = tla.infinity(tl.float32)
```

仓库测试：[tla_ut/scalar/infinity.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/scalar/infinity.py#L40)。

<a name="api-tla-maximum"></a>

## tla.maximum

签名：`tla.maximum(a, b, propagate_nan=tl.PropagateNan.NONE)`

返回：两个标量的较大值，类型由操作数按 Triton 规则提升后决定；该接口不执行归约。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 输入。 | 必填 |
| `b` | 第二个操作数 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 输入。 | 必填 |
| `propagate_nan` | 选择 NaN 传播规则 | 编译期 tl.PropagateNan | NONE / ALL | NONE 下恰有一个输入为 NaN 的组合暂不支持；ALL 的语义是传播任一输入的 NaN。 | tl.PropagateNan.NONE |

约束：float16 和 bfloat16 在类型提升前按原始操作数检查。Ascend950 前端拒绝任一操作数为 float64 的调用。

**调用示例**

```python
@tla.jit
def example(x, y):
    value = tla.maximum(x, y)
```

仓库测试：[tla_ut/scalar/maximum.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/scalar/maximum.py#L43)。

<a name="api-tla-max-value"></a>

## tla.max_value

签名：`tla.max_value(dtype)`

返回：指定 `dtype` 的最大有限值标量常量；整数返回该类型的上界。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 / float64 | — | 必填 |

**调用示例**

```python
@tla.jit
def example():
    value = tla.max_value(tl.int32)
```

仓库测试：[tla_ut/scalar/max_value.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/scalar/max_value.py#L54)。

<a name="api-tla-minimum"></a>

## tla.minimum

签名：`tla.minimum(a, b, propagate_nan=tl.PropagateNan.NONE)`

返回：两个标量的较小值，类型由操作数按 Triton 规则提升后决定；该接口不执行归约。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 输入。 | 必填 |
| `b` | 第二个操作数 | 标量（bool / int / float） | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 输入。 | 必填 |
| `propagate_nan` | 选择 NaN 传播规则 | 编译期 tl.PropagateNan | NONE / ALL | NONE 下恰有一个输入为 NaN 的组合暂不支持；ALL 的语义是传播任一输入的 NaN。 | tl.PropagateNan.NONE |

约束：float16 和 bfloat16 在类型提升前按原始操作数检查。Ascend950 前端拒绝任一操作数为 float64 的调用。

**调用示例**

```python
@tla.jit
def example(x, y):
    value = tla.minimum(x, y)
```

仓库测试：[tla_ut/scalar/minimum.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/tla_ut/scalar/minimum.py#L43)。
