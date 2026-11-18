# 标量接口规格

[返回接口目录](../README.md)

<a name="api-tla-set-atomic"></a>

## tla.set_atomic

签名：`tla.set_atomic(kind, dtype)`

返回：None。

编译表示：hivm.hir.set_atomic。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `kind` | 原子更新操作 | 编译期 tla.AtomicKind | NONE / ADD / MAX / MIN | NONE 恢复普通存储模式。 | 必填 |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B8：int8<br>B16：int16 / float16 / bfloat16<br>B32：int32 / float32<br>B64：不支持 | — | 必填 |

**调用示例**

```python
@tla.jit
def example(src_ub, y):
    tla.set_atomic(tla.AtomicKind.ADD, tl.float32)
    tla.store_ub_to_gm(src_ub, tla.view(y, (64,)))
    tla.set_atomic(tla.AtomicKind.NONE, tl.float32)
```

仓库测试：[tla_ut/dma/set_atomic.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/set_atomic.py#L35)。

<a name="api-tla-select"></a>

## tla.select

签名：`tla.select(condition, true_value, false_value)`

返回：分支提升后 dtype 的秩为 0 的 tl.tensor。

两个值分支提前求值。条件沿用 tl.where 到 int1 的转换，分支按 Triton 类型提升。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `condition` | 选择条件 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 tensor。 | 必填 |
| `true_value` | 条件为真时选取的值 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 tensor。 | 必填 |
| `false_value` | 条件为假时选取的值 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 tensor。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.select(x > 0, x, 0)
```

仓库测试：[tla_ut/scalar/scalar_frontend_contract.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/scalar_frontend_contract.py#L48)。

<a name="api-tla-abs"></a>

## tla.abs

签名：`tla.abs(x)`

返回：与输入转换后 dtype 一致的秩为 0 的 tl.tensor。

无符号整数及 bool 保持原值。有符号整数最小值按原 dtype 保留，例如 int32 的 −2³¹。浮点路径清除符号位，NaN payload 不作保证。Ascend 950PR 已验证 int32 最小值及邻近边界。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 tensor。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.abs(x)
```

<a name="api-xor"></a>

## ^

签名：`lhs ^ rhs`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `lhs` | 左操作数 | int / 标量整数 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 对整数位模式做 XOR，类型提升遵循 Triton。 | 必填 |
| `rhs` | 右操作数 | int / 标量整数 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 对整数位模式做 XOR，类型提升遵循 Triton。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x, y):
    value = x ^ y
```

<a name="api-tla-cast"></a>

## tla.cast

签名：`tla.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)`

返回：目标 dtype 的秩为 0 的 tl.tensor。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `input` | 待转换的标量 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 tensor。 | 必填 |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 不接受 bfloat16。按源和目标类型对执行 Triton cast。 | 必填 |
| `fp_downcast_rounding` | 浮点窄化舍入模式 | 编译期 str 或 None | None / rtne / rtz | 仅用于 Triton 支持的浮点窄化类型对。 | None |
| `bitcast` | 是否按位重解释 | 编译期 bool | False / True | True 需要等位宽。 | False |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.cast(x, tl.int32)
```

仓库测试：[tla_ut/scalar/cdiv.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/cdiv.py#L40)。

<a name="api-tla-cdiv"></a>

## tla.cdiv

签名：`tla.cdiv(x, div)`

返回：按 Triton 类型提升得到的秩为 0 的整数 tl.tensor。

计算式为 (x + (div − 1)) // div。验收范围为 x ≥ 0、div > 0，中间结果 div − 1 与 x + (div − 1) 均须在实际计算 dtype 的可表示范围内。Ascend 950PR 已验证 int32 整除、非整除以及中间和恰好等于 INT32_MAX 的边界。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | int / 标量整数 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 面向非负被除数、正除数的常规用法，中间计算不能溢出。 | 必填 |
| `div` | 除数 | int / 标量整数 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 面向非负被除数、正除数的常规用法，中间计算不能溢出。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.cdiv(x, 16)
```

仓库测试：[tla_ut/scalar/cdiv.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/cdiv.py#L42)。

<a name="api-tla-clamp"></a>

## tla.clamp

签名：`tla.clamp(x, min, max, propagate_nan=tl.PropagateNan.NONE)`

返回：提升后的浮点 dtype 的秩为 0 的 tl.tensor。

x、min、max 经类型提升后须为浮点类型。min 和 max 均不得为 NaN，且 min ≤ max。Ascend 950PR 的 FP32 正例使用运行时 x 和可表示为 FP32 的有序常量边界。详见[标量比较设备范围](#scalar-comparison-test-scope)。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `min` | 下界 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `max` | 上界 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `propagate_nan` | 选择 NaN 传播规则 | 编译期 tl.PropagateNan | NONE / ALL | 仅作用于 x。Ascend 950PR 的设备正例使用 NONE，x 不含 NaN。ALL 仅纳入 IR 验证。 | tl.PropagateNan.NONE |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.clamp(x, 0, 255)
```

仓库测试：[tla_ut/scalar/clamp.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/clamp.py#L46)。

<a name="api-tl-constexpr"></a>

## tl.constexpr

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

仓库测试：[tla_ut/scope/gm2l1.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/gm2l1.py#L22)。

<a name="api-tla-exp"></a>

## tla.exp

签名：`tla.exp(x)`

返回：e 的 x 次方，为 FP32 标量。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 要计算指数的标量。 | 标量 tl.tensor / Python float | 数据类型：float32 | tensor 的秩为 0；Python float 及其 tl.constexpr 包装按 float32 处理。 | 必填 |

正无穷得到正无穷，负无穷得到零，NaN 返回 NaN。 本轮按[标量数学函数转测范围](#scalar-unary-test-scope)验收。

调用示例

```python
@tla.jit
def example(x):
    y = tla.exp(x)
```

<a name="api-tla-infinity"></a>

## tla.infinity

签名：`tla.infinity(dtype)`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B8：不支持<br>B16：float16 / bfloat16<br>B32：float32<br>B64：不支持 | — | 必填 |

**调用示例**

```python
@tla.jit
def example():
    value = tla.infinity(tl.float32)
```

仓库测试：[tla_ut/scalar/infinity.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/infinity.py#L40)。

<a name="api-tla-log1p"></a>

## tla.log1p

签名：`tla.log1p(x)`

返回：ln(1+x)，为 FP32 标量。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 加一前的标量。 | 标量 tl.tensor / Python float | 数据类型：float32 | tensor 的秩为 0；Python float 及其 tl.constexpr 包装按 float32 处理。 | 必填 |

x 等于 −1 时返回负无穷，x 小于 −1 时返回 NaN。正无穷返回正无穷，NaN 返回 NaN。保留零的符号。 本轮按[标量数学函数转测范围](#scalar-unary-test-scope)验收。

调用示例

```python
@tla.jit
def example(x):
    y = tla.log1p(x)
```

<a name="api-tla-maximum"></a>

## tla.maximum

签名：`tla.maximum(a, b, propagate_nan=tl.PropagateNan.NONE)`

返回：操作数提升后的 dtype 的秩为 0 的 tl.tensor。

浮点设备验收见[标量比较设备范围](#scalar-comparison-test-scope)。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `b` | 第二个操作数 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `propagate_nan` | 选择 NaN 传播规则 | 编译期 tl.PropagateNan | NONE / ALL | Ascend 950PR 的浮点设备正例使用 NONE，两个输入均不含 NaN。ALL 仅纳入 IR 验证。 | tl.PropagateNan.NONE |

**调用示例**

```python
@tla.jit
def example(x, y):
    value = tla.maximum(x, y)
```

仓库测试：[tla_ut/scalar/maximum.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/maximum.py#L43)。

<a name="api-tla-max-value"></a>

## tla.max_value

签名：`tla.max_value(dtype)`

返回：指定 dtype 的最大有限值，为秩为 0 的 tl.tensor。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 / float64 | float64 常量仅纳入前端与 IR 验证。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    value = tla.max_value(tl.int32)
```

仓库测试：[tla_ut/scalar/max_value.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/max_value.py#L54)。

<a name="api-tla-minimum"></a>

## tla.minimum

签名：`tla.minimum(a, b, propagate_nan=tl.PropagateNan.NONE)`

返回：操作数提升后的 dtype 的秩为 0 的 tl.tensor。

浮点设备验收见[标量比较设备范围](#scalar-comparison-test-scope)。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `b` | 第二个操作数 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `propagate_nan` | 选择 NaN 传播规则 | 编译期 tl.PropagateNan | NONE / ALL | Ascend 950PR 的浮点设备正例使用 NONE，两个输入均不含 NaN。ALL 仅纳入 IR 验证。 | tl.PropagateNan.NONE |

**调用示例**

```python
@tla.jit
def example(x, y):
    value = tla.minimum(x, y)
```

仓库测试：[tla_ut/scalar/minimum.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/minimum.py#L43)。

<a name="api-tla-sigmoid"></a>

## tla.sigmoid

签名：`tla.sigmoid(x)`

返回：1/(1+exp(−x))，为 FP32 标量。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 要变换的标量。 | 标量 tl.tensor / Python float | 数据类型：float32 | tensor 的秩为 0；Python float 及其 tl.constexpr 包装按 float32 处理。 | 必填 |

x 等于零时返回 0.5。正无穷和负无穷分别返回 1 和 0，NaN 返回 NaN。 本轮按[标量数学函数转测范围](#scalar-unary-test-scope)验收。

调用示例

```python
@tla.jit
def example(x):
    y = tla.sigmoid(x)
```

<a name="scalar-unary-test-scope"></a>

## 标量数学函数转测范围

本轮覆盖 Ascend 950PR 上的 TLA 标量 FP32 功能。运行时输入和返回值都是 FP32 标量，Triton 前端用秩为 0 的 `tl.tensor` 表示。编译期 Python 浮点数按 FP32 验证 IR。实机用例通过单个 `@tla.jit` kernel 验证运行时标量，SIMD 写回只用于读取计算结果。

| 接口 | 正向用例 |
| --- | --- |
| `tla.exp` | 常规正负值、零、FP32 上溢和下溢边界、正负无穷、NaN |
| `tla.log1p` | 接近零的正负值、正零和负零的符号、范围缩减切换点、−1 附近、−1、低于 −1、较大正值、正负无穷、NaN |
| `tla.sigmoid` | 常规正负值、大幅值正负输入、零、正负无穷、NaN |

- 验收覆盖适配器 IR、结果秩与类型、设备编译和数值结果。负向抽测 bool、int、非 FP32 标量 tensor 和块 tensor，预期前端报 TypeError。数值对照 PyTorch FP32，同位置 NaN 视为匹配。验收容差为 `rtol=1e-5`、`atol=2**-126`。绝对容差允许目标对次正规结果冲零。`log1p(±0)` 另核对零的符号。
- 本轮范围外包括其他 dtype、块张量和 SIMD 向量的正向用例，以及 SIMT 和其他芯片。原生单指令映射、性能指标和 ULP 误差上限留待单独评审。

参考实现见 [triton-ascend PR #3268](https://github.com/triton-lang/triton-ascend/pull/3268)。用 Ascend 950PR、`2aa6cc727` wheel 和该 PR 改动验证了三项接口，15 项检查全部通过。转测进度见首页。

实现路径：`tla/scalar.py`。用例路径：`tla_ut/scalar/unary_math.py`。

<a name="scalar-comparison-test-scope"></a>

## 标量比较设备范围

Ascend 950PR 使用 CANN 9.1、2aa6cc727 wheel 和本次 TLA 源码验证。clamp、maximum、minimum 的 FP32 默认模式通过常规值、边界值和正无穷的读回验证。该范围要求操作数不含 NaN，clamp 还要求边界有序。

三者的 FP32 ALL 模式在相同工具链的设备指令选择阶段失败，诊断涉及 fmaximum 或 fminimum。转测中的设备正向矩阵使用 NONE。其他 dtype 或模式按对应接口的前端及 IR 范围验证，设备组合按目标单独补齐。
