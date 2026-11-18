# 语言接口规格

[返回接口目录](../README.md)

本页的标量为编译期 Python 值，或在 kernel 中以秩 0 的 tl.tensor 表示的运行时值。形状为 (1,) 的块张量仍属于块张量。SIMD 向量寄存器由 tla.simd 接口处理。

<a name="api-tla-sqrt"></a>

## tla.sqrt

签名：`tla.sqrt(x)`

返回：FP32 标量。

编译表示：math.sqrt。SIMD 向量平方根使用 [tla.simd.vsqrt](simd.md#api-tla-simd-vsqrt)。

Ascend 950PR 标量抽测覆盖 FP32 负数、零、正有限值、正无穷及 NaN。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | Python float / 运行时 FP32 标量 | 数据类型：float32 | Python float 必须被 Triton 推断为 float32。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    value = tla.sqrt(4.0)
```

仓库测试：[tla_ut/vector/vector_type.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/vector_type.py#L124)。

<a name="api-tla-symbolic"></a>

## tla.symbolic

范围结论：不需要。保留原提案供追溯。

原始提案：`tla.symbolic(name: str, dtype: str = "int32") -> Expr`

<a name="api-tla-truncmod"></a>

## tla.truncmod

签名：`tla.truncmod(a, b)`

返回：按 Triton 整数提升得到的秩为 0 的整数 tl.tensor。

整数除法向零截断，非零余数与 a 同号。编译期整数除数为 0 时在前端报错。Ascend 950PR 已验证 int32 正负除数组合和有符号最小值对 3 的余数。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | Python int / 运行时整数标量 | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 不接受 bool。禁止混合 signed 和 unsigned tensor。除数非零，signed 最小值 ÷ −1 未定义。 | 必填 |
| `b` | 第二个操作数 | Python int / 运行时整数标量 | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 不接受 bool。禁止混合 signed 和 unsigned tensor。除数非零，signed 最小值 ÷ −1 未定义。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.truncmod(x, 16)
```

仓库测试：[tla_ut/vector/vector_type.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/vector_type.py#L104)。

<a name="api-tl-assume"></a>

## tl.assume

签名：`tl.assume(cond)`

返回：None。

沿用 Triton 入口，适用场景仍受 TLA 编译路径和目标约束。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `cond` | 必须成立的条件 | 标量 int1 tl.tensor / bool | 数据类型：int1 | 条件为假时行为未定义。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    tl.assume(x >= 0)
```

<a name="api-tla-where"></a>

## tla.where

签名：`tla.where(condition, x, y)`

返回：Triton 类型提升后的标量。

编译表示：arith.select。两路值在选择前求值。SIMD 谓词选择使用 [tla.simd.vsel](simd.md#api-tla-simd-vsel)。

Ascend 950PR 已验证运行时 int1 条件和 FP32 分支值，包括无穷和 NaN。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `condition` | 选择条件 | 编译期 bool / 运行时布尔标量 | 数据类型：int1 | 必须是布尔条件。 | 必填 |
| `x` | 条件为真时选择的值 | Python bool / int / float / 运行时标量 | 数据类型：bool / 整数 / 浮点 | 遵循 Triton 标量类型提升。 | 必填 |
| `y` | 条件为假时选择的值 | Python bool / int / float / 运行时标量 | 数据类型：bool / 整数 / 浮点 | 遵循 Triton 标量类型提升。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.where(x > 0, x, 0)
```

用例路径为 tla_ut/vector/vector_type.py，入口为 test_scalar_where_ir。

## 范围说明

以下条目在当前整理表中没有可定位的 TLA 接口名。

| 原序号 | 范围结论 | 涉及场景 |
| --- | --- | --- |
| 216 | 规范不支持（triton里不能写break） | T.loop_break |
| 217 | 暂不支持 | T.serial for i in range(n) |
| 218 | 暂不支持 | T.unroll for i in tl.static_range(n) |
| 219 | 暂不支持 | T.vectorized |
