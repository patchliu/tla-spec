# DFX接口规格

[返回接口目录](../README.md)

<a name="api-tla-device-assert"></a>

## tla.device_assert

签名：`tla.device_assert(cond, msg='')`

返回：None。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `cond` | 需要满足的条件 | bool / 标量 tl.tensor | 数据类型：int1 | debug=False 不生成 op。debug=True 用户断言到设备的转换有缺口。 | 必填 |
| `msg` | 断言失败时的消息 | 编译期 str | 字符串 | — | '' |

**调用示例**

```python
@tla.jit
def example(x):
    tla.device_assert(x >= 0, 'expected a nonnegative value')
```

<a name="api-tla-device-print"></a>

## tla.device_print

签名：`tla.device_print(prefix, *args, hex=False)`

返回：None。

编译表示：hivm.hir.debug。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `prefix` | 输出文本前缀 | 编译期 str | 字符串 | — | 必填 |
| `hex` | 是否按十六进制打印 | 编译期 bool | False / True | — | False |
| `*args` | 依次打印的值 | 标量 / tla.buffer / buffer_access | 地址空间：GM / UB（缓冲区）<br>Ascend950 前端拒绝 float64 标量 | 在核作用域外调用。buffer_access 按单元素处理。 | 可省略 |

**调用示例**

```python
@tla.jit
def example(x):
    tla.device_print('x', x)
```

仓库测试：[tla_ut/debug/device_print.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/debug/device_print.py#L41)。
