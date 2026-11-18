# 主机接口规格

[返回接口目录](../README.md)

<a name="api-tla-jit"></a>

## tla.jit

签名：`tla.jit(fn=None, **kwargs)`

返回：装饰后的 TLA JIT 函数。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `fn` | 要装饰的 Python 函数 | Python callable 或 None | 可编译的 Python 函数 / None | 允许 @tla.jit 或 @tla.jit(...)。 | None |
| `**kwargs` | 传给 TLAJITFunction 的 JIT 选项 | 标识符到值的关键字参数 | TLAJITFunction 继承的 JIT 选项 | 例如 debug、noinline。非法选项沿用 Triton JIT 检查。 | 可省略 |

**调用示例**

```python
@tla.jit
def example(x, y):
    tile = tla.alloc((64,), tl.float32)
    tla.load_gm_to_ub(tla.view(x, (64,)), tile)
    tla.store_ub_to_gm(tile, tla.view(y, (64,)))
```

仓库测试：[tla_ut/scope/pointer_abi.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/pointer_abi.py#L12)。
