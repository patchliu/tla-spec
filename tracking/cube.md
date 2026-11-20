# Cube评审跟踪

[返回接口目录](../README.md)

<a name="cube-001"></a>

## CUBE-001 · tla.copy_ub_to_l1

[查看规格](../spec/cube.md#api-tla-copy-ub-to-l1)

- `copy_ub_to_l1` 仅保留 `src`、`dst`，不提供填充配置；物理布局由调用方准备。

结论：收紧为规格列出的 32B 对齐搬运，暂不支持手动设置 `pad_mode` 和 `pad_value`。多个 AIV 写者须写入互不重叠范围，AIV→AIC 之间须建立跨核依赖。

依据：AscendNPU-IR `c521aba2` 的 `bishengir/lib/Template/include/RegBase/DMAUtils.h` 中 `copy_ubuf_to_cbuf_intrin_core` 按 32B 块计算行间距，未提供非对齐地址回退。非对齐行距可能产生错误数据；规格外输入尚未统一在前端拒绝。

补充结论：FP32→FP32 纳入待转测范围，保留 32B 对齐要求（FP32 为 8 个元素），支持规格中的 2D 物理视图与 4D NZ 物理拷贝，不执行 ND2NZ 或类型转换。

补充依据：AscendNPU-IR `0e7d8bb2d0d6a39b83996e909eee1b973685e164` 的 `HIVMDMAOps.td::CopyOp` 接受 F32 且要求源、目的同类型；RegBase `Vector/Copy1D.cpp`、`Copy2D.cpp`、`Copy3D.cpp` 均注册 UB→L1 的 float 模板。TLA 的 `copy_ub_to_l1` 直接生成该 copy 路径。核对范围为 IR 类型约束与目标 lowering；本次规格更新未新增设备精度验证。

补充验证：在 Ascend950DT_9582、AscendNPU-IR `1389120d` 的 Process 路径上，FP32 的四维 NZ 物理拷贝与带源行距的二维物理拷贝均经 L1→UB 回读并与参考逐元素精确比对。二维正例的行宽为 8 个元素，源行距为 16 个元素，源行内偏移为 8 个元素。

<a name="cube-002"></a>

## CUBE-002 · tla.copy_l1_to_ub

[查看规格](../spec/cube.md#api-tla-copy-l1-to-ub)

结论：目的 UB 的有效地址和行间距须满足 32B 对齐，仅保留完整 NZ 源分块及规格中的 dtype。非对齐目的可能触发 MTE 地址错误，不能以末维连续和不越界替代对齐要求。

依据：AscendNPU-IR `c521aba2` 的 `bishengir/lib/Template/lib/RegBase/Cube/l12ub.cpp` 按 fractal 地址执行 32B 搬运；静态地址约束的前端诊断仍需补齐。

待评审扩展：上述设备和后端已验证同类型 FP32 从 (3,2,16,8) NZ 搬至 (32,24) ND，目标行距 32 个元素、行内偏移 8 个元素，源从非零偏移开始。AIC 通过 MTE1 向 AIV sub-block 0 建立依赖。后续使用预排 NZ 数据填充源分配及哨兵块，真机验证了源 stride (384,128,8,1) 和 (640,256,8,1)，分别覆盖仅 N 方向留空和 N、M 两个外层方向留空。两个用例各比较 768 个 FP32 元素，最大绝对误差为 0。5D、类型转换、FP32 C0=16 及末维不连续仍超出扩展范围。

后端依据：AscendNPU-IR `1389120d` 的 `L12UBOp` 经 RegBase `L12UBOpToLibraryCallPattern` 选择 `l12ub.cpp` 中注册的同类型 FP32 四维到二维模板。

<a name="cube-003"></a>

## CUBE-003 · tla.copy_l1_to_bt

[查看规格](../spec/cube.md#api-tla-copy-l1-to-bt)

结论：已关闭，本轮不需要。L1→BT→MMAD 路径仍存在 AIC trap，尚未定位到搬运、地址规划或消费端；修复后再重新评审。

<a name="cube-004"></a>

## CUBE-004 · tla.copy_l1_to_fb

[查看规格](../spec/cube.md#api-tla-copy-l1-to-fb)

- 目前只有预留入口，建议在完成后端映射前标记为不可转测。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="cube-005"></a>

## CUBE-005 · tla.copy_l0c_to_ub

[查看规格](../spec/cube.md#api-tla-copy-l0c-to-ub)

- 需限定 `dma_mode`、`pre_quant`、`pre_relu` 与目的子块的合法组合。

结论：源为 FP32；三种目的布局按规格提供 FP32/FP16/BF16 输出及无激活/ReLU。双目标仅提供 NZ2ND，且须显式 `sub_block_idx=None`。额外激活模式、任意 scale 及其他双目标布局暂不开放。

目的范围收紧为直接分配或等价 reshape 的连续、零偏移 UB。带额外行间距的目的视图存在设备执行不能完成的问题，即使地址满足 32B 对齐也不能纳入当前范围；目的 subview 暂不开放。待定位地址生成及同步后再评审该范围。

补充结论：NZ2NZ 来源收紧为规格列出的连续 4D NZ 布局；保留父 buffer 额外间距的来源子视图暂不开放。该限制排除本次发现的 12 个失败组合（3 种 dtype × 2 个子块 × 2 种 ReLU 配置）。相同来源的 NZ2ND/NZ2DN 路径通过本次对照检查，不套用该来源限制。

原因：AscendNPU-IR `c521aba2` 的 `bishengir/lib/Template/lib/RegBase/Cube/compat/Fixpipe/Fixpipe.cpp` 中，`copy_matrix_cc_to_ubuf_normal_4d_to_4d_core` 按切片后的逻辑 M 推导来源步长，未使用描述符保留的实际 stride。TA 生成的描述符正确。前端仍需补齐规格外来源的拒绝检查；文档收紧不等同于实现已经修复。后端按实际描述符处理 stride，并通过来源子视图精度检查后，再评审是否解除限制。

查阅版本：TA main `f1997518` 合入 PR #3267（`c8a02948`）与 PR #2982（`99373b0d`），实际使用 AscendNPU-IR `c521aba2`。三个 L0C 搬运接口已覆盖规格列出的 82 种显式 enum/dtype 组合（GM 18、UB 48、L1 16），另有视图及边界对抗检查；不代表任意 shape、stride、offset 或数值均已穷举。

<a name="cube-006"></a>

## CUBE-006 · tla.copy_l0c_to_l1

[查看规格](../spec/cube.md#api-tla-copy-l0c-to-l1)

- 需按目的布局限定类型转换和激活组合。

结论：本轮范围已确定，按规格进入待转测。NZ2ND/NZ2NZ 支持 FP32/FP16/BF16；NZ2DN 仅保留 FP16/BF16，FP32 路径暂不具备目标实现。激活仅提供无激活/ReLU；不提供双目标或子块选择。

补充结论：FP32 NZ2NZ 在固定 `channel_split=False` 下保留 C0=16 布局，不能未经实际布局转换直接接收为 C0=8 的 FP32 Cube 输入；该限制已写入规格。两项直接串接失败不能单独证明搬运损坏，不能通过修改 shape 元数据宣称转换完成。前端和消费端应按各自布局合同检查，串接链路须有明确转换。

验证边界：FP16/BF16 已检查后续 Cube 消费；FP32 的内部 L1 回读只证明所检查的搬运数据，所用 FP32 L1→UB 辅助路径不在公开规格内，不能据此宣称公开接口的完整 FP32 串接已通过。依据为上述 TA 集成版本及 AscendNPU-IR `c521aba2`。

<a name="cube-007"></a>

## CUBE-007 · tla.store_l0c_to_gm

[查看规格](../spec/cube.md#api-tla-store-l0c-to-gm)

- 需明确目的布局、类型转换与激活组合。

结论：本轮范围已确定，按规格进入待转测。NZ2ND/NZ2NZ/NZ2DN 支持 FP32/FP16/BF16 输出及无激活/ReLU；不提供双目标或子块选择。不提供额外 scale、激活模式或手动同步控制字段。

## 搬运长度边界与文档整理

- 查阅版本：AscendNPU-IR `0e7d8bb2` 及当前工作树的 L1→L0 增量实现。本轮仅核对源码和文档，未重新运行边界设备测试。
- UB→L1：`Template/include/RegBase/DMAUtils.h::copy_ubuf_to_cbuf_intrin_core` 的 burst_len、src_stride、dst_stride 为 uint16；步长以元素传入，随后按 32B 换算。规格对带行间距路径约束每行长度、步长不超过 65535 个元素，且按 32B 对齐。不能将其他 GM DMA 的 21-bit 字节长度上限套用到此接口。
- UB→L1：当前非连续二维路径只处理一个 4095 行主段及尾段；没有完整遍历多个主段。规格暂将该路径的行数限制为不超过 8189，避免 8190 及以上时遗漏中间行。容量约束仍须独立满足；后端补齐循环后可重新评审。
- GM→L1：`RegBase/Cube/compat/DMA/Cbuf/nd2nz.cpp` 将矩阵维度转换为 uint16；规格按 65535 上界与类型分块对齐的交集约束，不能把截断后数值当作合法输入。
- L0C→UB/L1/GM：`RegBase/Cube/compat/Fixpipe/Fixpipe.cpp` 的常规四维路径使用 uint16 矩阵维度；NZ2NZ 的 dst_stride 还计算目的行数乘块宽。规格同时限制维度及该乘积，不能仅检查单个维度。
- L1→UB：`RegBase/Cube/l12ub.cpp` 按完整块逐段搬运，不据单次原生指令长度宣称整个接口具有相同总长度上限。L1→L0 的 `RegBase/Cube/l12l0.cpp::checkedAddress` 检查地址、步长及容量；规格保留 512B 对齐与目标 64KiB 上限。
- 上述数值是当前实现不得越过的边界，不是整张矩阵能装入片上存储的保证；所有接口仍须满足实际分配和可用容量。更大形状或非默认模式不能据此自动纳入支持范围。
- 规格页按语言接口描述，源码路径和实现依据仅记于跟踪页；保留原有仓库测试链接。
