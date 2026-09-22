# TPMS 晶格系统新窗口交接文档

更新日期：2026-08-09  
工作区：`D:\shoe(1)\shoe`  
当前阶段：研究/中间工作台，尚未完全并入主应用

## 新窗口接手方式

先阅读本文件、根目录 [`CONTEXT.md`](../../CONTEXT.md) 和
[`ADR-0027`](../../docs/adr/0027-use-topology-constrained-provider-agnostic-transitions.md)，
然后优先处理：

1. [`01-transition-topology-correction-artifacts.md`](issues/01-transition-topology-correction-artifacts.md)
2. [`02-derived-stl-topology-and-chunk-stitching.md`](issues/02-derived-stl-topology-and-chunk-stitching.md)
3. [大网格内存策略](../tpms-memory-risk/issues/02-large-grid-memory-strategy.md)

建议在新窗口中直接说：

> 请先阅读 `.scratch/tpms-system-handoff/handoff.md`，从优先级 P0 的拓扑修正像素块诊断开始；必须先复现、做关闭/开启对照和采样收敛测试，再修改算法。

## 一、必须先知道的真实状态

| 项目 | 当前真实状态 |
| --- | --- |
| 当前主要 TPMS 工作台 | [`tests/intermediate_tests/tpms_filling_app.py`](../../tests/intermediate_tests/tpms_filling_app.py)，启动命令见下文。它不是根目录主应用的正式模块。 |
| 根目录主应用 | [`ui_app.py`](../../ui_app.py) 仍是原鞋底晶格工作台，只复用了增强后的 PyVista 渲染器；新 TPMS/Cell Map/自定义晶胞/通用过渡 UI 没有完整并入。 |
| 权威几何 | `ImplicitBody` 的连续求值器。它不是体素数组，也不是 STL。 |
| 隐式体显示 | 对权威求值器生成有限分辨率 `SampledImplicitField`，再由 VTK OpenGL GPU 零等值面光线投射显示。它不是直接对任意求值点进行无采样渲染。 |
| STL | 从权威求值器按独立的 STL 采样间距重新采样并执行 Marching Cubes 得到的派生网格；不复用显示场。 |
| G/D | 标准解析 TPMS 周期场、物理壁厚近似、独立 Cell Map/Frame 已实现。 |
| 自定义晶胞 | 水密 STL 可作为周期晶胞，通过 Cell Map UVW 重复求值；可检查并按用户勾选方向桥接 U/V/W 接缝。 |
| 通用过渡 | A/B 均可选 G、D、自定义晶胞，也允许同类型不同参数。有限宽度 Ramp、局部配准、拓扑修正和独立验证已实现。 |
| 过渡质量 | 仍未达成稳定生产质量。拓扑修正开启后出现规则像素小块；现有过渡 STL 验收仍不水密且多连通。 |
| GPU 数值计算 | 当前机器已检测到 `NVIDIA GeForce RTX 4060 Laptop GPU`；TPMS 场和几何后端均为 Numba CUDA。失败会回退 CPU。 |
| GPU 渲染 | 支持时使用 VTK OpenGL GPU 隐式等值面光线投射；STL 也由 GPU 光栅化，但大网格可能先生成仅用于显示的缓存代理。 |
| 内存问题 | 已降低临时数组峰值，但没有从根本上消除。全局场修复、稠密场、最终网格与拼接数组仍可能耗尽电脑系统 RAM。 |
| 导出策略 | 不合格网格允许导出，但必须给出水密性、绕向和连通量警告。 |

## 二、当前运行入口和环境

启动中间工作台：

```powershell
& '.venv\Scripts\python.exe' 'tests\intermediate_tests\tpms_filling_app.py'
```

生成旧版 G、D、G/D 示例：

```powershell
& '.venv\Scripts\python.exe' 'tests\intermediate_tests\tpms_filling_app.py' --generate-example
```

当前解释器和关键库：

- Python 3.12.7
- NumPy 2.4.4
- SciPy 1.17.1
- trimesh 4.11.5
- scikit-image 0.26.0
- PyVista 0.48.4
- 可选 GPU 依赖记录在 [`requirements-gpu.txt`](../../requirements-gpu.txt)

当前后端实测：

```text
geometry = Numba CUDA geometry / RTX 4060 Laptop GPU
tpms     = Numba CUDA / RTX 4060 Laptop GPU
```

## 三、核心数据流

```text
设计域 STL
  -> 清理/检查
  -> 设计域 ImplicitBody（有可靠内外符号时可做布尔运算）

Cell Map + Unit-cell provider
  -> G / D 解析场，或自定义 STL 周期求值器
  -> 晶格 ImplicitBody
  -> max(设计域场, 晶格场) 隐式交集
  -> 权威生成结果 ImplicitBody

权威 ImplicitBody
  -> 显示采样 -> SampledImplicitField -> VTK GPU 零等值面显示
  -> STL 专用采样 -> Marching Cubes -> 修复/碎片清理/切片优化 -> STL
```

过渡路径：

```text
Operand A + Operand B
  -> 分界面有符号距离
  -> 有限宽度 Ramp 权重
  -> 可选局部相位配准
  -> 可选拓扑约束修正
  -> 与设计域隐式交集
  -> 独立过渡质量验证
```

## 四、已经完成并有代码/测试支撑的内容

### 1. G/D 隐式晶格与设计域交集

- G、D 使用解析周期场，不复制单个晶胞网格。
- 壁厚通过场梯度转换为毫米尺度的 signed-distance-like 壳层，不只是任意等值面。
- G、D 各自拥有独立的 U/V/W 晶胞尺寸、壁厚、level、Cell Map 和 Frame。
- 与设计域通过隐式交集 `max(domain_sdf, lattice_field)` 限制。
- 解析场和后端在 [`core/implicit/tpms_compute.py`](../../core/implicit/tpms_compute.py)，工作台装配在 [`tpms_filling_app.py`](../../tests/intermediate_tests/tpms_filling_app.py)。

### 2. Cell Map 与 UVW Frame

- `fit_bounds`：仅在默认世界坐标 Frame 下，使 Cell Map 外边界精确等于设计域 AABB；用户尺寸是目标值，实际尺寸由整数晶胞数反算。
- `complete_cells`：保留用户输入的实际 U/V/W 尺寸，并用完整晶胞覆盖设计域，边界可超出 AABB。
- 旋转或手动 Frame 会进入完整晶胞覆盖，因为旋转规则网格不可能同时精确贴合世界轴 AABB 又保持晶胞尺寸和相位。
- Frame 由原点、U、V 定义，W 由右手叉积得到；“原点随 U/V 自动定位”会把原点放到当前 Frame 下设计域的 UVW 最小角。
- Cell Map 可独立预览，代码在 [`core/implicit/cell_map.py`](../../core/implicit/cell_map.py)。

### 3. 自定义 STL 晶胞

- 入口为 `prepare_stl_unit_cell()` 和 `make_periodic_lattice()`。
- 输入晶胞必须在清理后仍水密、绕向一致；非水密自定义晶胞不能提供可靠周期有符号场。
- 源模型 X/Y/Z 映射到 Cell Map U/V/W，不按晶胞数复制三角面片。
- 可选“目标特征厚度”通过世界空间 SDF 偏移实现；它不是精确的局部最小壁厚保证。
- U/V/W 接缝检查考虑当前 Cell Map 缩放、目标特征厚度和桥接深度。
- 用户明确勾选 U/V/W 后才会添加对应方向的局部桥接；警告不会阻止生成。
- 小晶胞显示中的周期性符号散点已通过 CUDA 多方向符号判定和显示专用小岛清理缓解；显示清理不修改权威几何或 STL。
- 实现在 [`core/implicit/custom_unit_cell.py`](../../core/implicit/custom_unit_cell.py)。

真实自定义晶胞验收产物：

- [`results/custom_unit_cell_acceptance/acceptance_report.json`](../../results/custom_unit_cell_acceptance/acceptance_report.json)
- 12 x 12 x 8 mm、0.8 mm 显示间距、12 块 STL 重建；导出水密且绕向一致，但有 8 个闭合连通分量，不是单连通。
- 主分量占 99.773% 面片，其余是设计域边界裁切形成的小闭合体，当前默认保留。

### 4. 通用晶胞过渡

- A/B 支持 G、D、Custom，允许 G-G、D-D 或同一自定义类型的不同参数。
- 每侧晶胞尺寸、特征厚度、Cell Map 和 Frame 可独立设置。
- 默认权重 `automatic` 解析为 `smootherstep`；高级模式支持 linear、smoothstep、smootherstep、sigmoid、cosine。
- 带外严格恢复各自原始场；局部配准和拓扑修正都限制在过渡带内。
- 过渡质量验证独立于显示分辨率和 STL 导出设置，报告跨带连接、最小特征、孤立块和分量数。
- 实现在 [`core/implicit/transition.py`](../../core/implicit/transition.py)。

### 5. 隐式体与 STL 渲染

- 隐式体：`vtkOpenGLGPUVolumeRayCastMapper` 的 IsoSurface 模式。
- STL：VTK/PyVista polygon actor，支持 PBR 插值。
- 显示质量：低、中、高、极佳，调整采样距离、插值、抗锯齿、SSAO/阴影等。
- 材质支持颜色、metallic、roughness；当前隐式体是 PBR 风格近似，网格是 VTK 原生 PBR。
- 白色 CAD 背景、四灯工作室光照、模型接地网格、深度近距离缩放已实现。
- 隐式和网格显示层均有驻留缓存；隐藏再显示可复用 actor/mapper/几何，受数量与内存预算限制。
- 剖切只更新 mapper clipping plane，不为每次拖动重新生成 STL 或隐式采样场。
- 渲染实现在 [`ui/viewers/pyvista_viewer.py`](../../ui/viewers/pyvista_viewer.py)。

### 6. 交互剖切 Gizmo

- X/Y/Z 轴拖动平移剖切面，轴间弧线拖动旋转。
- 左键仅选择并持续拖动命中的轴/弧；右键旋转视角；中键平移；滚轮缩放。
- hover 严格命中高亮，释放左键立即解锁。
- 独立模块在 [`tests/intermediate_tests/interactive_section_viewer.py`](../../tests/intermediate_tests/interactive_section_viewer.py)，已集成到 TPMS 工作台。

### 7. STL 重建、检查、修复、简化与导出

- STL 重建与显示缓存解耦。
- 支持三种明确计算模式：单次、分批场 + 单次 MC、分块采样 + 分块 MC + 拼接。
- 支持自适应砖块提取开关，但 GUI 中仅在“分块采样 + 分块 MC”且修复容差为 0 时真正启用。
- 修复容差大于 0 时，必须构造完整规则场做形态闭合，因此会安全回退到全局稠密场流程。
- 数值碎片清理只移除低于容差体积和相对体积阈值的闭合小分量。
- 面片简化使用 VTK C++（回退 Open3D），检查拓扑与几何偏差；不合格候选会尝试修复，仍不合格则回滚。
- “切片优化”使用拓扑保持、容差约束的 VTK C++ 处理，并有质量门控和回滚。
- 导出后重新加载 STL 并再次检查；不合格允许导出但警告。

### 8. 自动 GPU/CPU 后端

- TPMS G/D 场：优先 Numba CUDA，失败回退 NumPy CPU。
- 设计域/自定义 STL SDF：优先 CUDA BVH，失败回退 C++ BVH，再回退 PyVista/CPU 路径。
- 隐式交集和 Marching Cubes：优先 CUDA，失败回退 CPU。
- CUDA Marching Cubes 使用共享网格边标识，避免每个三角形传回三个重复顶点。
- 当前通用过渡的两个 operand 可以在 GPU 求值，但 Ramp、局部拓扑修正和验证仍在 CPU 数组上执行。
- 网格修复、连通性检查、数值碎片清理、VTK 简化主要使用系统 RAM/CPU/C++，不应写成“全部 GPU”。

## 五、关键参数及当前准确语义

### G/D 参数

| 参数 | 当前语义 |
| --- | --- |
| U/V/W 晶胞尺寸 | Cell Map 三方向目标周期，范围 0.5–100 mm。默认 Frame 下 `fit_bounds` 可能反算为略有不同的实际周期。 |
| 壁厚 | 物理毫米厚度近似，范围 0.05–20 mm，且必须小于最小晶胞尺寸的 45%。 |
| level / 中心面偏移 | 改变 TPMS 场的等值中心，范围 -2–2；它会改变两侧体积分数/形态，不等同于壁厚。 |
| Cell Map 边界 | `贴合设计域` 是贴合设计域 AABB，不是曲面共形网格；`完整晶胞扩展` 会超出后再由设计域裁切。 |
| Frame 原点、U、V | 控制晶胞方向与相位；W 自动保持右手正交。 |

### 隐式体显示参数

| 参数 | 当前语义 |
| --- | --- |
| 目标体素数，默认 4,000,000 | 只参与自动推荐显示采样间距，不是最大体素限制。 |
| 显示体素大小 | `SampledImplicitField` 的相邻采样点间距；只改变预览精度和显示内存，不改变权威 `ImplicitBody` 或 STL 重建。 |
| 自动推荐显示体素 | 综合目标体素数、每晶胞建议采样数和每壁厚建议采样数。 |
| 显示显存预算，默认 512 MB | 约束驻留显示场；超预算时增大显示体素。它不是系统总显存的预留保证。 |
| 隐式体分批采样 | 沿 X 分批求值，降低临时数组峰值；最终完整显示标量场仍要驻留。默认关闭，默认 4 批。 |
| 单晶胞采样估算 | 约为 `(U/h) x (V/h) x (W/h)`，其中 `h` 是实际显示体素大小。 |

### STL 重建参数

| 参数 | 当前语义 |
| --- | --- |
| STL 输入间距，默认 0.25 mm | 当前代码变量仍叫 `tolerance_mm`，但在网格提取中本质是采样间距输入。 |
| 自动推荐间距 | 每轴实际间距为 `min(用户输入, 特征厚度/3, 该轴晶胞周期/16)`。因此实际值可能小于输入。 |
| 精确使用输入间距 | 三轴直接使用用户值，不经过上述 `min`。 |
| 网格体素估算 | 在分配前显示实际 spacing、grid shape 和总采样点数。 |
| 修复容差，默认 0.20 mm | 仅闭合可由规则场表示、且不大于该尺度的局部间隙；大于 0 会要求全局规则场。 |
| 清理数值碎片 | 默认开启；只处理满足严格体积阈值的闭合小分量，不保证最终分量数为 1。 |
| 面片简化 | 0–90%，0 表示不简化；受拓扑和最大偏差门控。 |
| 自适应表面提取 | 默认勾选；GUI 中还必须选择“分块采样 + 分块 MC”且修复容差为 0 才进入 adaptive-bricks。 |

三种计算模式：

1. `single_pass`：完整场一次求值，单次 Marching Cubes；批数必须为 1。
2. `batched_field`：按批求值但组合成完整标量场，最后仍是一次全局 Marching Cubes。
3. `chunked_marching_cubes`：分块求值、每块 Marching Cubes、共享边标识拼接；可配合自适应砖块。

所有模式内部都可能出现“微分片”，它只是把一次 evaluator 调用限制在最多约 250,000 点，避免临时点数组过大，不代表用户开启了分批模式。

### 过渡参数

设点到分界面的有符号距离为 `d`，中心偏移为 `c`，用户宽度为 `W`：

```text
t = clip((d - c + W/2) / W, 0, 1)
```

- 过渡带严格是 `[c-W/2, c+W/2]`。
- `W=0.5 mm` 表示总宽度 0.5 mm，即两侧各 0.25 mm。
- 自动权重为 smootherstep；Sigmoid 只在明确选择后使用。
- Sigmoid 锐度范围 0.1–10，只改变带内 `t -> weight` 曲线，不改变物理宽度。
- 过渡最小特征厚度为 0 时，自动取两侧特征厚度较小值。
- 建议宽度不会覆盖用户输入：

```text
max(
  3 * min(两侧特征厚度),
  min(两侧最小周期) * (1 + 0.35 * 尺寸失配 + 0.5 * Frame 角度失配比例)
)
```

- 当前拓扑修正中心强度为 `16*t^2*(1-t)^2`，用平滑并集并向内偏移 `0.125*最小特征厚度`；该逻辑正是当前 P0 缺陷调查对象。

### 自定义晶胞参数

| 参数 | 当前语义 |
| --- | --- |
| 源 STL | 定义一个完整周期的几何；清理后必须水密且绕向一致。 |
| U/V/W 尺寸 | 把源模型包围盒映射到一个 Cell Map 周期的世界尺寸。 |
| 目标特征厚度 | 0 表示保留缩放后源模型厚度；非 0 通过 SDF 偏移调整特征厚度。 |
| U/V/W 桥接 | 只对用户勾选方向在周期接缝附近加材。 |
| 桥接深度 | 0 表示自动；必须小于所选方向晶胞尺寸的一半。 |

## 六、当前未解决问题

### P0：拓扑约束修正产生规则像素小块

状态：未解决，尚无可复现参数包和关闭/开启差分产物。  
Issue：[`01-transition-topology-correction-artifacts.md`](issues/01-transition-topology-correction-artifacts.md)

已知代码行为：拓扑修正不是离散形态操作，而是连续数组公式；但在有限分辨率采样后，修正后的零等值面仍可能呈现规则块状加材。不能在没有对照实验前简单归因于“体素太粗”。

### P0：过渡隐式体验证与导出 STL 拓扑不一致

状态：稳定复现，未解决。  
Issue：[`02-derived-stl-topology-and-chunk-stitching.md`](issues/02-derived-stl-topology-and-chunk-stitching.md)

现有 [`transition_acceptance`](../../tests/results/transition_acceptance/) 证据：

- 36 mm Ramp：权威过渡验证通过且分量数 1；0.55 mm adaptive-bricks STL 不水密，14 个分量。
- 0.5 mm Ramp：跨带连接通过但最小特征失败；0.55 mm adaptive-bricks STL 不水密，5 个分量。

这说明当前主要缺陷至少包含分块/自适应表面提取、块边界共享或设计域边界封闭问题，不能只修改过渡权重函数。

### P0：历史 TPMS 水密性和连通性问题未普遍解决

旧记录：[`TPMS-001`](../../docs/issues/TPMS-001-connectivity-and-watertightness.md)。

- 当前 G/D 的一个特定 `1.stl` 参数集可通过场修复得到水密、单分量结果。
- 这不构成任意 Cell Map、壁厚、Frame、设计域和过渡参数的普遍保证。
- 自定义晶胞验收虽然水密，但有 8 个闭合分量；“水密”与“连通量为 1”必须分开判断。

### P1：大网格系统 RAM 风险

状态：`needs-triage`。  
Issue：[`02-large-grid-memory-strategy.md`](../tpms-memory-risk/issues/02-large-grid-memory-strategy.md)

- 分批求值只降低临时数组；`batched_field` 仍保留完整标量场。
- `chunked_marching_cubes` 避免完整场，但最终 chunk 网格、拼接键、全局顶点/面数组仍在 CPU RAM。
- 修复容差大于 0 时必须构造完整规则场。
- GPU 不会消除系统内存问题；设备数组、主机传输和最终 trimesh 都会占 RAM。
- 当前没有隐藏最大体素上限，也没有完整的 out-of-core/memory-map 流水线。

### P1：测试工作台尚未产品化

状态：未决定集成方式。  
Issue：[`03-promote-intermediate-workbench.md`](issues/03-promote-intermediate-workbench.md)

- 主要 UI 文件约 6,900 行，算法、工作线程、UI 和导出流程仍集中在测试目录单文件中。
- `core/implicit` 已有较好的深模块边界，但 UI 服务层仍需拆分。
- 在整合前，不应把 `tests/intermediate_tests/tpms_filling_app.py` 当成稳定公共 API。

### P1：过渡拓扑约束不是严格数学保证

- 当前 operator 是平滑混合、局部相位平移和平滑并集偏移，不是拓扑优化或骨架级约束求解器。
- `validate_transition()` 是有限采样检查，默认最多 2,000,000 点；它可能受检查间距影响。
- 极窄过渡带、Frame/周期严重失配、自定义晶胞接缝不兼容时仍可失败。
- 验证失败只轻量提示，不阻止查看或导出，这是明确产品策略。

### P2：直接隐式切片尚未实现

- 当前切片仍面向 STL/三角网格。
- “对每一层二维隐式场直接提取闭合轮廓，绕过 STL”仍只是后续方向。
- 该方向同时适用于稠密和自适应三维表示，因为切片应直接调用权威 evaluator，而不是依赖已有三维体素场。

### P2：nTop 等级的隐式渲染尚未完全达到

- 当前交互显示是采样规则场 + VTK GPU 等值面光线投射，不是对压缩隐式表达式/窄带结构直接求交的专用商业内核。
- PBR 光照对隐式体是校准近似，不是 polygon actor 的完整 PBR BRDF。
- “精确隐式渲染”目前主要依赖更细显示采样；还没有独立的 evaluator ray-marching/区间求交器。

### P2：测试套件耗时

- 163 项常规中间测试约 51.86 秒。
- 3 项真实 TPMS 自适应测试约 288.30 秒。
- 一次性运行完整 166 项时，180 秒外部超时在约 83% 处终止且当时无失败；拆分后两组均全部通过。
- 后续应给真实验收测试增加明确 slow marker 和缓存 fixture，避免开发者误认为卡死。

## 七、2026-08-09 回归与验收证据

本次交接实际执行：

```text
163 passed, 30 warnings in 51.86s
3 passed, 12 warnings in 288.30s
```

即拆分执行的 166 项测试全部通过。通过表示对应回归断言满足，不表示上述端到端 STL 缺陷已解决。

重点产物：

- G/D Cell Map、水密单分量特定验收：[`acceptance_report.json`](../../tests/results/cell_map_topology_acceptance/acceptance_report.json)
- 过渡失败证据：[`transition_acceptance`](../../tests/results/transition_acceptance/)
- STL 性能基准：[`benchmark_optimized_report.json`](../../tests/results/performance_benchmark/benchmark_optimized_report.json)
- 自定义晶胞验收：[`custom_unit_cell_acceptance`](../../results/custom_unit_cell_acceptance/)
- 自定义显示散点清理：[`custom_unit_cell_speckle_fix`](../../results/custom_unit_cell_speckle_fix/)

G/D 特定验收结果：

- G：1,637,824 面，水密，绕向一致，1 个连通分量。
- D：1,372,840 面，水密，绕向一致，1 个连通分量。
- 两者都使用 0.45 mm 输入、0.45 mm 修复容差；实际采样间距受壁厚和周期约束。
- G 切片优化被接受；D 因质量/偏差门控回滚，保留原网格。

性能基准中的一个 G 示例：

- 旧记录总流水线约 76.66 秒、峰值 RSS 4.18 GiB、2,047,280 面。
- 当前优化记录约 40.45 秒、峰值 RSS 1.83 GiB、1,637,824 面。
- 这些数字只对应报告中的固定参数和 RTX 4060 环境，不能外推为所有模型。

## 八、代码结构和职责

### 权威隐式与采样

- [`core/implicit/field.py`](../../core/implicit/field.py)
  - `ImplicitBody`：权威 evaluator、bounds、可选保守 exclusion。
  - `SampledImplicitField`：显示用规则标量场。
  - `remove_small_negative_islands()`：仅显示缓存的小岛清理。

### Cell Map

- [`core/implicit/cell_map.py`](../../core/implicit/cell_map.py)
  - `CellMapFrame`：UVW 右手坐标系和世界/局部变换。
  - `CellMap`：晶胞数量、实际周期、索引范围、线框预览。

### 晶胞 provider

- [`core/implicit/tpms_compute.py`](../../core/implicit/tpms_compute.py)：G/D CUDA/CPU 场求值。
- [`core/implicit/tpms_exclusion.py`](../../core/implicit/tpms_exclusion.py)：G/D 保守表面排除场。
- [`core/implicit/custom_unit_cell.py`](../../core/implicit/custom_unit_cell.py)：自定义晶胞准备、周期求值、接缝检查与桥接。

### 过渡

- [`core/implicit/transition.py`](../../core/implicit/transition.py)
  - `TransitionSpec`、`TransitionOperand`
  - `transition_weights()`
  - `build_transition_body()`
  - `validate_transition()`

### 数值后端与表面提取

- [`core/implicit/geometry_compute.py`](../../core/implicit/geometry_compute.py)：CUDA SDF、交集、共享边 Marching Cubes 和 CPU 回退。
- [`core/implicit/surface_extraction.py`](../../core/implicit/surface_extraction.py)：稠密/自适应砖块规划、提取与拼接。
- [`core/cpp/cpp_mesh_sdf.cpp`](../../core/cpp/cpp_mesh_sdf.cpp)：C++ mesh SDF 后端。

### UI 与渲染

- [`tests/intermediate_tests/tpms_filling_app.py`](../../tests/intermediate_tests/tpms_filling_app.py)：当前完整工作台、参数、worker、重建、修复、导出。
- [`ui/viewers/pyvista_viewer.py`](../../ui/viewers/pyvista_viewer.py)：Mesh/Implicit layer、VTK actor、缓存、PBR 风格、剖切和相机交互。
- [`tests/intermediate_tests/interactive_section_viewer.py`](../../tests/intermediate_tests/interactive_section_viewer.py)：三轴/旋转弧剖切 Gizmo。

主要渲染对象：

- `_MeshLayerRecord`：派生 STL 或设计域网格的 `PolyData + actor + cache key`。
- `_ImplicitLayerRecord`：`SampledImplicitField + vtkImageData + vtkVolume actor`。
- `MeshData`：网格显示输入和材质，不是权威导出网格容器。
- `SampledImplicitField`：隐式显示输入，不是权威几何。
- Cell Map wireframe actor：只做布局预览，不参与晶格布尔计算。
- 剖切 Gizmo actors：X/Y/Z 箭头、旋转弧和剖切面，仅做交互控制。

## 九、工作树与版本风险

当前工作树不是干净状态，并且大量核心文件仍是未跟踪文件：

```text
修改：core/cpp/__init__.py、core/implicit/__init__.py、
      ui/viewers/base_viewer.py、ui/viewers/pyvista_viewer.py、ui_app.py
未跟踪：CONTEXT.md、docs/、core/implicit/*.py、
        core/cpp/cpp_mesh_sdf.cpp、tests/intermediate_tests/、.scratch/ 等
```

这意味着：

- 不能假设存在一个可回退的已提交 TPMS 基线。
- 不要使用 `git reset --hard`、`git checkout --` 或覆盖用户修改。
- 下一阶段开始前应先审查并形成一个明确 checkpoint commit；是否提交由用户决定。
- 运行产物分散在根目录 `results/` 和 `tests/results/`，后续应统一，但不要在未核对引用前移动或删除。

## 十、建议的下一步顺序

### 第一步：诊断拓扑修正像素块

固定同一设计域、A/B、Frame、过渡面、宽度、显示 spacing 和 STL spacing：

1. 关闭拓扑修正，保存权威场切片、显示场和 STL。
2. 开启拓扑修正，保存同样产物和逐点差分。
3. 分别用 0.5x、1x、2x 采样间距做收敛测试。
4. 区分权威场缺陷、显示采样缺陷和 Marching Cubes 缺陷。
5. 只有确认来源后才替换修正算子；禁止只靠提高分辨率掩盖。

### 第二步：修复 adaptive/chunk seam 拓扑

1. 对同一过渡 evaluator、同一 spacing 比较 dense、固定 chunk 和 adaptive-bricks。
2. 统计每个开放边是否位于 brick 边界或设计域 halo。
3. 验证 CUDA `edge_ids` 在相邻 brick 上完全一致。
4. 修复后要求水密性、绕向、连通分量、bounds 和体积与 dense 基线一致。
5. 在 0.5 mm 和 36 mm 两个过渡案例上回归。

### 第三步：再处理大内存与性能

在拓扑正确之后实现真正 out-of-core：

- 磁盘/内存映射场；
- 流式 chunk 顶点与面输出；
- 外部排序/哈希拼接；
- 峰值 host/device memory 预估与 OOM 可恢复；
- 中断后清理临时文件。

### 第四步：产品化

- 把重建、导出、worker、参数模型从 6,900 行工作台拆成服务模块。
- 决定并入 `ui_app.py` 还是建立新的正式 TPMS 应用入口。
- 为关键真实验收添加 slow marker、固定 fixture 和 JSON 结果比较。
- 再考虑直接隐式切片和更接近 nTop 的专用隐式渲染内核。

## 十一、不要再混淆的概念

- 权威隐式体 != 显示体素场 != STL。
- 水密 != 连通分量为 1。
- 目标体素数 != 最大体素限制。
- 显示体素大小 != STL 输入间距。
- STL 输入间距 != 修复容差。
- 分批场求值 != 分块 Marching Cubes。
- 微分片 != 用户选择的分批模式。
- CUDA 数值后端 != OpenGL 渲染后端。
- Cell Map 贴合 AABB != 曲面共形 Cell Map。
- 平滑权重 != 拓扑连通保证。

