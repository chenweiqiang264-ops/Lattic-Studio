# nTop `Mesh from Implicit Body` 的 `Tolerance` 语义核对

调研日期：2026-08-08  
来源范围：仅使用本机安装的 nTop 第一方文档；未使用论坛、博客转载或其他二手资料。  
文档快照说明：本机文档树包含 nTop 5.45 发布说明；目标 block 页面列出的最新接口版本为 2.5.0。

## 结论

1. **`Tolerance` 的官方定义是几何偏差上限。** `Mesh from Implicit Body` 2.5.0 的输入表将其定义为“网格相对隐式几何的最大允许偏差”；值越小，网格越精确，计算越慢。[S1]
2. **它不等于该 block 内部使用的 voxel size。** 同一官方页面的 Tips 明确给出：`Tolerance` 会转换为 voxel size，且 `voxel size = Tolerance / 2`。[S1] 因此，如果“规则采样间距”指体素边长或网格步长 `h`，则官方给出的关系是 `h = T / 2`，不是 `h = T`。
3. **不能把上述关系扩写成已证实的“规则点采样实现”。** nTop 把 `Voxel Grid` 定义为“在稀疏三维网格上离散化的距离场”，并把相关 block 的 `Voxel size` 仅描述为“Size of a voxel”。这些第一方页面没有说明网格是否全局均匀、采样点位于体素中心还是角点、插值方式、窄带宽度，或该 mesher 的内部遍历方式。[S2][S3] 可确认的是 voxel 尺寸关系，不能确认更具体的规则采样算法。
4. **它有明确的几何误差语义。** “maximum allowable deviation”是该参数的正式定义；启用 `Simplify` 时，官方还称减面会在“不违反给定 tolerance criterion”的前提下减少三角形。[S1] 但文档没有定义偏差采用 Hausdorff 距离、单向/双向距离、逐顶点距离还是逐面最大距离，因此不能给它补充未经披露的误差范数。
5. **`Tolerance` 本身没有被定义为自适应网格参数。** 当前 block 的自适应式减面语义来自独立的 `Simplify` 选项：官方将它类比旧版 `adaptivity`，后者会在低曲率区域减少单元；新版 `Simplify` 则以满足输入 tolerance 为约束减少三角形。[S1] 所以可以说“简化后的三角形密度可能随几何复杂度变化”，不能说“Tolerance 就是局部自适应采样间距”。
6. **显式的 Adaptive Tessellation 属于另一个 block。** `Mesh from Implicit by AT` 明确声明使用 Adaptive Tessellation，并通过在更平坦或较不复杂区域生成更少三角形来降低开销；其 `Tolerance` 仍定义为相对隐式几何的最大允许偏差。[S4] 这支持“误差容限”和“自适应三角化”是两个可同时存在但概念不同的维度。
7. **不要把稳定版的 `voxel size = T/2` 自动套到 Beta block。** `Mesh from Implicit Body 2` 的本地页面只重复了“最大允许偏差”的定义，没有出现 voxel size、采样间距或 adaptive tessellation 的说明。[S5]

## 官方原文摘录

### `Mesh from Implicit Body`

> “Tolerance: Translates to the voxel size. The voxel size is given by half the input tolerance.”

> “The maximum allowable deviation of the mesh from the implicit geometry. as this value gets smaller, the resulting mesh will be more precise, but the block will take longer to compute.”

> “The old version provides an ‘adaptivity’ input that can be used to obtain a mesh with fewer elements in areas of low curvature. In a similar spirit, the new version has a ‘simplify’ option that produces a mesh with the least amount of triangles while still conforming to the input tolerance.”

来源：[S1]，分别位于生成 HTML 第 28798、29170、28813 行。第 29170 行对应页面所列最新普通 overload 2.5.0；其扩展 overload 2.5.0 在第 29513 行给出相同的 `Tolerance` 定义。

### `Voxel Grid`

> “A distance field geometry discretized over a sparse three-dimensional grid.”

来源：[S2]，生成 HTML 第 28765 行。

### `Mesh from Implicit by AT`

> “Converts an Implicit Body to a Mesh using Adaptive Tessellation.”

> “It achieves this by creating fewer triangles in flatter or less complex regions, resulting in a lower overall triangle count.”

> “The maximum allowable deviation of the tessellation from the implicit geometry.”

来源：[S4]，生成 HTML 第 28779、28798 行。

## 判定边界

| 问题 | 判定 | 第一方证据边界 |
|---|---|---|
| `Tolerance` 是什么 | 网格相对隐式几何的最大允许偏差 | [S1] |
| 是否等于 voxel size / 规则步长 | 不等于；页面给出 `voxel size = Tolerance / 2` | [S1] |
| 是否证明均匀规则采样 | 否 | [S2][S3] 只说明稀疏三维 voxel grid 和 voxel 的尺寸，未披露采样布局 |
| 是否有几何误差语义 | 有 | [S1] 明确使用 maximum allowable deviation；未定义具体距离范数 |
| `Tolerance` 是否就是 adaptivity | 否 | [S1] 将 `Simplify`/旧版 `adaptivity` 作为独立控制 |
| nTop 是否提供真正自适应三角化 | 是，但在另一 block 中明确命名 | [S4] `Mesh from Implicit by AT` |
| Beta `Mesh from Implicit Body 2` 是否仍有 `T/2` 映射 | 当前证据不能确认 | [S5] 只保留最大偏差定义 |

## 第一方来源

### [S1] `Mesh from Implicit Body`

本机路径：`C:\ProgramData\nTopology\documentation\block-documentation\blocks\utilities\conversion\mesh-from-implicit-body.html`  
相关位置：`About this Block V2.00` 的 Tips；`Implicit Body, Scalar, Scalar, Bool, Bool, 2.5.0` 的 Inputs。  
SHA-256：`BE625EB936456C7D3B7204CFE22C40B18E30F604CB82280810FAF9852752B47F`

### [S2] `Voxel Grid` 类型文档

本机路径：`C:\ProgramData\nTopology\documentation\block-documentation\types\voxel-grid.html`  
相关位置：页面主定义。  
SHA-256：`7C1C11F3A32C3497DE84189DF62EED6ABD45B19A2A5919803B12654404AD27BD`

### [S3] `Voxel Grid from Implicit Body`

本机路径：`C:\ProgramData\nTopology\documentation\block-documentation\blocks\utilities\conversion\voxel-grid-from-implicit-body.html`  
相关位置：Inputs > `Voxel size`。  
SHA-256：`69EB89C0AC61F9723BE7585FE5183E27B2C6669AFC4F3B89D9B5FB6636D4E973`

### [S4] `Mesh from Implicit by AT`

本机路径：`C:\ProgramData\nTopology\documentation\block-documentation\blocks\utilities\conversion\mesh-from-implicit-by-at.html`  
相关位置：`Implicit Body, Scalar Field, Bool, 5.42.0` 的说明及 Inputs。  
SHA-256：`1BA94F0EBCC45D251117A62F2DEBD306137CD96D85BB732221DB96E3350872EA`

### [S5] `Mesh from Implicit Body 2`（Beta）

本机路径：`C:\ProgramData\nTopology\documentation\block-documentation\blocks\beta\utilities\mesh-from-implicit-body-2.html`  
相关位置：Inputs > `Tolerance`。  
SHA-256：`86F594E66851AA6BEEB9F23B7F870869DF16798A5B4B0168FB6C87C686E03BA4`

## 限制

- 本次未用二手材料补足 nTop 未公开的 mesher 内部实现。
- 本地页面没有为几何偏差给出数学范数，也没有描述 voxel 的采样点布局、插值或网格遍历方式。
- 因本地第一方文档已直接覆盖问题，本次没有引入可能与本机版本不一致的网页资料。
