Type: task
Status: resolved

# Field-first cleanup for floating lattice islands

生成晶格在局部采样不足、边界裁剪或隐式场重建时可能出现悬空小结构，造成 STL 多连通。需要将可确认的数值伪影与真实晶格分支区分开，并把安全清理放在隐式场层面。

## Answer

已实现：

- 新增 `remove_small_negative_field_components()`，使用 26 邻域标记隐式场负值分量。
- 仅对不接触规则场边界、相对主分量足够小且满足体积/物理尺寸阈值的分量进行清理。
- 单次计算和分批采样模式在 Marching Cubes 前执行场清理；分块 Marching Cubes 仍走网格后处理路径，避免为清理重新驻留完整场。
- `ExportMeshResult` 返回 `field_fragment_cleanup` 审计报告，UI 和 1.stl 验收脚本可显示清理数量。
- 大型或无法判定为数值伪影的独立分量不被删除。

验证：

- 场级孤立球回归通过。
- 边界分量保留回归通过。
- `resources/鞋底/1.stl` 极端 G 参数（10 mm 晶胞、0.9 mm 壁厚、1.0 mm 容差）从 2 个连通分量降为 1 个，水密保持为 True；场级清理移除 1 个、3 个采样点组成的孤立分量。
- 中间测试 167 passed，16 个 CUDA 专项因当前驱动访问冲突使用环境开关跳过；针对性拓扑测试 39 passed。

## Comments

- 2026-08-31：当前仍需后续处理“工程尺度的真实晶格分支断裂”。不能通过保留最大分量或放宽阈值解决，否则会误删有效晶格。
