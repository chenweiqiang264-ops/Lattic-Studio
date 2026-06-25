# 需求文档：多区域晶格系统

## 简介

本系统在现有鞋底晶格生成系统基础上，扩展支持多区域划分功能。用户可以将鞋底划分为多个区域，为每个区域独立配置晶格参数（方法、单胞尺寸、等值面等），系统将为每个区域生成晶格并合并为最终结果。这是现有晶格生成功能的直接扩展，允许在不同区域应用不同的晶格配置。

## 术语表

- **System**: 多区域晶格系统
- **Region**: 鞋底上用户定义的独立区域，可具有不同的晶格参数
- **Region_List**: 区域列表，显示所有已定义区域的界面组件
- **Lattice_Parameters**: 晶格参数，包括方法（gyroid/voronoi/tile）、单胞尺寸、等值面、分辨率等
- **Segmented_Mesh**: 分割网格，来自"模型分割"选项卡的分割结果
- **Whole_Sole**: 整体鞋底，使用完整的鞋底模型作为区域
- **Region_Source**: 区域来源，可以是分割结果或整体鞋底
- **Combined_Lattice**: 合并晶格，将所有区域的晶格合并后的最终结果
- **Region_Preview**: 区域预览，在 3D 视图中显示选中区域的边界
- **Lattice_Generation**: 晶格生成，为单个区域生成晶格的过程
- **Export_Result**: 导出结果，将合并后的晶格保存为 STL 文件

## 需求

### 需求 1：区域晶格设计选项卡

**用户故事：** 作为鞋底设计师，我希望在主界面中有一个专门的选项卡用于多区域晶格设计，以便管理和配置不同区域的晶格参数。

#### 验收标准

1. THE System SHALL 在主界面选项卡中添加"区域晶格设计"选项卡
2. THE "区域晶格设计"选项卡 SHALL 位于"晶格生成"和"模型分割"选项卡之后
3. THE System SHALL 在"区域晶格设计"选项卡中显示区域管理界面
4. THE System SHALL 在"区域晶格设计"选项卡中显示晶格参数配置界面
5. WHEN 用户切换到"区域晶格设计"选项卡时，THE System SHALL 保持其他选项卡的状态不变

### 需求 2：区域定义与管理

**用户故事：** 作为用户，我希望能够定义和管理多个区域，以便为不同区域配置不同的晶格参数。

#### 验收标准

1. THE System SHALL 提供"添加区域"按钮用于创建新区域
2. WHEN 用户点击"添加区域"时，THE System SHALL 显示区域来源选择对话框
3. THE System SHALL 支持"使用分割结果"作为 Region_Source
4. THE System SHALL 支持"使用整体鞋底"作为 Region_Source
5. WHEN 用户选择"使用分割结果"时，THE System SHALL 显示可用的 Segmented_Mesh 列表（正侧/负侧）
6. WHEN 用户选择"使用整体鞋底"时，THE System SHALL 使用完整鞋底模型作为区域
7. THE System SHALL 为每个新区域自动生成唯一的区域名称（如"区域 1"、"区域 2"）
8. THE System SHALL 允许用户编辑区域名称
9. THE System SHALL 提供"删除区域"按钮用于移除选中的区域
10. THE System SHALL 支持至少 10 个 Region 的同时管理

### 需求 3：区域列表显示

**用户故事：** 作为用户，我希望看到所有已定义区域的列表，以便选择和管理区域。

#### 验收标准

1. THE System SHALL 在界面中显示 Region_List
2. THE Region_List SHALL 显示每个区域的名称
3. THE Region_List SHALL 显示每个区域的来源类型（分割结果/整体鞋底）
4. THE Region_List SHALL 支持单选模式，一次只能选中一个区域
5. WHEN 用户选中一个区域时，THE System SHALL 在右侧显示该区域的晶格参数
6. WHEN 用户选中一个区域时，THE System SHALL 在 3D 视图中高亮显示该区域
7. WHEN Region_List 为空时，THE System SHALL 显示提示信息"暂无区域，请点击'添加区域'开始"

### 需求 4：区域晶格参数配置

**用户故事：** 作为用户，我希望为每个区域独立配置晶格参数，以便生成不同类型和密度的晶格。

#### 验收标准

1. THE System SHALL 为选中的区域显示晶格方法选择器（gyroid/voronoi_2_5d/tile_unit）
2. WHEN 晶格方法为 gyroid 时，THE System SHALL 显示 Gyroid 参数组（单胞尺寸、等值面、分辨率）
3. WHEN 晶格方法为 voronoi_2_5d 时，THE System SHALL 显示 Voronoi 参数组（胞元尺寸、杆径、Z 向层数）
4. WHEN 晶格方法为 tile_unit 时，THE System SHALL 显示 Tile 参数组（单元尺寸缩放、平铺间距系数、边界安全距离）
5. THE System SHALL 允许用户修改选中区域的晶格参数
6. WHEN 用户修改参数时，THE System SHALL 立即保存到该区域的配置中
7. WHEN 没有选中区域时，THE System SHALL 禁用参数配置界面并显示提示"请先选择一个区域"
8. THE System SHALL 为每个区域独立存储 Lattice_Parameters

### 需求 5：区域晶格生成

**用户故事：** 作为用户，我希望为每个区域生成晶格并合并为最终结果，以便预览和导出多区域晶格。

#### 验收标准

1. THE System SHALL 提供"生成所有区域晶格"按钮
2. WHEN 用户点击"生成所有区域晶格"时，THE System SHALL 验证至少存在一个区域
3. WHEN 区域列表为空时，THE System SHALL 显示错误信息"请先添加至少一个区域"
4. THE System SHALL 按顺序为每个区域生成晶格
5. WHEN 为区域生成晶格时，THE System SHALL 使用该区域的网格和 Lattice_Parameters
6. THE System SHALL 在状态日志中显示每个区域的生成进度
7. THE System SHALL 将所有区域的晶格合并为 Combined_Lattice
8. WHEN 生成完成时，THE System SHALL 在 3D 视图中显示 Combined_Lattice
9. WHEN 生成失败时，THE System SHALL 显示错误信息并指明失败的区域

### 需求 6：区域预览

**用户故事：** 作为用户，我希望在 3D 视图中预览选中的区域，以便确认区域范围是否正确。

#### 验收标准

1. WHEN 用户在 Region_List 中选中一个区域时，THE System SHALL 在 3D 视图中显示该区域的网格
2. THE System SHALL 使用不同的颜色区分不同区域（如青色、橙色、绿色等）
3. THE System SHALL 支持显示模式切换（仅区域/区域+鞋底/仅鞋底）
4. WHEN 用户取消选择区域时，THE System SHALL 恢复显示完整鞋底
5. THE System SHALL 在预览时保持 3D 视图的交互功能（旋转、平移、缩放）

### 需求 7：导出多区域晶格

**用户故事：** 作为用户，我希望导出合并后的多区域晶格，以便用于后续制造或分析。

#### 验收标准

1. THE System SHALL 提供"导出合并晶格"按钮
2. WHEN 用户点击"导出合并晶格"时，THE System SHALL 验证 Combined_Lattice 是否已生成
3. WHEN Combined_Lattice 未生成时，THE System SHALL 显示错误信息"请先生成区域晶格"
4. THE System SHALL 打开文件保存对话框让用户选择导出路径
5. THE System SHALL 将 Combined_Lattice 保存为 STL 文件
6. THE System SHALL 在状态日志中显示导出成功信息和文件路径
7. WHEN 导出失败时，THE System SHALL 显示错误信息

### 需求 8：区域配置持久化

**用户故事：** 作为用户，我希望系统能够保存和加载区域配置，以便复用设计方案。

#### 验收标准

1. THE System SHALL 提供"保存配置"按钮
2. WHEN 用户点击"保存配置"时，THE System SHALL 将所有区域定义和参数保存到 JSON 文件
3. THE System SHALL 提供"加载配置"按钮
4. WHEN 用户点击"加载配置"时，THE System SHALL 从 JSON 文件加载区域定义和参数
5. THE System SHALL 在配置文件中保存区域名称、来源类型、来源路径、晶格方法和所有参数
6. WHEN 配置文件格式错误时，THE System SHALL 显示明确的错误信息
7. THE System SHALL 在加载配置后更新 Region_List 和参数界面

### 需求 9：错误处理与用户反馈

**用户故事：** 作为用户，我希望系统在遇到错误时能够提供清晰的错误信息，以便我知道如何修正。

#### 验收标准

1. WHEN 区域网格文件缺失时，THE System SHALL 显示错误信息"区域网格文件不存在"
2. WHEN 区域网格为空时，THE System SHALL 显示警告信息"区域网格为空，无法生成晶格"
3. WHEN 晶格生成失败时，THE System SHALL 显示错误信息并指明失败原因
4. WHEN 用户尝试删除最后一个区域时，THE System SHALL 允许删除（不强制保留区域）
5. THE System SHALL 在状态日志中记录所有操作和错误信息
6. THE System SHALL 在错误对话框中提供"查看日志"按钮

### 需求 10：与现有功能的兼容性

**用户故事：** 作为开发者，我希望新功能能够与现有代码兼容，以便不影响现有的晶格生成和模型分割功能。

#### 验收标准

1. THE System SHALL 保持"晶格生成"选项卡的所有现有功能不变
2. THE System SHALL 保持"模型分割"选项卡的所有现有功能不变
3. THE System SHALL 复用现有的晶格生成函数（geometry.py 中的函数）
4. THE System SHALL 复用现有的网格加载和处理函数
5. THE System SHALL 复用现有的 3D 视图组件（GLMeshViewer）
6. THE System SHALL 复用现有的配置系统（config.py）
7. THE System SHALL 不修改现有的缓存机制
8. THE System SHALL 在新选项卡中使用与现有选项卡一致的 UI 风格
