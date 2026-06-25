# 代码重构详细指南

## 📋 重构原则

1. **渐进式重构**：每次只改动一小部分，保持程序可运行
2. **测试驱动**：每个阶段完成后立即测试
3. **保留备份**：所有原文件都有 .backup 备份
4. **文档先行**：先规划再执行

## 🎯 阶段2：重构 core 层

### 步骤 2.1：移动 mesh_slicer.py

```bash
# 1. 复制文件
Copy-Item "mesh_slicer.py" "domain/slicer/mesh_slicer.py"

# 2. 更新导入（如果有的话）
# mesh_slicer.py 中可能导入了其他模块，需要更新路径
```

**需要修改的导入**：
- 无需修改（mesh_slicer.py 相对独立）

### 步骤 2.2：移动 C++ 扩展

```bash
# 移动 C++ 扩展文件
Copy-Item "cpp_distance_field.cp312-win_amd64.pyd" "core/cpp/"
Copy-Item "cpp_mesh_slicer.cp312-win_amd64.pyd" "core/cpp/"
```

### 步骤 2.3：移动 geometry.py

**这是最复杂的一步，因为 geometry.py 包含多个功能**

当前 geometry.py 的结构：
- Gyroid 生成
- Voronoi 2.5D 生成
- Tile 单元生成
- 网格操作工具

**拆分策略**：
1. 保留 geometry.py 在 core/ 目录（暂时不拆分）
2. 未来可以拆分为：
   - core/implicit/tpms.py（Gyroid）
   - core/implicit/tile.py（Tile）
   - core/geometry/mesh_ops.py（网格操作）

```bash
# 先简单移动
Copy-Item "geometry.py" "core/geometry.py"
```

### 步骤 2.4：移动 voronoi_implicit_new.py

```bash
Copy-Item "voronoi_implicit_new.py" "core/implicit/voronoi.py"
```

## 🎯 阶段3：重构 domain 层

### 步骤 3.1：移动 region_manager.py

```bash
Copy-Item "region_manager.py" "domain/region/region_manager.py"
```

**需要更新的导入**：
```python
# 原来：from geometry import ...
# 改为：from core.geometry import ...
```

### 步骤 3.2：移动 plane_region_manager.py

```bash
Copy-Item "plane_region_manager.py" "domain/region/plane_region_manager.py"
```

### 步骤 3.3：移动 project_manager.py

```bash
Copy-Item "project_manager.py" "domain/project/project_manager.py"
```

### 步骤 3.4：移动 cache_info.py

```bash
Copy-Item "cache_info.py" "domain/project/cache_manager.py"
```

## 🎯 阶段4：重构 ui/viewers

### 步骤 4.1：移动 viewer.py

```bash
Copy-Item "viewer.py" "ui/viewers/base_viewer.py"
```

### 步骤 4.2：移动 viewer_pyvista.py

```bash
Copy-Item "viewer_pyvista.py" "ui/viewers/pyvista_viewer.py"
```

**需要更新的导入**：
```python
# 原来：from viewer import GLMeshViewer
# 改为：from ui.viewers.base_viewer import GLMeshViewer
```

### 步骤 4.3：移动 viewer_open3d.py

```bash
Copy-Item "viewer_open3d.py" "ui/viewers/open3d_viewer.py"
```

### 步骤 4.4：移动 startup_dialog.py

```bash
Copy-Item "startup_dialog.py" "ui/dialogs/startup_dialog.py"
```

## 🎯 阶段5：拆分 ui_app.py（最关键）

### 步骤 5.1：创建 base_panel.py

创建面板基类，提供通用功能。

### 步骤 5.2：提取 lattice_panel.py

从 ui_app.py 提取：
- `_build_lattice_tab()`
- `_on_method_changed()`
- `_sync_method_fields()`
- `_apply_recommended_params()`
- 所有晶格相关的控件和方法

### 步骤 5.3：提取 slice_panel.py

从 ui_app.py 提取：
- `_build_slice_tab()`
- 所有分割相关的方法
- 分割控件

### 步骤 5.4：提取 region_panel_a.py

从 ui_app.py 提取：
- `_build_region_lattice_tab()`
- 区域A相关方法

### 步骤 5.5：提取 region_panel_b.py

从 ui_app.py 提取：
- `_build_plane_region_tab()`
- 区域B相关方法

### 步骤 5.6：创建 main_window.py

保留：
- 窗口初始化
- 菜单栏
- 工具栏
- 布局管理
- 事件协调

## 🎯 阶段6：更新导入路径

创建一个 Python 脚本自动更新所有导入：

```python
# update_imports.py
import re
from pathlib import Path

replacements = {
    'from geometry import': 'from core.geometry import',
    'from voronoi_implicit_new import': 'from core.implicit.voronoi import',
    'from mesh_slicer import': 'from domain.slicer.mesh_slicer import',
    'from region_manager import': 'from domain.region.region_manager import',
    'from plane_region_manager import': 'from domain.region.plane_region_manager import',
    'from project_manager import': 'from domain.project.project_manager import',
    'from viewer import': 'from ui.viewers.base_viewer import',
    'from viewer_pyvista import': 'from ui.viewers.pyvista_viewer import',
    'from viewer_open3d import': 'from ui.viewers.open3d_viewer import',
    'from startup_dialog import': 'from ui.dialogs.startup_dialog import',
}

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# 更新所有 Python 文件
for py_file in Path('.').rglob('*.py'):
    if '.backup' not in str(py_file) and '__pycache__' not in str(py_file):
        update_file(py_file)
        print(f'Updated: {py_file}')
```

## 🎯 阶段7：测试清单

### 基础功能测试
- [ ] 程序能正常启动
- [ ] 主窗口正常显示
- [ ] 菜单栏功能正常

### 晶格生成测试
- [ ] 加载鞋底模型
- [ ] Gyroid 生成
- [ ] Voronoi 2.5D 生成
- [ ] Voronoi Implicit 生成
- [ ] Tile Unit 生成
- [ ] 参数调整生效
- [ ] 预览功能正常

### 模型分割测试
- [ ] 启用分割功能
- [ ] 调整分割平面
- [ ] 执行分割
- [ ] 导出分割结果

### 区域设计测试
- [ ] 区域设计A：添加/删除区域
- [ ] 区域设计A：生成区域晶格
- [ ] 区域设计B：添加/删除平面区域
- [ ] 区域设计B：生成平面区域晶格

### 工程管理测试
- [ ] 新建工程
- [ ] 保存工程
- [ ] 打开工程
- [ ] 工程参数正确加载

### 导出功能测试
- [ ] 导出晶格
- [ ] 导出组合模型
- [ ] 导出分割结果

## 📝 注意事项

1. **每次移动文件后立即测试**
2. **遇到导入错误立即修复**
3. **保持 git 提交频率**（如果使用版本控制）
4. **记录遇到的问题和解决方案**

## 🚨 常见问题

### Q: 导入路径错误
A: 检查 `__init__.py` 是否存在，检查相对路径是否正确

### Q: 循环导入
A: 重新设计模块依赖关系，使用延迟导入

### Q: 功能失效
A: 检查是否所有依赖都已正确移动和更新

## 📊 进度追踪

使用 REFACTORING_PLAN.md 追踪整体进度。
