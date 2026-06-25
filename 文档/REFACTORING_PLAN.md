# 代码重构计划

## 目标
将单体应用重构为模块化架构，提高代码可维护性和可扩展性。

## 目录结构
```
shoe_lattice_system/
├── core/           # 核心算法层（纯计算）
├── domain/         # 业务逻辑层
├── ui/             # UI层
├── utils/          # 工具模块
├── resources/      # 资源文件
└── tests/          # 测试
```

## 重构阶段

### ✅ 阶段1：创建目录结构（已完成）
- [x] 创建所有目录
- [x] 创建所有 __init__.py 文件

### 🔄 阶段2：移动和重构 core 层（进行中）
- [ ] 移动 geometry.py → core/geometry/
- [ ] 移动 voronoi_implicit_new.py → core/implicit/voronoi.py
- [ ] 移动 mesh_slicer.py → core/slicer/mesh_slicer.py
- [ ] 移动 C++ 扩展 → core/cpp/
- [ ] 创建 core/implicit/base.py（隐式函数基类）
- [ ] 创建 core/implicit/tpms.py（从 geometry.py 提取）

### ⏳ 阶段3：移动和重构 domain 层
- [ ] 移动 region_manager.py → domain/region/
- [ ] 移动 plane_region_manager.py → domain/region/
- [ ] 移动 project_manager.py → domain/project/
- [ ] 移动 cache_info.py → domain/project/cache_manager.py

### ⏳ 阶段4：移动和重构 ui 层
- [ ] 移动 viewer.py → ui/viewers/base_viewer.py
- [ ] 移动 viewer_pyvista.py → ui/viewers/
- [ ] 移动 viewer_open3d.py → ui/viewers/
- [ ] 移动 startup_dialog.py → ui/dialogs/

### ⏳ 阶段5：拆分 ui_app.py（最复杂）
- [ ] 创建 ui/panels/base_panel.py
- [ ] 提取 ui/panels/lattice_panel.py
- [ ] 提取 ui/panels/slice_panel.py
- [ ] 提取 ui/panels/region_panel_a.py
- [ ] 提取 ui/panels/region_panel_b.py
- [ ] 创建 ui/main_window.py（主窗口框架）

### ⏳ 阶段6：更新导入路径
- [ ] 更新所有文件的 import 语句
- [ ] 更新 main.py 入口

### ⏳ 阶段7：测试和验证
- [ ] 测试程序启动
- [ ] 测试晶格生成
- [ ] 测试模型分割
- [ ] 测试区域设计
- [ ] 测试工程保存/加载
- [ ] 测试导出功能

### ⏳ 阶段8：清理和优化
- [ ] 删除旧文件
- [ ] 优化代码
- [ ] 添加文档注释
- [ ] 创建 requirements.txt

## 注意事项
1. 每个阶段完成后都要测试程序是否可运行
2. 保留原文件备份（添加 .backup 后缀）
3. 逐步提交，便于回滚
4. 更新导入路径时要仔细检查

## 当前进度
- 阶段1：✅ 完成 - 创建目录结构
- 阶段2：✅ 完成 - 移动 core 层文件
- 阶段3：✅ 完成 - 移动 domain 层文件
- 阶段4：✅ 完成 - 移动 ui 层文件
- 阶段5：🔄 准备开始 - 拆分 ui_app.py
- 阶段6：✅ 部分完成 - 更新导入路径（主要文件已更新）

## 测试结果
- ✅ 所有模块导入测试通过
- ⏳ 待测试：程序启动
- ⏳ 待测试：功能完整性
