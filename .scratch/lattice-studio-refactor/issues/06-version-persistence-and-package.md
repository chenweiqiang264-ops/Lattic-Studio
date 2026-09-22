# Version persistence and build the distribution

Type: task
Status: ready-for-agent
Blocked by: 04, 05

Add workspace schema migrations, automatic NVIDIA runtime probing suitable for packaged systems, and the PyInstaller `onedir` build with required resources and native extensions.

## Comments

2026-09-20：完成 0.1.1 onedir 与 Inno Setup 安装包。修复冻结 EXE 的 CUDA 探测递归启动，补齐 Python DLL。使用最终安装包安装后验证 GPU/CPU TPMS、STL 导出回读及主窗口开关，随后真实卸载并确认安装目录、快捷方式、注册项及测试进程无残留。详细证据和限制见 `docs/PACKAGING_ACCEPTANCE_0.1.1.md`。干净 Windows 验证及人工交互建模验收仍未完成；本记录不表示整个重构事项均已完成。
