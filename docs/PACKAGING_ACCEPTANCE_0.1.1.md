# Lattice Studio 0.1.1 安装包验证记录

验证日期：2026-09-20。

## 交付文件

- 安装包：`dist/installer/LatticeStudio-Setup-0.1.1-x64.exe`
- 文件大小：321,930,891 字节（约 307 MiB）。
- SHA256：`BCBE328671E4C661962E225C2216A93D48E310B3C89CAAD26F8CEE8C4CB6BD0D`
- Windows x64 安装向导，默认按用户安装，不要求管理员权限。
- 分发此 Setup 文件即可；安装后的 EXE 必须与其依赖目录一起保留。

## 本次修正

1. 显式打包当前 Python 发行版需要的 ffi、OpenSSL、bz2、lzma、expat 和 sqlite DLL，修复安装后 `_ctypes` 等模块加载失败。
2. 冻结程序的 CUDA 探测改用独立 `--cuda-probe-worker` 入口和 JSON 结果文件，避免将应用 EXE 当作 Python 解释器传入 `-c`，造成递归启动。
3. 增加安装版 `--verify-runtime` 自检入口，实际运行计算、STL 导出回读及窗口启动关闭；不再以进程短暂存活作为成功依据。
4. 卸载仅删除安装器管理的文件及空目录，保留后来创建的用户文件。验收脚本只观察卸载结果，不补删残留来制造通过结果。

## 实际验证结果

使用最终 Setup 安装到 `build/installer-acceptance-0.1.1/app` 后运行验证，再执行安装器自带的卸载程序。自检期间 PATH 仅保留 Windows 路径，清除 Python、Conda、CUDA 路径变量。

| 检查项 | 结果与范围 |
| --- | --- |
| PyInstaller onedir 与 Inno Setup 构建 | 成功 |
| Python 原生依赖 | `_ctypes`、`_ssl`、`_bz2`、`_lzma`、`_sqlite3`、`pyexpat` 均从安装目录加载 |
| C++ 扩展 | distance field、mesh SDF、mesh slicer、mask 四个扩展加载成功；本项不代表全部 C++ 算法经过功能测试 |
| GPU 数值计算 | 自动选择 Numba CUDA，识别 NVIDIA GeForce RTX 4060 Laptop GPU；完成 G 型晶格 35,937 个采样点计算，与 NumPy 参考结果在容差内一致 |
| CPU 回退 | 显式禁用 CUDA 后选择 NumPy CPU，同一计算通过 |
| STL 重建与文件回读 | GPU 和 CPU 两次均重建、导出并回读 36,352 个三角面片 |
| Qt 主窗口 | 两次自动验证均显示窗口并正常关闭，测试安装版没有遗留进程 |
| 真实卸载 | EXE、安装目录、快捷方式、卸载注册项均移除；测试程序残留进程为 0 |
| 针对性回归 | CUDA 探测及设备常驻流水线相关测试合计 12 项通过 |

卸载此次删除的是临时测试安装。测试日志、JSON 和导出的 STL 保留在测试目录的 `app` 子目录之外，没有被卸载或补删。

## 证据与复现

- `build/release-build-0.1.1.log`：onedir 构建日志。
- `build/release-installer-0.1.1.log`：安装包构建日志。
- `build/installer-acceptance-0.1.1/install.log`：安装日志。
- `build/installer-acceptance-0.1.1/runtime.json`：安装版 GPU 验证。
- `build/installer-acceptance-0.1.1/runtime-cpu.json`：安装版 CPU 验证。
- `build/installer-acceptance-0.1.1/uninstall.log`：卸载日志。
- `build/installer-acceptance-0.1.1/uninstall-result.json`：卸载结果，`TestScriptDeletedInstallFiles=false`。

重新验证时使用新的测试目录，且不能覆盖已有注册安装：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File packaging/verify_installer.ps1
```

## 尚未完成的验收

- 当前是开发机上的环境变量隔离验证，不能替代干净 Windows 虚拟机或另一台电脑的验证；本轮没有可用的干净 Windows 环境。
- 人工界面操作验收未完成：截图接口报错，之后用户按 Esc 停止电脑操作，未再继续界面控制。自动窗口开关通过不等于交互建模、渲染效果或全部业务流程通过。
- CPU 路径通过禁用 CUDA 验证，并非在无 NVIDIA 显卡的物理电脑上验证。其他 GPU/驱动组合尚未验证；CUDA 加速仍需要兼容的 NVIDIA 显卡和驱动。
- 本次不是全部业务功能的回归或压力测试。

## 2026-09-22 工作区后端迁移后复验

工作区命令与 Qt 投影迁移完成后，重新执行 `build_installer.ps1
-RebuildOnedir`，并使用新的测试目录运行验收。安装包为
`dist/installer/LatticeStudio-Setup-0.1.1-x64.exe`，构建后大小为
322,019,613 字节，SHA256 为
`B40C30BFBD45FBE13949E433756B18F353DF4006AFCF0BFBD29860A163F2ED0F`。

证据目录：
`build/installer-acceptance-workspace-projection-20260922-145200`。

- GPU 运行时检查通过：Numba CUDA 使用 NVIDIA GeForce RTX 4060 Laptop GPU，
  完成 35,937 个采样点和 36,352 个 STL 三角面片的验证。
- 强制 CPU 回退检查通过：NumPy CPU 完成同一运行时验证。
- 两次自动 Qt 主窗口打开和关闭均通过。
- 卸载后 EXE、安装目录、快捷方式和注册项均不存在，残留进程为 0；
  `TestScriptDeletedInstallFiles=false`。

这仍是开发机上的隔离环境验证，不替代干净 Windows 或另一台物理电脑的验收。
