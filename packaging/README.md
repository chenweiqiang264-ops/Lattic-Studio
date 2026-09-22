# Windows packaging

当前 0.1.1 安装包的实际验证结果与未覆盖项见 [安装包验证记录](../docs/PACKAGING_ACCEPTANCE_0.1.1.md)。

- `lattice_studio.spec`: PyInstaller `onedir` 定义。
- `build_onedir.ps1`: 从仓库根目录执行的构建入口。
- `hooks/`: 仅打包程序需要的运行时初始化。
- `requirements/`: 构建依赖和可选 GPU 运行依赖。

构建输出位于 `dist/LatticeStudio/`，临时分析文件位于 `build/`。
## Windows distribution

The application is built as a PyInstaller `onedir` package and then wrapped
with Inno Setup. The installer includes the application runtime, native C++
extensions, and the CUDA NVVM/libdevice files used by Numba. The NVIDIA driver
is intentionally not bundled and must be installed on machines that should use
CUDA acceleration.

Build the application directory and installer:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\packaging\build_installer.ps1 -RebuildOnedir
```

If `dist\LatticeStudio` already exists, build only the installer:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\packaging\build_installer.ps1
```

The resulting installer is written to `dist\installer`. It installs per-user
under `%LOCALAPPDATA%\Programs\Lattice Studio`, so administrator privileges
are not required. The complete installed directory must remain intact; do not
copy only the executable out of it.

Run the local installer smoke test:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\packaging\verify_installer.ps1
```

This test installs into a fresh `build/installer-acceptance-*` directory. It
clears Python/Conda/CUDA environment paths, then runs the installed executable's
`--verify-runtime <report.json>` command. That command imports the native Python
dependencies and C++ extensions, compares an actual TPMS calculation with its
CPU reference, exports and reopens an STL, and opens/closes the Qt workbench.
The check is repeated with CUDA explicitly disabled to test CPU fallback.

The uninstaller is then executed. The script observes EXE/DLL directory,
shortcut, process, and uninstall-registration removal; it **never deletes
leftovers to make the test pass**. Logs and JSON reports remain in the test
directory. `-Phase Install -TestRoot <path>` retains the installed application
for interactive testing; after closing it, use `-Phase Uninstall -TestRoot
<same-path>` to complete verification. A previously registered installation
blocks a new acceptance installation to protect that existing installation.

Uninstall removes installer-owned binaries, dependencies and shortcuts. User
files created later inside the installation directory are not recursively
erased; an otherwise empty installation directory is removed. User-saved
workspaces and exports outside it are retained.

This is local packaging isolation, not a substitute for testing on a separate
clean Windows machine. CUDA acceleration requires a supported NVIDIA GPU and
driver; the application falls back to CPU when CUDA is unavailable.
