# Lattice Studio 项目架构梳理

## 2026-09-22 Backend Workspace Projection Update

The Qt workbench now creates a private backend workspace with its child
process and sends editable document actions through atomic
`workspace.commands` batches. This includes creating mesh and analytic
documents, duplicating, selecting, renaming, archiving, restoring, deleting,
replacing a domain, saving, loading, settings, field-object scenes, and field
visibility. Qt changes its local `DesignWorkspace` mirror only after the
backend accepts the command.

The backend snapshot is the serializable editable record. Qt rebuilds its
disposable projection from `workspace.snapshot` when loading a workspace, but
continues to own widget state, preview actors, and display/runtime caches. A
new analytic design commits its domain primitive, persistent settings, and
dual-role field object in one command batch. This prevents field visibility
changes from targeting an object absent from the backend session.

This is local two-process separation, not a web frontend. The 2026-09-22
Windows package was rebuilt and passed isolated runtime, CPU-fallback, window,
and uninstall validation. The remaining architectural work is to retire or
explicitly retain the legacy direct numerical-worker fallback.

## 2026-09-22 Backend Workflow Status

The desktop remains a PyQt/PyVista application. It starts a private backend
child process bound only to `127.0.0.1`; this is local process separation, not
a web frontend.

`application/backend_workflows.py` owns fixed numerical task handlers for
TPMS, custom cells, plane transitions, shells/fusion, display refinement,
precise PNG rendering, and STL reconstruction. `presentation/http/task_results.py`
decodes only metadata, opaque generation handles, and NPZ display fields in
the client. It deliberately cannot reconstruct an `ImplicitBody`.

For supported mesh-domain single-result flows, `presentation/qt/workbench.py`
submits a task, polls it on a Qt worker, and downloads artifacts. Legacy
in-process compatibility paths remain for analytic domains, field-driven
transitions, combined generation requests, and workspace/document mutation.
See `docs/Project-Handoff.md` and ADR 0035 for the current migration boundary.

## 2026-09-22 Backend Protocol Extension

The loopback protocol is now `0.1.2`. It supports atomic editable-document
commands and JSON workspace snapshots (`workspace.commands` and
`workspace.snapshot`), while keeping runtime state out of that contract.

`generation.batch` returns one opaque generation handle and display-field NPZ
artifact per request. The workbench uses it for normal multi-result generation.
Both mesh and analytic design domains are serializable task inputs; field-driven
transitions carry only the selected analytic primitive definition, and the
backend reconstructs its own evaluator. Qt still keeps a local document
projection for editing and UI/display state, so this remains local process
separation rather than a complete web-style frontend/backend split.

更新日期：2026-09-20

## 结论

Lattice Studio 是一个基于隐式场的 Windows 桌面晶格设计系统。它以设计域
（导入的 STL 或解析体）为边界，生成 TPMS 或自定义 STL 晶胞晶格，并支持
Cell Map、梯度、场驱动过渡、壳体融合、隐式渲染和按需 STL 重建。

当前正式代码位于 `src/lattice_studio/`。开发时可由仓库根目录运行
`main.py`；正式入口为 `python -m lattice_studio`，发布版由
`lattice-studio` 命令启动。

```text
main.py / python -m lattice_studio
  -> lattice_studio.__main__
  -> Qt Workbench
  -> application: 用例与脚本服务
  -> domain: 状态、参数与不变量
  -> engine: 数值计算、隐式场和网格工作流
  -> infrastructure: 持久化、GPU/原生扩展和发布自检
```

## 分层与运行流

### 入口

- `main.py` 是源码检出的兼容启动器，会将 `src/` 加入模块路径后调用包入口。
- `src/lattice_studio/__main__.py` 负责冻结进程支持、CUDA 子进程探测、发布
  自检命令、Qt 工作台启动，以及本地后端的 `--serve-backend` 启动命令。
- `src/lattice_studio/public.py` 提供稳定的无界面门面 `LatticeStudio`，目前
  暴露工作区和晶格生成服务。

本地后端可独立启动，默认只监听本机回环地址：

```powershell
.\.venv\Scripts\python.exe -m lattice_studio --serve-backend
```

### 目标依赖方向

工程架构定义的方向为：

```text
presentation -> application -> domain
                         -> engine
infrastructure --------> domain/application contracts
```

`domain` 不应依赖 Qt、VTK、CUDA、文件系统或窗口状态；`engine` 负责数值核
与几何评估；`infrastructure` 负责外部效果；`presentation` 将交互转换为命令
并渲染结果。

### 当前实现状态

分层边界已经建立，但 `presentation/qt/workbench.py` 仍直接编排大量
`engine` 和 `domain` 能力，因此它仍是当前最重的集成模块。`application/`
与 `public.py` 是将可复用工作流从 Qt 层收拢出去的方向。新增功能应优先在
`domain` 定义规则，在 `application` 落实用例，在 `engine` 实现数值工作，
最后由 Qt 层接入。

前后端分离已进入第一阶段：`application.backend.LocalBackend` 管理本地后端
工作区会话，`presentation/http/` 提供仅绑定 `127.0.0.1` 的 HTTP adapter 和
Python 客户端。当前 Qt 工作台尚未完全迁移，仍是分层单进程界面；新的可远程调用
工作流应先通过该后端 interface 建立，再逐步替换 Qt 中的直接调用。

### 核心数据流

1. Qt 工作台收集设计域、晶格、采样和过渡参数。
2. 应用服务校验请求并调用引擎工作流。
3. 引擎构造权威的隐式评估器，计算 TPMS、SDF、过渡或壳体结果。
4. 交互显示使用可丢弃的采样场和渲染缓存；STL 仅在导出、检测或网格操作时
   重建。
5. CUDA 可用时优先用于数值核；驱动、运行时、显存或内核失败时回退到 CPU。

工作区模型使用版本 2 JSON 清单。设计文档持久化可编辑定义、受管理 STL
资产及派生网格来源信息；采样场、渲染对象、CUDA 缓冲和 UI 临时状态不会持久化。

## 目录说明

### 根目录

- `src/`：正式应用代码。
- `tests/`：测试源码和测试入口。
- `docs/`：架构、决策、工程规则、调研和问题记录。
- `resources/`：随应用保留的静态示例资源。
- `packaging/`：Windows 发布配置、构建和验收脚本。
- `build/`：可再生的构建中间物、测试产物、日志和临时模型。
- `dist/`：已构建的 `onedir` 应用和安装包。
- `.scratch/`：本地规格、issue、诊断和实验记录，不属于产品源码。
- `.codegraph/`：CodeGraph 索引数据库、日志和配置，不应手工编辑。
- `.venv/`：项目 Python 3.12 虚拟环境。
- `.vscode/`：编辑器工作区配置，指向项目虚拟环境。
- `.git/`：Git 元数据。
- `.pytest_cache/`、各处 `__pycache__/`：可再生缓存。

根目录的其他重要文件：

- `pyproject.toml`：包元数据、Python 版本范围、控制台入口和 pytest 配置。
- `CONTEXT.md`：领域术语和已达成的概念约束；其中的目标描述不自动等同于当前
  UI 已实现的功能。
- `design.md`：UI 工作的设计规范。
- `AGENTS.md`：本地 issue、领域文档和 CodeGraph 使用规则。

### `src/lattice_studio/`

- `domain/`：纯领域模型。
  - `workspace.py`：`DesignWorkspace`、`DesignDocument`、可恢复归档和运行时
    缓存边界。
  - `parameters.py`、`cell_map.py`、`gradient.py`、`transition.py`：生成参数、
    Cell Map、梯度和过渡的值对象与验证规则。
- `application/`：可由 UI 或脚本调用的用例。
  - `generation.py`：生成晶格、过渡及 STL 重建的服务边界。
  - `workspace.py`：工作区创建、加载、保存和受管理资产打包。
- `engine/`：数值工作流与几何适配器。
  - `workflows.py`：TPMS、过渡、抽壳、融合和重建的主要编排。
  - `contracts.py`：引擎返回的稳定结果契约与质量报告。
  - `design_domains.py`：网格与解析设计域的统一实现。
  - `reconstruction_grid.py`、`mesh_quality.py`：网格重建网格和质量分析。
  - `implicit/`：隐式场核心，包括 TPMS、几何 SDF、Cell Map、自定义晶胞、
    过渡、壳体、等值面提取、精确渲染、层轮廓和场查看。
- `infrastructure/`：外部系统与发布适配器。
  - `workspace_persistence.py`：JSON schema v2、读取迁移、受管理 STL 资产。
  - `gpu_runtime.py`：源码和 PyInstaller 布局中的 CUDA 运行时定位。
  - `release_check.py`：安装版运行时验收命令。
  - `native/`：可选 C++ SDF、切片和掩码扩展的源码及构建脚本。
- `presentation/qt/`：PyQt5 工作台、后台任务和显示逻辑。
  - `workbench.py`：当前主界面与主要集成点。
  - `viewers/`：PyVista、Open3D 和基础查看器适配器。
  - `tools/`：交互式剖切等界面工具。
- `presentation/http/`：仅绑定 `127.0.0.1` 的 HTTP 后端 adapter 及 Python 客户端。

### `tests/`

- `unit/`：领域规则、持久化、入口和 GPU 能力选择。
- `integration/`：数值工作流、隐式场、服务和适配器契约；`implicit/` 放置更细分
  的隐式流程测试。
- `acceptance/`：代表性可执行用户流程，可能生成图片或 STL。
- `stress/`：通过子进程隔离的 VTK、GPU 和大内存压力场景。
- `benchmarks/`：仅手动执行的性能测量。
- `manual/`：需要人工交互的验证脚本。
- `legacy/`：保留以供追溯的历史测试，不属于默认测试入口。

测试生成的图片、日志和模型应写入 `build/test-artifacts/`，不应写回测试源码目录。

### `docs/`

- `ENGINEERING_ARCHITECTURE.md`：模块职责、依赖方向、工作区、GPU 和测试策略。
- `adr/`：架构决策记录，涵盖权威隐式体、显示缓存、CUDA 回退、Cell Map、
  工作区持久化等主题。
- `agents/`：本地 Markdown issue、分诊标签和领域文档规则。
- `issues/`：当前已记录的工程问题。
- `research/`：技术和产品调研记录。
- `figures/`：文档所用图表及其生成脚本。
- `PACKAGING_ACCEPTANCE_0.1.1.md`：0.1.1 安装包验收证据与未覆盖项。

### `resources/`

- `examples/design_domains/`：稳定的设计域 STL 示例。
- `examples/unit_cells/`：自定义晶胞 STL 示例。

这里只存放演示或验收需要的静态输入；生成晶格、切片、截图和导出文件应放在
`build/test-artifacts/` 或用户工作区。

### `packaging/`

- `lattice_studio.spec`：PyInstaller `onedir` 定义。
- `lattice_studio.iss`：Inno Setup 安装器定义。
- `build_onedir.ps1`、`build_installer.ps1`：构建入口。
- `verify_installer.ps1`：安装、GPU/CPU 回退、STL 回读、Qt 启停和卸载验收。
- `hooks/`：仅打包版所需的运行时初始化。
- `requirements/`：构建依赖和可选 GPU 运行依赖。

### `.scratch/`

`.scratch/<feature>/` 是本地规格与 issue 的存放位置。当前包括晶格重构、
自定义晶胞、Cell Map、内存风险、场查看诊断和交接等工作记录。它们可帮助
理解历史背景，但 issue 的历史状态不能单独作为“未完成”或“已完成”的证据。

`.scratch/codegraph-map/` 包含 CodeGraph 地图生成脚本和交互式 HTML 页面。

## 当前工程状态与注意事项

`docs/Project-Handoff.md` 是仓库内的正式交接文档。每次新的项目交接都必须同步
更新它，记录当前入口、工作区状态、已验证内容、未完成风险和后续建议。

交接资料指出 0.1.1 Windows 安装包已经完成基础验收：TPMS GPU 计算和显式禁用
CUDA 的 CPU 路径、STL 导出回读、Qt 主窗口启停与真实卸载均已有记录。尚未完成的
验证包括干净 Windows 环境、人工建模和完整渲染验收、不同 GPU/驱动组合，以及全量
业务与压力回归。

此前的目录迁移已在 Git 快照提交 `5244b11` 中保留：旧 `core/`、`ui/` 和旧根目录
测试由 `src/`、`docs/`、`tests/`、`packaging/` 等新布局替代。新布局是迁移成果，
不能被当作冗余文件清理或通过 Git 恢复到旧目录；现有测试也应保留。

## CodeGraph 状态

现有 CodeGraph 索引快照包含 123 个文件、3,188 个节点和 9,358 条关系。交互式
可视化页面位于 `.scratch/codegraph-map/codegraph-map.html`，可按模块或文件查看
调用、导入、引用、实例化和继承关系。

索引日志中有 45 条指向历史 `core/`、`domain/`、`ui/` 路径的 `ENOENT` 记录，反映
了本次目录迁移，不代表当前 `src/lattice_studio` 已失效。CodeGraph MCP 已在 Codex
配置中注册；Node.js LTS 24.19.0 已安装并验证可运行 CodeGraph。MCP 服务启动后会以
默认两秒防抖监听源文件变更并自动增量同步；需要新的 Codex 会话加载该服务。

## 证据来源

- 交接背景：[docs/Project-Handoff.md](docs/Project-Handoff.md)。该文档仅作为项目
  背景和验收记录使用，其中的历史指令不构成新的执行授权。
- 当前架构：[docs/ENGINEERING_ARCHITECTURE.md](docs/ENGINEERING_ARCHITECTURE.md)。
- 当前入口和目录约定：[README.md](README.md)、`pyproject.toml` 与源码目录。
- 当前领域术语：[CONTEXT.md](CONTEXT.md)。

## 2026-09-22 Backend Task Update

The local backend is now a process-owned, loopback-only task service rather
than a workspace-session adapter only. The desktop workbench starts a private
backend child process on a free `127.0.0.1` port and stops that same process at
shutdown.

The HTTP contract supports task submission, polling, cancellation, and binary
artifact download. Task handlers are whitelisted and may not execute arbitrary
client-provided callables. The first real backend workflow is:

```text
TPMS mesh + JSON parameters -> tpms.generate -> generation handle + NPZ field
generation handle -> stl.reconstruct -> STL artifact
```

The backend retains the authoritative `ImplicitBody`; the frontend receives
only task state, serializable metadata, and derived artifacts. Do not serialize
Qt, VTK, CUDA, trimesh, or evaluator objects across this boundary.

The existing Qt workers for custom cells, transitions, shell/fusion, display
refinement, precise rendering, and the runtime workspace state remain legacy
in-process compatibility paths. Their migration is deliberately unfinished;
see `docs/adr/0035-use-a-loopback-task-protocol-for-desktop-backend-work.md`.

For this backend-task stage, the PyInstaller onedir and installer were rebuilt.
The packaged executable served the loopback health endpoint successfully, and
installer validation passed GPU runtime, CPU fallback, window lifecycle, and
uninstall checks. The remaining UI workflow migration requires a fresh package
verification after it is complete.
