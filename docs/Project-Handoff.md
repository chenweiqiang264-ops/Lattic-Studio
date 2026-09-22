# Project-Handoff

## 2026-09-22 Workspace Projection Migration Update

Current branch: `feature/local-backend-separation`. The working tree contains
uncommitted backend-separation work on top of the prior continuation changes.

The Qt workbench now owns a private `backend_workspace_id` for its local child
process. It commits document creation, import/replacement, activation,
duplication, rename, archive/restore/delete, settings, field-object scenes,
and field visibility through `workspace.commands` before changing the Qt
projection. Save first reconciles the current local mirror to the backend and
then uses backend persistence; load requests a backend snapshot and rebuilds
the disposable Qt projection from it.

New document initialization is atomic at the protocol boundary. In particular,
an analytic design's domain primitive, settings, and selectable dual-role
field object are sent together. A regression test verifies that repeated field
visibility toggles preserve the local transition result and update the backend
snapshot as well.

Verified after this follow-up:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pytest tests\integration\implicit\test_field_object_workspace_actions.py::test_preview_toggle_preserves_transition_and_revision tests\integration\implicit\test_field_object_workspace_actions.py::test_rename_requires_confirmation_and_preserves_results tests\integration\implicit\test_field_object_workspace_actions.py::test_clear_selected_archive_requires_confirmation_and_preserves_other_designs tests\integration\implicit\test_field_object_workspace_actions.py::test_empty_workspace_disables_rename_and_empty_recycle_actions -q
codegraph.cmd sync
codegraph.cmd status
```

The targeted workbench regression suite passed `5 passed`. CodeGraph is
current with 134 files, 3,570 nodes, and 10,765 edges. The rebuilt
`dist/installer/LatticeStudio-Setup-0.1.1-x64.exe` passed isolated runtime,
GPU, forced CPU fallback, Qt lifecycle, and genuine uninstall validation in
`build/installer-acceptance-workspace-projection-20260922-145200`. Qt retains
a display/runtime projection and legacy direct-worker compatibility paths, so
this is not a web frontend or a claim that every UI operation is remote-only.
The rebuilt installer is 322,019,613 bytes with SHA256
`B40C30BFBD45FBE13949E433756B18F353DF4006AFCF0BFBD29860A163F2ED0F`.

## 2026-09-22 Protocol and Generation Migration Update

Current branch: `feature/local-backend-separation`. The working tree contains
uncommitted backend-separation work on top of the prior continuation changes.

The loopback protocol is now `0.1.2`. `LocalBackend` exposes two new
workspace operations: `workspace.snapshot` returns JSON-safe editable
definitions, while `workspace.commands` atomically applies a fixed command
batch for document create/replace, activation, lifecycle, settings, and field
objects. The command service applies to an isolated candidate workspace before
committing, so a rejected command leaves the backend session unchanged.

`generation.batch` accepts two to sixteen fixed TPMS/custom/transition
requests and publishes an opaque generation handle and separate display-field
NPZ artifact for each request. The Qt workbench uses that batch path for
combined results. It now serializes both mesh and analytic design domains for
generation work. A field-driven transition transports only the selected
`ImplicitPrimitive` DTO; the backend reconstructs the driver evaluator locally.
No `ImplicitBody`, sampled display field, Qt, VTK, CUDA, or trimesh object
crosses HTTP.

The remaining separation boundary is precise: Qt still owns its editable
`DesignWorkspace` projection, save/load UI orchestration, and display/runtime
caches. The new workspace command API is ready, but those UI mutation actions
have not yet been switched to it. Do not call the project fully frontend/backend
separated until that projection and remaining in-process compatibility workers
are retired or deliberately retained by a new ADR.

Verified after this update:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pytest tests\integration\test_local_backend_api.py tests\integration\test_local_backend_tasks.py tests\integration\test_workbench_shutdown.py -q
.\.venv\Scripts\python.exe -m pytest tests\integration\test_design_domain_reload.py -q
codegraph.cmd status
codegraph.cmd sync
```

The HTTP/task/shutdown contract suite passed `20 passed`; the design-domain
reload suite passed `3 passed`. CodeGraph reports an up-to-date index with 133
files, 3,542 nodes, and 10,630 edges. CUDA emitted only low-occupancy warnings
on small fixtures. Packaging has not been rebuilt after this source change;
rebuild the onedir and installer and rerun installer verification before any
release claim.

## 2026-09-22 Loopback Workflow Migration Update

Current branch: `feature/local-backend-separation`. The working tree contains
uncommitted continuation work after `ee9fe6a`.

The local child backend now owns these fixed task kinds:

- `tpms.generate`, `custom.generate`, and plane-driven `transition.generate`;
- `shell.generate` and `shell.union`, consuming an opaque generation handle;
- `display.refine`, `render.precise`, and `stl.reconstruct`.

Each generation keeps its `ImplicitBody` in the backend process. The Qt client
stores only `BackendGenerationResult` handles plus downloaded display-field
NPZ data. It never recreates an evaluator from a display field. Backend
precise renders publish a PNG through a dependency-free writer; reconstructed
meshes are downloaded as STL artifacts.

Workbench adapters currently route a mesh-domain single TPMS/custom/plane
transition, backend-handle STL reconstruction, backend-handle shell/fusion,
backend display refinement, and backend precise rendering through the task
client. The child-process readiness probe still uses a short timeout, while
the returned client uses 30 seconds for large artifact downloads.

Compatibility paths remain for analytic design domains, field-driven
transitions, and combined multi-result generation requests. Qt also remains
the authority for editable `DesignWorkspace` documents, save/load, and UI
state. Therefore this is a substantial numerical-workflow migration, not yet
complete frontend/backend separation. Continue with explicit serializable
workspace/document commands and a backend batch task before claiming that
goal.

Verified in this workspace after the continuation changes:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src
.\.venv\Scripts\python.exe -m pytest tests\integration\test_local_backend_tasks.py tests\integration\test_workbench_shutdown.py -q
```

Result: `14 passed`. The contract suite covers TPMS-to-STL, custom cells,
plane transitions, shell union, display refinement, precise-render PNGs,
artifact download, cancellation, and child-process shutdown. Numba emitted
only low-occupancy CUDA warnings.

Final verification after the proxy-bypass follow-up:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_local_backend_api.py tests\integration\test_local_backend_tasks.py tests\integration\test_workbench_shutdown.py -q
powershell -ExecutionPolicy Bypass -File .\packaging\build_onedir.ps1
powershell -ExecutionPolicy Bypass -File .\packaging\build_installer.ps1
powershell -ExecutionPolicy Bypass -File .\packaging\verify_installer.ps1 -TestRoot build\installer-acceptance-backend-final-20260922-133202
```

The focused suite passed `17 passed`. A fresh `dist/LatticeStudio` health
check passed. The installer acceptance evidence is in
`build/installer-acceptance-backend-final-20260922-133202`: GPU runtime,
forced CPU fallback, Qt window open/close, and uninstall all passed, with no
remaining install files or processes. `LocalBackendClient` now disables proxy
use explicitly, which prevents a configured system proxy from returning 502
for the private `127.0.0.1` backend.

## 2026-09-22 Local Backend Separation Update

Current branch: `feature/local-backend-separation`.

The baseline backup commit is `5244b11`; the initial workspace-loopback
foundation is `c896de3`; the current branch tip contains the backend task
stage. It adds:

- `application.tasks` owns asynchronous task state, cancellation, progress,
  and backend-managed artifacts.
- `presentation.http.local_backend` exposes task submission, polling,
  cancellation, and artifact download on `127.0.0.1` only.
- `presentation.http.backend_process` starts and stops the desktop-owned local
  backend child process. It is invoked by the Qt workbench lifecycle.
- `application.backend_workflows` implements backend-owned `tpms.generate`
  and `stl.reconstruct`; the generation handle remains authoritative in the
  backend and the client receives a display-field NPZ or STL artifact.

Verified in this workspace:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_local_backend_api.py tests\integration\test_local_backend_tasks.py tests\integration\test_workbench_shutdown.py -q
.\.venv\Scripts\python.exe -m compileall -q src
```

The task tests cover success, failure, cancellation, binary artifact download,
backend child-process lifecycle, and a real TPMS-to-STL vertical workflow.

Remaining risk: the Qt workbench still uses in-process workers for custom-cell,
transition, shell/fusion, display refinement, precise rendering, and its
runtime workspace model. Do not transfer `ImplicitBody`, VTK, Qt, CUDA, or
trimesh objects across HTTP; complete those migrations through backend handles
and derived artifacts as specified in ADR 0035.

Packaging evidence for this backend-task stage: `packaging/build_onedir.ps1` and
`packaging/build_installer.ps1` completed; the rebuilt onedir executable was
started with `--serve-backend` and returned the task protocol health payload.
`packaging/verify_installer.ps1` completed in
`build/installer-acceptance-backend-20260922-122300`: GPU runtime, explicitly
disabled CUDA CPU fallback, window open/close, and uninstall all passed. A
future migration of the remaining UI workflows must rebuild and verify again.

Full-suite note: `pytest -q -x` stopped after 63 passing tests at
`tests/integration/test_cell_map_and_topology_export.py::test_topology_aware_simplification_repairs_non_watertight_candidate`.
The expectation was `repair_attempted=True`, while the current simplification
path reported `False`. A separate unconstrained full run later aborted in
PyVista/VTK interactor initialization with a Windows access violation around
44 percent progress. Neither failure path includes the backend task modules;
the focused 21-test task/workspace/shutdown suite passed.

交接日期：2026-09-20。用途：让新的对话接手 Lattice Studio 的工程管理和后续开发。

本文件按 handoff 技能要求保存在系统临时目录。文件内仓库路径均相对于 `D:\shoe(1)\shoe`；技能路径使用 `%USERPROFILE%\.codex\skills`。不包含账号、密钥或其他凭据。临时目录可能被系统清理，开启新对话时请直接附上此文件。

## 1. 接手时先了解什么

项目是 Windows 桌面隐式晶格设计系统，核心工作包括 TPMS、自定义 STL 晶胞、Cell Map、梯度与场驱动过渡、壳体融合、隐式渲染、STL 重建，以及剖切、逐层等值线和场可视化。用户希望软件具备成熟工程结构，最终能安装到其他电脑使用，交互与显示长期参考 nTop。

最近一项完成交付是 0.1.1 Windows 安装包，以及安装后基础计算、窗口开关和真实卸载验证。当前用户要求是生成交接文档，方便新对话管理工程；没有授权新对话自动执行新的大范围重构或恢复旧功能块方案。

请按以下顺序建立背景，随后只阅读当前任务需要的源文件：

1. `AGENTS.md` 和 `docs/agents/`：本地 issue、领域文档规则。
2. `README.md`、`docs/ENGINEERING_ARCHITECTURE.md`：当前入口与职责划分。
3. `CONTEXT.md` 及相关 `docs/adr/`：术语与决策；其中有目标状态，不能一概视为实现状态。
4. UI 任务先读根目录 `design.md`，这是用户明确要求持续遵守的设计规范。
5. 发布任务读 `packaging/README.md` 和 `docs/PACKAGING_ACCEPTANCE_0.1.1.md`。
6. 继续架构整理时读 `.scratch/lattice-studio-refactor/spec.md` 和其 `issues/`，但其中“Current state”描述的是重构前基线，不是现在的文件布局。

## 2. 当前运行入口与环境

- 仓库根目录：`D:\shoe(1)\shoe`，Windows PowerShell。
- 项目解释器：`.venv\Scripts\python.exe`，Python 3.12 系列；`pyproject.toml` 限制为 `>=3.12,<3.13`。
- `.vscode/settings.json` 已指向项目 `.venv`。不要随手切换系统 Python 或另一个 Conda 环境。
- 源码便捷入口：`main.py`；正式包入口：`src/lattice_studio/__main__.py`。
- 包入口调用 Qt 工作台 `src/lattice_studio/presentation/qt/workbench.py`。
- 历史 `tests/intermediate_tests/tpms_filling_app.py` 和 `ui_app.py` 已不应再作为生产入口；聊天里的旧 IDE 标签不能作为现状依据。
- 脚本接口位置：`src/lattice_studio/public.py`。完整职责和扩展规则引用架构文档，不在此复制。

从仓库根目录启动：

```powershell
.\.venv\Scripts\python.exe main.py
```

正式模块启动方式及 editable 安装见 `README.md`。`pyproject.toml` 的依赖列表目前为空，因此不能把单独 `pip install .` 当成完整开发环境安装方案；依赖与构建入口还需参考 `packaging/requirements/` 和构建脚本。

## 3. 工作区与版本控制注意事项

本次交接时 `git status --short` 仍有大量旧路径删除、修改，以及未跟踪的 `src/`、`docs/`、`packaging/`、测试子目录等。它们包括先前已进行的工程迁移，不能作为“无用文件”清理，也不能用 Git reset/checkout/clean 恢复到旧结构。交接轮未创建 Git 提交。

用户曾表示已经备份工程并允许清理真正冗余代码，但随后明确说“先别删 test 文件了”。最新约束是保留测试；现有历史删除记录不代表现在可以继续删除测试。

`.scratch/` 是本地规格与 issue，具体约定见 `docs/agents/issue-tracker.md`。issue 的旧状态可能未与实现同步，不能仅凭 `ready-for-agent` 或历史 Comments 判断某项完全未做或已完成。

## 4. 已交付安装包与验证边界

安装包：`dist/installer/LatticeStudio-Setup-0.1.1-x64.exe`。本次交接已确认文件仍存在。对应 onedir：`dist/LatticeStudio/`。不要误发旧版 0.1.0。

详细修复、文件大小、SHA256、验证数据和日志位置均在 `docs/PACKAGING_ACCEPTANCE_0.1.1.md`。该报告是本轮发布验收的引用依据。

状态摘要：最终安装包实际安装后，TPMS GPU 计算和显式禁用 CUDA 的 CPU 路径、STL 导出回读、Qt 主窗口打开关闭通过；真实卸载移除了测试安装目录、快捷方式和注册项，没有残留测试进程。验证脚本没有补删安装文件来制造通过结果。

以下仍未完成，不能对用户宣称已验证：

- 干净 Windows 虚拟机或另一台电脑上的安装运行。
- 人工交互建模及完整渲染效果验收。
- 无 NVIDIA 显卡的物理电脑和其他 GPU/驱动组合验证。
- 全业务流程、全压力矩阵的本次发布回归。

之前人工界面验证遇到截图接口错误，随后用户按 Esc 停止电脑操作，已停止该轮电脑控制。新的界面测试应明确是新任务下的操作，不要默默恢复被停止的输入。之前打开过独立 onedir GUI，不能假定现在仍在运行，也不能批量终止用户进程；确需处理时核对完整可执行路径。

## 5. 最新代码改动的导航

最近安装包问题的具体说明引用验收报告，下列路径用于接手定位：

| 关注点 | 源文件 |
| --- | --- |
| 程序入口、冻结 worker 分发 | `src/lattice_studio/__main__.py` |
| 冻结 CUDA 探测与子进程隔离 | `src/lattice_studio/engine/implicit/cuda_probe.py` |
| 安装版运行时验收命令 | `src/lattice_studio/infrastructure/release_check.py` |
| DLL 收集与 onedir 定义 | `packaging/lattice_studio.spec` |
| Windows 安装向导与卸载规则 | `packaging/lattice_studio.iss` |
| 构建入口 | `packaging/build_onedir.ps1`、`packaging/build_installer.ps1` |
| 实际安装、GPU/CPU 自检与卸载 | `packaging/verify_installer.ps1` |
| 最近针对性回归 | `tests/integration/test_cuda_probe.py`、`tests/integration/test_cuda_resident_pipeline.py` |

修改源码后，不能仅重打 Inno 安装器而继续使用旧 onedir；需要按 `packaging/README.md` 的 `-RebuildOnedir` 路径重新构建。无源码变动时，不必为了交接重复耗时构建。

安装版自检只覆盖报告所列范围。CUDA、OpenGL 渲染、C++ 本地扩展是不同加速路径，不能以任一成功推断其他路径全部正常。当前发行目标的 NVIDIA/CPU 策略见 ADR-0033。

## 6. 历史需求如何继承

历史对话很多，且中途出现过回退和改口。以下是下一位代理容易误判的重点：

- 用户曾推进 nTop 式可引用功能块、输出属性、显示开关和右侧查看工具，之后明确要求恢复到首版功能块改造前。不要因 `CONTEXT.md` 的功能块术语或 ADR-0030 尚存在，就自动重启该改造；以当前代码和新需求为准。
- 用户要求工程目录清晰、保留必要代码并支持软件分发；重构要保持已有行为与数值结果，而不是趁机重做交互。
- 用户希望操作真正完成、必要时自己运行验证，反感反复确认；授权明确的常规实施继续做。报告要区分代码检查、自动化测试、实际界面测试和跨电脑验证。
- 用户强调多设计隔离、首次空工作区、可恢复删除、工作区持久化与导入资源打包副本。约束与现有模型请引用 `CONTEXT.md`、ADR-0029/0034 和工作区模块，不另建平行状态系统。
- 用户明确要求以后 UI 先读 `design.md`；次要和高级参数采用折叠分组，参数控件不应被普通滚轮意外改值。
- 历史出现过基本几何体尺寸编辑闪退、场查看卡死、操纵器缺失、预览覆盖生成结果、固定层高使模型沿查看方向延伸等问题。这些是需关注的回归场景，不等于交接时仍复现，不能直接宣布未修复或已全部修复。
- 权威隐式体、显示采样场、STL 重建采样场必须区分。场连续性不等于几何连通性；权重插值也不能单独保证过渡连接。相关决策优先查领域文档、ADR-0001/0004/0027/0028。
- 文献检索和 Word 整理是历史任务；没有在本次核实成果位置，不应虚构论文清单或文档路径。

## 7. 后续开发建议

新会话首先确认用户本次要管理或修改的事项，并核对当前工作树；不要自动把全部历史请求列为待实施任务。

若继续发行验收，优先补干净 Windows 和人工代表性流程验证，按既有报告追加证据。若继续工程整理，按架构完成条件逐项找实现与测试证据，不要仅按目录名称宣布重构完成。

测试布局引用 `tests/README.md`。Windows VTK/OpenGL 压力测试应使用子进程隔离，相关入口为 `tests/stress/run_implicit_stress_matrix.py` 和 `tests/stress/implicit_pipeline_worker.py`；先读参数和场景定义再运行，不要无边界启动所有内存临界场景。针对性测试通过后，只有新改动或未解问题才扩大测试。

此前用户要求从 GitHub 下载 CodeGraph skill，随后中断该轮。当前能看到 `add-lang`、`agent-eval` 技能目录，但这不足以证明所需 CodeGraph skill 或 MCP 已完整安装配置。此项未在本次完成或验证；若用户继续要求，先确认已有安装与准确来源，避免重复安装。

## Suggested skills

按任务需要选择，先检查新会话实际可用技能并完整阅读对应 `SKILL.md`：

- `diagnosing-bugs`：卡死、闪退、GPU 回退或性能异常的复现与定位。
- `codebase-design`：模块职责、接口和依赖整理。
- `domain-modeling`：维护 `CONTEXT.md` 与 ADR，避免重复领域概念。
- `computer-use:computer-use`：安装版真实界面验收；遵守用户停止操作信号。
- `code-review`：有明确比较基线时审查改动，不把全体脏工作区误归入单次任务。
- `skill-installer`：继续 GitHub skill 安装请求。
- `nature-academic-search`：需要恢复晶格过渡文献调研时使用。
- `handoff`：再次交接时更新交接摘要；用户此次指定的技能位于 `%USERPROFILE%\.codex\skills\handoff\SKILL.md`。

历史提到的 `grill-with-docs` 不在本轮技能列表中，不能假定新会话可用。用户既有确认继续有效，不应因为切换会话再次进行整套需求盘问。

## 新对话可直接使用的开场提示

> 请先阅读附件 Project-Handoff.md，再读取仓库 AGENTS.md、README.md 和 docs/ENGINEERING_ARCHITECTURE.md，按本次任务阅读相关领域文档。当前项目入口为 main.py / lattice_studio，工作树包含未提交的工程迁移，请保留。以当前代码和已记录的验证证据为准，区分历史目标与已实现功能。接下来我要进行的任务是：【填写具体任务】。
