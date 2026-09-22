# Test layout

- `unit/`: 纯领域规则、持久化、入口和 GPU 能力选择。
- `integration/`: 数值工作流、隐式场和适配器契约。
- `acceptance/`: 可执行的代表性用户流程，可能生成图片或 STL。
- `stress/`: 子进程隔离的崩溃、GPU、VTK 和内存压力场景。
- `benchmarks/`: 只手动运行的性能测量，不属于快速测试。
- `manual/`: 需要人工交互的旧式验证脚本。
- `legacy/`: 保留的历史测试源码，仅用于追溯，不属于当前测试入口。

所有生成产物统一写入 `build/test-artifacts/`，测试源码目录不保存图片、日志或模型。
