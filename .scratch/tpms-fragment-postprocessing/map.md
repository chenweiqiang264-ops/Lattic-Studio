# TPMS 悬空结构后处理工作项

- [spec](spec.md)：后处理目标、约束和验收标准。
- [01](issues/01-field-first-floating-component-cleanup.md)：隐式场优先的悬空小分量清理，已完成。

## Decisions so far

- 权威几何仍是隐式求值器；场清理只修改本次导出采样副本，不修改权威求值器。
- “悬空小结构”与“真实多连通晶格”必须分开处理；后者进入质量警告和后续连接策略。
