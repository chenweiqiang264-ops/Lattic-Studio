# Agent Instructions

## Agent skills

### Issue tracker

Issues 使用 Local Markdown，存放在 `.scratch/<feature>/`。详见 `docs/agents/issue-tracker.md`。

### Triage labels

使用默认 canonical labels：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。详见 `docs/agents/triage-labels.md`。

### Domain docs

使用 single-context 布局：根目录 `CONTEXT.md` 和 `docs/adr/`。详见 `docs/agents/domain.md`。

### Project handoff

交接文档存放在 `docs/Project-Handoff.md`。每次完成或进行新的项目交接时，必须
同步更新该文档，明确当前入口、工作区状态、已验证内容、未完成风险和后续建议。
