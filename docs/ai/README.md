# AI Docs Router

Use this file to decide what to read for a task.

## Always Read

1. `../../AGENTS.md`
2. `principles.yaml`
3. `error_contract.md`

## Read By Task

| Task | Read |
|------|------|
| Exchange/API integration, pagination, data fetch | `exchange_api_quirks.md` |
| Rust/PyO3 build, extension loading, Rust test execution | `build_pitfalls.md` |
| Logging behavior/levels/tags | `logging_guide.md` |
| Order logic, risk, Python/Rust boundary | `architecture.md`, `decisions.md` |
| Code review | `code_review_prompt.md` |
| Common implementation mistakes | `pitfalls.md` |
| Running tools/tests/backtests/optimizer | `commands.md` |
| Feature-specific changes | `features/README.md` + relevant file in `features/` |
| Deep incident context | `debugging_case_studies.md`, `suite_optimizer_memory_investigation.md` |

## Rules For AI Docs

1. Keep core docs short and normative.
2. Avoid repeating the same rule across many files.
3. Keep historical notes separate from core instructions.
4. Include code references only where they add non-obvious context.
