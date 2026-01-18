# age-adjusted-ebooks

## Stack
- Language: Python
- Test: `pytest tests/`

## Coding Rules
- Type hints required
- No hardcoded paths or credentials
- Error handling for all external calls
- Follow existing patterns in codebase

## Test Policy
- Run tests before commit: `pytest tests/`
- Coverage target: 80%
- TDD for new features

## Delegation Rules

**Claude (orchestrator) handles:**
- Planning and architecture decisions
- Updating documentation (READMEs, STATUS.md)
- Code review and final approval

**Delegate to executors:**
- Code generation (>20 lines) → use `delegate_code` tool
- Research queries → use `delegate_research` tool
- Always provide `context_files` to prevent hallucination

## MCP Tools
```
delegate_code(task, context_files, output_path, executor="auto")
delegate_research(query, output_path, mode="deep")
delegation_status()
run_tests(test_command, working_dir)
```

## Project Notes
Age-appropriate ebook content adjustment system
