# Scratchpad

## Purpose
Ephemeral files for session coordination between Claude and delegated agents.

## Contents
| File | Purpose |
|------|---------|
| current_task.md | What's actively being worked on |
| delegation_log.md | Record of delegated tasks and results |
| test_failures.md | Current test failures being debugged |

## For AI Agents

### Usage Rules
1. **Write freely** - This is scratch space, not permanent docs
2. **Clean up** - Delete files at session end
3. **Coordinate** - Use for multi-agent handoffs
4. **Don't commit** - Add to .gitignore

### current_task.md
```markdown
# Current Task
[What you're working on]

## Context
[Key files, decisions, constraints]

## Progress
- [x] Step 1
- [ ] Step 2
- [ ] Step 3
```

### delegation_log.md
```markdown
## [Timestamp] Delegated to [Executor]
**Task:** [Brief description]
**Context files:** [List]
**Output:** [Path or summary]
**Result:** [Success/Failed + notes]
```

### test_failures.md
Use format from systematic-debug skill.

### Cleanup
At session end:
```bash
rm -rf _scratch/*.md
# Or keep for next session if work continues
```
