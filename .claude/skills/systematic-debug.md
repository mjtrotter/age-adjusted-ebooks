---
name: systematic-debug
description: Systematic debugging - hypothesis, reproduce, narrow, verify
triggers:
  - "debug"
  - "fix bug"
  - "not working"
  - "error"
  - "failing"
---

# Systematic Debugging Skill

## When to Use
- Something isn't working as expected
- Test failures
- Runtime errors
- Unexpected behavior

## Process

### Step 1: Gather Information
Before making any changes:

```
1. Read the error message completely
2. Identify the failing file and line number
3. Read the relevant code (use Read tool)
4. Check recent changes (git diff, git log)
5. Understand expected vs actual behavior
```

### Step 2: Form Hypothesis
Based on information gathered:

```
1. What could cause this specific error?
2. List 2-3 most likely causes
3. Rank by probability
4. Start with most likely
```

### Step 3: Reproduce
Create minimal reproduction:

```
1. Write a test that demonstrates the bug
2. Or create a minimal script that triggers it
3. Confirm you can reproduce consistently
4. Document reproduction steps in _scratch/test_failures.md
```

### Step 4: Narrow Down
Isolate the root cause:

```
1. Add logging/print statements at key points
2. Check inputs and outputs at each stage
3. Binary search: comment out half the code
4. Identify the exact line/condition that fails
```

### Step 5: Fix
Make minimal change:

```
1. Fix only the root cause
2. Don't refactor surrounding code
3. Don't add "while I'm here" improvements
4. Keep the fix focused and reviewable
```

### Step 6: Verify
Confirm the fix works:

```
1. Run the reproduction test - should pass
2. Run full test suite - no regressions
3. Test edge cases related to the fix
4. Update any relevant documentation
```

## Anti-Patterns to Avoid

❌ **Shotgun debugging** - Making random changes hoping something works
❌ **Blame external** - Assuming it's a library bug without evidence
❌ **Fix symptoms** - Adding try/except without understanding cause
❌ **Over-fix** - Refactoring entire module to fix one bug

## Delegation Pattern

For investigation (not fixing):

```python
# Delegate code analysis to gather context
delegate_code(
    task="Analyze why this function might return None unexpectedly.
          List possible causes based on the code flow.",
    context_files=["src/module.py", "tests/test_module.py"],
    output_path="_scratch/analysis.md",
    executor="qwen"  # Use QwenAgent for analysis (preserves quota)
)
```

## Logging Template

Add to `_scratch/test_failures.md`:

```markdown
## Bug: [Brief description]
**Date:** [Today]
**Error:** [Error message]
**File:** [File:line]

### Reproduction
[Steps or test code]

### Hypothesis
1. [Most likely cause]
2. [Second possibility]
3. [Third possibility]

### Investigation
- [x] Checked input values: [findings]
- [x] Added logging at line X: [findings]
- [ ] Binary search narrowing

### Root Cause
[What was actually wrong]

### Fix
[What was changed and why]
```
