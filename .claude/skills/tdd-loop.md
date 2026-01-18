---
name: tdd-loop
description: Test-Driven Development loop - write tests first, then implement
triggers:
  - "implement with tests"
  - "TDD"
  - "test first"
  - "add feature with tests"
---

# TDD Loop Skill

## When to Use
- Implementing new features
- Adding new functions or classes
- User explicitly requests TDD approach

## Process

### Step 1: Understand Requirements
Read the relevant files to understand:
- What needs to be implemented
- Existing patterns and conventions
- Related test files

### Step 2: Write Tests First
Before any implementation:

```
1. Create/update test file in tests/ directory
2. Write test cases covering:
   - Happy path (expected behavior)
   - Edge cases (empty, null, boundaries)
   - Error cases (invalid input, failures)
3. Run tests - they should FAIL (red phase)
```

### Step 3: Implement Minimum Code
Write the simplest code that makes tests pass:

```
1. Focus only on passing the current failing test
2. Don't over-engineer or add extra features
3. Run tests after each change
4. Continue until all tests pass (green phase)
```

### Step 4: Refactor
With passing tests as safety net:

```
1. Improve code structure
2. Remove duplication
3. Improve naming
4. Run tests after each refactor
5. Tests must stay green
```

### Step 5: Document
Update relevant READMEs with new functionality.

## Delegation Pattern

For complex implementations, delegate the code generation:

```python
# First, write tests yourself (orchestrator)
# Then delegate implementation:
delegate_code(
    task="Implement function X that passes these tests: [paste test code]",
    context_files=["tests/test_x.py", "src/related_module.py"],
    output_path="src/x.py"
)
```

## Example Workflow

User: "Add a function to validate email addresses"

1. Create `tests/test_email_validator.py`:
   ```python
   def test_valid_email():
       assert validate_email("user@example.com") == True

   def test_invalid_email_no_at():
       assert validate_email("userexample.com") == False

   def test_empty_email():
       assert validate_email("") == False
   ```

2. Run: `pytest tests/test_email_validator.py` → FAIL

3. Implement `src/email_validator.py`

4. Run: `pytest tests/test_email_validator.py` → PASS

5. Update `src/README.md` with new module
