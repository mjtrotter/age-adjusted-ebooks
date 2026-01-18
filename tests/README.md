# Tests

## Purpose
Test files mirroring the `src/` structure.

## Contents
| File | Tests For |
|------|-----------|
| (Add test files as you create them) | |

## For AI Agents

### Running Tests
```bash
# All tests
{{TEST_COMMAND}}

# Specific file
{{TEST_COMMAND}} tests/test_specific.py

# With coverage
{{TEST_COMMAND}} --cov=src --cov-report=html
```

### Test Structure
```python
def test_function_name():
    # Given - setup
    input_data = ...

    # When - action
    result = function(input_data)

    # Then - assertion
    assert result == expected
```

### Naming Convention
- Test files: `test_<module>.py`
- Test functions: `test_<function>_<scenario>`
- Example: `test_validate_email_with_empty_string`

### Coverage Target
- Minimum: 80%
- Goal: 90%+

### Common Fixtures
(Document shared fixtures here)

### Mocking Patterns
(Document how to mock external services)
