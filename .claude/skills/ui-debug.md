---
name: ui-debug
description: Debug UI/frontend issues using Chrome extension
triggers:
  - "UI issue"
  - "component not rendering"
  - "frontend bug"
  - "React"
  - "display issue"
  - "CSS"
  - "layout"
---

# UI Debug Skill

## When to Use
- Component not rendering correctly
- Layout/CSS issues
- Interactive elements not working
- Console errors in browser
- Visual discrepancies

## Prerequisites

Chrome extension tools available:
- `mcp__claude-in-chrome__tabs_context_mcp`
- `mcp__claude-in-chrome__navigate`
- `mcp__claude-in-chrome__read_page`
- `mcp__claude-in-chrome__read_console_messages`
- `mcp__claude-in-chrome__computer` (screenshot)
- `mcp__claude-in-chrome__javascript_tool`

## Process

### Step 1: Get Browser Context

```
1. Call mcp__claude-in-chrome__tabs_context_mcp to get tab info
2. Navigate to the affected page
3. Take a screenshot to see current state
```

### Step 2: Check Console Errors

```python
# Get console errors
mcp__claude-in-chrome__read_console_messages(
    tabId=<tab_id>,
    onlyErrors=True,
    pattern="error|warning|undefined"
)
```

Common issues:
- `undefined is not a function` → Missing import or wrong prop
- `Cannot read property of null` → Component not mounted or bad selector
- `Failed to fetch` → API endpoint issue

### Step 3: Inspect DOM Structure

```python
# Read page accessibility tree
mcp__claude-in-chrome__read_page(
    tabId=<tab_id>,
    filter="all",
    depth=10
)

# Or find specific elements
mcp__claude-in-chrome__find(
    tabId=<tab_id>,
    query="button with text Submit"
)
```

### Step 4: Check Component State (React)

```python
# Execute JavaScript to inspect React state
mcp__claude-in-chrome__javascript_tool(
    tabId=<tab_id>,
    action="javascript_exec",
    text='''
    // Find React component (works with React DevTools)
    const element = document.querySelector('[data-testid="my-component"]');
    const fiber = element?._reactRootContainer?._internalRoot?.current;
    console.log('Component state:', fiber?.memoizedState);
    '''
)
```

### Step 5: Test Interactions

```python
# Click and observe
mcp__claude-in-chrome__computer(
    tabId=<tab_id>,
    action="left_click",
    coordinate=[x, y]
)

# Wait and take screenshot
mcp__claude-in-chrome__computer(
    tabId=<tab_id>,
    action="wait",
    duration=1
)

mcp__claude-in-chrome__computer(
    tabId=<tab_id>,
    action="screenshot"
)
```

## Common UI Issues & Solutions

### Component Not Rendering
1. Check if component is imported correctly
2. Check conditional rendering logic
3. Check if data is loaded (loading state)
4. Check for errors in parent components

### Styling Issues
1. Check CSS specificity conflicts
2. Check for missing CSS imports
3. Check responsive breakpoints
4. Inspect computed styles in DevTools

### Event Handlers Not Working
1. Check if handler is bound correctly
2. Check for event propagation issues (stopPropagation)
3. Check if element is disabled or hidden
4. Check for overlapping elements blocking clicks

## Delegation Pattern

For quick Haiku analysis of screenshots:

```python
# Take screenshot first, then analyze with vision model
# Note: Delegate to Haiku for fast, cheap analysis
delegate_code(
    task="Analyze this UI screenshot. Identify:
          1. Any obvious layout issues
          2. Missing elements
          3. Broken styling
          [Include screenshot reference]",
    context_files=["src/components/MyComponent.tsx"],
    output_path="_scratch/ui_analysis.md",
    executor="qwen"  # Fast local analysis
)
```

## Network Request Debugging

```python
# Check API calls
mcp__claude-in-chrome__read_network_requests(
    tabId=<tab_id>,
    urlPattern="/api/"
)
```

Look for:
- Failed requests (4xx, 5xx status)
- Missing requests (API not called)
- Wrong request payload
- CORS errors
