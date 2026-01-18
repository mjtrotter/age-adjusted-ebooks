# {{PROJECT_NAME}}

> {{PROJECT_DESCRIPTION}}

## Quick Start

```bash
# Install dependencies
{{INSTALL_COMMAND}}

# Run tests
{{TEST_COMMAND}}

# Start development
{{DEV_COMMAND}}
```

## Project Structure

```
{{PROJECT_NAME}}/
├── .claude/              # AI orchestration config
│   ├── settings.json     # MCP servers and tools
│   ├── delegation/       # Delegation toolkit
│   └── skills/           # Reusable skills (TDD, debug, etc.)
├── src/                  # Source code
├── tests/                # Test files
├── planning/             # Project status and session logs
├── research/             # Research queries and results (if needed)
└── _scratch/             # Ephemeral session coordination
```

## Development Workflow

1. **Check status:** Read `planning/STATUS.md`
2. **Understand code:** Read folder READMEs before modifying
3. **Use TDD:** Write tests first for new features
4. **Delegate:** Use `delegate_code` for code generation
5. **Update docs:** Update READMEs when adding/changing files

## AI Orchestration

This project uses the Delegation Toolkit for AI-assisted development:

- **Jules:** Complex multi-file changes (15/day)
- **Gemini CLI:** Fast code generation (1500/day)
- **QwenAgent:** Local fallback (unlimited)
- **Perplexity:** Research queries

See `CLAUDE.md` for detailed delegation rules.

## License

{{LICENSE}}
