#!/bin/bash
# Delegation Toolkit Setup Script
# Run this after cloning the template to configure your project

set -e

echo "🛠️  Delegation Toolkit Setup"
echo "============================"
echo ""

# Get project info
read -p "Project name: " PROJECT_NAME
read -p "Brief description: " PROJECT_DESCRIPTION
read -p "Language (python/typescript/other): " LANGUAGE
read -p "Framework (if any): " FRAMEWORK
read -p "Database (if any, or 'none'): " DATABASE
read -p "Test command (e.g., 'pytest tests/'): " TEST_COMMAND
read -p "Build command (e.g., 'npm run build'): " BUILD_COMMAND
read -p "Install command (e.g., 'pip install -e .'): " INSTALL_COMMAND
read -p "Dev command (e.g., 'npm run dev'): " DEV_COMMAND

# Create CLAUDE.md from template
echo ""
echo "📝 Creating CLAUDE.md..."
sed -e "s/{{PROJECT_NAME}}/$PROJECT_NAME/g" \
    -e "s/{{LANGUAGE}}/$LANGUAGE/g" \
    -e "s/{{FRAMEWORK}}/$FRAMEWORK/g" \
    -e "s/{{DATABASE}}/$DATABASE/g" \
    -e "s/{{TEST_COMMAND}}/$TEST_COMMAND/g" \
    -e "s/{{BUILD_COMMAND}}/$BUILD_COMMAND/g" \
    -e "s/{{PROJECT_NOTES}}/Add project-specific notes here/g" \
    CLAUDE.md.template > CLAUDE.md

# Update README.md
echo "📝 Updating README.md..."
sed -i '' -e "s/{{PROJECT_NAME}}/$PROJECT_NAME/g" \
    -e "s/{{PROJECT_DESCRIPTION}}/$PROJECT_DESCRIPTION/g" \
    -e "s/{{INSTALL_COMMAND}}/$INSTALL_COMMAND/g" \
    -e "s/{{TEST_COMMAND}}/$TEST_COMMAND/g" \
    -e "s/{{DEV_COMMAND}}/$DEV_COMMAND/g" \
    -e "s/{{LICENSE}}/MIT/g" \
    README.md 2>/dev/null || true

# Update other templates
echo "📝 Updating folder READMEs..."
find . -name "README.md" -exec sed -i '' -e "s/{{TEST_COMMAND}}/$TEST_COMMAND/g" {} \; 2>/dev/null || true

# Update STATUS.md with today's date
TODAY=$(date +%Y-%m-%d)
sed -i '' -e "s/{{DATE}}/$TODAY/g" planning/STATUS.md 2>/dev/null || true

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
.venv/
venv/
.env

# Node
node_modules/
dist/
.next/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Project
_scratch/*.md
!_scratch/README.md
*.log
.claude/delegation_quota.json
EOF

# Install FastMCP if Python project
if [ "$LANGUAGE" = "python" ]; then
    echo ""
    echo "📦 Installing FastMCP..."
    pip install fastmcp 2>/dev/null || echo "⚠️  Could not install fastmcp. Install manually: pip install fastmcp"
fi

# Cleanup
rm -f CLAUDE.md.template

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review and customize CLAUDE.md"
echo "2. Set up your development environment"
echo "3. Run: claude to start coding with AI assistance"
echo ""
echo "Available delegation tools:"
echo "  - delegate_code(task, context_files, output_path)"
echo "  - delegate_research(query, output_path)"
echo "  - delegation_status()"
echo "  - run_tests(test_command)"
