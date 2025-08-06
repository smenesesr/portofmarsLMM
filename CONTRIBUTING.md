# Contributing to Mars Simulation Framework

Thank you for your interest in contributing to the Mars Resource Management Game simulation framework! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- OpenAI API key (for running simulations)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/mars-simulation.git
   cd mars-simulation
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use Black for code formatting: `black src/ scripts/ tests/`
- Use type hints where appropriate
- Write docstrings for all public functions and classes

### Testing
- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Use pytest for testing framework

### Documentation
- Update README.md if adding new features
- Add docstrings to new functions/classes
- Update configuration examples if needed
- Include usage examples for new functionality

## 🔄 Contribution Workflow

### 1. Create an Issue
Before starting work, create an issue describing:
- The problem you're solving
- Proposed solution approach
- Any breaking changes

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 3. Make Changes
- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed
- Follow existing code patterns

### 4. Test Your Changes
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_game_engine.py

# Run with coverage
pytest --cov=src tests/

# Format code
black src/ scripts/ tests/
flake8 src/ scripts/ tests/
```

### 5. Commit Changes
Use clear, descriptive commit messages:
```bash
git add .
git commit -m "feat: add risk-aware agent implementation"
# or
git commit -m "fix: resolve health calculation bug in game engine"
# or
git commit -m "docs: update API documentation for new features"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots/examples if applicable
- Checklist of completed items

## 🎯 Types of Contributions

### 🐛 Bug Fixes
- Fix existing functionality that isn't working correctly
- Include test cases that reproduce the bug
- Explain the root cause in PR description

### ✨ New Features
- Add new AI agents or game mechanics
- Implement new analysis capabilities
- Add support for new models or APIs
- Include comprehensive tests and documentation

### 📚 Documentation
- Improve README or other documentation
- Add code examples and tutorials
- Fix typos or unclear explanations
- Add API documentation

### 🔧 Infrastructure
- Improve CI/CD pipeline
- Add new testing capabilities
- Optimize performance
- Improve code organization

### 📊 Analysis & Research
- Add new visualization capabilities
- Implement statistical analysis methods
- Create example experiments
- Add benchmark datasets

## 🧪 Experiment Contributions

### Adding New Experiments
1. Create experiment configuration in `experiments/configs/`
2. Add analysis notebook in `notebooks/experiments/`
3. Document methodology and results
4. Include reproducible setup instructions

### Example Experiment Structure
```
experiments/
├── configs/
│   └── your_experiment.yaml
├── notebooks/
│   └── your_experiment_analysis.ipynb
└── results/
    └── your_experiment_results.csv
```

## 📝 Code Review Process

### For Contributors
- Respond to feedback promptly
- Make requested changes in separate commits
- Keep PR scope focused and manageable
- Test changes thoroughly

### For Reviewers
- Be constructive and specific in feedback
- Test the changes locally when possible
- Check for code style and documentation
- Verify tests are comprehensive

## 🏷️ Release Process

### Version Numbering
We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number bumped
- [ ] Changelog updated
- [ ] Release notes prepared

## 🤝 Community Guidelines

### Be Respectful
- Use inclusive language
- Be patient with newcomers
- Provide constructive feedback
- Respect different perspectives

### Communication Channels
- **Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Pull Requests**: Code contributions
- **Email**: Sensitive or private matters

## 📋 Checklist for Pull Requests

Before submitting a PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] PR description explains the changes
- [ ] No sensitive information (API keys, etc.) included
- [ ] Breaking changes are clearly marked

## 🆘 Getting Help

If you need help:

1. **Check existing documentation** - README, docs/, and code comments
2. **Search existing issues** - Your question might already be answered
3. **Create a discussion** - For general questions about usage
4. **Create an issue** - For specific bugs or feature requests
5. **Contact maintainers** - For sensitive or urgent matters

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to the Mars Simulation Framework! 🚀
