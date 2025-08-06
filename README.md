# Mars Resource Management Game - AI Simulation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive framework for simulating AI behavior in resource management scenarios using the Mars colonization game paradigm.

## 🎯 Overview

This project simulates AI decision-making in a resource allocation game where agents must balance:
- **System Health**: Maintaining colony infrastructure
- **Point Collection**: Maximizing rewards
- **Risk Management**: Handling random events and uncertainty

## 🚀 Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd mars-simulation

# Install dependencies
pip install -r requirements.txt

# Run a basic simulation
python scripts/run_simulation.py --model gpt-4 --runs 10

# Launch interactive analysis
jupyter lab notebooks/analysis/
```

## 📁 Project Structure

```
mars-simulation/
├── 📁 src/                    # Core source code
│   ├── game_engine.py         # Main game mechanics
│   ├── ai_agents.py          # AI agent implementations
│   ├── data_processing.py    # Data handling utilities
│   └── visualization.py     # Plotting and analysis tools
├── 📁 scripts/               # Executable scripts
│   ├── run_simulation.py     # Main simulation runner
│   ├── batch_experiments.py  # Batch processing
│   └── data_analysis.py     # Analysis pipeline
├── 📁 notebooks/             # Jupyter notebooks for exploration
│   ├── exploration/          # Data exploration
│   ├── experiments/          # Specific experiments
│   └── analysis/            # Results analysis
├── 📁 data/                  # Data files
│   ├── config/              # Game configuration
│   ├── raw/                 # Raw experimental data
│   └── processed/           # Processed datasets
├── 📁 results/               # Experimental results
├── 📁 tests/                 # Unit tests
└── 📁 docs/                  # Documentation
```

## 🎮 Game Mechanics

### Core Rules
- **Starting Health**: 15 units
- **Maximum Health**: 25 units
- **Investment Range**: 0-10 units per round
- **Automatic Decay**: -5 health per round
- **Win Condition**: Maximize points over 11 rounds
- **Fail Condition**: Health reaches 0 (score becomes 0)

### Event System
- **30 unique events** with varying impacts
- **Dynamic event frequency** based on system health:
  - Health 16-25: 1 event per round
  - Health 9-15: 2 events per round  
  - Health <9: 3 events per round

## 🤖 AI Models Supported

- **GPT-4** / **GPT-3.5-turbo**
- **Claude** (Anthropic)
- **Custom risk profiles** based on human behavioral data

## 📊 Key Features

- **Automated Simulation Pipeline**: Run hundreds of games efficiently
- **Risk Profile Integration**: Incorporate human behavioral data
- **Robust Result Extraction**: Parse AI outputs reliably
- **Comprehensive Analysis**: Statistical analysis and visualization
- **Batch Processing**: Handle multiple models and configurations
- **Interactive Exploration**: Jupyter notebooks for deep dives

## 🛠️ Usage Examples

### Basic Simulation
```python
from src.game_engine import MarsGame
from src.ai_agents import GPTAgent

# Initialize game and agent
game = MarsGame()
agent = GPTAgent(model="gpt-4", temperature=0.7)

# Run simulation
result = game.simulate(agent, rounds=11)
print(f"Final score: {result.score}")
```

### Batch Experiments
```bash
# Run multiple models with different configurations
python scripts/batch_experiments.py \
    --models gpt-4 gpt-3.5-turbo \
    --runs 50 \
    --output results/comparison_study.csv
```

### Risk Profile Analysis
```python
from src.data_processing import load_risk_profiles
from src.ai_agents import RiskAwareAgent

# Load human behavioral data
risk_profiles = load_risk_profiles("data/raw/human_responses.csv")

# Run simulations with different risk profiles
for risk_level in risk_profiles:
    agent = RiskAwareAgent(risk_level=risk_level)
    results = run_batch_simulation(agent, runs=100)
```

## 📈 Analysis Capabilities

- **Performance Metrics**: Score distribution, survival rates, strategy analysis
- **Model Comparison**: Statistical comparison across AI models
- **Risk Impact**: How risk tolerance affects decision-making
- **Strategy Patterns**: Investment behavior analysis
- **Event Response**: How agents react to different game events

## 🔧 Configuration

Game parameters can be configured via `data/config/game_config.yaml`:

```yaml
game:
  starting_health: 15
  max_health: 25
  rounds: 11
  investment_range: [0, 10]

simulation:
  default_runs: 50
  temperature: 0.7
  delay_between_calls: 1.0

models:
  gpt-4:
    api_key_env: "OPENAI_API_KEY"
    max_tokens: 1000
  gpt-3.5-turbo:
    api_key_env: "OPENAI_API_KEY"
    max_tokens: 800
```

## 🧪 Running Experiments

### 1. Single Model Test
```bash
python scripts/run_simulation.py --model gpt-4 --runs 10 --output test_results.csv
```

### 2. Model Comparison
```bash
python scripts/batch_experiments.py --config experiments/model_comparison.yaml
```

### 3. Risk Profile Study
```bash
python scripts/risk_analysis.py --data data/raw/human_responses.csv --runs 100
```

## 📊 Results and Visualization

Results are automatically saved in structured formats:
- **CSV files**: Raw simulation data
- **JSON files**: Experiment metadata
- **PNG files**: Automated visualizations

Access analysis notebooks:
```bash
jupyter lab notebooks/analysis/experiment_overview.ipynb
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Jupyter Lab (for notebooks)
- See `requirements.txt` for full dependencies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{mars_simulation_2024,
  title={Mars Resource Management Game - AI Simulation Framework},
  author={Your Team},
  year={2024},
  url={https://github.com/your-username/mars-simulation}
}
```

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/mars-simulation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/mars-simulation/discussions)
- **Documentation**: [Full Documentation](docs/)

---

**Built with ❤️ for AI research and behavioral analysis**
