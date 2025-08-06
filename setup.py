"""
Setup script for Mars Resource Management Game simulation framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
else:
    requirements = [
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "openai>=1.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "tqdm>=4.64.0"
    ]

setup(
    name="mars-simulation",
    version="1.0.0",
    author="Your Research Team",
    author_email="your-email@example.com",
    description="AI simulation framework for Mars resource management game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/mars-simulation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "analysis": [
            "scikit-learn>=1.1.0",
            "statsmodels>=0.13.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "mars-simulate=scripts.run_simulation:main",
            "mars-batch=scripts.batch_experiments:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["data/config/*.yaml", "data/config/*.csv"],
    },
    zip_safe=False,
)
