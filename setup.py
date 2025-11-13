"""Setup script for JunGrad."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="jungrad",
    version="0.1.0",
    description="A robust N-D autograd library with comprehensive ops, NN layers, and optimizers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="JunGrad Contributors",
    url="https://github.com/junkim100/JunGrad",
    packages=find_packages(exclude=["tests", "examples", "*.tests", "*.tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "black>=24.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "build>=1.2.1",
            "twine>=4.0.2",
        ],
        "sparse": [
            "scipy>=1.10.0",
        ],
        "viz": [
            "graphviz>=0.20.0",
        ],
        "tutorial": [
            "datasets>=4.4.0",
            "matplotlib>=3.7.0",
            "scikit-learn>=1.2.0",
            "ipykernel>=6.0.0",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/junkim100/JunGrad",
        "Documentation": "https://github.com/junkim100/JunGrad#readme",
        "Issues": "https://github.com/junkim100/JunGrad/issues",
        "Source": "https://github.com/junkim100/JunGrad",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
