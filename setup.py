#!/usr/bin/env python3
"""Setup script for SIGLA package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sigla",
    version="0.2.0",
    author="SIGLA Team",
    description="Lightweight knowledge capsule management system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.9.0",
        "transformers>=4.20.0",
    ],
    extras_require={
        "core": [
            "faiss-cpu>=1.7.0",
            "sentence-transformers>=2.2.0",
        ],
        "server": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
        ],
        "llm": [
            "transformers>=4.21.0",
            "torch>=1.12.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "all": [
            "faiss-cpu>=1.7.0",
            "sentence-transformers>=2.2.0",
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "transformers>=4.21.0",
            "torch>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sigla=sigla.__main__:main",
        ],
    },
) 