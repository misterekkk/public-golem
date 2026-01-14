# HisTree: High-Performance Histogram-Based Ensemble Library
[![Python 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Flake8 Linting](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml)
[![Tests](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

HisTree is a high-performance machine learning library implementing advanced histogram-based ensemble algorithms (Random Forest, Gradient Boosting, and **Sequential Weighted Forest**) from scratch.

Built with Python and optimized using **Numba (JIT)**, this project bridges the gap between readable Python code and C++ level performance. It features state-of-the-art engineering techniques such as **in-place data partitioning, histogram subtraction, and parallel processing**, achieving training speeds comparable to LightGBM and significantly outperforming standard scikit-learn implementations.

Designed as a **technical recruitment project** for the **GOLEM Student Research Group**, this library demonstrates a deep understanding of low-level algorithm optimization, memory management, and CPU cache utilization.

## Key Features

### 1. Algorithms & Models
* **Histogram Random Forest (HRF):** A high-performance Bagging ensemble that replaces expensive exact sorting with histogram-based split finding. This reduces computational complexity from $O(N \log N)$ to $O(N)$, offering significantly faster training and lower memory footprint compared to standard implementations.
* **Histogram Gradient Boosting (HGB):** An efficient GBDT implementation utilizing depth-wise tree growth combined with GOSS (Gradient-based One-Side Sampling). It optimizes training speed by focusing on samples with large gradients while maintaining a low memory footprint through efficient histogram subtraction.
* **Histogram Sequential Weighted Forest (HSWF):** An efficient implementation of Histogram-based AdaBoost. It trains trees sequentially, utilizing adaptive sample weighting to focus on difficult cases, combined with weight trimming to accelerate training by ignoring negligible samples.

### 2. Engineering & Optimization
* **Contiguous Memory Processing:** Ensures that data subsets used in heavy computations are stored as contiguous blocks in memory. This reduces the overhead of random memory access compared to using raw unsorted indices.
* **Smart Histogram Subtraction:** Accelerates training by ~50%. Instead of calculating statistics from scratch for every node, the algorithm reuses parent data to instantly derive child node values.
* **Native Machine Code (JIT):** Uses Numba to compile critical Python functions into optimized machine code, allowing the library to run at the speed of C++ while maintaining Python's simplicity.
* **Multi-Core Scaling:** Automatically distributes heavy computations across all available CPU cores, maximizing hardware utilization for faster training.

### 3. Compatibility
* **Scikit-Learn API:** Fully compatible with `sklearn` estimators. Supports `.fit()`, `.predict()`, `Pipeline`, and `GridSearchCV`.

## Benchmarks

Tests performed on a synthetic dataset (1,000,000 samples, 50 features).

| Model | Training Time (s) | RMSE | Performance Note |
| :--- | :--- | :--- | :--- |
| **HisTree HRF** | **20.53s** | 44.78 | **~20x faster** than sklearn `RandomForestRegressor` (417s) |
| **HisTree HGB** | **4.92s** | 18.62 | Competitive with **LightGBM** (3.70s) |
| **HisTree HSWF** | **7.84s** | 31.65 | **~73x faster** than sklearn `AdaBoostRegressor` (569s) |

*Note: Benchmarks may vary depending on hardware configuration.*

*Benchmarks performed on: MacBook M4 Pro, 16GB RAM*

## Project Structure

```text
Projekt-Wstepny-GOLEM/
├── notebooks/      
├── src/
│   └── histree/    
│       ├── gb/     
│       ├── rf/     
│       ├── swf/    
│       └── __init__.py
├── tests/          
├── .flake8         
├── LICENSE         
├── pyproject.toml  
├── README.md       
└── requirements.txt
```
## Installation
Clone the repository and install the required dependencies:
```bash

git clone https://github.com/misterekkk/public-golem.git
cd public-golem

# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

pip install -e .

```

## Usage Example
Histogram-Based Random Forest

```python
from histree import HistogramRandomForestRegressor

# Initialize efficient histogram-based RF
model = HistogramRandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features=0.9,
    n_bins=64,           # Number of histogram bins
    bootstrap=True,
    n_jobs=-1,           # Use all CPU cores
    random_state=42
)

# Train and predict (Standard sklearn API)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Histogram-Based Gradient Boosting

```python
from histree import HistogramGradientBoostingRegressor

model = HistogramGradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=20,
    reg_lambda=1.0,      # L2 Regularization
    n_bins=64,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Histogram-Based Sequential Weighted Forest

```python
from histree import HistogramSequentialWeightedForestRegressor

model = HistogramSequentialWeightedForestRegressor(
    n_estimators=100,
    learning_rate=1.0,
    max_depth=12,
    reg_lambda=1.0,
    n_bins=64,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Requirements

- Python 3.12+
- numpy
- pandas
- scikit-learn
- numba
- lightgbm (optional, for benchmarking comparisons)

## License

Distributed under the [MIT License](https://opensource.org/licenses/MIT). See [LICENSE](LICENSE) for more information.