# Density Ratio-based Causal Discovery from Bivariate Continuous-Discrete Data

This code package contains the implementation of **"Density Ratio-based Causal Discovery from Bivariate Continuous-Discrete Data" (DRCD)**, a novel method for causal discovery from mixed data.

---

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
  - [Using DRCD Only](#using-drcd-only)
  - [Running Experiments for DRCD Only](#running-experiments-for-drcd-only)
  - [Running Comparison Experiments](#running-comparison-experiments)
  - [Checking Assumptions for DRCD](#checking-assumptions-for-drcd)
- [Comparison Methods](#comparison-methods)

---

## Overview

This code package provides DRCD, a causal discovery method for bivariate data with one continuous and one discrete variable.
The code package includes both the implementation of DRCD and experimental code for evaluation on synthetic and real-world datasets.

---

## Quick Start

### Using DRCD Only

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Main components:
   - `DRCD.py`: Core implementation of the DRCD algorithm
   - `DRCD_Tutorial.ipynb`: Tutorial notebook demonstrating DRCD on the UCI Heart Disease Dataset

---

### Running Experiments for DRCD Only

1. Install the required packages:
   ```bash
   pip install -r requirements_expr_DRCDonly.txt
   ```

2. Run experiments:
   - **Recommended:**  
     Use the notebook:
     ```bash
     Experiments_DRCDonly.ipynb
     ```
   - **Alternative:**  
     Execute the Python scripts:
     ```bash
     python experiments_artificial_DRCD.py   # For synthetic data experiments
     python experiments_real_DRCD.py         # For real-world data experiments
     ```

3. Experiment outputs:
   - Synthetic data results:  
     LaTeX table at:
     ```
     ./results/artificial/accuracy_table_with_overall_DRCDonly.tex
     ```
   - Real-world data results:  
     Bipartite graph PDF at:
     ```
     ./results/real/bipartite_graphs_DRCDonly.pdf
     ```

---

### Running Comparison Experiments

1. Install the extended requirements:
   ```bash
   pip install -r requirements_expr_all.txt
   ```

2. Set up comparison methods:
   ```bash
   sh setup.sh
   ```
   This script will:
   - Clone and download necessary repositories
   - Build and compile required components
   - Set up all comparison methods

3. Additional prerequisites:
   - C++ installation with Python bindings
   - MATLAB installation with Python interface

4. Run experiments:
   - **Recommended:**  
     Use the notebook:
     ```bash
     Experiments.ipynb
     ```
   - **Alternative:**  
     Execute the Python scripts:
     ```bash
     python experiments_artificial.py   # For synthetic data experiments
     python experiments_real.py         # For real-world data experiments
     ```

5. Experiment outputs:
   - Synthetic data results:  
     LaTeX table at:
     ```
     ./results/artificial/accuracy_table_with_overall.tex
     ```
   - Real-world data results:  
     Bipartite graph PDF at:
     ```
     ./results/real/bipartite_graphs.pdf
     ```

---

### Checking Assumptions for DRCD

Before applying DRCD, it is important to verify that the underlying assumptions regarding the conditional distributions are satisfied.

In particular, DRCD may fail if the conditional distributions of the continuous variable, given different values of the discrete variable, are both normal or Laplace distributions with the same scale parameter but different means.

To assist with this verification, we provide the following notebook:

```bash
distribution_scale_and_type_test.ipynb
```

This notebook checks whether two samples (corresponding to different discrete variable values) have the same scale and whether their distributions are close to normal or Laplace distributions.
It serves as a supplementary tool to confirm the applicability of DRCD to your data.

---

## Comparison Methods

This repository incorporates the following comparison methods: some by obtaining their official implementations, and one implemented by ourselves:

- **LiM** (Causal discovery for linear mixed data)  
  - [Paper](https://proceedings.mlr.press/v177/zeng22a.html) | [Repository](https://github.com/cdt15/lingam)  
  - Language: Python

- **CRACK** (Classification and regression based packing of data)  
  - [Paper](https://dl.acm.org/doi/10.1007/978-3-030-10928-8_39) | [Repository](https://eda.rg.cispa.io/prj/crack/)  
  - Language: C++

- **HCM** (Hybrid Causal Structure Learning Algorithm for Mixed-type Data)  
  - [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20707) | [Repository](https://github.com/DAMO-DI-ML/AAAI2022-HCM)  
  - Language: Python

- **GSF** (Generalized Score Functions for Causal Discovery)  
  - [Paper](https://dl.acm.org/doi/10.1145/3219819.3220104) | [Repository](https://github.com/Biwei-Huang/Generalized-Score-Functions-for-Causal-Discovery)  
  - Language: MATLAB

- **MIC** (Mixed Causal Structure Discovery with Application to Prescriptive Pricing)  
  - [Paper](https://www.ijcai.org/proceedings/2018/711)  
  - Language: Python (Custom implementation without external packages)

