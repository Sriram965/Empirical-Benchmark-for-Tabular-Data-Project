# Tabular Data Benchmark: When and Why Does Deep Learning Fail to Beat Gradient Boosting?

## Overview
A rigorous empirical benchmark comparing 9 machine learning model families 
across 60+ real-world tabular datasets. This project investigates under what 
conditions tree-based models outperform deep learning, and builds a meta-model 
to predict which algorithm will win on a new dataset.

## Models Compared
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- MLP (Multi-Layer Perceptron)
- ResNet
- TabNet
- FT-Transformer
- TabPFN

## Project Structure
- `data/` — raw downloaded datasets from OpenML
- `notebooks/` — exploratory analysis notebooks
- `src/` — reusable Python modules
- `results/` — benchmark results saved as parquet files
- `figures/` — publication-quality plots
- `report/` — final writeup
- `experiments/` — individual experiment runs with notes

## Status
🔄 Currently in Phase 2: Environment Setup and Project Structure

## Author
Yembadi Sri Ramvarma