# Bostan_house_prediction_swatantra-

This project implements a modular machine learning workflow to predict house prices using the Boston Housing dataset. This repository contains a complete ML pipeline to predict house prices using the Boston Housing dataset.
The project is modular, version-controlled, and automated using GitHub Actions.

📌 Instructions
Dataset is loaded manually from the UCI repository.
Regression models: Linear Regression, Decision Tree, Random Forest.
CI/CD setup via GitHub Actions.
📁 Branches
main: Contains only this README initially.
reg: Implements basic regression models.
hyper: Adds hyperparameter tuning.
🚀 Setup
conda create -n mlops_a1 python=3.10
conda activate mlops_a1
pip install -r requirements.txt
---

## 📂 Branch Overview

| Branch Name | Description |
|-------------|-------------|
| `main`      | Contains only README initially, later merged code |
| `reg`       | Implements 3 regression models and evaluation (MSE, R²) |
| `hyper`     | Includes hyperparameter tuning using `GridSearchCV` |

---

## 📈 Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

## 🔍 Hyperparameter Tuning (`GridSearchCV`)

| Model              | Tuned Hyperparameters |
|-------------------|------------------------|
| Decision Tree      | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Random Forest      | `n_estimators`, `max_depth`, `min_samples_split` |
| Linear Regression  | No tunable params, included for baseline |

---

## 🧪 Performance Metrics (Example Output)

> Replace these values with actual output from your GitHub Actions
