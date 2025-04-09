# 🧪 Grid Search Backtester for IMC Prosperity

This script performs a **grid search** over algorithm parameters to find optimal configurations for your trading strategy in the IMC Prosperity competition.

It works by replacing parameter placeholders in a template algorithm file, running a backtest for each combination, and collecting profits for comparison.

---

## 📁 Requirements

- Python 3.10+
---

## 🧩 How It Works

1. You define a **parameter grid** as a dictionary.
2. The script generates every combination of parameters.
3. It replaces `{{PARAM}}` placeholders in your strategy file with values from the grid.
4. It writes each version to a temporary file.
5. It runs the backtest using `prosperity3bt`.
6. It returns a list of results with corresponding profits.

---

## 🛠️ Setup

3. Prepare your strategy template with placeholders like `{{PARAM1}}`, `{{banana}}`, etc. In sample_algorithm.py, ctrl+f for `{{PARAM1}}` and `{{PARAM2}}` to find all placeholders.