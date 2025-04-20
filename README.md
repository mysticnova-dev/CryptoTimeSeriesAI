

# 🧠 Cryptocurrency Forecasting with Deep Learning and Machine Learning

This project forecasts short-term Bitcoin price movements using engineered blockchain indicators and state-of-the-art modeling techniques:

- 🔁 **LSTM** – Memory-based sequential modeling
- 🧠 **Transformer** – Attention-based deep learning for temporal data
- ⚡ **XGBoost** – High-performance gradient boosting as a baseline

The models are trained with temporal data, extensive preprocessing, and hyperparameter tuning using Ray Tune.

---

## 📦 Quickstart

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/mysticnova-dev/CryptoTimeSeriesAI.git
cd CryptoTimeSeriesAI
```

### 2️⃣ Install Dependencies

We recommend using a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All outputs are saved under `./LSTM/Models/`, `./Transformer/Models/`, or `./XGBoost/Models/`.

---

## 📊 Dataset

The dataset consists of daily Bitcoin blockchain metrics including price, supply, volume, mining difficulty, and transactional activity. See the full [data dictionary](#📊-dataset-description) below.

Ensure the dataset is named `bitcoin_dataset.csv` and placed in the `Data/` directory or update the path in the `run_*.py` scripts.

---

## 🛠️ Project Structure

```
├── utils/
│   ├── data_cleaning_utils.py           # Handles missing values
│   └── generate_btc_features.py         # Feature engineering logic
├── Transformer/
│   └── Models/                          # Saved Transformer models, plots, and metrics
├── LSTM/
│   └── Models/                          # Saved LSTM models, plots, and metrics
├── XGBoost/
│   └── Models/                          # Saved XGBoost models, plots, and metrics
├── Data/
│   └── bitcoin_dataset.csv              # Input CSV file (daily blockchain metrics)
├── PDFs/
│   ├── Transformer_Demo.pdf             # Transformer explanation and visuals
│   ├── XGBoost_regressor_demo.pdf       # XGBoost demonstration
│   └── LSTM_Demo.pdf                    # LSTM explanation and visuals
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

| Feature | Description |
|--------|-------------|
| `Date` | Date of observation |
| `btc_market_price` | Average USD market price across major exchanges |
| `btc_total_bitcoins` | Total bitcoins mined |
| `btc_market_cap` | USD market cap of circulating supply |
| `btc_trade_volume` | USD value traded daily |
| `btc_hash_rate`, `btc_difficulty` | Mining difficulty and hash rate |
| `btc_miners_revenue`, `btc_transaction_fees` | Miner rewards & transaction fees |
| `btc_cost_per_transaction` | Cost per TX for miners |
| `btc_n_transactions`, `btc_n_unique_addresses` | On-chain activity metrics |
| `btc_output_volume`, `btc_estimated_transaction_volume` | Total value of outputs |

See code for full feature usage and transformations.

---

## ✨ Feature Engineering

Features are generated using domain-relevant indicators:

- **Volatility Metrics:** Bollinger Bands, ATR
- **Momentum Indicators:** ROC, Momentum
- **Volume Patterns:** OBV, RVOL, VPT
- **Supply-Demand Ratios:** Market cap / supply
- **Log-transformed Variables**
- **Missingness Flags**

---

## 🧠 Model Overview

### 🔁 LSTM
- Recurrent neural network with memory cells
- Tunable sequence length and hidden layers
- Sliding-window sequence generation
- Dropout + learning rate scheduling
- Early stopping for generalization

### 🧠 Transformer
- Positional encoding + self-attention
- Multi-head architecture with MLP decoder
- Flattened token sequence → global prediction
- Ray Tune + HyperOpt search space
- Full visualization and model checkpointing

### ⚡ XGBoost
- High-performance gradient-boosted trees
- Tabular feature format (no sequence modeling)
- Robust to noise and fast to train
- Serves as a predictive baseline

---

## 🔍 Hyperparameter Tuning

Uses Ray Tune with `ASHAScheduler` and `HyperOptSearch`:

- 100 trial budget with early stopping
- Parallel trials with CPU/GPU resource control
- Automatic trial logging and saving
- Best model auto-retrained and exported

---

## 📈 Evaluation & Visualization

Each model is evaluated using:

| Metric | Meaning |
|--------|---------|
| `MSE` | Mean Squared Error |
| `RMSE` | Root Mean Squared Error |
| `MAE` | Mean Absolute Error |
| `R²` | Coefficient of determination |

You also get:

- 📉 Training vs Validation Loss
- 📈 Actual vs Forecasted Prices
- 🔍 Zoomed Forecasts
- 📊 Residuals Plot
- 📄 Metric + Config Reports

---

## 🔬 Best Results XGBoost Tuned

| Metric | Value |
|--------|-------|
| MSE    | 45325.7883 |
| RMSE   | 212.8985 |
| MAE    | 119.4872 |
| R²     | 0.9712 |

Best Hyperparameters:

```yaml
colsample_bytree: 0.7447698266738357
gamma: 7.692991551918608
learning_rate: 0.03012583048868712
max_depth: 3
min_child_weight: 3.0
n_estimators: 100
reg_alpha: 0.0003163525705898181
reg_lambda: 0.0023388797183377724
subsample: 0.521394012077824
```

---

## ✅ Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Minimum required packages:
```
numpy
pandas
scikit-learn
matplotlib
torch
xgboost
ray[default]
hyperopt
```

---

## 🚀 Future Work

- 🔁 **Hypercomplex Neural Net Forecasting**: Extend the architecture using quaternion, octonion, and even sedenion-based neural networks to capture entangled market signals across multi-dimensional manifolds, enabling richer representation learning for temporal financial data.
- 🧠 **Bayesian Smoothing**: Apply Bayesian inference for uncertainty-aware predictions and to regularize noisy outputs in volatile market zones.
- 📉 **Residual Modeling**: Predict model residuals as a secondary signal and recursively refine the primary forecasts using residual correction loops.
- 🧮 **Hierarchical Forecasting**: Integrate taxonomic structures (e.g., price → trend + volatility components) with specialized submodels.
- 📈 **Feature Importance from Attention Maps**: Extract interpretability insights by aggregating self-attention weights from transformer heads.

---

## 🧠 Author

**Leonard Burtenshaw**  
AI Engineer | Forecast Architect | Data Science Specialist  
[LinkedIn](https://linkedin.com/in/leoharrisai/)
