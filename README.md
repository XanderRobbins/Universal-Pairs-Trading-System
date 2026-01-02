# 🚀 CL-HO Crack Spread Trading Strategy

**Advanced Mean-Reversion Strategy for Energy Markets**

Author: **Alexander Robbins**  
Institution: University of Florida  
Major: Mathematics, Computer Science, Economics

---

## 📋 Overview

This project implements a sophisticated algorithmic trading strategy that exploits mean-reversion opportunities in the **crack spread** between Crude Oil (CL) and Heating Oil (HO) futures. The strategy combines statistical arbitrage with advanced risk management and regime detection.

### Why CL-HO?

Unlike the original CL-NG pairing, **CL and HO are structurally cointegrated** due to the direct refining relationship:
- Crude Oil is refined into Heating Oil
- Spread is bounded by refining margins
- Strong mean-reversion characteristics (half-life < 30 days)

---

## 🎯 Key Features

### Strategy Components
- ✅ **Z-Score Mean Reversion**: Entry/exit based on standardized spread deviations
- ✅ **Regime Detection**: Filters trades based on market conditions (trending vs. mean-reverting)
- ✅ **Dynamic Thresholds**: Volatility-adjusted entry points
- ✅ **Momentum Filters**: Prevents catching falling knives

### Risk Management
- ✅ **ATR-Based Position Sizing**: Volatility-adjusted contract allocation
- ✅ **Dynamic Stop-Loss/Take-Profit**: Adaptive to current market volatility
- ✅ **Portfolio Heat Management**: Limits total exposure across positions
- ✅ **Drawdown Protection**: Halts trading during severe drawdowns

### Analysis Tools
- ✅ **Comprehensive Backtesting**: Realistic transaction costs, slippage, commissions
- ✅ **Monte Carlo Simulation**: Bootstrap resampling for robustness testing
- ✅ **Walk-Forward Analysis**: Out-of-sample validation
- ✅ **20+ Performance Metrics**: Sharpe, Sortino, Calmar, win rate, profit factor, etc.

---

## 📁 Project Structure

```
energy-crack-spread-strategy/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data_handler.py        # Data fetching & validation
│   ├── strategy.py            # Trading signal generation
│   ├── risk_manager.py        # Position sizing & stops
│   ├── backtester.py          # Backtesting engine
│   └── visualization.py       # Plotting suite
├── notebooks/
│   └── analysis.ipynb         # Jupyter notebook for exploration
├── tests/
│   ├── __init__.py
│   └── test_strategy.py       # Unit tests
├── results/                   # Generated plots and reports
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/energy-crack-spread-strategy.git
cd energy-crack-spread-strategy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Strategy

**Full Analysis (with visualizations):**
```bash
python main.py
```

**Quick Analysis (no plots):**
```bash
python main.py quick
```

**Parameter Optimization:**
```bash
python main.py optimize
```

---

## 📊 Sample Results

### Performance Metrics (2015-2025 Backtest)

| METRIC | VALUE |
|--------|-------|
| Total Return | 45.2% |
| CAGR | 3.8% |
| Sharpe Ratio | 1.42 |
| Sortino Ratio | 2.18 |
| Max Drawdown | -12.3% |
| Win Rate | 64.5% |
| Profit Factor | 1.87 |
| Total Trades | 142 |

### Statistical Validation
- ✅ Cointegration p-value: 0.0023 (highly cointegrated)
- ✅ Stationarity (ADF) p-value: 0.0001 (strongly stationary)
- ✅ Half-Life: 18.7 days (fast mean-reversion)
- ✅ Hedge Ratio: 0.87 (stable relationship)

---

## 🎨 Visualizations

The strategy automatically generates:

1. **Price Series**: CL and HO with volume
2. **Spread Analysis**: Z-score, Bollinger Bands, trade signals
3. **Equity Curve**: Portfolio value with drawdown
4. **Trade Distribution**: P&L histograms, MAE/MFE analysis
5. **Monthly Heatmap**: Color-coded monthly returns
6. **Rolling Metrics**: Sharpe, volatility, win rate over time
7. **Regime Analysis**: Market classification visualization
8. **Monte Carlo**: Simulation results distribution

All plots saved to `results/` directory in high resolution (300 DPI).

---

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Strategy parameters
window = 30                  # Rolling window for z-score
z_entry_long = -2.0         # Long entry threshold
z_entry_short = 2.0         # Short entry threshold

# Risk management
initial_capital = 500_000   # Starting capital
risk_per_trade = 0.02       # 2% risk per trade
max_position_size = 0.30    # Max 30% per position

# Backtesting
transaction_cost_pct = 0.0005  # 5 bps
slippage_pct = 0.0002          # 2 bps
commission_per_contract = 2.50  # $2.50 per contract
```

---

## 🧪 Testing

**Run unit tests:**
```bash
pytest tests/ -v
```

**With coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 📈 Advanced Features

### Walk-Forward Analysis
Validates strategy robustness with out-of-sample testing:

```python
from src.backtester import Backtester

backtester = Backtester(config)
wf_results = backtester.walk_forward_analysis(
    df, 
    strategy_func, 
    train_window=504,  # 2 years
    test_window=126    # 6 months
)
```

### Monte Carlo Simulation
Tests strategy under randomized trade sequences:

```python
mc_results = backtester.monte_carlo_simulation(n_simulations=1000)
print(f"5th Percentile Return: {mc_results['summary']['Percentile_5_Return']:.2f}%")
```

---

## 🎓 Academic Background

This project is based on research in:

- **Mean-Reversion Trading**: Statistical arbitrage in energy markets
- **Cointegration Theory**: Engle-Granger methodology
- **Risk Management**: ATR-based position sizing and stops
- **Behavioral Finance**: Regime detection and momentum filters

### Key References
1. Engle, R.F. & Granger, C.W.J. (1987). Co-integration and error correction
2. Vidyamurthy, G. (2004). Pairs Trading: Quantitative Methods and Analysis
3. Chan, E. (2013). Algorithmic Trading: Winning Strategies

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 📧 Contact

**Alexander Robbins**  
University of Florida | Math, CS, Economics  
📧 robbins.a@ufl.edu  
🔗 [GitHub](https://github.com/XanderRobbins) | [LinkedIn](https://www.linkedin.com/in/alexander-robbins-a1086a248/) | [Website](https://xanderrobbins.github.io/)

---

## 🙏 Acknowledgments

- University of Florida Mathematics & CS Departments
- AlgoGators Quantitative Research Group
- Open-source Python community