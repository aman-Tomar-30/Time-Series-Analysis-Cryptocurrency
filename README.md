# 📈 Time-Series Forecasting with Cryptocurrency Analytics Dashboard

<div align="center">
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
</div>

A powerful and interactive cryptocurrency analytics platform that leverages advanced time-series forecasting models (ARIMA, SARIMA, Prophet, and LSTM) to predict cryptocurrency prices. Built with Streamlit for a seamless user experience.

<div align="center">
[![View Dashboard](https://img.shields.io/badge/View-Dashboard-brightgreen)](https://time-series-analysis-cryptocurrency.streamlit.app/)
</div>


---

## 🌟 Features

### 📊 **Multi-Cryptocurrency Support**
- Analyze **15+ cryptocurrencies** including Bitcoin, Ethereum, Solana, and more
- Easy switching between different cryptocurrencies
- Automatic discovery of available crypto data files

### 🤖 **Advanced Forecasting Models**
- **ARIMA** - AutoRegressive Integrated Moving Average for trend analysis
- **SARIMA** - Seasonal ARIMA for capturing seasonal patterns
- **Prophet** - Facebook's robust forecasting tool with multiple seasonalities
- **LSTM** - Deep Learning model for complex pattern recognition

### 📈 **Comprehensive Analytics**
- Real-time price trends with moving averages (MA7, MA30, MA50)
- Trading volume analysis
- Market capitalization tracking
- Volatility metrics and price change indicators
- Historical data visualization

### 🔍 **Multi-Crypto Comparison**
- Compare multiple cryptocurrencies side-by-side
- Normalized price comparison charts
- Performance statistics table
- Correlation analysis

### 💡 **Interactive Interface**
- Dark-themed, modern UI with glassmorphism effects
- Dynamic charts with Plotly
- Customizable date ranges
- Adjustable forecast horizons (7-90 days)
- Real-time model training progress

---

## 📁 Project Structure

```
cryptocurrency-analytics-dashboard/
├── .devcontainer/              # Dev container configuration
├── data/                       # Cryptocurrency price data
│   ├── aave_prices.csv
│   ├── bitcoin_prices.csv
│   ├── binancecoin_prices.csv
│   ├── cosmos_prices.csv
│   ├── eos_prices.csv
│   ├── ethereum_prices.csv
│   ├── filecoin_prices.csv
│   ├── maker_prices.csv
│   ├── monero_prices.csv
│   ├── ripple_prices.csv
│   ├── solana_prices.csv
│   ├── tether_prices.csv
│   ├── tezos_prices.csv
│   ├── tron_prices.csv
│   └── vechain_prices.csv
├── streamlit_app.py            # Main Streamlit application
├── data_src.py                 # Data fetching script (CoinGecko API)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- **Virtual environment** (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/crypto-analytics-dashboard.git
   cd crypto-analytics-dashboard
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Fetch cryptocurrency data** (if needed)
   ```bash
   python data_src.py
   ```
   This will fetch the latest price data for all 15 cryptocurrencies using the CoinGecko API.

5. **Run the dashboard**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Open in browser**
   - The dashboard will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

---

## 📦 Dependencies

### Core Libraries
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.14.0
```

### Machine Learning & Statistics
```
scikit-learn>=1.3.0
statsmodels>=0.14.0
prophet>=1.1.0
tensorflow>=2.13.0
```

> **Note**: Full dependencies are listed in `requirements.txt`

---

## 💻 Usage Guide

### 1. **Selecting a Cryptocurrency**

- Open the **sidebar** on the left
- Click the **"Choose a cryptocurrency"** dropdown
- Select from 15 available cryptocurrencies
- The entire dashboard updates instantly!

### 2. **Viewing Analytics (Overview Tab)**

The Overview tab displays:
- **Current Price** with percentage change
- **Highest/Lowest Price** in selected period
- **Average Price** and **Volatility**
- **Price Trend Chart** with moving averages
- **Trading Volume** bar chart
- **Market Capitalization** trend

### 3. **Generating Predictions (Model Predictions Tab)**

1. Select cryptocurrency
2. Choose date range in sidebar
3. Set forecast days (7-90 days)
4. Select models to train (ARIMA, SARIMA, Prophet, LSTM)
5. Click **"🚀 Train Models and Generate Predictions"**
6. Wait for training to complete (2-5 minutes)
7. View predictions, metrics, and forecasts!

### 4. **Comparing Cryptocurrencies (Multi-Crypto Compare Tab)**

1. Enable **"📊 Compare Multiple Cryptos"** in sidebar
2. Select cryptocurrencies to compare
3. Navigate to **"Multi-Crypto Compare"** tab
4. View normalized price comparison chart
5. Analyze performance statistics table

### 5. **Downloading Data (Raw Data Tab)**

- View complete dataset with technical indicators
- Filter by date range
- Download as CSV with one click

---

## 🤖 Model Details

### ARIMA (AutoRegressive Integrated Moving Average)
**Best for:** Short-term predictions, linear trends

**Parameters:**
- Order: (5, 1, 0)
- Training time: 5-10 seconds
- Recommended data: 90+ days

**Pros:**
- Fast training
- Good for stable trends
- Low computational cost

**Cons:**
- Limited for non-linear patterns
- Assumes stationarity

---

### SARIMA (Seasonal ARIMA)
**Best for:** Data with seasonal patterns

**Parameters:**
- Order: (1, 1, 1)
- Seasonal order: (1, 1, 1, 12)
- Training time: 15-30 seconds
- Recommended data: 180+ days

**Pros:**
- Captures seasonality
- Better than ARIMA for cyclic data
- Handles weekly/monthly patterns

**Cons:**
- Slower than ARIMA
- Requires more data

---

### Prophet (Facebook's Forecasting Tool)
**Best for:** Overall trend analysis, handling missing data

**Parameters:**
- Daily, weekly, yearly seasonality enabled
- Changepoint prior scale: 0.05
- Training time: 20-40 seconds
- Recommended data: 180+ days

**Pros:**
- Robust to missing data
- Multiple seasonality support
- Automatic trend detection
- Intuitive results

**Cons:**
- Can overfit on short-term noise
- Less interpretable parameters

---

### LSTM (Long Short-Term Memory)
**Best for:** Complex patterns, highest accuracy

**Architecture:**
```python
Sequential([
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
```

**Parameters:**
- Sequence length: 60 days
- Epochs: 30 (with early stopping)
- Training time: 2-5 minutes
- Recommended data: 365+ days

**Pros:**
- Highest accuracy
- Captures long-term dependencies
- Learns complex patterns

**Cons:**
- Slowest training
- Requires more data
- GPU recommended for large datasets

---

## 📊 Supported Cryptocurrencies

| # | Cryptocurrency | Symbol | Icon |
|---|----------------|--------|------|
| 1 | Bitcoin | BTC | ₿ |
| 2 | Ethereum | ETH | Ξ |
| 3 | Solana | SOL | ◎ |
| 4 | Binance Coin | BNB | B |
| 5 | Ripple | XRP | X |
| 6 | Tron | TRX | T |
| 7 | Aave | AAVE | Å |
| 8 | Cosmos | ATOM | ⚛ |
| 9 | Vechain | VET | V |
| 10 | Tether | USDT | $ |
| 11 | Tezos | XTZ | ꜩ |
| 12 | Monero | XMR | ɱ |
| 13 | Maker | MKR | Ⓜ |
| 14 | Filecoin | FIL | ⨎ |
| 15 | EOS | EOS | ◈ |

---

## 📈 Data Format

### CSV File Structure
```csv
timestamp,price
1738022400000,101958.46953745594
1738108800000,101313.11264498268
1738195200000,103718.97939813645
...
```

**Columns:**
- `timestamp` - Unix timestamp in milliseconds
- `price` - Price in USD

**Optional columns** (automatically estimated if missing):
- `volume` - Trading volume in USD
- `market_cap` - Market capitalization

---

## 🔧 Configuration

### Adjusting Model Parameters

**ARIMA Order:**
```python
# In train_arima_model() function
model = ARIMA(data, order=(5,1,0))
# Change to: order=(p, d, q)
# p = AR terms, d = differencing, q = MA terms
```

**SARIMA Seasonal Order:**
```python
# In train_sarima_model() function
seasonal_order=(1,1,1,12)
# Change to: (P, D, Q, s)
# s = seasonal period (7 for weekly, 30 for monthly)
```

**LSTM Epochs:**
```python
# In train_lstm_model() function
train_lstm_model(data, epochs=30)
# Increase for better accuracy (slower)
# Decrease for faster training (less accurate)
```

### Customizing Forecast Horizon

In the sidebar:
- **Minimum:** 7 days
- **Maximum:** 90 days
- **Default:** 30 days

---

## 📊 Performance Metrics

The dashboard calculates the following metrics:

| Metric | Description | Best Value |
|--------|-------------|------------|
| **RMSE** | Root Mean Square Error | Lower |
| **MAE** | Mean Absolute Error | Lower |
| **R²** | Coefficient of Determination | Higher (0-1) |
| **MAPE** | Mean Absolute Percentage Error | Lower |

### Interpreting Metrics

**RMSE = $1,500**
- Average prediction error is $1,500

**MAE = $1,200**
- On average, predictions are off by $1,200

**R² = 0.95**
- Model explains 95% of price variance (excellent!)

**MAPE = 2.5%**
- Average prediction error is 2.5%

---

## 🎨 UI Features

### Dark Theme Design
- Modern gradient backgrounds
- Glassmorphism effects
- Crypto-specific color coding
- Smooth animations

### Interactive Charts
- Zoom and pan
- Hover for details
- Download as PNG
- Responsive design

### Custom Fonts
- **Space Grotesk** - Headers
- **Rajdhani** - Body text
- **Monospace** - Metrics

---

## 🔄 Data Updates

### Fetching New Data

Run the data fetching script:
```bash
python data_src.py
```

This script:
1. Connects to CoinGecko API
2. Fetches latest price data for all 15 cryptos
3. Saves to `data/` folder as CSV files
4. Updates existing files or creates new ones

### API Configuration

The `data_src.py` script uses the CoinGecko API:
- **Free tier:** No API key required
- **Rate limit:** 50 calls/minute
- **Historical data:** Up to 365 days
- **Update frequency:** Daily recommended

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: No module named 'prophet'**
```bash
# Windows
pip install prophet

# macOS/Linux (if above fails)
conda install -c conda-forge prophet
```

**2. TensorFlow Installation Issues**
```bash
# Use specific version
pip install tensorflow==2.13.0

# For Apple Silicon Macs
pip install tensorflow-macos
pip install tensorflow-metal
```

**3. Dashboard Not Opening**
```bash
# Check if port 8501 is in use
# Try different port
streamlit run streamlit_app.py --server.port 8502
```

**4. Model Training Fails**
- Ensure you have enough data (minimum 90 days)
- Check for missing values in CSV
- Verify timestamp format (milliseconds)
- Try training with fewer models first

**5. Slow LSTM Training**
```python
# Reduce epochs in streamlit_app.py
train_lstm_model(price_data, epochs=20)  # Instead of 30

# Or reduce sequence length
def train_lstm_model(data, seq_length=30, epochs=30)  # Instead of 60
```

---

## 📚 Technical Details

### Data Processing Pipeline

1. **Load CSV** → Read cryptocurrency price data
2. **Convert Timestamps** → Milliseconds to datetime
3. **Estimate Missing Data** → Calculate volume/market cap
4. **Calculate Indicators** → MA7, MA30, volatility, etc.
5. **Filter by Date** → Apply user-selected range
6. **Train Models** → Generate predictions
7. **Visualize** → Display charts and metrics

### Caching Strategy

The app uses Streamlit's caching:
```python
@st.cache_data
def load_crypto_data(filepath, crypto_name):
    # Data cached for faster loading
    # Reloads only when file changes
```

### State Management

- Session state for model predictions
- URL parameters for sharing configs
- Local storage for user preferences

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/crypto-analytics-dashboard.git

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

---


## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Aman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👤 Author

**Aman**

- GitHub: [@aman-Tomar-30](https://github.com/aman-Tomar-30)
- LinkedIn: [Aman Tomar](https://linkedin.com/in/tomaraman)
- Email: amantomar2609@gmail.com

---

## 🙏 Acknowledgments

- **CoinGecko** - For providing free cryptocurrency data API
- **Streamlit** - For the amazing web framework
- **Facebook Prophet** - For the robust forecasting library
- **TensorFlow** - For deep learning capabilities
- **Plotly** - For interactive visualizations

---

## 📞 Support

If you encounter any issues or have questions:

1. **Check the Troubleshooting section** above
2. **Search existing GitHub issues**
3. **Create a new issue** with:
   - Description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots (if applicable)
   - System information (OS, Python version)

---

## 🔥 Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/yourusername/crypto-analytics-dashboard.git
cd crypto-analytics-dashboard
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Fetch data
python data_src.py

# Run dashboard
streamlit run streamlit_app.py

# Open browser at http://localhost:8501
```

**That's it! Start analyzing cryptocurrencies! 🚀**

---

<div align="center">

**Made with ❤️ by Aman**

**© 2026 | All Rights Reserved**

</div>