import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# ML/Stats Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Time-Series Forecasting with Cryptocurrency Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .crypto-header {
        background: linear-gradient(135deg, rgba(15, 12, 41, 0.7) 0%, rgba(48, 43, 99, 0.7) 100%);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 50px rgba(0, 255, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .crypto-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f5ff 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: 2px;
    }
    
    .crypto-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        color: #00f5ff;
        margin-top: 0.5rem;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        color: #00f5ff !important;
        font-family: 'Rajdhani', sans-serif !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        background: linear-gradient(135deg, #00f5ff 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #00f5ff 0%, #667eea 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        border: none;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0, 245, 255, 0.5);
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        color: #00f5ff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CRYPTO DISCOVERY AND SELECTION
# ============================================================================

@st.cache_data
def discover_crypto_files(data_folder='data'):
    """
    Discover all cryptocurrency CSV files in the data folder
    Returns a dictionary with crypto names and file paths
    """
    crypto_files = {}
    
    # Check if data folder exists
    if not os.path.exists(data_folder):
        st.error(f"❌ Data folder '{data_folder}' not found!")
        return crypto_files
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
    
    if not csv_files:
        st.warning(f"⚠️ No CSV files found in '{data_folder}' folder!")
        return crypto_files
    
    # Process each file
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        # Extract crypto name from filename (e.g., 'bitcoin_prices.csv' -> 'Bitcoin')
        crypto_name = filename.replace('_prices.csv', '').replace('_', ' ').title()
        crypto_files[crypto_name] = filepath
    
    return crypto_files

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

@st.cache_data
def load_crypto_data(filepath, crypto_name):
    """
    Load cryptocurrency data from CSV file
    Expected CSV format: timestamp (milliseconds), price
    """
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
        
        # Convert timestamp (in milliseconds) to datetime
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            st.error("CSV must have 'timestamp' or 'date' column!")
            return None
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date').reset_index(drop=True)
        
        # Estimate volume if not present
        if 'volume' not in df.columns:
            np.random.seed(42)
            base_volume = 35e9  # 35 billion USD base volume
            price_volatility = df['price'].pct_change().abs().fillna(0)
            df['volume'] = base_volume * (1 + price_volatility * 10) * np.random.uniform(0.9, 1.1, len(df))
        
        # Estimate market cap if not present
        if 'market_cap' not in df.columns:
            # Use approximate supply based on crypto (you can customize this)
            supply_estimates = {
                'Bitcoin': 19.6e6,
                'Ethereum': 120e6,
                'Cardano': 35e9,
                'Ripple': 100e9,
                'Solana': 400e6,
                'Polkadot': 1.1e9,
                'Dogecoin': 140e9,
                'Avalanche': 300e6,
                'Chainlink': 500e6,
                'Polygon': 10e9
            }
            supply = supply_estimates.get(crypto_name, 100e6)  # Default 100M
            df['market_cap'] = df['price'] * supply
        
        # Calculate technical indicators
        df['MA7'] = df['price'].rolling(window=7).mean()
        df['MA30'] = df['price'].rolling(window=30).mean()
        df['MA50'] = df['price'].rolling(window=50).mean()
        df['volatility'] = df['price'].rolling(window=30).std()
        df['price_change'] = df['price'].pct_change() * 100
        df['returns'] = df['price'].pct_change()
        df['momentum'] = ((df['price'] - df['MA7']) / df['MA7']) * 100
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading {crypto_name} data: {str(e)}")
        return None

# ============================================================================
# CRYPTO INFO DATABASE
# ============================================================================

CRYPTO_INFO = {
    'Bitcoin': {
        'symbol': 'BTC',
        'color': '#F7931A',
        'icon': '₿',
        'description': 'The first and most valuable cryptocurrency'
    },
    'Ethereum': {
        'symbol': 'ETH',
        'color': '#627EEA',
        'icon': 'Ξ',
        'description': 'Smart contract platform and second-largest crypto'
    },
    'Solana': {
        'symbol': 'SOL',
        'color': '#14F195',
        'icon': '◎',
        'description': 'High-performance blockchain'
    },
    'Binancecoin': {
        'symbol': 'BNB',
        'color': '#F3BA2F',
        'icon': 'B',
        'description': 'Binance exchange utility token'
    },
    'Eos': {
        'symbol': 'EOS',
        'color': '#443F54',
        'icon': '◈',
        'description': 'Scalable blockchain platform for decentralized applications'
    },

    'Filecoin': {
        'symbol': 'FIL',
        'color': '#0090FF',
        'icon': '⨎',
        'description': 'Decentralized storage network for digital data'
    },

    'Maker': {
        'symbol': 'MKR',
        'color': '#1AAB9B',
        'icon': 'Ⓜ',
        'description': 'Governance token for the MakerDAO and DAI stablecoin'
    },

    'Monero': {
        'symbol': 'XMR',
        'color': '#FF6600',
        'icon': 'ɱ',
        'description': 'Privacy-focused cryptocurrency with anonymous transactions'
    },

    'Ripple': {
        'symbol': 'XRP',
        'color': '#23292F',
        'icon': 'X',
        'description': 'Fast, low-cost digital payment protocol'
    },

    'Solana': {
        'symbol': 'SOL',
        'color': '#14F195',
        'icon': '◎',
        'description': 'High-performance blockchain with low transaction fees'
    },

    'Tether': {
        'symbol': 'USDT',
        'color': '#26A17B',
        'icon': '$',
        'description': 'US dollar–pegged stablecoin for trading and liquidity'
    },

    'Tezos': {
        'symbol': 'XTZ',
        'color': '#2C7DF7',
        'icon': 'ꜩ',
        'description': 'Self-amending blockchain with on-chain governance'
    },

    'Tron': {
        'symbol': 'TRX',
        'color': '#EC0622',
        'icon': 'T',
        'description': 'Blockchain platform for decentralized content and entertainment'
    },
    
    'Aave': {
        'symbol': 'AAVE',
        'color': '#B6509E',
        'icon': 'Å',
        'description': 'Decentralized lending and borrowing protocol'
    },

    'Cosmos': {
        'symbol': 'ATOM',
        'color': '#2E3148',
        'icon': '⚛',
        'description': 'Interoperable blockchain ecosystem connecting multiple networks'
    },

    'Vechain': {
        'symbol': 'VET',
        'color': '#15BDFF',
        'icon': 'V',
        'description': 'Enterprise-focused blockchain for supply chain management'
    }
}

def get_crypto_info(crypto_name):
    """Get crypto info or return defaults"""
    return CRYPTO_INFO.get(crypto_name, {
        'symbol': crypto_name[:3].upper(),
        'color': '#00f5ff',
        'icon': '●',
        'description': f'{crypto_name} cryptocurrency'
    })

# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_arima_model(data, order=(5,1,0)):
    """Train ARIMA model"""
    with st.spinner('🔄 Training ARIMA model...'):
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        return fitted_model

def train_sarima_model(data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """Train SARIMA model"""
    with st.spinner('🔄 Training SARIMA model...'):
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        return fitted_model

def train_prophet_model(df):
    """Train Prophet model"""
    with st.spinner('🔄 Training Prophet model...'):
        prophet_df = df[['date', 'price']].copy()
        prophet_df.columns = ['ds', 'y']
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)
        return model

def create_sequences(data, seq_length=60):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_lstm_model(data, seq_length=60, epochs=50):
    """Train LSTM model"""
    with st.spinner('🔄 Training LSTM model... This may take a few minutes.'):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        X, y = create_sequences(scaled_data, seq_length)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        return model, scaler, history

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def make_arima_predictions(model, steps=30):
    """Generate ARIMA predictions"""
    forecast = model.forecast(steps=steps)
    return forecast

def make_sarima_predictions(model, steps=30):
    """Generate SARIMA predictions"""
    forecast = model.forecast(steps=steps)
    return forecast

def make_prophet_predictions(model, periods=30):
    """Generate Prophet predictions"""
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast['yhat'].tail(periods).values

def make_lstm_predictions(model, scaler, last_sequence, steps=30):
    """Generate LSTM predictions"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0, 0])
        current_sequence = np.append(current_sequence[1:], pred[0, 0])
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(actual, predicted):
    """Calculate model performance metrics"""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Discover available cryptocurrencies
    crypto_files = discover_crypto_files('data')
    
    if not crypto_files:
        st.error("❌ No cryptocurrency data found! Please add CSV files to the 'data' folder.")
        st.info("""
        **Expected folder structure:**
        ```
        data/
        ├── bitcoin_prices.csv
        ├── ethereum_prices.csv
        └── ... (other crypto CSVs)
        ```
        
        **CSV format:**
        ```
        timestamp,price
        1738022400000,101958.46
        1738108800000,101313.11
        ...
        ```
        """)
        st.stop()
    
    # Header with dynamic crypto count
    st.markdown(f"""
        <div class="crypto-header">
            <h1 class="crypto-title"> 📈 TIME-SERIES FORECASTING WITH CRYPTOCURRENCY ANALYTICS DASHBOARD</h1>
            <p class="crypto-subtitle">{len(crypto_files)} Cryptocurrencies Available | ARIMA • SARIMA • Prophet • LSTM</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - CRYPTO SELECTION
    # ========================================================================
    
    st.sidebar.markdown("### 🪙 SELECT CRYPTOCURRENCY")
    st.sidebar.markdown("---")
    
    # Cryptocurrency selector
    crypto_names = sorted(crypto_files.keys())
    selected_crypto = st.sidebar.selectbox(
        "Choose a cryptocurrency",
        crypto_names,
        index=0,
        help="Select which cryptocurrency to analyze"
    )
    
    # Display crypto info
    crypto_info = get_crypto_info(selected_crypto)
    st.sidebar.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(15, 12, 41, 0.7) 0%, rgba(48, 43, 99, 0.7) 100%); 
                padding: 1.5rem; border-radius: 15px; margin: 1rem 0; 
                border: 2px solid {crypto_info['color']}; box-shadow: 0 5px 20px rgba(0, 245, 255, 0.2);'>
        <h2 style='color: {crypto_info['color']}; margin: 0; font-size: 2rem;'>{crypto_info['icon']} {selected_crypto}</h2>
        <p style='color: #00f5ff; margin: 0.5rem 0 0 0; font-weight: 700;'>Symbol: {crypto_info['symbol']}</p>
        <p style='color: #ffffff; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>{crypto_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load selected cryptocurrency data
    selected_filepath = crypto_files[selected_crypto]
    df = load_crypto_data(selected_filepath, selected_crypto)
    
    if df is None:
        st.stop()
    
    # ========================================================================
    # SIDEBAR - FILTERS AND SETTINGS
    # ========================================================================
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 ANALYSIS CONTROLS")
    
    # Date range filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    # Ensure default start date is within bounds
    default_start_date = max(min_date, max_date - timedelta(days=365))
    
    date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(default_start_date, max_date),
    min_value=min_date,
    max_value=max_date
    )
    # Filter data
    if len(date_range) == 2:
        mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()
    
    # Prediction settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔮 PREDICTION SETTINGS")
    forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)
    
    # Model selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 MODELS")
    models_to_run = st.sidebar.multiselect(
        "Select Models to Train",
        ["ARIMA", "SARIMA", "Prophet", "LSTM"],
        default=["ARIMA", "Prophet"]
    )
    
    # Data info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 DATA INFO")
    st.sidebar.metric("Records", f"{len(filtered_df):,}")
    st.sidebar.metric("Date Range", f"{(filtered_df['date'].max() - filtered_df['date'].min()).days} days")
    st.sidebar.metric("Min Price", f"${filtered_df['price'].min():,.2f}")
    st.sidebar.metric("Max Price", f"${filtered_df['price'].max():,.2f}")
    st.sidebar.metric("Current Price", f"${filtered_df['price'].iloc[-1]:,.2f}")

    
    # Compare cryptos option
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("📊 Compare Multiple Cryptos"):
        compare_cryptos = st.sidebar.multiselect(
            "Select cryptos to compare",
            [c for c in crypto_names if c != selected_crypto],
            default=[]
        )
    else:
        compare_cryptos = []
    
    # ========================================================================
    # MAIN CONTENT TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🤖 Model Predictions", 
        "📈 Comparison", 
        "📋 Raw Data",
        "🔍 Multi-Crypto Compare"
    ])
    
    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    with tab1:
        st.markdown(f"### 💰 {selected_crypto} Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = filtered_df['price'].iloc[-1]
            price_change = ((filtered_df['price'].iloc[-1] - filtered_df['price'].iloc[0]) / filtered_df['price'].iloc[0]) * 100
            st.metric("Current Price", f"${current_price:,.2f}", f"{price_change:+.2f}%")
        
        with col2:
            st.metric("Highest Price", f"${filtered_df['price'].max():,.2f}")
        
        with col3:
            st.metric("Average Price", f"${filtered_df['price'].mean():,.2f}")
        
        with col4:
            current_vol = filtered_df['volatility'].iloc[-1]
            st.metric("Volatility (30D)", f"${current_vol:,.2f}")
        
        st.markdown("---")
        
        # Price chart
        st.markdown(f"### 📈 {selected_crypto} Price Trend")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['price'],
            mode='lines',
            name='Price',
            line=dict(color=crypto_info['color'], width=2),
            fill='tozeroy',
            fillcolor=f"rgba({int(crypto_info['color'][1:3], 16)}, {int(crypto_info['color'][3:5], 16)}, {int(crypto_info['color'][5:7], 16)}, 0.1)"
        ))
        
        fig.add_trace(go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['MA7'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='#00d395', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['MA30'],
            mode='lines',
            name='30-Day MA',
            line=dict(color='#667eea', width=2, dash='dot')
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=500,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume and Market Cap
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 📊 {selected_crypto} Trading Volume")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=filtered_df['date'],
                y=filtered_df['volume']/1e9,
                marker_color=crypto_info['color']
            ))
            fig_vol.update_layout(
                template='plotly_dark',
                height=350,
                xaxis_title="Date",
                yaxis_title="Volume (Billion USD)"
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            st.markdown(f"### 💎 {selected_crypto} Market Cap")
            fig_mcap = go.Figure()
            fig_mcap.add_trace(go.Scatter(
                x=filtered_df['date'],
                y=filtered_df['market_cap']/1e12,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#00d395', width=2),
                fillcolor='rgba(0, 211, 149, 0.1)'
            ))
            fig_mcap.update_layout(
                template='plotly_dark',
                height=350,
                xaxis_title="Date",
                yaxis_title="Market Cap (Trillion USD)"
            )
            st.plotly_chart(fig_mcap, use_container_width=True)
    
    # ========================================================================
    # TAB 2: MODEL PREDICTIONS
    # ========================================================================
    with tab2:
        st.markdown(f"### 🤖 Train Models for {selected_crypto}")
        
        if st.button("🚀 Train Models and Generate Predictions", type="primary"):
            
            price_data = filtered_df['price'].values
            last_date = filtered_df['date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            
            predictions = {}
            metrics = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_models = len(models_to_run)
            
            for idx, model_name in enumerate(models_to_run):
                status_text.text(f"Training {model_name} for {selected_crypto}...")
                
                try:
                    if model_name == "ARIMA":
                        arima_model = train_arima_model(price_data[-365:])
                        arima_forecast = make_arima_predictions(arima_model, forecast_days)
                        predictions['ARIMA'] = arima_forecast
                        
                        train_size = int(len(price_data) * 0.8)
                        val_data = price_data[train_size:]
                        val_model = train_arima_model(price_data[:train_size])
                        val_pred = make_arima_predictions(val_model, len(val_data))
                        metrics['ARIMA'] = calculate_metrics(val_data, val_pred)
                    
                    elif model_name == "SARIMA":
                        sarima_model = train_sarima_model(price_data[-365:])
                        sarima_forecast = make_sarima_predictions(sarima_model, forecast_days)
                        predictions['SARIMA'] = sarima_forecast
                        
                        train_size = int(len(price_data) * 0.8)
                        val_data = price_data[train_size:]
                        val_model = train_sarima_model(price_data[:train_size])
                        val_pred = make_sarima_predictions(val_model, len(val_data))
                        metrics['SARIMA'] = calculate_metrics(val_data, val_pred)
                    
                    elif model_name == "Prophet":
                        prophet_model = train_prophet_model(filtered_df[-365:])
                        prophet_forecast = make_prophet_predictions(prophet_model, forecast_days)
                        predictions['Prophet'] = prophet_forecast
                        
                        train_size = int(len(filtered_df) * 0.8)
                        val_df = filtered_df.iloc[train_size:].copy()
                        train_df = filtered_df.iloc[:train_size].copy()
                        val_model = train_prophet_model(train_df)
                        val_pred = make_prophet_predictions(val_model, len(val_df))
                        metrics['Prophet'] = calculate_metrics(val_df['price'].values, val_pred)
                    
                    elif model_name == "LSTM":
                        lstm_model, scaler, history = train_lstm_model(price_data, epochs=30)
                        scaled_data = scaler.transform(price_data.reshape(-1, 1))
                        last_sequence = scaled_data[-60:]
                        lstm_forecast = make_lstm_predictions(lstm_model, scaler, last_sequence.flatten(), forecast_days)
                        predictions['LSTM'] = lstm_forecast
                        
                        seq_length = 60
                        X, y = create_sequences(scaled_data, seq_length)
                        split = int(0.8 * len(X))
                        X_test, y_test = X[split:], y[split:]
                        y_pred = lstm_model.predict(X_test, verbose=0)
                        y_test_inv = scaler.inverse_transform(y_test)
                        y_pred_inv = scaler.inverse_transform(y_pred)
                        metrics['LSTM'] = calculate_metrics(y_test_inv.flatten(), y_pred_inv.flatten())
                    
                    progress_bar.progress((idx + 1) / total_models)
                    
                except Exception as e:
                    st.error(f"Error training {model_name}: {str(e)}")
            
            status_text.text("✅ Training complete!")
            progress_bar.empty()
            
            # Display predictions
            if predictions:
                st.markdown("---")
                st.markdown(f"### 📊 {selected_crypto} Prediction Results")
                
                fig_pred = go.Figure()
                
                # Historical data
                fig_pred.add_trace(go.Scatter(
                    x=filtered_df['date'].tail(90),
                    y=filtered_df['price'].tail(90),
                    mode='lines',
                    name='Historical',
                    line=dict(color=crypto_info['color'], width=2)
                ))
                
                # Predictions
                colors = {'ARIMA': '#f093fb', 'SARIMA': '#f5576c', 'Prophet': '#00d395', 'LSTM': '#667eea'}
                
                for model_name, forecast in predictions.items():
                    fig_pred.add_trace(go.Scatter(
                        x=future_dates,
                        y=forecast,
                        mode='lines+markers',
                        name=f'{model_name} Forecast',
                        line=dict(color=colors.get(model_name, '#ffffff'), width=2, dash='dash')
                    ))
                
                fig_pred.update_layout(
                    template='plotly_dark',
                    height=600,
                    title=f"{selected_crypto} Price Predictions - All Models",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Metrics table
                st.markdown("### 📈 Model Performance Metrics")
                
                if metrics:
                    metrics_df = pd.DataFrame(metrics).T
                    metrics_df = metrics_df.round(2)
                    st.dataframe(metrics_df, use_container_width=True)
                
                # Prediction statistics
                st.markdown(f"### 📊 {selected_crypto} Forecast Statistics")
                
                cols = st.columns(len(predictions))
                for idx, (model_name, forecast) in enumerate(predictions.items()):
                    with cols[idx]:
                        last_price = filtered_df['price'].iloc[-1]
                        predicted_price = forecast[-1]
                        change = ((predicted_price - last_price) / last_price) * 100
                        
                        st.metric(
                            f"{model_name} ({forecast_days}D)",
                            f"${predicted_price:,.2f}",
                            f"{change:+.2f}%"
                        )
        
        else:
            st.info(f"👆 Click the button above to train models and generate predictions for {selected_crypto}")
    
    # ========================================================================
    # TAB 3: COMPARISON
    # ========================================================================
    with tab3:
        st.markdown("### ⚖️ Model Comparison Guide")
        
        comparison_data = {
            'Model': ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
            'Type': ['Statistical', 'Statistical', 'Statistical', 'Deep Learning'],
            'Best For': [
                'Linear trends, short-term',
                'Seasonal patterns',
                'Multiple seasonality, holidays',
                'Complex patterns, long sequences'
            ],
            'Training Speed': ['Fast', 'Medium', 'Medium', 'Slow'],
            'Accuracy': ['Medium', 'Medium-High', 'High', 'Very High'],
            'Data Required': ['Low', 'Medium', 'Medium', 'High']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🎯 When to Use Each Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **ARIMA**
            - Quick forecasts
            - Linear trends
            - Stationary data
            - Short-term predictions
            
            **SARIMA**
            - Seasonal patterns
            - Weekly/Monthly cycles
            - Medium-term forecasts
            """)
        
        with col2:
            st.markdown("""
            **Prophet**
            - Holiday effects
            - Multiple seasonalities
            - Missing data handling
            - Business forecasting
            
            **LSTM**
            - Complex patterns
            - Long-term dependencies
            - Large datasets
            - Highest accuracy needs
            """)
    
    # ========================================================================
    # TAB 4: RAW DATA
    # ========================================================================
    with tab4:
        st.markdown(f"### 📋 {selected_crypto} Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(filtered_df))
        with col2:
            st.metric("Date Range", f"{(filtered_df['date'].max() - filtered_df['date'].min()).days} days")
        with col3:
            st.metric("Missing Values", filtered_df.isnull().sum().sum())
        
        st.markdown("---")
        st.markdown(f"### 📊 {selected_crypto} Data Table")
        
        display_df = filtered_df.copy()
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}")
        display_df['volume'] = display_df['volume'].apply(lambda x: f"${x:,.0f}")
        display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(
            display_df[['date', 'price', 'volume', 'market_cap', 'MA7', 'MA30']].tail(100),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {selected_crypto} Data (CSV)",
            data=csv,
            file_name=f"{selected_crypto.lower()}_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # ========================================================================
    # TAB 5: MULTI-CRYPTO COMPARISON
    # ========================================================================
    with tab5:
        st.markdown("### 🔍 Multi-Cryptocurrency Comparison")
        
        if compare_cryptos:
            # Load comparison data
            comparison_data = {selected_crypto: filtered_df}
            
            for crypto in compare_cryptos:
                filepath = crypto_files[crypto]
                comp_df = load_crypto_data(filepath, crypto)
                if comp_df is not None:
                    # Filter to same date range
                    if len(date_range) == 2:
                        mask = (comp_df['date'].dt.date >= date_range[0]) & (comp_df['date'].dt.date <= date_range[1])
                        comparison_data[crypto] = comp_df[mask].copy()
            
            # Price comparison chart
            st.markdown("### 📈 Price Comparison (Normalized to 100)")
            
            fig_compare = go.Figure()
            
            for crypto_name, crypto_df in comparison_data.items():
                # Normalize prices to 100 for comparison
                normalized_price = (crypto_df['price'] / crypto_df['price'].iloc[0]) * 100
                crypto_info_comp = get_crypto_info(crypto_name)
                
                fig_compare.add_trace(go.Scatter(
                    x=crypto_df['date'],
                    y=normalized_price,
                    mode='lines',
                    name=f"{crypto_name} ({crypto_info_comp['symbol']})",
                    line=dict(color=crypto_info_comp['color'], width=2)
                ))
            
            fig_compare.update_layout(
                template='plotly_dark',
                height=500,
                xaxis_title="Date",
                yaxis_title="Normalized Price (Base 100)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Performance comparison
            st.markdown("### 📊 Performance Statistics")
            
            perf_data = []
            for crypto_name, crypto_df in comparison_data.items():
                perf_data.append({
                    'Cryptocurrency': crypto_name,
                    'Current Price': f"${crypto_df['price'].iloc[-1]:,.2f}",
                    'Min Price': f"${crypto_df['price'].min():,.2f}",
                    'Max Price': f"${crypto_df['price'].max():,.2f}",
                    'Avg Price': f"${crypto_df['price'].mean():,.2f}",
                    'Return %': f"{((crypto_df['price'].iloc[-1] / crypto_df['price'].iloc[0] - 1) * 100):+.2f}%",
                    'Volatility': f"${crypto_df['volatility'].mean():,.2f}"
                })
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
            
        else:
            st.info("👈 Enable 'Compare Multiple Cryptos' in the sidebar and select cryptocurrencies to compare")
            
            # Show available cryptos
            st.markdown("### 🪙 Available Cryptocurrencies")
            
            cols = st.columns(3)
            for idx, crypto_name in enumerate(crypto_names):
                with cols[idx % 3]:
                    info = get_crypto_info(crypto_name)
                    st.markdown(f"""
                    <div style='background: rgba(15, 12, 41, 0.5); padding: 1rem; border-radius: 10px; 
                                margin: 0.5rem 0; border: 1px solid {info['color']};'>
                        <h4 style='color: {info['color']}; margin: 0;'>{info['icon']} {crypto_name}</h4>
                        <p style='color: #00f5ff; margin: 0.3rem 0;'>{info['symbol']}</p>
                        <p style='color: #ffffff; font-size: 0.85rem; margin: 0;'>{info['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem; background: rgba(15, 12, 41, 0.7); border-radius: 15px;'>
            <p style='color: #00f5ff; font-size: 1.2rem; margin: 0;'>
                <strong>🚀 Time-Series Forecasting with Cryptocurrency Analytics Dashboard</strong>
            </p>
            <p style='color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-top: 0.5rem;'>
                {len(crypto_files)} Cryptos Available | ARIMA • SARIMA • Prophet • LSTM
            </p>
            <p style='color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-top: 0.5rem;'>
                copyright &copy; 2026 | Made By Aman | All rights reserved
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()