import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM, Dropout
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------
# Enhanced Streamlit App
# ----------------------

st.set_page_config(
    page_title="𝘽𝙐𝙇𝙇𝙍𝙄𝘿𝙀",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern look
# Modern CSS for Streamlit with contemporary design
st.markdown(
    """
    <style>
        /* Import modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Root variables for consistent theming */
        :root {
            --primary-color: #6366f1;
            --secondary-color: #8b5cf6;
            --accent-color: #06b6d4;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #f1f5f9;
            --border-color: #e2e8f0;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-accent: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        /* Main container styling */
        .reportview-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
            min-height: 100vh;
        }
        
        .main .block-container {
            padding: 2rem 3rem;
            max-width: 1200px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            margin: 2rem auto;
            box-shadow: var(--shadow-lg);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Typography improvements */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary);
            font-weight: 600;
            line-height: 1.25;
            margin-bottom: 1rem;
        }

        h1 {
            font-size: 2.5rem;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1.5rem;
        }

        h2 {
            font-size: 2rem;
            color: var(--text-primary);
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }

        h3 {
            font-size: 1.5rem;
            color: var(--primary-color);
        }

        p, .stMarkdown {
            color: var(--text-secondary);
            line-height: 1.6;
            font-size: 1rem;
        }

        /* Header styling */
        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }

        /* Button styling */
        .stButton > button {
            background: var(--gradient-primary);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 500;
            font-size: 1rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: var(--shadow-md);
            cursor: pointer;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        }

        .stButton > button:active {
            transform: translateY(0);
            box-shadow: var(--shadow-sm);
        }

        /* Input field styling */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            border-radius: 12px;
            border: 2px solid var(--border-color);
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: var(--bg-primary);
        }

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
            outline: none;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            border-right: 1px solid var(--border-color);
        }

        /* Metric styling */
        [data-testid="metric-container"] {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
        }

        [data-testid="metric-container"]:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }

        /* DataFrame styling */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--shadow-md);
        }

        /* Progress bar styling */
        .stProgress > div > div > div {
            background: var(--gradient-primary);
            border-radius: 10px;
        }

        /* Alert/info boxes */
        .stAlert {
            border-radius: 12px;
            border: none;
            box-shadow: var(--shadow-sm);
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            background: var(--primary-color);
            color: white;
            box-shadow: var(--shadow-sm);
        }

        /* Chart containers */
        .js-plotly-plot, .stPlotlyChart {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: var(--shadow-md);
            background: var(--bg-primary);
        }

        /* Expander styling */
        .streamlit-expander {
            border-radius: 12px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            overflow: hidden;
        }

        /* File uploader styling */
        .stFileUploader {
            border-radius: 12px;
            border: 2px dashed var(--border-color);
            background: var(--bg-secondary);
            transition: all 0.3s ease;
        }

        .stFileUploader:hover {
            border-color: var(--primary-color);
            background: rgba(99, 102, 241, 0.05);
        }

        /* Custom animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .main .block-container > div {
            animation: fadeIn 0.6s ease-out;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem 1.5rem;
                margin: 1rem;
                border-radius: 16px;
            }
            
            h1 { font-size: 2rem; }
            h2 { font-size: 1.5rem; }
        }

        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            :root {
                --text-primary: #f8fafc;
                --text-secondary: #cbd5e1;
                --bg-primary: #1e293b;
                --bg-secondary: #0f172a;
                --bg-tertiary: #334155;
                --border-color: #475569;
            }
            
            .main .block-container {
                background: rgba(30, 41, 59, 0.95);
                border: 1px solid rgba(71, 85, 105, 0.3);
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------
# Sidebar: Controls
# ----------------------
with st.sidebar:
    st.header('SETTINGS')
    user_input = st.text_input('Enter Stock Ticker', 'AAPL')
    start_date = st.date_input('Start Date', pd.to_datetime('2012-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2024-12-31'))

    st.markdown('---')
    st.subheader('Model & Prediction')
    window_size = st.slider('Sequence Window Size (days)', 30, 200, 100)
    test_split = st.slider('Test set percentage', 5, 50, 30)
    forecast_days = st.slider('Forecast next N days', 0, 30, 5)
    show_ma = st.checkbox('Show Moving Averages', value=True)
    ma_short = st.number_input('MA Short (days)', min_value=5, max_value=200, value=100)
    ma_long = st.number_input('MA Long (days)', min_value=50, max_value=500, value=250)
    show_bbands = st.checkbox('Show Bollinger Bands', value=False)

    st.markdown('---')
    st.info('This application predicts stock trends using an LSTM model.')
    st.info('\nData is fetched from Yahoo Finance and predictions are made on historical closing prices.')
    st.markdown('Upload model (optional):')
    uploaded_model = st.file_uploader('Upload keras_model.h5 to override local file', type=['h5'])

# ----------------------
# Helper functions
# ----------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    ticker = yf.Ticker(ticker, session=None)
    df = ticker.history(start=start, end=end, progress=False, threads=True, auto_adjust=True)
    return df

@st.cache_resource
def load_or_get_model(uploaded):
    # Priority: uploaded file -> local keras_model.h5
    if uploaded is not None:
        try:
            # streamlit's uploaded file is a BytesIO; keras can load from file-like
            model = load_model(uploaded)
            return model
        except Exception as e:
            st.warning(f"Uploaded model couldn't be loaded: {e}")
    try:
        model = load_model('keras_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def create_sequences(data_array, window):
    x = []
    y = []
    for i in range(window, len(data_array)):
        x.append(data_array[i-window:i, 0])
        y.append(data_array[i, 0])
    return np.array(x), np.array(y)


def forecast_future(model, recent_scaled, scaler_obj, days, window):
    # recent_scaled: array of shape (window, ) scaled
    future = []
    seq = recent_scaled.copy()
    for _ in range(days):
        inp = seq.reshape(1, window, 1)
        pred = model.predict(inp, verbose=0)
        future.append(pred[0, 0])
        seq = np.append(seq[1:], pred[0, 0])
    future = np.array(future).reshape(-1, 1)
    future_inv = scaler_obj.inverse_transform(future)
    return future_inv.flatten()

# ----------------------
# Main App Flow - Mathematically Correct Version
# ----------------------
st.title('𝙎𝙩𝙤𝙘𝙠 𝙏𝙧𝙚𝙣𝙙 𝙋𝙧𝙚𝙙𝙞𝙘𝙩𝙞𝙤𝙣𝙨 ⚡')
st.markdown('A simple and interactive web application to visualize and predict stock prices using an LSTM model.')

# Load data
with st.spinner('Fetching data from Yahoo Finance...'):
    df = load_data(user_input, start_date, end_date)

if df is None or df.empty:
    st.error(f"No data found for: {user_input}. Check the ticker or date range.")
    st.stop()

# ----------------------
# MATHEMATICALLY CORRECT CALCULATIONS
# ----------------------

# Ensure data is sorted by date (ascending)
df = df.sort_index()

# Basic price statistics
current_price = float(df['Close'].iloc[-1])  # Last price
opening_price = float(df['Close'].iloc[0])   # First price in dataset
highest_price = float(df['High'].max())      # Highest price in period
lowest_price = float(df['Low'].min())        # Lowest price in period

# Total return calculation: ((Final - Initial) / Initial) * 100
total_return = ((current_price - opening_price) / opening_price) * 100

# Daily returns calculation: (Price_today - Price_yesterday) / Price_yesterday
df = df.copy()  # Avoid SettingWithCopyWarning
df['Daily_Return'] = df['Close'].pct_change()  # This gives decimal form
df['Daily_Return_Pct'] = df['Daily_Return'] * 100  # Convert to percentage

# Remove NaN values for calculations
valid_returns = df['Daily_Return'].dropna()
valid_returns_pct = df['Daily_Return_Pct'].dropna()

# Statistical measures
avg_daily_return = float(valid_returns_pct.mean())  # Average daily return %
median_daily_return = float(valid_returns_pct.median())  # Median daily return %
volatility = float(valid_returns_pct.std())  # Standard deviation of daily returns
total_days = len(df)
trading_days = len(valid_returns)  # Exclude first day (no return calculation possible)

# Moving averages (only calculate where we have enough data)
df['MA_Short'] = df['Close'].rolling(window=ma_short, min_periods=ma_short).mean()
df['MA_Long'] = df['Close'].rolling(window=ma_long, min_periods=ma_long).mean()

# Count up/down/flat days (excluding first day with NaN return)
up_days = int((valid_returns > 0).sum())
down_days = int((valid_returns < 0).sum()) 
flat_days = int((valid_returns == 0).sum())

# Verify our counts
assert up_days + down_days + flat_days == trading_days, "Day counts don't match!"

# Volume statistics
total_volume = int(df['Volume'].sum())
avg_volume = float(df['Volume'].mean())
max_volume = int(df['Volume'].max())

# Price range calculations
price_range = highest_price - lowest_price
price_range_pct = (price_range / lowest_price) * 100

# ----------------------
# DISPLAY BASIC INFO WITH VERIFIED CALCULATIONS
# ----------------------
st.subheader(f'📊 Stock Analysis for {user_input}')
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('Current Price', f"${current_price:.2f}")
with col2:
    delta_color = "normal" if total_return >= 0 else "inverse"
    st.metric('Total Return', f"{total_return:.2f}%", 
              delta=f"{total_return:.2f}%")
with col3:
    st.metric('Avg Daily Return', f"{avg_daily_return:.3f}%")
with col4:
    st.metric('Volatility (StdDev)', f"{volatility:.3f}%")

# Additional metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Highest Price', f"${highest_price:.2f}")
with col2:
    st.metric('Lowest Price', f"${lowest_price:.2f}")
with col3:
    st.metric('Price Range', f"${price_range:.2f}")
with col4:
    st.metric('Avg Volume', f"{avg_volume:,.0f}")

# Data verification info
st.subheader('📋 Data Overview & Verification')
col1, col2 = st.columns([3, 1])
with col1:
    # Show last 10 rows with calculated returns
    display_df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return_Pct']].round(5)
    st.dataframe(display_df, use_container_width=True)
with col2:
    st.info(f"""
    Period: {start_date} to {end_date}
    
    Total Calendar Days: {total_days}
    \nTrading Days: {trading_days}
    
    Return Calculation 🢃
    \n• First Price: ${opening_price:.2f}
    \n• Last Price: ${current_price:.2f}
    \n• Total Change: ${current_price - opening_price:.2f}
    \n• Total Return: {total_return:.2f}%
    
    Day Count Verification 🢃
    \n• Up Days: {up_days}
    \n• Down Days: {down_days}  
    \n• Flat Days: {flat_days}
    \n• Total: {up_days + down_days + flat_days}
    """)
st.markdown("---")
# ----------------------
# VISUALIZATION 1: Price Trend (Mathematically Accurate)
# ----------------------
st.subheader('📈 Price Trend Over Time')
fig1 = go.Figure()

# Main price line
fig1.add_trace(go.Scatter(
    x=df.index, 
    y=df['Close'], 
    mode='lines',
    name='Close Price',
    line=dict(color='blue', width=2),
    hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
))

# Add moving averages only where they exist (no NaN values)
if show_ma:
    ma_short_data = df['MA_Short'].dropna()
    if len(ma_short_data) > 0:
        fig1.add_trace(go.Scatter(
            x=ma_short_data.index,
            y=ma_short_data,
            mode='lines',
            name=f'{ma_short}-Day MA',
            line=dict(color='orange', width=1),
            hovertemplate=f'{ma_short}-Day MA: $%{{y:.2f}}<extra></extra>'
        ))
    
    ma_long_data = df['MA_Long'].dropna()
    if len(ma_long_data) > 0:
        fig1.add_trace(go.Scatter(
            x=ma_long_data.index,
            y=ma_long_data,
            mode='lines',
            name=f'{ma_long}-Day MA',
            line=dict(color='red', width=1),
            hovertemplate=f'{ma_long}-Day MA: $%{{y:.2f}}<extra></extra>'
        ))

fig1.update_layout(
    title=f'{user_input} Stock Price Movement',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    hovermode='x unified',
    height=400
)
st.plotly_chart(fig1, use_container_width=True)
st.markdown("---")
# ----------------------
# VISUALIZATION 2: Monthly Performance (Correct Calculation)
# ----------------------
# --- Robust monthly returns + plotting + metrics (drop into your app) ---
# Assumes: `df` has a 'Close' column, `user_input` is defined, and imports for pandas/plotly/streamlit exist

# 1) get last close of each month and month-over-month return (in %)
monthly_close = df['Close'].resample('M').last()
monthly_returns = monthly_close.pct_change().dropna() * 100  # Series or maybe DataFrame

# 2) if monthly_returns is a DataFrame, pick the first column (safer than crashing)
if isinstance(monthly_returns, pd.DataFrame):
    monthly_returns = monthly_returns.iloc[:, 0]

if len(monthly_returns) > 0:
    # ensure index are datetimes and build strings like 'YYYY-MM'
    months_dt = pd.to_datetime(monthly_returns.index)
    month_labels = [d.strftime('%Y-%m') for d in months_dt]

    # ensure returns is a 1D numpy array
    returns = monthly_returns.values.flatten()

    # prepare DataFrame for plotting (both columns are 1-D)
    monthly_data = pd.DataFrame({
        'Month': month_labels,
        'Return': returns
    })

    # color bars
    colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in returns]

    # Plotly bar chart
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=monthly_data['Month'],
        y=monthly_data['Return'],
        name='Monthly Return %',
        marker_color=colors,
        hovertemplate='Month: %{x}<br>Return: %{y:.2f}%<extra></extra>'
    ))
    fig2.update_layout(
        title=f'{user_input} Monthly Returns',
        xaxis_title='Month',
        yaxis_title='Return (%)',
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- statistics with month labels ---
    sr = pd.Series(data=returns, index=month_labels)  # simple Series keyed by YYYY-MM strings
    avg_monthly_return = sr.mean()
    best_month_label = sr.idxmax()
    best_month_val = sr.max()
    worst_month_label = sr.idxmin()
    worst_month_val = sr.min()

    # helper for emoji arrow
    def arrow_emoji(val):
        if val > 0:
            return '🔼'
        elif val < 0:
            return '🔻'
        else:
            return '⏺️'

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Monthly Return", f"{avg_monthly_return:.2f}%")
    with col2:
        st.metric("Best Month", f"{arrow_emoji(best_month_val)} {best_month_label} ({best_month_val:.2f}%)")
    with col3:
        st.metric("Worst Month", f"{arrow_emoji(worst_month_val)} {worst_month_label} ({worst_month_val:.2f}%)")
else:
    st.info("Not enough data for monthly analysis. Need data spanning multiple months.")
st.markdown("---")
# ----------------------
# VISUALIZATION 3: Trading Days Distribution (Verified)
# ----------------------
st.subheader('🥧 Trading Days Distribution (Pie Chart)')

# Double-check our calculations
up_pct = (up_days / trading_days) * 100
down_pct = (down_days / trading_days) * 100
flat_pct = (flat_days / trading_days) * 100

# Verify percentages add up to 100%
total_pct = up_pct + down_pct + flat_pct
assert abs(total_pct - 100.0) < 0.001, f"Percentages don't add to 100%: {total_pct}"

fig3 = go.Figure()
fig3.add_trace(go.Pie(
    labels=['Up Days', 'Down Days', 'Flat Days'],
    values=[up_days, down_days, flat_days],
    hole=0.4,
    marker=dict(colors=['green', 'red', 'gray']),
    textinfo='label+percent+value',
    hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
))

fig3.update_layout(
    title=f'{user_input} Price Movement Distribution ({trading_days} Trading Days)',
    height=400
)

col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(fig3, use_container_width=True)
with col2:
    st.write("**Verified Summary:**")
    st.write(f"• Up Days: {up_days} ({up_pct:.1f}%)")
    st.write(f"• Down Days: {down_days} ({down_pct:.1f}%)")  
    st.write(f"• Flat Days: {flat_days} ({flat_pct:.1f}%)")
    st.write(f"• **Total: {up_days + down_days + flat_days} ({total_pct:.1f}%)**")
    
    # Additional statistics
    if trading_days > 0:
        win_rate = up_pct
        st.write(f"\n**Win Rate: {win_rate:.1f}%**")
        
        if up_days > 0 and down_days > 0:
            avg_up_return = float(valid_returns_pct[valid_returns_pct > 0].mean())
            avg_down_return = float(valid_returns_pct[valid_returns_pct < 0].mean())
            st.write(f"• Avg Up Day: +{avg_up_return:.2f}%")
            st.write(f"• Avg Down Day: {avg_down_return:.2f}%")
st.markdown("---")
# ----------------------
# VISUALIZATION 4: Volume Analysis (Accurate)
# ----------------------
st.subheader('📊 Trading Volume Analysis')

fig4 = go.Figure()

# Volume bars
fig4.add_trace(go.Bar(
    x=df.index,
    y=df['Volume'],
    name='Daily Volume',
    marker_color='lightblue',
    hovertemplate='Date: %{x}<br>Volume: %{y:,}<extra></extra>'
))

# Add average volume line
fig4.add_hline(
    y=avg_volume, 
    line_dash="dash", 
    line_color="red",
    annotation_text=f"Avg Volume: {avg_volume:,.0f}"
)

# Add median volume line for comparison
median_volume = float(df['Volume'].median())
fig4.add_hline(
    y=median_volume,
    line_dash="dot",
    line_color="orange", 
    annotation_text=f"Median Volume: {median_volume:,.0f}"
)

fig4.update_layout(
    title=f'{user_input} Trading Volume',
    xaxis_title='Date',
    yaxis_title='Volume',
    height=350
)
st.plotly_chart(fig4, use_container_width=True)

# Volume statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Volume", f"{total_volume:,}")
with col2:
    st.metric("Average Volume", f"{avg_volume:,.0f}")
with col3:
    st.metric("Median Volume", f"{median_volume:,.0f}")
with col4:
    st.metric("Max Volume", f"{max_volume:,}")
st.markdown("---")
# ----------------------
# LSTM MODEL PREDICTION (With Verified Calculations)
# ----------------------
st.subheader('🤖 LSTM Model Predictions')

# Prepare data with correct calculations
prices = df['Close'].values.reshape(-1, 1)
train_len = int(np.ceil(len(prices) * (1 - test_split/100.0)))

# Verify split calculation
test_len = len(prices) - train_len
actual_test_split = (test_len / len(prices)) * 100

st.write(f"**Data Split Verification:**")
st.write(f"• Total Data Points: {len(prices)}")
st.write(f"• Training Points: {train_len} ({((train_len/len(prices))*100):.1f}%)")
st.write(f"• Testing Points: {test_len} ({actual_test_split:.1f}%)")
st.write(f"• Requested Test Split: {test_split}%")

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Create training and testing datasets
train_data = scaled_data[:train_len]
test_data = scaled_data[train_len - window_size:]  # Include window_size points before test

# Create sequences for testing
x_test, y_test = create_sequences(test_data, window_size)

if len(x_test) == 0:
    st.error(f"Not enough data for testing. Need at least {window_size + 1} points after training split.")
    st.stop()

x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Load and use model
with st.spinner('Loading and running LSTM model...'):
    model = load_or_get_model(uploaded_model)

if model is None:
    st.warning('⚠️ No LSTM model found. Upload a model file to enable predictions.')
    st.info('You can still view the stock analysis above.')
else:
    # Make predictions
    with st.spinner('Generating predictions...'):
        predictions_scaled = model.predict(x_test, verbose=0)
        predictions = scaler.inverse_transform(predictions_scaled)
        actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate accuracy metrics (mathematically correct)
    mae = float(np.mean(np.abs(predictions - actual_values)))
    rmse = float(np.sqrt(np.mean((predictions - actual_values)**2)))
    
    # MAPE calculation with proper handling of zero values
    non_zero_mask = actual_values != 0
    if np.sum(non_zero_mask) > 0:
        mape = float(np.mean(np.abs((actual_values[non_zero_mask] - predictions[non_zero_mask]) / actual_values[non_zero_mask])) * 100)
    else:
        mape = float('inf')  # Handle case where all actual values are zero
    
    # R-squared calculation
    ss_res = np.sum((actual_values - predictions) ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Mean Absolute Error', f"${mae:.3f}")
    with col2:
        st.metric('Root Mean Square Error', f"${rmse:.3f}")
    with col3:
        if mape != float('inf'):
            st.metric('Mean Absolute % Error', f"{mape:.2f}%")
        else:
            st.metric('Mean Absolute % Error', "N/A")
    with col4:
        st.metric('R-squared', f"{r_squared:.4f}")
    
    # Verify prediction data
    st.write(f"**Prediction Verification:**")
    st.write(f"• Test sequences created: {len(x_test)}")
    st.write(f"• Predictions generated: {len(predictions)}")
    st.write(f"• Actual values: {len(actual_values)}")
    
    # Plot predictions vs actual
    st.subheader('🎯 Predictions vs Actual Prices')
    
    # Create test dates (ensure correct alignment)
    test_start_idx = train_len
    prediction_dates = df.index[test_start_idx:test_start_idx + len(predictions)]
    
    # Verify date alignment
    if len(prediction_dates) != len(predictions):
        st.error(f"Date alignment error: {len(prediction_dates)} dates vs {len(predictions)} predictions")
        st.stop()
    
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=prediction_dates,
        y=actual_values.flatten(),
        mode='lines',
        name='Actual Prices',
        line=dict(color='blue', width=2),
        hovertemplate='Date: %{x}<br>Actual: $%{y:.2f}<extra></extra>'
    ))
    
    fig5.add_trace(go.Scatter(
        x=prediction_dates,
        y=predictions.flatten(),
        mode='lines',
        name='Predicted Prices',
        line=dict(color='red', width=2, dash='dot'),
        hovertemplate='Date: %{x}<br>Predicted: $%{y:.2f}<extra></extra>'
    ))
    
    fig5.update_layout(
        title=f'{user_input} Actual vs Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("---")
    # Future forecast
    if forecast_days > 0:
        st.subheader(f'🔮 Future Forecast ({forecast_days} days)')
        
        with st.spinner('Generating future forecast...'):
            # Use last window_size points for forecasting
            last_sequence = scaled_data[-window_size:].reshape(-1)
            future_predictions = forecast_future(model, last_sequence, scaler, forecast_days, window_size)
        
        # Create future dates (business days only)
        last_date = df.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        # Verify forecast data
        st.write(f"**Forecast Verification:**")
        st.write(f"• Last training date: {last_date}")
        st.write(f"• Forecast start date: {future_dates[0]}")
        st.write(f"• Forecast end date: {future_dates[-1]}")
        st.write(f"• Forecast points: {len(future_predictions)}")
        
        fig6 = go.Figure()
        
        # Show last 30 days of actual data
        recent_data = df['Close'].tail(30)
        fig6.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data.values,
            mode='lines',
            name='Recent Actual',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Show forecast
        fig6.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange', width=2),
            marker=dict(size=6),
            hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
        ))
        
        fig6.update_layout(
            title=f'{user_input} Future Price Forecast',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=400
        )
        st.plotly_chart(fig6, use_container_width=True)
        
        # Show forecast values in table
        st.write("**Forecasted Prices:**")
        forecast_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Day': [d.strftime('%A') for d in future_dates],
            'Forecasted_Price': future_predictions,
            'Formatted_Price': [f"${price:.2f}" for price in future_predictions]
        })
        
        # Calculate forecast statistics
        forecast_change = future_predictions[-1] - current_price
        forecast_change_pct = (forecast_change / current_price) * 100
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(forecast_df[['Date', 'Day', 'Formatted_Price']], use_container_width=True)
        with col2:
            st.write("**Forecast Summary:**")
            st.write(f"• Current Price: ${current_price:.2f}")
            st.write(f"• Forecast End: ${future_predictions[-1]:.2f}")
            st.write(f"• Expected Change: {forecast_change:+.2f}")
            st.write(f"• Expected Return: {forecast_change_pct:+.2f}%")
st.markdown("---")
# ----------------------
# MATHEMATICALLY VERIFIED SUMMARY
# ----------------------
st.subheader('💡 Key Insights (Verified Calculations)')

col1, col2 = st.columns(2)

with col1:
    st.write("**1. Performance Summary:**")
    performance_emoji = "🟢" if total_return > 0 else "🔴" if total_return < 0 else "🟡"
    st.write(f"{performance_emoji} **I. Total Return:** {total_return:.3f}%")
    st.write(f"  From  ${opening_price:.2f}  \tto  \t${current_price:.2f}")
    
    trend_emoji = "🟢" if avg_daily_return > 0 else "🔴" if avg_daily_return < 0 else "🟡"
    st.write(f"{trend_emoji} **II. Avg Daily Return:** {avg_daily_return:.3f}%")
    st.write(f"  Median: {median_daily_return:.3f}%")
    
    volatility_level = "High" if volatility > 3 else "Medium" if volatility > 1.5 else "Low"
    st.write(f"📊 **III. Volatility:** {volatility:.3f}% ({volatility_level})")
    st.write(f"  Range: ${lowest_price:.2f} - ${highest_price:.2f}")

with col2:
    st.write("**2. Trading Activity (Verified):**")
    st.write(f"📈 **I. Bullish Days:** {up_days}/{trading_days} ({up_pct:.1f}%)")
    if up_days > 0:
        avg_gain = float(valid_returns_pct[valid_returns_pct > 0].mean())
        st.write(f"  Avg Gain: +{avg_gain:.3f}%")
    
    st.write(f"📉 **II. Bearish Days:** {down_days}/{trading_days} ({down_pct:.1f}%)")
    if down_days > 0:
        avg_loss = float(valid_returns_pct[valid_returns_pct < 0].mean())
        st.write(f"  Avg Loss: {avg_loss:.3f}%")
    
    st.write(f"📊 **III. Volume:** {avg_volume:,.0f} avg, {max_volume:,} max")

if model is not None and 'mape' in locals():
    st.write("**3. Model Performance (Verified):**")
    if mape != float('inf'):
        accuracy = max(0, 100 - mape)
        accuracy_level = "Excellent" if accuracy > 95 else "Good" if accuracy > 90 else "Medium" if accuracy > 80 else "Fair" if accuracy > 70 else "Poor"
        st.write(f"🎯 **Prediction Accuracy:** {accuracy:.2f}% ({accuracy_level})")
    else:
        st.write(f"🎯 **Prediction Accuracy:** Cannot calculate (zero actual values)")
    
    st.write(f"📏 **Average Error:** ${mae:.3f} ({(mae/current_price)*100:.2f}% of current price)")
    st.write(f"📐 **R-squared:** {r_squared:.4f}")
    st.write(f"  Test samples: {len(predictions)}")

# Final verification summary
st.write("---")
st.write("**🔍 Calculation Verification Summary:**")
verification_checks = [
    f"✅ Data points: {len(df)} total, {trading_days} with returns",
    f"✅ Return calculation: ({current_price:.2f} - {opening_price:.2f}) / {opening_price:.2f} * 100 = {total_return:.3f}%",
    f"✅ Day counts: {up_days} + {down_days} + {flat_days} = {up_days + down_days + flat_days} = {trading_days}",
    f"✅ Percentages: {up_pct:.1f}% + {down_pct:.1f}% + {flat_pct:.1f}% = {total_pct:.1f}%"
]

if model is not None:
    verification_checks.append(f"✅ Train/Test split: {train_len} / {test_len} = {((train_len/len(prices))*100):.1f}% / {actual_test_split:.1f}%")
    if 'predictions' in locals():
        verification_checks.append(f"✅ Predictions: {len(x_test)} test sequences → {len(predictions)} predictions")

for check in verification_checks:
    st.write(check)

# --- Footer ---
st.markdown("---")
st.markdown("Developed by **Jeet Bhowmick**")



