import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    return pd.read_csv("TSLA.csv")

df = load_data()

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Deep Learning Model",
    ["SimpleRNN", "Stacked LSTM"]
)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model(model_name):
    if model_name == "SimpleRNN":
        return tf.keras.models.load_model(
            "tesla_simple_rnn.keras",
            compile=False
        )
    else:
        return tf.keras.models.load_model(
            "tesla_stacked_lstm_model.keras",
            compile=False
        )

model = load_model(model_choice)

# =====================================
# HEADER
# =====================================
st.title("📈 Tesla Stock Price Prediction Dashboard")

st.markdown("""
### Deep Learning Models Used
- SimpleRNN
- Stacked LSTM

### Forecast Horizons
- 1 Day
- 5 Days
- 10 Days

### Target Variable
- Tesla Close Price
""")

# =====================================
# DATASET OVERVIEW
# =====================================
st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.dataframe(df.head(), use_container_width=True)

# =====================================
# KPI CARDS
# =====================================
latest_close = float(df["Close"].iloc[-1])
highest_close = float(df["Close"].max())
lowest_close = float(df["Close"].min())

st.subheader("📌 Tesla Stock KPIs")

c1, c2, c3 = st.columns(3)

c1.metric("Latest Close", f"${latest_close:.2f}")
c2.metric("Highest Close", f"${highest_close:.2f}")
c3.metric("Lowest Close", f"${lowest_close:.2f}")

# =====================================
# CLOSE PRICE TREND
# =====================================
st.subheader("📈 Tesla Closing Price Trend")

fig1, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(df["Close"])
ax1.set_title("Tesla Closing Price")
ax1.set_xlabel("Days")
ax1.set_ylabel("Price ($)")
ax1.grid(True)

st.pyplot(fig1)

# =====================================
# VOLUME TREND
# =====================================
st.subheader("📊 Trading Volume")

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(df["Volume"])
ax2.set_title("Tesla Trading Volume")
ax2.set_xlabel("Days")
ax2.set_ylabel("Volume")
ax2.grid(True)

st.pyplot(fig2)

# =====================================
# PREPROCESSING
# =====================================
price_column = "Close"

data = df[[price_column]].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

TIME_STEP = 60

def create_sequences(dataset, time_step):
    X = []

    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])

    return np.array(X)

X = create_sequences(scaled_data, TIME_STEP)

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

# =====================================
# MODEL PREDICTIONS
# =====================================
predictions = model.predict(X, verbose=0)

predictions = scaler.inverse_transform(predictions)

actual = data[TIME_STEP:]

# =====================================
# ACTUAL VS PREDICTED
# =====================================
st.subheader(f"🤖 Actual vs Predicted ({model_choice})")

fig3, ax3 = plt.subplots(figsize=(12, 5))

ax3.plot(actual, label="Actual")
ax3.plot(predictions, label="Predicted")

ax3.set_title("Actual vs Predicted Tesla Prices")
ax3.legend()
ax3.grid(True)

st.pyplot(fig3)

# =====================================
# METRICS
# =====================================
mae = np.mean(np.abs(actual - predictions))
mse = np.mean((actual - predictions) ** 2)
rmse = np.sqrt(mse)

st.subheader("📉 Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric("MAE", f"{mae:.4f}")
m2.metric("MSE", f"{mse:.4f}")
m3.metric("RMSE", f"{rmse:.4f}")

# =====================================
# FORECAST FUNCTION
# =====================================
def forecast_future(model, scaled_data, scaler, time_step, days):

    temp_input = scaled_data[-time_step:].flatten().tolist()

    output = []

    for _ in range(days):

        x_input = np.array(
            temp_input[-time_step:]
        )

        x_input = x_input.reshape(
            1,
            time_step,
            1
        )

        pred = model.predict(
            x_input,
            verbose=0
        )[0][0]

        output.append(pred)

        temp_input.append(pred)

    output = np.array(output).reshape(-1, 1)

    output = scaler.inverse_transform(output)

    return output.flatten()

# =====================================
# FORECASTS
# =====================================
forecast_1 = forecast_future(
    model,
    scaled_data,
    scaler,
    TIME_STEP,
    1
)

forecast_5 = forecast_future(
    model,
    scaled_data,
    scaler,
    TIME_STEP,
    5
)

forecast_10 = forecast_future(
    model,
    scaled_data,
    scaler,
    TIME_STEP,
    10
)

# =====================================
# FORECAST CARDS
# =====================================
st.subheader("🔮 Future Forecasts")

f1, f2, f3 = st.columns(3)

f1.metric(
    "1-Day Forecast",
    f"${forecast_1[-1]:.2f}"
)

f2.metric(
    "5-Day Forecast",
    f"${forecast_5[-1]:.2f}"
)

f3.metric(
    "10-Day Forecast",
    f"${forecast_10[-1]:.2f}"
)

# =====================================
# FORECAST CHART
# =====================================
st.subheader("📈 Future Forecast Trend")

fig4, ax4 = plt.subplots(figsize=(12, 5))

historical = data[-60:].flatten()

ax4.plot(
    range(len(historical)),
    historical,
    label="Historical"
)

future_x = range(
    len(historical),
    len(historical) + len(forecast_10)
)

ax4.plot(
    future_x,
    forecast_10,
    marker="o",
    label="Forecast"
)

ax4.set_title("10-Day Tesla Forecast")
ax4.legend()
ax4.grid(True)

st.pyplot(fig4)

# =====================================
# FORECAST TABLE
# =====================================
forecast_df = pd.DataFrame({
    "Day": np.arange(1, 11),
    "Predicted Close Price": forecast_10
})

st.subheader("📋 10-Day Forecast Table")

st.dataframe(
    forecast_df,
    use_container_width=True
)

# =====================================
# MODEL COMPARISON NOTE
# =====================================
st.subheader("📚 Model Comparison")

st.info("""
Use the sidebar to switch between:

• SimpleRNN

• Stacked LSTM

Compare:
- Actual vs Predicted Charts
- MAE
- MSE
- RMSE

and determine which model performs better.
""")

# =====================================
# FOOTER
# =====================================
st.markdown("---")

st.markdown(
    "🚀 Tesla Stock Price Prediction using Deep Learning (SimpleRNN & Stacked LSTM)"
)
