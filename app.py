
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Tesla Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------------
# LOAD DATA & MODEL
# --------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("TSLA.csv")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tesla_stacked_lstm_model.keras")

df = load_data()
model = load_model()

# --------------------------------
# HEADER
# --------------------------------
st.title("📈 Tesla Stock Prediction Dashboard")
st.markdown(
    "Deep Learning-based Tesla Stock Price Forecasting using LSTM Networks"
)

# --------------------------------
# DATA OVERVIEW
# --------------------------------
st.subheader("📊 Dataset Overview")

st.dataframe(df.head(), use_container_width=True)

# --------------------------------
# KPI CARDS
# --------------------------------
latest_close = float(df["Close"].iloc[-1])
highest_close = float(df["Close"].max())
lowest_close = float(df["Close"].min())

col1, col2, col3 = st.columns(3)

col1.metric("Latest Close", f"${latest_close:.2f}")
col2.metric("Highest Close", f"${highest_close:.2f}")
col3.metric("Lowest Close", f"${lowest_close:.2f}")

# --------------------------------
# STOCK PRICE CHART
# --------------------------------
st.subheader("📈 Tesla Closing Price Trend")

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df["Close"])
ax1.set_title("Tesla Closing Price")
ax1.set_xlabel("Trading Days")
ax1.set_ylabel("Price ($)")
ax1.grid(True)

st.pyplot(fig1)

# --------------------------------
# VOLUME CHART
# --------------------------------
st.subheader("📊 Trading Volume")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df["Volume"])
ax2.set_title("Tesla Trading Volume")
ax2.set_xlabel("Trading Days")
ax2.set_ylabel("Volume")
ax2.grid(True)

st.pyplot(fig2)

# --------------------------------
# PRICE COLUMN
# --------------------------------
price_column = "Close"
data = df[[price_column]].values

# --------------------------------
# SCALING
# --------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# --------------------------------
# SEQUENCE CREATION
# --------------------------------
time_step = 60

def create_sequences(data, time_step):
    X = []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
    return np.array(X)

X = create_sequences(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# --------------------------------
# PREDICTIONS
# --------------------------------
predictions = model.predict(X, verbose=0)
predictions = scaler.inverse_transform(predictions)

actual = data[time_step:]

# --------------------------------
# ACTUAL VS PREDICTED
# --------------------------------
st.subheader("🤖 Actual vs Predicted Stock Prices")

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(actual, label="Actual Price")
ax3.plot(predictions, label="Predicted Price")
ax3.set_title("Actual vs Predicted Tesla Prices")
ax3.legend()
ax3.grid(True)

st.pyplot(fig3)

# --------------------------------
# PERFORMANCE METRICS
# --------------------------------
mae = np.mean(np.abs(actual - predictions))
mse = np.mean((actual - predictions) ** 2)
rmse = np.sqrt(mse)

st.subheader("📉 Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric("MAE", f"{mae:.4f}")
m2.metric("MSE", f"{mse:.4f}")
m3.metric("RMSE", f"{rmse:.4f}")

# --------------------------------
# NEXT DAY PREDICTION
# --------------------------------
last_sequence = scaled_data[-time_step:]
input_data = last_sequence.reshape(1, time_step, 1)

next_price = model.predict(input_data, verbose=0)
next_price = scaler.inverse_transform(next_price)

st.subheader("🔮 Next-Day Forecast")

st.success(
    f"Predicted Next Tesla Closing Price: ${next_price[0][0]:.2f}"
)


# --------------------------------
# MULTI-DAY FORECASTING
# --------------------------------
st.subheader("🔮 Future Forecasts")

def forecast_future(model, scaled_data, scaler, time_step, days):
    temp_input = scaled_data[-time_step:].flatten().tolist()
    forecasts = []

    for _ in range(days):
        x_input = np.array(temp_input[-time_step:])
        x_input = x_input.reshape(1, time_step, 1)

        pred = model.predict(x_input, verbose=0)[0][0]

        forecasts.append(pred)
        temp_input.append(pred)

    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts = scaler.inverse_transform(forecasts)

    return forecasts.flatten()

# Generate forecasts
forecast_1 = forecast_future(model, scaled_data, scaler, time_step, 1)
forecast_5 = forecast_future(model, scaled_data, scaler, time_step, 5)
forecast_10 = forecast_future(model, scaled_data, scaler, time_step, 10)

# Metrics Cards
c1, c2, c3 = st.columns(3)

c1.metric(
    "1-Day Forecast",
    f"${forecast_1[-1]:.2f}"
)

c2.metric(
    "5-Day Forecast",
    f"${forecast_5[-1]:.2f}"
)

c3.metric(
    "10-Day Forecast",
    f"${forecast_10[-1]:.2f}"
)

# Forecast Chart
st.subheader("📈 Future Price Forecast")

fig4, ax4 = plt.subplots(figsize=(12, 5))

ax4.plot(
    range(len(data[-60:])),
    data[-60:],
    label="Historical Close Price"
)

future_x = range(
    len(data[-60:]),
    len(data[-60:]) + len(forecast_10)
)

ax4.plot(
    future_x,
    forecast_10,
    marker="o",
    label="10-Day Forecast"
)

ax4.set_title("Tesla Future Price Forecast")
ax4.set_xlabel("Days")
ax4.set_ylabel("Price ($)")
ax4.legend()
ax4.grid(True)

st.pyplot(fig4)

# Forecast Table
forecast_df = pd.DataFrame({
    "Day": np.arange(1, 11),
    "Forecast Price": forecast_10
})

st.subheader("📋 Forecast Table")
st.dataframe(forecast_df, use_container_width=True)


```
```

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("---")
st.markdown(
    "🚀 Built with Streamlit | TensorFlow | LSTM | Tesla Stock Analysis"
)

