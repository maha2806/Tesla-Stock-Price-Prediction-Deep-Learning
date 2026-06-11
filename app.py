import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Prediction using Deep Learning")
st.markdown("Upload your dataset and trained model to predict stock prices using RNN/LSTM.")

# ---------------------------
# UPLOAD DATASET
# ---------------------------
st.subheader("📂 Upload Dataset (CSV)")
uploaded_file = st.file_uploader("Upload your stock price CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    st.write(df.head())
else:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

# ---------------------------
# SELECT COLUMN
# ---------------------------
price_column = st.selectbox("Select Price Column", df.columns)

data = df[[price_column]].values

# ---------------------------
# SCALE DATA
# ---------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# ---------------------------
# UPLOAD MODEL
# ---------------------------
st.subheader("🧠 Upload Trained Model (.keras)")
model_file = st.file_uploader("Upload your trained Keras model", type=["keras", "h5"])

if model_file is not None:
    with open("temp_model.keras", "wb") as f:
        f.write(model_file.read())

    model = tf.keras.models.load_model("temp_model.keras")
    st.success("Model loaded successfully!")
else:
    st.warning("Please upload a trained model file.")
    st.stop()

# ---------------------------
# CREATE SEQUENCES
# ---------------------------
time_step = st.slider("Time Step (Sequence Length)", 10, 120, 60)

def create_sequences(data, time_step):
    X = []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
    return np.array(X)

X = create_sequences(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ---------------------------
# PREDICTIONS
# ---------------------------
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

actual = data[time_step:]

# ---------------------------
# VISUALIZATION
# ---------------------------
st.subheader("📉 Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(actual, label="Actual Price")
ax.plot(predictions, label="Predicted Price")
ax.set_title("Stock Price Prediction")
ax.legend()

st.pyplot(fig)

# ---------------------------
# METRICS
# ---------------------------
mae = np.mean(np.abs(actual - predictions))
mse = np.mean((actual - predictions) ** 2)
rmse = np.sqrt(mse)

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.5f}")
col2.metric("MSE", f"{mse:.5f}")
col3.metric("RMSE", f"{rmse:.5f}")

# ---------------------------
# FUTURE PREDICTION
# ---------------------------
st.subheader("🔮 Next Step Prediction")

last_sequence = scaled_data[-time_step:]
input_data = last_sequence.reshape(1, time_step, 1)

next_price = model.predict(input_data)
next_price = scaler.inverse_transform(next_price)

st.success(f"Predicted Next Price: {next_price[0][0]:.2f}")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("🚀 Built using Streamlit | Deep Learning Stock Price Prediction")
