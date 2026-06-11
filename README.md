# Tesla-Stock-Price-Prediction-Deep-Learning
Deep Learning-based Tesla Stock Price Prediction using SimpleRNN, LSTM, and Stacked LSTM models with data visualization, model comparison, and Streamlit deployment.

## Project Overview

This project predicts Tesla stock prices using Deep Learning techniques, specifically:

- SimpleRNN
- LSTM
- Stacked LSTM

The objective is to forecast Tesla stock closing prices using historical stock market data and compare the performance of different recurrent neural network architectures.

---

## Problem Statement

Stock prices are sequential in nature and contain temporal dependencies.

This project applies Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) Networks to predict Tesla stock prices and analyze their forecasting performance.

---

## Dataset

Tesla Historical Stock Dataset

Features:

- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

Target Variable:

- Close Price

Dataset Source:

https://drive.google.com/file/d/1BHzdUi6-iKz7a3tnZunxcp_Td-7I24C7/view

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow
- Keras
- Streamlit

---

## Project Workflow

### Data Collection

Loaded Tesla stock dataset.

### Data Cleaning

- Missing value analysis
- Null value treatment

### Exploratory Data Analysis

- Trend Analysis
- Distribution Analysis
- Correlation Analysis
- Volume Analysis
- Price Movement Analysis

### Feature Engineering

Created additional features:

- Price Range
- Daily Return
- Moving Average (7 Days)
- Moving Average (30 Days)

### Data Scaling

Applied MinMaxScaler for neural network training.

### Sequence Creation

Prepared time-series sequences for:

- 1 Day Prediction
- 5 Day Prediction
- 10 Day Prediction

### Model Development

Implemented:

#### Model 1

SimpleRNN

#### Model 2

LSTM

#### Model 3

Stacked LSTM

### Hyperparameter Optimization

Used:

- Keras Tuner
- EarlyStopping
- ModelCheckpoint

### Model Evaluation

Evaluation Metrics:

- MAE
- MSE
- RMSE

### Deployment

Interactive Streamlit application deployed for real-time Tesla stock prediction.

---

## Model Comparison

| Model       | MAE      | MSE      | RMSE     | Performance Insight                                          |
| ----------- | -------- | -------- | -------- | ------------------------------------------------------------ |
| **SimpleRNN** | 0.015351 | 0.000528 | 0.022989 | ⭐ Best performance (lowest error, most accurate predictions) |
| **LSTM** | 0.017728 | 0.000653 | 0.025545 | Moderate performance, slight increase in error               |
| **Stacked LSTM** | 0.020515 | 0.000951 | 0.030845 | Weakest performance, highest deviation                       |

Final Selected Model: Simple RNN

✅ Simple RNN

---

## Business Impact

### Automated Trading

Supports buy/sell decision making.

### Portfolio Optimization

Assists investors in managing risk.

### Financial Forecasting

Provides future stock trend insights.

### Research Applications

Acts as a baseline for advanced forecasting models.

---

## Streamlit Application

Features:

- Tesla Stock Overview
- Historical Trend Visualization
- Deep Learning Prediction
- Actual vs Predicted Graph
- Model Performance Metrics

---

## Future Improvements

- News Sentiment Analysis
- Social Media Sentiment
- Transformer Models
- GRU Networks
- Macroeconomic Indicators Integration

---

## Author

Sita Bharatula

Master of Computer Applications

Chandigarh University

---

## License

This project is developed for educational and research purposes.
