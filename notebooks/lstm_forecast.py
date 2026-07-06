"""
LSTM forecasting model for Rossmann Store 262 daily sales.
Alternative to the SARIMA model built in sarima_analysis.Rmd, using the
same train/test split and log transformation for direct comparison.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# ============================================
# Part 0: Load and prepare data (same pipeline as ARIMA)
# ============================================
rossmann = pd.read_csv("../data/train.csv", low_memory=False)
store262 = rossmann[(rossmann['Store'] == 262) & (rossmann['Open'] == 1)].copy()
store262['Date'] = pd.to_datetime(store262['Date'])
store262 = store262.sort_values('Date')

raw_sales = store262['Sales'].values
data = np.log(raw_sales)  # log transformation, consistent with ARIMA
print("Total observations:", len(data))

# ============================================
# Part 1: Create sequences (following course's Part 2.2 multi-step approach)
# ============================================
def create_sequences(data, look_back, look_forward):
    X, y = [], []
    for i in range(len(data) - look_back - look_forward + 1):
        X.append(data[i:(i + look_back)])
        y.append(data[(i + look_back):(i + look_back + look_forward)])
    return np.array(X), np.array(y)

look_back = 18      # 3 full seasonal cycles (6 days/cycle)
look_forward = 10   # forecast horizon, matches ARIMA's test set size

X, y = create_sequences(data, look_back, look_forward)
print("X shape:", X.shape, "y shape:", y.shape)

# ============================================
# Part 2: Train/test split
# Last sequence's target = the final 10 points of the series (matches ARIMA's Y_test)
# ============================================
split_index = len(X) - 1
X_train, X_test = X[:split_index], X[split_index:split_index+1]
y_train, y_test = y[:split_index], y[split_index:split_index+1]

X_train = X_train.reshape(-1, look_back, 1)
X_test = X_test.reshape(-1, look_back, 1)

print("X_train:", X_train.shape, "X_test:", X_test.shape)

# ============================================
# Part 3: Build and train LSTM model
# ============================================
tf.random.set_seed(66)
model = Sequential([
    LSTM(32, activation='relu', input_shape=(look_back, 1)),
    Dense(look_forward)
])
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# ============================================
# Part 4: Forecast and convert back to original scale (EUR)
# ============================================
y_pred = model.predict(X_test, verbose=0)

pred_original = np.exp(y_pred[0])
actual_original = np.exp(y_test[0])

print("\nPredicted (EUR):", np.round(pred_original, 1))
print("Actual (EUR):", actual_original)

# ============================================
# Part 5: RMSE
# ============================================
rmse_lstm = np.sqrt(np.mean((pred_original - actual_original) ** 2))
print("\nLSTM RMSE (original scale):", rmse_lstm)

# ============================================
# Part 6: Plot forecast superimposed on raw data
# ============================================
n = len(raw_sales)
test_start = n - 10

plt.figure(figsize=(12, 5))
plt.plot(range(n), raw_sales, label="Actual (Raw Data)", color="black")
plt.plot(range(test_start, n), pred_original, label="LSTM Forecast",
         color="red", linewidth=2)
plt.axvline(x=test_start, linestyle="--", color="gray")
plt.xlabel("Time")
plt.ylabel("Sales (EUR)")
plt.title("LSTM Forecast vs Actual Sales (Store 262)")
plt.legend()
plt.savefig("../lstm_forecast_plot.png", dpi=150, bbox_inches='tight')
plt.show()
