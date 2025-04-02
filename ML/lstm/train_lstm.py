from data_processing import fetch_stock_data, preprocess_data, create_lstm_input
from lstm_model import build_lstm_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from sklearn.metrics import r2_score

def train_lstm_model(ticker):
    stock_data = fetch_stock_data(ticker, '2008-01-01', '2021-12-31')
    stock_data, scaler = preprocess_data(stock_data)

    X, y = create_lstm_input(stock_data, target_column='Adj Close', lookback=20)
    
    train_size = int(0.7 * len(X))
    dev_size = int(0.15 * len(X))
    
    # Perform the splits: 70% training, 15% development, 15% testing
    X_train, y_train = X[:train_size], y[:train_size]
    X_dev, y_dev = X[train_size:train_size + dev_size], y[train_size:train_size + dev_size]
    X_test, y_test = X[train_size + dev_size:], y[train_size + dev_size:]

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    # Train the model using the training and development sets
    history = model.fit(X_train, y_train, epochs=90, batch_size=32, validation_data=(X_dev, y_dev), verbose=0)

    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['mean_squared_error'], label='Training MSE')
    plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
    plt.title('Training vs Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

    model.save(f"lstm_{ticker}_model.h5")

    test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}")

    y_pred = model.predict(X_test)

    # Inverse transform the predictions and actual values to original scale
    y_test_padded = np.zeros((len(y_test), scaler.min_.shape[0]))
    y_pred_padded = np.zeros((len(y_pred), scaler.min_.shape[0]))
    y_test_padded[:, 0] = y_test.flatten()
    y_pred_padded[:, 0] = y_pred.flatten()
    
    y_test_rescaled = scaler.inverse_transform(y_test_padded)[:, 0]
    y_pred_rescaled = scaler.inverse_transform(y_pred_padded)[:, 0]

    mape_test = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
    rmse_test = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    r2_test = r2_score(y_test_rescaled, y_pred_rescaled)

    print(f"\nMean Absolute Percentage Error (MAPE) for Test Set: {mape_test * 100:.2f}%")
    print(f"Root Mean Squared Error (RMSE) for Test Set: {rmse_test:.2f}")
    print(f"R² Score for Test Set: {r2_test:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label='Actual Prices', color='blue', marker='o')
    plt.plot(y_pred_rescaled, label='Predicted Prices', color='red', marker='x')
    plt.title('Predicted vs Actual Adjusted Close Prices for Test Set')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.5)
    plt.title('Scatter Plot of Predicted vs Actual Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.show()
