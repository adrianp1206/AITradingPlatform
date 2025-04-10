# train_xgboost_wandb.py

import sys
import os
import xgboost as xgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from joblib import dump
import matplotlib.pyplot as plt
import wandb
import signal

# Add the parent directory (if needed) so that your data_processing module can be found
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_processing import fetch_stock_data_alpha, calculate_technical_indicators

# Define custom exception for timeout.
class TimeoutException(Exception):
    pass

# Define the timeout handler.
def timeout_handler(signum, frame):
    raise TimeoutException("Run exceeded 10-minute time limit.")

def train_xgboost(
    ticker,
    start_date='2008-01-01',
    end_date='2021-12-31',
    feature_subset=None,
    n_splits=5,
    params=None,
    save_model=True
):
    # Set default model parameters if not provided.
    if params is None:
        params = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    
    # Fetch data and calculate technical indicators.
    data = fetch_stock_data_alpha(ticker, 'NL9PDOM5JWRPAT9O', start_date=start_date, end_date=end_date)
    data = calculate_technical_indicators(data)
    
    fundamental_cols = [
        'DE Ratio', 'Return on Equity', 'Price/Book',
        'Profit Margin', 'Diluted EPS', 'Beta'
    ]
    data = data.drop(columns=[col for col in fundamental_cols if col in data.columns])
    
    # Create target: Predict tomorrow's price movement (shift the Close column by -1).
    data['Future_Close'] = data['Close'].shift(-1)
    data['Price_Change'] = data['Future_Close'] - data['Close']
    data['Target'] = (data['Price_Change'] > 0).astype(int)
    data = data.dropna()
    
    # Feature selection: Drop target-related columns ("Price_Change", "Target", "Close") if no subset is provided.
    if feature_subset is None:
        X = data.drop(columns=['Price_Change', 'Target', 'Close'], errors='ignore')
    else:
        X = data[feature_subset]
    y = data['Target']

    # Set up time series cross-validation.
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy_list.append(accuracy_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred, zero_division=0))
        recall_list.append(recall_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        roc_auc_list.append(roc_auc_score(y_test, y_pred_proba))
    
    cv_metrics = {
        'accuracy': np.mean(accuracy_list),
        'precision': np.mean(precision_list),
        'recall': np.mean(recall_list),
        'f1_score': np.mean(f1_list),
        'roc_auc': np.mean(roc_auc_list),
    }
    
    print("Cross-validation metrics:")
    for key, value in cv_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Retrain final model on all data.
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)
    
    # Optionally display confusion matrix for full training set.
    y_final_pred = final_model.predict(X)
    cm = confusion_matrix(y, y_final_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Full Training Data) for {ticker}")
    plt.show()
    
    # Save model if required.
    if save_model:
        model_filename = f"xgboost_{ticker}.joblib"
        dump(final_model, model_filename)
        print(f"Final model retrained on all data saved to {model_filename}")
    
    return cv_metrics, final_model

def run_experiment():
    # Register the timeout handler and set an alarm for 600 seconds (10 minutes).
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)
    
    try:
        # Initialize a wandb run.
        wandb.init(project="stock-xgb")
        config = wandb.config

        # Build the hyperparameters dict from wandb.config.
        params = {
            'learning_rate': config.learning_rate,
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'min_child_weight': config.min_child_weight,
            'subsample': config.subsample,
            'colsample_bytree': config.colsample_bytree,
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1,
        }
        
        feature_subset = [
            'HT_DCPERIOD', 'MEDPRICE', 'HT_DCPHASE', 'MACD_signal', 'HT_TRENDMODE',
            'STOCH_fastd', 'AROON_UP', 'HT_SINE', 'MINUS_DM', 'HT_PHASOR_quadrature',
            'BB_middle', 'NATR', 'ADXR', 'TYPPRICE', 'MIDPRICE', 'RSI', 'BB_lower',
            'AROONOSC', 'AROON_DOWN', 'ADOSC', 'DEMA', 'PLUS_DM', 'CMO', 'AVGPRICE',
            'MOM', 'ROC', 'STOCH_slowk', 'TRIMA', 'T3', 'MIDPOINT', 'SMA', 'CCI',
            'STOCH_fastk'
        ]
        
        # Run the training; adjust ticker and date ranges as necessary.
        cv_metrics, final_model = train_xgboost(
            ticker="JNJ",
            start_date='2008-01-01',
            end_date='2021-12-31',
            feature_subset=feature_subset,
            params=params,
            save_model=False  # Usually, you may not want to save models during a sweep.
        )
        
        # Cancel the alarm if run finishes in time.
        signal.alarm(0)
        wandb.log(cv_metrics)
        wandb.finish()
    
    except TimeoutException as e:
        print("Timeout exceeded: skipping this run.")
        wandb.log({"run_skipped_due_to_timeout": True})
        wandb.finish()
        return

if __name__ == '__main__':
    run_experiment()
