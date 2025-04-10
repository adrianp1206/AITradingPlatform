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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_processing import fetch_stock_data_alpha, calculate_technical_indicators

def train_xgboost(
    ticker,
    start_date='2008-01-01',
    end_date='2021-12-31',
    feature_subset=None,
    n_splits=5,
    params=None,
    save_model=True,
    plot_cm=True  # New parameter to control plotting the confusion matrix
):
    import matplotlib.pyplot as plt

    # Set default model parameters if none provided
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
    
    # -----------------------------------------------------
    # 1. Fetch data and calculate technical indicators
    # -----------------------------------------------------
    data = fetch_stock_data_alpha(ticker, 'NL9PDOM5JWRPAT9O', start_date=start_date, end_date=end_date)
    data = calculate_technical_indicators(data)
    
    fundamental_cols = [
        'DE Ratio', 'Return on Equity', 'Price/Book',
        'Profit Margin', 'Diluted EPS', 'Beta'
    ]
    data = data.drop(columns=[col for col in fundamental_cols if col in data.columns])

    # ---------------------------------------------------------------------
    # 2. Create target: Predict tomorrow's price movement (shift Close by -1)
    # ---------------------------------------------------------------------
    data['Future_Close'] = data['Close'].shift(-1)
    data['Price_Change'] = data['Future_Close'] - data['Close']
    data['Target'] = (data['Price_Change'] > 0).astype(int)
    data = data.dropna()

    # -------------------------------------------------------
    # 3. Feature Selection: Ensure consistency with backtesting.
    # -------------------------------------------------------
    if feature_subset is None:
        # Exclude target related columns and "Close" for consistency if no subset is provided.
        X = data.drop(columns=['Price_Change', 'Target', 'Close'], errors='ignore')
    else:
        X = data[feature_subset]
    y = data['Target']

    # -----------------------------------------------------
    # 4. Time Series Cross-Validation to assess performance
    # -----------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Re-train final model on the entire dataset using the chosen params.
    # ------------------------------------------------------------------
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    # Optionally, show the confusion matrix on the full training set only if desired.
    # if plot_cm:
    #     y_final_pred = final_model.predict(X)
    #     cm = confusion_matrix(y, y_final_pred)
    #     ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot(cmap="Blues")
    #     plt.title(f"Confusion Matrix (Full Training Data) for {ticker}")
    #     plt.show()

    # ----------------------------
    # 6. Save the final trained model
    # ----------------------------
    if save_model:
        model_filename = f"xgboost_{ticker}_new.joblib"
        dump(final_model, model_filename)
        print(f"Final model retrained on all data saved to {model_filename}")

    return cv_metrics, final_model

