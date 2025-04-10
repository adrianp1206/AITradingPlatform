import wandb

sweep_config = {
    'method': 'bayes',  # Options: 'grid', 'random', or 'bayes'
    'name': 'xgboost-sweep',
    'program': 'train_boost_wandb.py',
    'metric': {
        'name': 'roc_auc',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.01, 0.05, 0.1, 0.2]
        },
        'n_estimators': {
            'values': [50, 100, 200]
        },
        'max_depth': {
            'values': [4, 6, 8]
        },
        'min_child_weight': {
            'values': [1, 3, 5]
        },
        'subsample': {
            'values': [0.75, 1.0]
        },
        'colsample_bytree': {
            'values': [0.75, 1.0]
        }
    },
    'count': 25
}

sweep_id = wandb.sweep(sweep_config, project="stock-xgb")
print("Sweep ID:", sweep_id)
