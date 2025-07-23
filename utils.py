import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import gc
from datetime import datetime
import os

# 1) CONFIGS
model_configs = {
    'LightGBM': {
        'estimator_class': lgb.LGBMRegressor,
        'params': {
            'n_estimators': [500, 1000],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [20, 31],
            'max_depth': [5, 10, -1],
            'min_child_samples': [20, 50],
        }
    },
    'RandomForest': {
        'estimator_class': RandomForestRegressor,
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'GradientBoosting': {
        'estimator_class': GradientBoostingRegressor,
        'params': {
            'n_estimators': [300, 500],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
}

# 2) GENERATORS
def param_product_grid(param_dict):
    keys = list(param_dict.keys())
    value_lists = [
        v if isinstance(v, (list, tuple)) else [v]
        for v in [param_dict[k] for k in keys]
    ]
    for combo in itertools.product(*value_lists):
        yield dict(zip(keys, combo))

def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# 3) EVALUATION
def compute_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

# 4) MAIN TRAINING FUNCTION
def train_and_evaluate_models_batch(
        X_train, y_train,
        X_test, y_test, 
        selected_models=('LightGBM',),
        batch_size=10,
        top_k=1,
        early_stop_rounds=50,
        save_batches=True,
        tscv=TimeSeriesSplit(n_splits=5, test_size=45000),
        random_state=42):
    
    print("Initializing model training...")
    
    models_dict = {}
    cv_results = []
    test_results = []
    best_scores = {}

    for model_name in selected_models:
        if model_name not in model_configs:
            print(f"[WARN] {model_name} not in model_configs, skip.")
            continue

        cfg = model_configs[model_name]
        Est = cfg['estimator_class']
        param_space = cfg['params']

        print(f"\n===== {model_name} SEARCH (batch={batch_size}) =====")
        gen = param_product_grid(param_space)
        batch_counter = 0
        best_heap = []  # list of tuples (cv_rmse, params, aux_info)

        for param_batch in batched(gen, batch_size):
            batch_counter += 1
            batch_records = []
            print(f"  Batch {batch_counter} size={len(param_batch)} ...")

            for params in param_batch:
                params = dict(params)
                # common defaults
                if model_name == 'LightGBM':
                    params.setdefault('random_state', random_state)
                    params.setdefault('verbosity', -1)
                else:
                    params.setdefault('random_state', random_state)

                fold_rmses = []
                fold_best_iters = []

                # Cross-validation
                for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train), start=1):
                    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

                    if model_name == 'LightGBM':
                        est = Est(**params)
                        est.fit(
                            X_tr, y_tr,
                            eval_set=[(X_va, y_va)],
                            eval_metric='rmse',
                            callbacks=[lgb.early_stopping(stopping_rounds=early_stop_rounds,
                                                          verbose=False)]
                        )
                        pred = est.predict(X_va, num_iteration=est.best_iteration_)
                        best_iter = est.best_iteration_ or est.n_estimators
                        fold_best_iters.append(best_iter)
                        del est
                    else:
                        est = Est(**params)
                        est.fit(X_tr, y_tr)
                        pred = est.predict(X_va)
                        # No concept of best iteration for non-LightGBM models
                        del est

                    mse = mean_squared_error(y_va, pred)
                    rmse = mse ** 0.5

                    fold_rmses.append(rmse)

                mean_rmse = float(np.mean(fold_rmses))
                std_rmse  = float(np.std(fold_rmses))
                avg_best_iter = int(np.mean(fold_best_iters)) if fold_best_iters else None

                batch_records.append({
                    'Model': model_name,
                    'Params': params,
                    'CV_RMSE': mean_rmse,
                    'CV_RMSE_std': std_rmse,
                    'Avg_Best_Iter': avg_best_iter
                })

                # update best_heap
                best_heap.append((mean_rmse, params, avg_best_iter))
                best_heap.sort(key=lambda x: x[0])
                if len(best_heap) > top_k:
                    best_heap = best_heap[:top_k]

            if save_batches:
                # Create model_checkpoints directory if it doesn't exist
                os.makedirs('model_checkpoints', exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join('model_checkpoints', f'{model_name}_batch_{batch_counter}_{ts}.pkl')
                print(f"Saving batch results to: {save_path}")
                joblib.dump(batch_records, save_path)

            # dump memory
            del batch_records
            gc.collect()

        # pick best from saved batches
        if save_batches:
            print("Loading all batch results to find best model...")
            all_records = []
            checkpoint_files = [f for f in os.listdir('model_checkpoints') if f.startswith(f'{model_name}_batch_') and f.endswith('.pkl')]
            for file in checkpoint_files:
                file_path = os.path.join('model_checkpoints', file)
                batch_data = joblib.load(file_path)
                all_records.extend(batch_data)
            
            # Find best model from all batches
            best_record = min(all_records, key=lambda x: x['CV_RMSE'])
            best_rmse = best_record['CV_RMSE']
            best_params = best_record['Params']
            best_iter = best_record['Avg_Best_Iter']
        else:
            best_rmse, best_params, best_iter = best_heap[0]

        print(f"  >> Best CV RMSE {best_rmse:.4f} params={best_params} avg_best_iter={best_iter}")

        # retrain on full train
        final_params = dict(best_params)
        if model_name == 'LightGBM':
            # Use best_iter (if found) to limit n_estimators
            if best_iter is not None:
                final_params['n_estimators'] = best_iter
            final_params['verbosity'] = -1
            final_params.setdefault('random_state', random_state)
            final_model = Est(**final_params)
            final_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=early_stop_rounds, verbose=False)]
            )
        else:
            final_params.setdefault('random_state', random_state)
            final_model = Est(**final_params)
            final_model.fit(X_train, y_train)

        # test metrics
        if model_name == 'LightGBM':
            y_pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration_)
            used_iter = final_model.best_iteration_
        else:
            y_pred_test = final_model.predict(X_test)
            used_iter = None

        test_rmse, test_mae, test_r2 = compute_metrics(y_test, y_pred_test)

        cv_results.append({
            'Model': model_name,
            'CV_RMSE': best_rmse,
            'CV_RMSE_std': np.nan, 
            'Best_Params': final_params,
            'Avg_Best_Iter': best_iter
        })

        test_results.append({
            'Model': model_name,
            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae,
            'Test_R2': test_r2,
            'Final_Best_Iteration': used_iter
        })

        models_dict[model_name] = final_model
        best_scores[model_name] = best_rmse

    return models_dict, cv_results, test_results, best_scores


if __name__ == "__main__":
    print("This is a utility module and should not be run directly.")