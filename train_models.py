import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


def load_data():
    train_df = pd.read_csv('train.csv')
    store_df = pd.read_csv('store.csv')
    print(f"Loaded {len(train_df)} training records")
    return train_df, store_df


def create_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    
    df['CompetitionOpenNMonthAgo'] = (df['Year'] - df['CompetitionOpenSinceYear']) * 12 + (df['Month'] - df['CompetitionOpenSinceMonth'])
    
    df['Promo2LastsForNWeeks'] = np.where(
        df['Promo2'] == 1,
        np.maximum(0, (df['Year'] - df['Promo2SinceYear']) * 52 + (df['WeekOfYear'] - df['Promo2SinceWeek'])),
        0
    )
    
    months_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                  'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    promo2_last_update_months = []
    
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Computing Promo2 features"):
        if row.Promo2 == 0:
            promo2_last_update_months.append(np.nan)
            continue
        if row.Promo2LastsForNWeeks == 0:
            promo2_last_update_months.append(np.nan)
            continue
        if row.Promo2LastsForNWeeks < 5:
            promo2_last_update_months.append(0)
            continue
        
        month_str = row.PromoInterval
        months = month_str.split(',')
        month_numbers = [months_map[month] for month in months]
        cur_month = row.Month
        last_update_month = min(month_numbers)
        flag = 1
        
        for month in month_numbers:
            if month <= cur_month:
                last_update_month = max(month, last_update_month)
                flag = 0
        
        last_update_year = row.Year - flag
        promo2_last_update_date = pd.to_datetime(f"{last_update_year}-{last_update_month}-01")
        
        if promo2_last_update_date < row.Date:
            if not flag:
                promo2_last_update_months.append(row.Month - last_update_month)
            else:
                promo2_last_update_months.append(row.Month + (12 - last_update_month))
        else:
            promo2_last_update_months.append(np.nan)
    
    df['Promo2LastUpdateNMonthAgo'] = promo2_last_update_months
    return df


def prepare_data(train_df, store_df):
    joined_df = pd.merge(train_df, store_df, on='Store')
    joined_df = joined_df[joined_df['Open'] == 1]
    joined_df = joined_df[~(joined_df['Customers'] == 0)]
    joined_df = joined_df[~(joined_df['Sales'] == 0)]
    joined_df = create_date_features(joined_df)
    print(f"After filtering: {len(joined_df)} records")
    return joined_df


def fill_missing_values(df):
    df['HasCompetition'] = df['CompetitionDistance'].notna().astype(int)
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(0)
    df['HasCompetitionData'] = df['CompetitionOpenSinceYear'].notna().astype(int)
    df['CompetitionOpenNMonthAgo'] = df['CompetitionOpenNMonthAgo'].fillna(0)
    df['HasPromo2UpdateData'] = df['Promo2LastUpdateNMonthAgo'].notna().astype(int)
    df['Promo2LastUpdateNMonthAgo'] = df['Promo2LastUpdateNMonthAgo'].fillna(0)
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    return df


def encode_categorical(df):
    cat_cols = ["StoreType", "Assortment", "StateHoliday"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
    return df


def encode_categorical_drop_first(df):
    cat_cols = ["StoreType", "Assortment", "StateHoliday"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    return df


def prepare_features(df):
    useless_features = ['Store', 'Date', 'Open', 'CompetitionOpenSinceMonth', 
                       'CompetitionOpenSinceYear', 'Promo2SinceWeek', 
                       'Promo2SinceYear', 'PromoInterval', 'Year', 'Month', 'WeekOfYear']
    
    full_features_temp = df.drop(columns=['Sales']).copy()
    full_target = df['Sales'].copy()
    full_features_encoded = encode_categorical(full_features_temp)
    
    cols_to_drop_final = [col for col in useless_features if col in full_features_encoded.columns]
    full_features_final = full_features_encoded.drop(columns=cols_to_drop_final, errors='ignore')
    
    print(f"Features: {list(full_features_final.columns)}")
    
    return full_features_final, full_target, df['Date']


def prepare_features_for_linear(df):
    useless_features = ['Store', 'Date', 'Open', 'CompetitionOpenSinceMonth', 
                       'CompetitionOpenSinceYear', 'Promo2SinceWeek', 
                       'Promo2SinceYear', 'PromoInterval', 'Year', 'Month', 'WeekOfYear']
    
    full_features_temp = df.drop(columns=['Sales']).copy()
    full_target = df['Sales'].copy()
    full_features_encoded = encode_categorical_drop_first(full_features_temp)
    
    cols_to_drop_final = [col for col in useless_features if col in full_features_encoded.columns]
    full_features_final = full_features_encoded.drop(columns=cols_to_drop_final, errors='ignore')
    
    print(f"Features for linear models (drop_first=True): {list(full_features_final.columns)}")
    
    return full_features_final, full_target, df['Date']


def time_based_split(features, target, dates, train_part=0.8, val_part=0.1, test_part=0.1):
    df_for_splitting = pd.concat([features, target, dates], axis=1)
    df_sorted = df_for_splitting.sort_values(by='Date').reset_index(drop=True)
    
    train_size = int(len(df_sorted) * train_part)
    val_size = int(len(df_sorted) * val_part)
    
    train_df = df_sorted[:train_size]
    val_df = df_sorted[train_size : train_size + val_size]
    test_df = df_sorted[train_size + val_size :]
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    X_train = train_df.drop(columns=['Sales', 'Date'])
    y_train = train_df['Sales']
    X_val = val_df.drop(columns=['Sales', 'Date'])
    y_val = val_df['Sales']
    X_test = test_df.drop(columns=['Sales', 'Date'])
    y_test = test_df['Sales']
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, binary_features, numeric_features, filename='processed_data.pkl'):
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'binary_features': binary_features,
        'numeric_features': numeric_features
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved processed data to {filename}")


def load_processed_data(filename='processed_data.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded processed data from {filename}")
        return data
    return None


def scale_numeric_features(X_train, X_val, X_test, numeric_features):
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_val_scaled[numeric_features] = scaler.transform(X_val[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    
    print(f"Scaled {len(numeric_features)} numeric features")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def apply_pca(X_train, X_val, X_test, n_components=10):
    pca = PCA(n_components=n_components)
    
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    explained_var = pca.explained_variance_ratio_
    total_var = explained_var.sum()
    
    print(f"PCA: {n_components} components explain {total_var:.4f} of variance")
    
    return X_train_pca, X_val_pca, X_test_pca, pca


def remove_redundant_onehot(X_train, X_val, X_test):
    cols_to_drop = []
    
    onehot_groups = {
        'StoreType': ['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d'],
        'Assortment': ['Assortment_a', 'Assortment_b', 'Assortment_c'],
        'StateHoliday': ['StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c']
    }
    
    for group_name, cols in onehot_groups.items():
        existing_cols = [c for c in cols if c in X_train.columns]
        if len(existing_cols) > 1:
            cols_to_drop.append(existing_cols[0])
    
    X_train_reduced = X_train.drop(columns=cols_to_drop)
    X_val_reduced = X_val.drop(columns=cols_to_drop)
    X_test_reduced = X_test.drop(columns=cols_to_drop)
    
    print(f"Removed {len(cols_to_drop)} redundant one-hot columns: {cols_to_drop}")
    print(f"Features reduced from {X_train.shape[1]} to {X_train_reduced.shape[1]}")
    
    return X_train_reduced, X_val_reduced, X_test_reduced


def rmspe_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    eps = 1e-6
    perc_error = (y_true - y_pred) / (y_true + eps)
    rmspe = float(np.sqrt(np.mean(perc_error ** 2)))
    return "rmspe", rmspe


def rmspe(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    perc_error = (y_true - y_pred) / (y_true + eps)
    return float(np.sqrt(np.mean(perc_error ** 2)))


def train_xgboost(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        "objective": "reg:squarederror",
        "eta": 0.1,
        "max_depth": 7,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "seed": 42,
        "nthread": -1,
        'disable_default_eval_metric': 1
    }
    
    evals_result = {}
    
    def lr_decay(boosting_round):
        if boosting_round <= 1000:
            return 0.1
        return 0.05
    
    xgb_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "eval")],
        custom_metric=rmspe_metric,
        evals_result=evals_result,
        verbose_eval=True,
        callbacks=[xgb.callback.LearningRateScheduler(lr_decay)],
    )
    
    print("\nXGBoost model trained successfully")
    return xgb_model, evals_result


def plot_xgboost_metrics(evals_result):
    train_rmspe = evals_result["train"]["rmspe"]
    val_rmspe = evals_result["eval"]["rmspe"]
    rounds_rmspe = range(len(train_rmspe))
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds_rmspe, train_rmspe, label="train RMSPE")
    plt.plot(rounds_rmspe, val_rmspe, label="val RMSPE")
    plt.xlabel("Boosting round")
    plt.ylabel("RMSPE")
    plt.title("RMSPE по итерациям обучения")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/xgboost_rmspe.png', dpi=300)
    plt.close()
    print("Saved: plots/xgboost_rmspe.png")


def evaluate_xgboost(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pred = model.predict(dtest)
    
    rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
    rmspe_final = rmspe_metric(y_pred, dtest)[1]
    
    print(f"RMSE на тестовом наборе (XGBoost): {rmse_final:.4f}")
    print(f"RMSPE на тестовом наборе (XGBoost): {rmspe_final:.4f}")
    
    return y_pred


def plot_xgboost_feature_importance(model):
    importance_weight = model.get_score(importance_type="weight")
    importance_gain = model.get_score(importance_type="gain")
    importance_cover = model.get_score(importance_type="cover")
    importance_total_gain = model.get_score(importance_type="total_gain")
    importance_total_cover = model.get_score(importance_type="total_cover")
    
    all_features_used = sorted(
        set(importance_weight.keys())
        | set(importance_gain.keys())
        | set(importance_cover.keys())
        | set(importance_total_gain.keys())
        | set(importance_total_cover.keys())
    )
    
    rows = []
    for f in all_features_used:
        rows.append({
            "feature": f,
            "weight": importance_weight.get(f, 0.0),
            "gain": importance_gain.get(f, 0.0),
            "cover": importance_cover.get(f, 0.0),
            "total_gain": importance_total_gain.get(f, 0.0),
            "total_cover": importance_total_cover.get(f, 0.0),
        })
    
    importance_df = pd.DataFrame(rows)
    
    for col in ["weight", "gain", "cover", "total_gain", "total_cover"]:
        s = importance_df[col].to_numpy(dtype=float)
        total = s.sum()
        if total > 0:
            importance_df[col + "_norm"] = s / total
        else:
            importance_df[col + "_norm"] = s
    
    importance_df = importance_df.sort_values("total_gain", ascending=False).reset_index(drop=True)
    
    print("\nТоп-20 признаков по суммарному information gain:")
    print(importance_df.head(20))
    
    top_n = 20
    top_by_gain = (
        importance_df
        .sort_values("gain", ascending=False)
        .head(top_n)
        .sort_values("gain", ascending=True)
    )
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_by_gain["feature"], top_by_gain["gain"])
    plt.gca().invert_yaxis()
    plt.xlabel("Средний information gain")
    plt.title(f"Top-{top_n} признаков по gain (XGBoost)")
    plt.tight_layout()
    plt.savefig('plots/xgboost_importance_gain.png', dpi=300)
    plt.close()
    print("Saved: plots/xgboost_importance_gain.png")
    
    top_by_weight = importance_df.sort_values("weight", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_by_weight["feature"], top_by_weight["weight"])
    plt.gca().invert_yaxis()
    plt.xlabel("Частота использования в сплитах (weight)")
    plt.title(f"Top-{top_n} признаков по количеству сплитов (XGBoost)")
    plt.tight_layout()
    plt.savefig('plots/xgboost_importance_weight.png', dpi=300)
    plt.close()
    print("Saved: plots/xgboost_importance_weight.png")


def train_random_forest(X_train, y_train):
    rf_params = {
        "n_estimators": 10,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42
    }
    
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    print("\nRandomForest model trained successfully")
    
    return rf_model


def evaluate_random_forest(model, X_train, y_train, X_val, y_val, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    print("\nRandomForest RMSPE metrics:")
    print(f"  Train RMSPE: {rmspe(y_train, y_train_pred):.6f}")
    print(f"  Val   RMSPE: {rmspe(y_val, y_val_pred):.6f}")
    print(f"  Test  RMSPE: {rmspe(y_test, y_test_pred):.6f}")
    
    return y_test_pred


def plot_random_forest_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    print("\nТоп-20 признаков по важности в случайном лесе:")
    print(feat_imp_df.head(20))
    
    top_n = 20
    top_feats = (
        feat_imp_df
        .head(top_n)
        .sort_values("importance", ascending=True)
    )
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_feats["feature"], top_feats["importance"])
    plt.xlabel("Feature importance")
    plt.title(f"Top-{top_n} признаков по важности (RandomForest)")
    plt.tight_layout()
    plt.savefig('plots/rf_importance.png', dpi=300)
    plt.close()
    print("Saved: plots/rf_importance.png")


def analyze_features(X_train):
    print("\n" + "="*50)
    print("Feature Analysis")
    print("="*50)
    
    binary_features = []
    numeric_features = []
    
    for col in X_train.columns:
        unique_vals = X_train[col].unique()
        n_unique = len(unique_vals)
        min_val = X_train[col].min()
        max_val = X_train[col].max()
        
        if n_unique == 2 and set(unique_vals).issubset({0, 1}):
            binary_features.append(col)
        else:
            numeric_features.append(col)
    
    print(f"\nBinary features ({len(binary_features)}):")
    for feat in sorted(binary_features):
        print(f"  - {feat}")
    
    print(f"\nNumeric features ({len(numeric_features)}):")
    for feat in sorted(numeric_features):
        min_val = X_train[feat].min()
        max_val = X_train[feat].max()
        mean_val = X_train[feat].mean()
        print(f"  - {feat:40s} | min: {min_val:10.2f} | max: {max_val:10.2f} | mean: {mean_val:10.2f}")
    
    return binary_features, numeric_features


def train_lasso(X_train, y_train, X_val, y_val, alpha=1.0):
    lasso_model = Lasso(alpha=alpha, max_iter=5000, random_state=42)
    lasso_model.fit(X_train, y_train)
    print(f"\nLasso model trained successfully (alpha={alpha})")
    return lasso_model


def evaluate_lasso(model, X_train, y_train, X_val, y_val, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    print("\nLasso RMSPE metrics:")
    print(f"  Train RMSPE: {rmspe(y_train, y_train_pred):.6f}")
    print(f"  Val   RMSPE: {rmspe(y_val, y_val_pred):.6f}")
    print(f"  Test  RMSPE: {rmspe(y_test, y_test_pred):.6f}")
    
    return y_test_pred


def plot_lasso_coefficients(model, feature_names):
    coefficients = model.coef_
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients)
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    
    non_zero = (coef_df['abs_coefficient'] > 1e-6).sum()
    print(f"\nLasso: {non_zero}/{len(feature_names)} non-zero coefficients")
    print("\nТоп-20 признаков по абсолютному значению коэффициента:")
    print(coef_df.head(20))
    
    top_n = 20
    top_coefs = (
        coef_df
        .head(top_n)
        .sort_values("coefficient", ascending=True)
    )
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if c < 0 else 'green' for c in top_coefs["coefficient"]]
    plt.barh(top_coefs["feature"], top_coefs["coefficient"], color=colors)
    plt.xlabel("Coefficient value")
    plt.title(f"Top-{top_n} признаков по важности (Lasso)")
    plt.tight_layout()
    plt.savefig('plots/lasso_coefficients.png', dpi=300)
    plt.close()
    print("Saved: plots/lasso_coefficients.png")


def train_mlp(X_train, y_train, X_val, y_val):
    mlp_params = {
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "batch_size": 256,
        "learning_rate": "adaptive",
        "learning_rate_init": 0.001,
        "max_iter": 1,
        "random_state": 42,
        "warm_start": True,
        "verbose": False
    }
    
    mlp_model = MLPRegressor(**mlp_params)
    
    train_rmspe_history = []
    val_rmspe_history = []
    epochs_logged = []
    
    max_epochs = 20
    best_val_rmspe = float('inf')
    patience = 10
    no_improvement = 0
    log_interval = 5
    
    print("\nTraining MLP with periodic logging:")
    for epoch in tqdm(range(max_epochs), desc="MLP training"):
        mlp_model.fit(X_train, y_train)
        
        if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == max_epochs - 1:
            y_train_pred = mlp_model.predict(X_train)
            y_val_pred = mlp_model.predict(X_val)
            
            train_rmspe_val = rmspe(y_train, y_train_pred)
            val_rmspe_val = rmspe(y_val, y_val_pred)
            
            train_rmspe_history.append(train_rmspe_val)
            val_rmspe_history.append(val_rmspe_val)
            epochs_logged.append(epoch + 1)
            
            print(f"Epoch {epoch+1}: Train RMSPE = {train_rmspe_val:.6f}, Val RMSPE = {val_rmspe_val:.6f}")
            
            if val_rmspe_val < best_val_rmspe:
                best_val_rmspe = val_rmspe_val
                no_improvement = 0
            else:
                no_improvement += log_interval
        
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nMLP model trained successfully ({epoch+1} epochs)")
    
    return mlp_model, train_rmspe_history, val_rmspe_history, epochs_logged


def plot_mlp_training(train_rmspe_history, val_rmspe_history, epochs_logged):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_logged, train_rmspe_history, label="Train RMSPE", marker='o')
    plt.plot(epochs_logged, val_rmspe_history, label="Val RMSPE", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("RMSPE")
    plt.title("MLP Training: RMSPE по эпохам")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mlp_training.png', dpi=300)
    plt.close()
    print("Saved: plots/mlp_training.png")


def evaluate_mlp(model, X_train, y_train, X_val, y_val, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    print("\nMLP RMSPE metrics:")
    print(f"  Train RMSPE: {rmspe(y_train, y_train_pred):.6f}")
    print(f"  Val   RMSPE: {rmspe(y_val, y_val_pred):.6f}")
    print(f"  Test  RMSPE: {rmspe(y_test, y_test_pred):.6f}")
    
    return y_test_pred


def plot_mlp_weights(model, feature_names):
    first_layer_weights = model.coefs_[0]
    avg_weights = np.abs(first_layer_weights).mean(axis=1)
    
    weight_df = pd.DataFrame({
        "feature": feature_names,
        "avg_abs_weight": avg_weights
    }).sort_values("avg_abs_weight", ascending=False).reset_index(drop=True)
    
    print("\nТоп-20 признаков по средним весам первого слоя MLP:")
    print(weight_df.head(20))
    
    top_n = 20
    top_weights = (
        weight_df
        .head(top_n)
        .sort_values("avg_abs_weight", ascending=True)
    )
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_weights["feature"], top_weights["avg_abs_weight"])
    plt.xlabel("Average absolute weight")
    plt.title(f"Top-{top_n} признаков по средним весам (MLP)")
    plt.tight_layout()
    plt.savefig('plots/mlp_weights.png', dpi=300)
    plt.close()
    print("Saved: plots/mlp_weights.png")


def main():
    processed_data = load_processed_data()
    
    if processed_data is None:
        train_df, store_df = load_data()
        
        df = prepare_data(train_df, store_df)
        df = fill_missing_values(df)
        
        features, target, dates = prepare_features(df)
        
        X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(features, target, dates)
        
        binary_features, numeric_features = analyze_features(X_train)
        
        save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, binary_features, numeric_features)
    else:
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        binary_features = processed_data['binary_features']
        numeric_features = processed_data['numeric_features']
        
        print(f"Loaded data: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
        print(f"Features: {len(binary_features)} binary, {len(numeric_features)} numeric")
    
    print("\n" + "="*50)
    print("Preparing data for linear models (removing multicollinearity)")
    print("="*50)
    
    X_train_linear, X_val_linear, X_test_linear = remove_redundant_onehot(X_train, X_val, X_test)
    
    binary_features_linear = [f for f in binary_features if f in X_train_linear.columns]
    numeric_features_linear = [f for f in numeric_features if f in X_train_linear.columns]
    
    X_train_scaled_linear, X_val_scaled_linear, X_test_scaled_linear, scaler_linear = scale_numeric_features(
        X_train_linear, X_val_linear, X_test_linear, numeric_features_linear
    )
    
    X_train_pca_linear, X_val_pca_linear, X_test_pca_linear, pca_linear = apply_pca(
        X_train_scaled_linear, X_val_scaled_linear, X_test_scaled_linear, n_components=10
    )
    
    # print("\n" + "="*50)
    # print("Training XGBoost")
    # print("="*50)
    # xgb_model, evals_result = train_xgboost(X_train, y_train, X_val, y_val)
    # 
    # plot_xgboost_metrics(evals_result)
    # evaluate_xgboost(xgb_model, X_test, y_test)
    # plot_xgboost_feature_importance(xgb_model)
    # 
    # print("\n" + "="*50)
    # print("Training RandomForest")
    # print("="*50)
    # rf_model = train_random_forest(X_train, y_train)
    # 
    # evaluate_random_forest(rf_model, X_train, y_train, X_val, y_val, X_test, y_test)
    # plot_random_forest_feature_importance(rf_model, list(X_train.columns))
    
    print("\n" + "="*50)
    print("Training Lasso (Linear Regression with L1)")
    print("="*50)
    lasso_model = train_lasso(X_train_pca_linear, y_train, X_val_pca_linear, y_val, alpha=10.0)
    
    evaluate_lasso(lasso_model, X_train_pca_linear, y_train, X_val_pca_linear, y_val, X_test_pca_linear, y_test)
    plot_lasso_coefficients(lasso_model, [f"PC{i+1}" for i in range(X_train_pca_linear.shape[1])])
    
    print("\n" + "="*50)
    print("Training MLP (Multi-layer Perceptron)")
    print("="*50)
    mlp_model, train_rmspe_history, val_rmspe_history, epochs_logged = train_mlp(X_train_pca_linear, y_train, X_val_pca_linear, y_val)
    
    plot_mlp_training(train_rmspe_history, val_rmspe_history, epochs_logged)
    evaluate_mlp(mlp_model, X_train_pca_linear, y_train, X_val_pca_linear, y_val, X_test_pca_linear, y_test)
    plot_mlp_weights(mlp_model, [f"PC{i+1}" for i in range(X_train_pca_linear.shape[1])])
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()

