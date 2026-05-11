#实验服务：模型对比、特征消融、指标计算与结果导出。
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from backtest_service import run_backtest
from config import (
    BATCH_SIZE,
    BUY_THRESHOLD,
    EPOCHS,
    RESULT_DIR,
    SELL_THRESHOLD,
    TIME_STEP,
    TRANSACTION_FEE,
    ensure_directories,
)
from data_service import format_stock_label, prepare_stock_dataset, resolve_stock_name
from model_service import AttentionLSTM, FEATURES, VanillaLSTM, train_regression_model


EXPERIMENT_EPOCHS = min(EPOCHS, 30)
MODEL_COMPARISON_CSV = "model_comparison.csv"
ABLATION_RESULT_CSV = "ablation_result.csv"

TECHNICAL_INDICATORS = ["MA5", "MA10", "MACD", "RSI", "VOLATILITY"]
MARKET_INDEX_FEATURES = ["大盘指数", "INDEX_RET"]
RETURN_FEATURES = ["RET", "LOG_RET", "INDEX_RET"]
PRICE_FEATURES = ["开盘价", "最高价", "最低价", "收盘价", "成交额(千元)"]


def _validate_features(features):
    missing = [name for name in features if name not in FEATURES]
    if missing:
        raise ValueError(f"实验特征不存在：{missing}")
    if "收盘价" not in features:
        raise ValueError("实验特征必须包含收盘价")


def _scale_train_test_data(pandas_df, features):
    _validate_features(features)
    data = pandas_df[features]
    train_size = int(len(data) * 0.8)
    if train_size <= TIME_STEP + 1 or len(data) - train_size <= TIME_STEP + 1:
        raise ValueError("数据量不足，无法完成训练/测试划分，请扩大日期范围。")

    train_raw = data.iloc[:train_size]
    test_raw = data.iloc[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    train_df = pd.DataFrame(train_scaled, columns=features, index=train_raw.index)
    test_df = pd.DataFrame(test_scaled, columns=features, index=test_raw.index)
    return train_df, test_df, scaler


def _create_regression_dataset(df, features, time_step=TIME_STEP):
    close_idx = features.index("收盘价")
    values = df[features].values
    X, y, prev_close = [], [], []
    for i in range(len(values) - time_step - 1):
        X.append(values[i : i + time_step])
        y.append(values[i + time_step, close_idx])
        prev_close.append(values[i + time_step - 1, close_idx])
    return np.array(X), np.array(y), np.array(prev_close)


def _make_loader(X, y, batch_size=BATCH_SIZE):
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def _inverse_close(values, scaler, features):
    close_idx = features.index("收盘价")
    dummy = np.zeros((len(values), len(features)))
    dummy[:, close_idx] = np.asarray(values).reshape(-1)
    return scaler.inverse_transform(dummy)[:, close_idx]


def _predict_torch_model(model, X_test, scaler, features):
    model.eval()
    preds = []
    loader = DataLoader(torch.tensor(X_test, dtype=torch.float32), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for x in loader:
            preds.append(model(x).detach().cpu().numpy())
    pred_scaled = np.concatenate(preds).reshape(-1)
    return _inverse_close(pred_scaled, scaler, features)


def _direction_probabilities(pred_price, prev_price):
    pred_return = (pred_price - prev_price) / np.maximum(np.abs(prev_price), 1e-8)
    probabilities = 1 / (1 + np.exp(-pred_return * 30))
    return np.clip(probabilities, 0.05, 0.95)


def _price_metrics(real_price, pred_price):
    mse = mean_squared_error(real_price, pred_price)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(real_price, pred_price)
    mape = float(np.mean(np.abs((real_price - pred_price) / np.maximum(np.abs(real_price), 1e-8))) * 100)
    return {
        "MSE": float(mse),
        "RMSE": rmse,
        "MAE": float(mae),
        "MAPE": mape,
    }


def _direction_metrics(real_price, pred_price, prev_price):
    real_direction = (real_price > prev_price).astype(int)
    pred_direction = (pred_price > prev_price).astype(int)
    return {
        "Accuracy": float(accuracy_score(real_direction, pred_direction)),
        "Precision": float(precision_score(real_direction, pred_direction, zero_division=0)),
        "Recall": float(recall_score(real_direction, pred_direction, zero_division=0)),
        "F1-score": float(f1_score(real_direction, pred_direction, zero_division=0)),
    }


def _evaluate_prediction(name, real_price, pred_price, prev_price, pandas_df):
    probabilities = _direction_probabilities(pred_price, prev_price)
    backtest = run_backtest(
        real_price,
        pred_price,
        probabilities,
        pandas_df,
        buy_threshold=BUY_THRESHOLD,
        sell_threshold=SELL_THRESHOLD,
        fee=TRANSACTION_FEE,
    )
    row = {"模型": name}
    row.update(_price_metrics(real_price, pred_price))
    row.update(_direction_metrics(real_price, pred_price, prev_price))
    row.update(
        {
            "总收益率": float(backtest.total_return),
            "最大回撤": float(backtest.max_drawdown),
            "夏普比率": float(backtest.sharpe),
            "交易轮数": int(len(backtest.trade_cycles)),
        }
    )
    return row


def _prepare_feature_experiment(pandas_df, features):
    train_df, test_df, scaler = _scale_train_test_data(pandas_df, features)
    X_train, y_train, _ = _create_regression_dataset(train_df, features)
    X_test, y_test, prev_test = _create_regression_dataset(test_df, features)

    real_price = _inverse_close(y_test, scaler, features)
    prev_price = _inverse_close(prev_test, scaler, features)
    train_loader = _make_loader(X_train, y_train)
    test_loader = _make_loader(X_test, y_test)
    return train_loader, test_loader, X_train, y_train, X_test, real_price, prev_price, scaler


def _normalize_experiment_dataframe(pandas_df):
    df = pandas_df.copy().dropna()
    if "交易日期" in df.columns:
        df["交易日期"] = pd.to_datetime(df["交易日期"])
    return df


def _train_lstm(model_class, train_loader, test_loader, input_size, epochs):
    model = model_class(input_size)
    train_regression_model(model, train_loader, test_loader, epochs=epochs)
    return model


def run_model_comparison(pandas_df, epochs=EXPERIMENT_EPOCHS):
    pandas_df = _normalize_experiment_dataframe(pandas_df)
    features = FEATURES
    (
        train_loader,
        test_loader,
        X_train,
        y_train,
        X_test,
        real_price,
        prev_price,
        scaler,
    ) = _prepare_feature_experiment(pandas_df, features)

    flat_train = X_train.reshape((X_train.shape[0], -1))
    flat_test = X_test.reshape((X_test.shape[0], -1))

    rows = []
    rows.append(_evaluate_prediction("Naive Baseline", real_price, prev_price, prev_price, pandas_df))

    linear_model = LinearRegression()
    linear_model.fit(flat_train, y_train)
    linear_pred = _inverse_close(linear_model.predict(flat_test), scaler, features)
    rows.append(_evaluate_prediction("Linear Regression", real_price, linear_pred, prev_price, pandas_df))

    forest_model = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
    forest_model.fit(flat_train, y_train)
    forest_pred = _inverse_close(forest_model.predict(flat_test), scaler, features)
    rows.append(_evaluate_prediction("Random Forest", real_price, forest_pred, prev_price, pandas_df))

    vanilla_model = _train_lstm(VanillaLSTM, train_loader, test_loader, len(features), epochs)
    vanilla_pred = _predict_torch_model(vanilla_model, X_test, scaler, features)
    rows.append(_evaluate_prediction("Vanilla LSTM", real_price, vanilla_pred, prev_price, pandas_df))

    attention_model = _train_lstm(AttentionLSTM, train_loader, test_loader, len(features), epochs)
    attention_pred = _predict_torch_model(attention_model, X_test, scaler, features)
    rows.append(_evaluate_prediction("Attention-LSTM", real_price, attention_pred, prev_price, pandas_df))

    return rows


def _without(features_to_remove):
    remove_set = set(features_to_remove)
    return [feature for feature in FEATURES if feature not in remove_set]


def run_ablation_experiment(pandas_df, epochs=EXPERIMENT_EPOCHS):
    pandas_df = _normalize_experiment_dataframe(pandas_df)
    experiments = [
        ("Full Features", FEATURES),
        ("No Technical Indicators", _without(TECHNICAL_INDICATORS)),
        ("No Market Index", _without(MARKET_INDEX_FEATURES)),
        ("No Return Features", _without(RETURN_FEATURES)),
        ("Only Price Features", PRICE_FEATURES),
    ]

    rows = []
    for experiment_name, features in experiments:
        train_loader, test_loader, _, _, X_test, real_price, prev_price, scaler = _prepare_feature_experiment(
            pandas_df, features
        )
        model = _train_lstm(AttentionLSTM, train_loader, test_loader, len(features), epochs)
        pred_price = _predict_torch_model(model, X_test, scaler, features)
        metrics = _price_metrics(real_price, pred_price)
        metrics.update(_direction_metrics(real_price, pred_price, prev_price))
        rows.append({"实验名称": experiment_name, "特征数量": len(features), **metrics})

    return rows


def _format_rows(rows):
    formatted_rows = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, float):
                item[key] = round(value, 6)
            else:
                item[key] = value
        formatted_rows.append(item)
    return formatted_rows


def _export_csv(rows, filename):
    ensure_directories()
    path = os.path.join(RESULT_DIR, filename)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return filename


def run_experiments(stock_code, start_date, end_date, epochs=EXPERIMENT_EPOCHS):
    stock_name = resolve_stock_name(stock_code, allow_remote=True)
    stock_label = format_stock_label(stock_code, stock_name)
    lstm_df, _, _ = prepare_stock_dataset(stock_code, start_date, end_date)
    pandas_df = _normalize_experiment_dataframe(lstm_df)

    model_rows = _format_rows(run_model_comparison(pandas_df, epochs=epochs))
    ablation_rows = _format_rows(run_ablation_experiment(pandas_df, epochs=epochs))

    model_csv = _export_csv(model_rows, MODEL_COMPARISON_CSV)
    ablation_csv = _export_csv(ablation_rows, ABLATION_RESULT_CSV)

    return {
        "stock_code": stock_code,
        "stock_label": stock_label,
        "start_date": start_date,
        "end_date": end_date,
        "epochs": epochs,
        "model_rows": model_rows,
        "ablation_rows": ablation_rows,
        "model_csv": model_csv,
        "ablation_csv": ablation_csv,
    }
