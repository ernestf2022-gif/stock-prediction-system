#实验服务：模型对比、特征消融、指标计算与结果导出。
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    EPOCHS,
    RANDOM_SEED,
    RESULT_DIR,
    TIME_STEP,
    ensure_directories,
)
from data_service import format_stock_label, prepare_stock_dataset, resolve_stock_name
from model_service import FEATURES
from plot_service import plot_price_prediction, setup_matplotlib


EXPERIMENT_EPOCHS = min(EPOCHS, 30)
MODEL_COMPARISON_CSV = "model_comparison.csv"
ABLATION_RESULT_CSV = "ablation_result.csv"
NO_MARKET_INDEX_ABLATION_NAME = "No Market Index"
NO_MARKET_INDEX_ABLATION_MODELS = ("Vanilla LSTM", "ARIMA", "DNN", "CNN")

TECHNICAL_INDICATORS = ["MA5", "MA10", "MACD", "RSI", "VOLATILITY"]
MARKET_INDEX_FEATURES = ["大盘指数", "INDEX_RET"]
RETURN_FEATURES = ["RET", "LOG_RET", "INDEX_RET"]
PRICE_FEATURES = ["开盘价", "最高价", "最低价", "收盘价", "成交额(千元)"]


def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


class VanillaLSTM(nn.Module):
    """普通 LSTM：用于模型对比和特征消融中的收盘价预测。"""

    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class DNNRegressor(nn.Module):
    """全连接深度神经网络：用于模型对比实验。"""

    def __init__(self, input_size, time_step=TIME_STEP, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * time_step, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class CNNRegressor(nn.Module):
    """一维 CNN：用于捕捉时间窗口内的局部变化模式。"""

    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


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


def train_regression_model(model, train_loader, test_loader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                criterion(model(x), y)


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


def _price_metrics(real_price, pred_price):
    mse = mean_squared_error(real_price, pred_price)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(real_price, pred_price)
    mape = float(np.mean(np.abs((real_price - pred_price) / np.maximum(np.abs(real_price), 1e-8))) * 100)
    r2 = r2_score(real_price, pred_price)
    return {
        "RMSE": rmse,
        "MAE": float(mae),
        "MAPE": mape,
        "R²": float(r2),
    }


def _evaluate_prediction(name, real_price, pred_price):
    row = {"模型": name}
    row.update(_price_metrics(real_price, pred_price))
    return row


def _prediction_dates(pandas_df, prediction_len):
    if "交易日期" in pandas_df.columns:
        return pandas_df["交易日期"].iloc[-prediction_len:]
    return np.arange(1, prediction_len + 1)


def _safe_filename_token(value):
    return str(value).replace(" ", "_").replace(".", "_").replace("/", "_").replace("\\", "_")


def _save_model_prediction_plots(prediction_curves, stock_label, output_prefix=None):
    ensure_directories()
    setup_matplotlib()
    safe_prefix = _safe_filename_token(output_prefix or "model_comparison")
    images = []

    for item in prediction_curves:
        safe_model_name = _safe_filename_token(item["model_name"])
        filename = f"{safe_prefix}_{safe_model_name}_price_prediction.png"
        output_path = os.path.join(RESULT_DIR, filename)
        plot_price_prediction(
            item["dates"],
            item["real_price"],
            item["pred_price"],
            stock_label=stock_label,
            model_name=item["model_name"],
            output_path=output_path,
        )
        images.append(filename)

    return images


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


def _predict_arima(pandas_df, forecast_len):
    close_series = pandas_df["收盘价"].astype(float).reset_index(drop=True)
    train_size = int(len(close_series) * 0.8)
    forecast_steps = TIME_STEP + forecast_len
    try:
        fitted = ARIMA(close_series.iloc[:train_size], order=(5, 1, 0)).fit()
        forecast = np.asarray(fitted.forecast(steps=forecast_steps), dtype=float)
        return forecast[TIME_STEP : TIME_STEP + forecast_len]
    except Exception as exc:
        print(f"ARIMA 训练失败，使用上一交易日价格作为兜底预测：{exc}")
        test_close = close_series.iloc[train_size:].to_numpy(dtype=float)
        return test_close[TIME_STEP - 1 : TIME_STEP - 1 + forecast_len]


def run_model_comparison(pandas_df, epochs=EXPERIMENT_EPOCHS, include_predictions=False):
    set_random_seed()
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

    rows = []
    prediction_curves = []
    dates = _prediction_dates(pandas_df, len(real_price))

    def add_prediction(name, pred_price):
        rows.append(_evaluate_prediction(name, real_price, pred_price))
        if include_predictions:
            prediction_curves.append(
                {
                    "model_name": name,
                    "dates": dates,
                    "real_price": real_price,
                    "pred_price": pred_price,
                }
            )

    arima_pred = _predict_arima(pandas_df, len(real_price))
    add_prediction("ARIMA", arima_pred)

    dnn_model = _train_lstm(DNNRegressor, train_loader, test_loader, len(features), epochs)
    dnn_pred = _predict_torch_model(dnn_model, X_test, scaler, features)
    add_prediction("DNN", dnn_pred)

    cnn_model = _train_lstm(CNNRegressor, train_loader, test_loader, len(features), epochs)
    cnn_pred = _predict_torch_model(cnn_model, X_test, scaler, features)
    add_prediction("CNN", cnn_pred)

    vanilla_model = _train_lstm(VanillaLSTM, train_loader, test_loader, len(features), epochs)
    vanilla_pred = _predict_torch_model(vanilla_model, X_test, scaler, features)
    add_prediction("Vanilla LSTM", vanilla_pred)

    if include_predictions:
        return rows, prediction_curves
    return rows


def _ablation_full_features_from_model_rows(model_rows):
    for row in model_rows:
        if row.get("模型") == "Vanilla LSTM":
            return {
                "模型": "Vanilla LSTM",
                "实验名称": "Full Features",
                "特征数量": len(FEATURES),
                "RMSE": row["RMSE"],
                "MAE": row["MAE"],
                "MAPE": row["MAPE"],
                "R²": row["R²"],
            }
    return None


def _without(features_to_remove):
    remove_set = set(features_to_remove)
    return [feature for feature in FEATURES if feature not in remove_set]


def run_ablation_experiment(pandas_df, epochs=EXPERIMENT_EPOCHS, full_features_row=None):
    set_random_seed()
    pandas_df = _normalize_experiment_dataframe(pandas_df)
    experiments = [
        ("No Technical Indicators", _without(TECHNICAL_INDICATORS)),
        ("No Market Index", _without(MARKET_INDEX_FEATURES)),
        ("No Return Features", _without(RETURN_FEATURES)),
        ("Only Price Features", PRICE_FEATURES),
    ]

    rows = []
    if full_features_row is not None:
        rows.append(full_features_row)
    else:
        train_loader, test_loader, _, _, X_test, real_price, _, scaler = _prepare_feature_experiment(
            pandas_df, FEATURES
        )
        model = _train_lstm(VanillaLSTM, train_loader, test_loader, len(FEATURES), epochs)
        pred_price = _predict_torch_model(model, X_test, scaler, FEATURES)
        metrics = _price_metrics(real_price, pred_price)
        rows.append({"模型": "Vanilla LSTM", "实验名称": "Full Features", "特征数量": len(FEATURES), **metrics})

    for experiment_name, features in experiments:
        train_loader, test_loader, _, _, X_test, real_price, _, scaler = _prepare_feature_experiment(
            pandas_df, features
        )
        model = _train_lstm(VanillaLSTM, train_loader, test_loader, len(features), epochs)
        pred_price = _predict_torch_model(model, X_test, scaler, features)
        metrics = _price_metrics(real_price, pred_price)
        rows.append({"模型": "Vanilla LSTM", "实验名称": experiment_name, "特征数量": len(features), **metrics})

        if experiment_name == NO_MARKET_INDEX_ABLATION_NAME:
            arima_pred_price = _predict_arima(pandas_df, len(real_price))
            arima_metrics = _price_metrics(real_price, arima_pred_price)
            rows.append({"模型": "ARIMA", "实验名称": experiment_name, "特征数量": 1, **arima_metrics})

            dnn_model = _train_lstm(DNNRegressor, train_loader, test_loader, len(features), epochs)
            dnn_pred_price = _predict_torch_model(dnn_model, X_test, scaler, features)
            dnn_metrics = _price_metrics(real_price, dnn_pred_price)
            rows.append({"模型": "DNN", "实验名称": experiment_name, "特征数量": len(features), **dnn_metrics})

            cnn_model = _train_lstm(CNNRegressor, train_loader, test_loader, len(features), epochs)
            cnn_pred_price = _predict_torch_model(cnn_model, X_test, scaler, features)
            cnn_metrics = _price_metrics(real_price, cnn_pred_price)
            rows.append({"模型": "CNN", "实验名称": experiment_name, "特征数量": len(features), **cnn_metrics})

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


def split_ablation_rows_for_display(rows):
    main_rows = []
    market_index_model_rows = []
    comparison_models = set(NO_MARKET_INDEX_ABLATION_MODELS)
    seen_comparison_rows = set()

    for row in rows or []:
        model_name = row.get("模型") or "Vanilla LSTM"
        experiment_name = row.get("实验名称")
        display_row = row if "模型" in row else {"模型": model_name, **row}

        if experiment_name == NO_MARKET_INDEX_ABLATION_NAME and model_name in comparison_models:
            key = (model_name, experiment_name)
            if key not in seen_comparison_rows:
                market_index_model_rows.append(display_row)
                seen_comparison_rows.add(key)

        if model_name != "Vanilla LSTM":
            continue
        main_rows.append(display_row)

    market_index_model_rows.sort(
        key=lambda item: NO_MARKET_INDEX_ABLATION_MODELS.index(item.get("模型"))
        if item.get("模型") in NO_MARKET_INDEX_ABLATION_MODELS
        else len(NO_MARKET_INDEX_ABLATION_MODELS)
    )
    return main_rows, market_index_model_rows


def _export_csv(rows, filename):
    ensure_directories()
    path = os.path.join(RESULT_DIR, filename)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return filename


def _experiment_output_names(output_prefix=None):
    if not output_prefix:
        return MODEL_COMPARISON_CSV, ABLATION_RESULT_CSV

    safe_prefix = str(output_prefix).replace(".", "_").replace("/", "_").replace("\\", "_")
    return f"{safe_prefix}_model_comparison.csv", f"{safe_prefix}_ablation_result.csv"


def run_experiments(stock_code, start_date, end_date, epochs=EXPERIMENT_EPOCHS, output_prefix=None):
    stock_name = resolve_stock_name(stock_code, allow_remote=True)
    stock_label = format_stock_label(stock_code, stock_name)
    lstm_df, _, _ = prepare_stock_dataset(stock_code, start_date, end_date)
    pandas_df = _normalize_experiment_dataframe(lstm_df)

    model_rows_raw, model_prediction_curves = run_model_comparison(pandas_df, epochs=epochs, include_predictions=True)
    full_features_row = _ablation_full_features_from_model_rows(model_rows_raw)
    model_rows = _format_rows(model_rows_raw)
    all_ablation_rows = _format_rows(
        run_ablation_experiment(pandas_df, epochs=epochs, full_features_row=full_features_row)
    )
    ablation_rows, market_index_model_rows = split_ablation_rows_for_display(all_ablation_rows)

    model_csv_name, ablation_csv_name = _experiment_output_names(output_prefix)
    model_csv = _export_csv(model_rows, model_csv_name)
    ablation_csv = _export_csv(all_ablation_rows, ablation_csv_name)
    model_prediction_images = _save_model_prediction_plots(
        model_prediction_curves,
        stock_label,
        output_prefix=output_prefix,
    )

    return {
        "stock_code": stock_code,
        "stock_label": stock_label,
        "start_date": start_date,
        "end_date": end_date,
        "epochs": epochs,
        "model_rows": model_rows,
        "ablation_rows": ablation_rows,
        "market_index_model_rows": market_index_model_rows,
        "model_csv": model_csv,
        "ablation_csv": ablation_csv,
        "model_prediction_images": model_prediction_images,
    }
