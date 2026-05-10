import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backtest_service import make_strategy_suggestion, run_backtest
from config import (
    BATCH_SIZE,
    BUY_THRESHOLD,
    END_DATE,
    EPOCHS,
    FUTURE_DAYS,
    SELL_THRESHOLD,
    START_DATE,
    STOCK_CODE,
    TIME_STEP,
    TRANSACTION_FEE,
)
from data_service import format_stock_label, prepare_stock_dataset, resolve_stock_name
from plot_service import plot_loss_curves, plot_prediction_and_backtest, plot_roc_curve, setup_matplotlib


FEATURES = [
    "开盘价",
    "最高价",
    "最低价",
    "收盘价",
    "成交额(千元)",
    "MA5",
    "MA10",
    "MACD",
    "RSI",
    "VOLATILITY",
    "大盘指数",
    "RET",
    "LOG_RET",
    "INDEX_RET",
]


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(out), dim=1)
        context = (weights * out).sum(dim=1)
        return self.fc(context)


class LSTMDirection(nn.Module):
    """分类模型：预测涨跌方向。"""

    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])


def create_dataset(df, features=FEATURES, time_step=TIME_STEP):
    X, y = [], []
    close_idx = features.index("收盘价")
    for i in range(len(df) - time_step - 1):
        X.append(df.iloc[i : (i + time_step)].values)
        y.append(df.iloc[i + time_step, close_idx])
    return np.array(X), np.array(y)


def create_direction_dataset(df, time_step=TIME_STEP):
    X, y = [], []
    for i in range(len(df) - time_step - 1):
        X.append(df.iloc[i : (i + time_step)].values)
        today_close = df.iloc[i + time_step - 1]["收盘价"]
        tomorrow_close = df.iloc[i + time_step]["收盘价"]
        y.append(1 if tomorrow_close > today_close else 0)
    return np.array(X), np.array(y)


def scale_train_test_data(pandas_df, features=FEATURES):
    data = pandas_df[features]
    train_size = int(len(data) * 0.8)

    train_data_raw = data.iloc[:train_size]
    test_data_raw = data.iloc[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data_raw)
    test_scaled = scaler.transform(test_data_raw)

    train_df = pd.DataFrame(train_scaled, columns=features, index=train_data_raw.index)
    test_df = pd.DataFrame(test_scaled, columns=features, index=test_data_raw.index)
    scaled_df = pd.concat([train_df, test_df])
    return train_df, test_df, scaled_df, scaler


def build_data_loaders(train_data, test_data, batch_size=BATCH_SIZE):
    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)
    X_train_dir, y_train_dir = create_direction_dataset(train_data)
    X_test_dir, y_test_dir = create_direction_dataset(test_data)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    train_loader_dir = DataLoader(
        TensorDataset(torch.tensor(X_train_dir, dtype=torch.float32), torch.tensor(y_train_dir, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader_dir = DataLoader(
        TensorDataset(torch.tensor(X_test_dir, dtype=torch.float32), torch.tensor(y_test_dir, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, train_loader_dir, test_loader, test_loader_dir


def train_regression_model(model, train_loader, test_loader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_loss_history = []
    test_loss_history = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        train_loss_history.append(epoch_loss / batch_count if batch_count else 0)

        model.eval()
        test_epoch_loss = 0.0
        test_batch_count = 0
        with torch.no_grad():
            for x, y in test_loader:
                test_loss = criterion(model(x), y)
                test_epoch_loss += test_loss.item()
                test_batch_count += 1
        test_loss_history.append(test_epoch_loss / test_batch_count if test_batch_count else 0)

    return train_loss_history, test_loss_history


def train_direction_model(model, train_loader, test_loader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loss_history = []
    test_loss_history = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        train_loss_history.append(epoch_loss / batch_count if batch_count else 0)

        model.eval()
        test_epoch_loss = 0.0
        test_batch_count = 0
        with torch.no_grad():
            for x, y in test_loader:
                test_loss = criterion(model(x), y)
                test_epoch_loss += test_loss.item()
                test_batch_count += 1
        test_loss_history.append(test_epoch_loss / test_batch_count if test_batch_count else 0)

    return train_loss_history, test_loss_history


def train_models(train_loader, train_loader_dir, test_loader, test_loader_dir, input_size, epochs=EPOCHS):
    model = AttentionLSTM(input_size)
    dir_model = LSTMDirection(input_size)

    reg_loss_history, reg_test_loss_history = train_regression_model(model, train_loader, test_loader, epochs)
    dir_loss_history, dir_test_loss_history = train_direction_model(
        dir_model, train_loader_dir, test_loader_dir, epochs
    )

    return {
        "model": model,
        "direction_model": dir_model,
        "reg_loss_history": reg_loss_history,
        "reg_test_loss_history": reg_test_loss_history,
        "dir_loss_history": dir_loss_history,
        "dir_test_loss_history": dir_test_loss_history,
    }


def predict_prices(model, test_loader, scaler, features=FEATURES):
    model.eval()
    preds, reals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x).numpy())
            reals.append(y.numpy())

    preds = np.concatenate(preds)
    reals = np.concatenate(reals)

    close_idx = features.index("收盘价")
    dummy = np.zeros((len(preds), len(features)))
    dummy[:, close_idx] = preds.flatten()
    pred_price = scaler.inverse_transform(dummy)[:, close_idx]
    dummy[:, close_idx] = reals.flatten()
    real_price = scaler.inverse_transform(dummy)[:, close_idx]
    return pred_price, real_price


def evaluate_prices(real_price, pred_price):
    mse = mean_squared_error(real_price, pred_price)
    mae = mean_absolute_error(real_price, pred_price)
    avg_price = real_price.mean()
    error_rate = mae / avg_price
    return mse, mae, avg_price, error_rate


def predict_direction(direction_model, test_loader_dir):
    direction_model.eval()
    probabilities = []
    dir_reals = []
    with torch.no_grad():
        for x, y in test_loader_dir:
            out = direction_model(x)
            probability = torch.softmax(out, dim=1)[:, 1]
            probabilities.append(probability.numpy())
            dir_reals.append(y.numpy())

    probabilities = np.concatenate(probabilities)
    dir_reals = np.concatenate(dir_reals)
    dir_preds = (probabilities >= 0.5).astype(int)
    return probabilities, dir_reals, dir_preds


def evaluate_direction(dir_reals, dir_preds):
    cls_accuracy = accuracy_score(dir_reals, dir_preds)
    cls_precision = precision_score(dir_reals, dir_preds, zero_division=0)
    cls_recall = recall_score(dir_reals, dir_preds, zero_division=0)
    cls_f1 = f1_score(dir_reals, dir_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(dir_reals, dir_preds, labels=[0, 1]).ravel()
    return {
        "accuracy": cls_accuracy,
        "precision": cls_precision,
        "recall": cls_recall,
        "f1": cls_f1,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def predict_future_prices(model, scaled_df, scaler, pandas_df, features=FEATURES, future_days=FUTURE_DAYS):
    future_prices = []
    future_dates = []
    close_idx = features.index("收盘价")
    temp_window = scaled_df.iloc[-TIME_STEP:].values.copy()
    next_date = pd.Timestamp(pandas_df["交易日期"].max())

    model.eval()
    for _ in range(future_days):
        next_date = next_date + pd.offsets.BDay(1)
        future_dates.append(next_date.strftime("%Y-%m-%d"))

        input_tensor = torch.tensor(temp_window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor).squeeze().detach().cpu().numpy().item()

        dummy = np.zeros((1, len(features)))
        dummy[:, close_idx] = pred
        next_price = scaler.inverse_transform(dummy)[0, close_idx]
        future_prices.append(next_price)

        new_row = temp_window[-1].copy()
        new_row[close_idx] = pred
        temp_window = np.vstack([temp_window[1:], new_row])

    return future_dates, future_prices


def run_pipeline(stock_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE, stock_name=None):
    """股票数据爬取 + LSTM预测 + 回测 + 可视化完整流程。"""
    stock_name = stock_name or resolve_stock_name(stock_code, allow_remote=False)
    stock_label = format_stock_label(stock_code, stock_name)
    lstm_df, _, _ = prepare_stock_dataset(stock_code, start_date, end_date)

    setup_matplotlib()
    pandas_df = lstm_df.copy().dropna()
    train_data, test_data, scaled_df, scaler = scale_train_test_data(pandas_df)
    train_loader, train_loader_dir, test_loader, test_loader_dir = build_data_loaders(train_data, test_data)

    trained = train_models(
        train_loader,
        train_loader_dir,
        test_loader,
        test_loader_dir,
        input_size=len(FEATURES),
        epochs=EPOCHS,
    )
    model = trained["model"]
    direction_model = trained["direction_model"]

    pred_price, real_price = predict_prices(model, test_loader, scaler)
    mse, mae, avg_price, error_rate = evaluate_prices(real_price, pred_price)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"平均股价: {avg_price:.2f}")
    print(f"MAE误差率: {error_rate:.2%}")

    probabilities, dir_reals, dir_preds = predict_direction(direction_model, test_loader_dir)
    cls_metrics = evaluate_direction(dir_reals, dir_preds)
    print(f"分类准确率: {cls_metrics['accuracy']:.4f}")
    print(f"分类精确率: {cls_metrics['precision']:.4f}")
    print(f"分类召回率: {cls_metrics['recall']:.4f}")
    print(f"分类F1: {cls_metrics['f1']:.4f}")
    cm = cls_metrics["confusion_matrix"]
    print(f"混淆矩阵: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")

    backtest = run_backtest(
        real_price,
        pred_price,
        probabilities,
        pandas_df,
        buy_threshold=BUY_THRESHOLD,
        sell_threshold=SELL_THRESHOLD,
        fee=TRANSACTION_FEE,
    )
    print(f"数据截止日期: {pandas_df['交易日期'].max()}")
    print(
        f"{stock_label} 总收益率: {backtest.total_return:.4f}, "
        f"最大回撤: {backtest.max_drawdown:.4f}, 年化夏普: {backtest.sharpe:.4f}"
    )

    future_dates, future_prices = predict_future_prices(model, scaled_df, scaler, pandas_df)
    print("📢 未来10天预测价格：")
    for date_text, price in zip(future_dates, future_prices):
        print(f"{date_text}: {price:.2f}")

    suggestion = make_strategy_suggestion(probabilities[-1], BUY_THRESHOLD, SELL_THRESHOLD)
    print(f"📢 策略建议: {suggestion}")

    plot_loss_curves(
        trained["reg_loss_history"],
        trained["reg_test_loss_history"],
        trained["dir_loss_history"],
        trained["dir_test_loss_history"],
        EPOCHS,
        stock_label=stock_label,
    )
    plot_roc_curve(dir_reals, probabilities, stock_label=stock_label)
    plot_prediction_and_backtest(
        backtest.dates,
        backtest.real_price,
        backtest.pred_price,
        backtest.signal_bt,
        backtest.dates_bt,
        backtest.cumulative_return,
        backtest.buy_hold,
        backtest.max_dd_idx,
        stock_label=stock_label,
    )

    return backtest.trade_cycles
