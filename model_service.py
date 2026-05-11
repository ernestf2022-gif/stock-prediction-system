#模型服务：训练、预测、输出结果。
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backtest_service import run_high_frequency_backtest
from config import (
    BATCH_SIZE,
    BUY_THRESHOLD,
    DIRECTION_RETURN_THRESHOLD,
    END_DATE,
    EPOCHS,
    FUTURE_DAYS,
    INITIAL_CAPITAL,
    LOT_SIZE,
    MAX_POSITION_PCT,
    MIN_EXPECTED_RETURN,
    SELL_THRESHOLD,
    SLIPPAGE,
    START_DATE,
    STOCK_CODE,
    STOP_LOSS,
    TAKE_PROFIT,
    TIME_STEP,
    TRANSACTION_FEE,
)
from data_service import format_stock_label, prepare_stock_dataset, resolve_stock_name
from plot_service import plot_high_frequency_backtest, plot_loss_curves, plot_price_prediction, setup_matplotlib


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


class VanillaLSTM(nn.Module):
    """普通 LSTM：用于收盘价预测。"""

    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class VanillaLSTMDirection(nn.Module):
    """增强 LSTM：输出下一周期上涨概率，用于交易信号。"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        output_size = hidden_size * 2
        self.norm = nn.LayerNorm(output_size)
        self.attention = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out)
        weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(weights * out, dim=1)
        return self.fc(context).squeeze(-1)


def create_dataset(df, features=FEATURES, time_step=TIME_STEP):
    X, y = [], []
    close_idx = features.index("收盘价")
    for i in range(len(df) - time_step - 1):
        X.append(df.iloc[i : (i + time_step)].values)
        y.append(df.iloc[i + time_step, close_idx])
    return np.array(X), np.array(y)


def create_direction_dataset(df, time_step=TIME_STEP, min_return=DIRECTION_RETURN_THRESHOLD):
    X, y = [], []
    raw_close = df.attrs.get("raw_close")
    for i in range(len(df) - time_step - 1):
        X.append(df.iloc[i : (i + time_step)].values)
        if raw_close is not None and len(raw_close) == len(df):
            today_close = raw_close[i + time_step - 1]
            tomorrow_close = raw_close[i + time_step]
            next_return = tomorrow_close / max(today_close, 1e-8) - 1
            y.append(1 if next_return > min_return else 0)
        else:
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
    train_df.attrs["raw_close"] = train_data_raw["收盘价"].to_numpy(dtype=float)
    test_df.attrs["raw_close"] = test_data_raw["收盘价"].to_numpy(dtype=float)
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
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    train_loader_dir = DataLoader(
        TensorDataset(torch.tensor(X_train_dir, dtype=torch.float32), torch.tensor(y_train_dir, dtype=torch.long)),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    train_labels = train_loader.dataset.tensors[1].float()
    positive_count = float(train_labels.sum().item())
    negative_count = float(len(train_labels) - positive_count)
    pos_weight_value = negative_count / max(positive_count, 1.0)
    pos_weight_value = min(max(pos_weight_value, 0.2), 5.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32))
    train_loss_history = []
    test_loss_history = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        train_loss_history.append(epoch_loss / batch_count if batch_count else 0)

        model.eval()
        test_epoch_loss = 0.0
        test_batch_count = 0
        with torch.no_grad():
            for x, y in test_loader:
                test_loss = criterion(model(x), y.float())
                test_epoch_loss += test_loss.item()
                test_batch_count += 1
        test_loss_history.append(test_epoch_loss / test_batch_count if test_batch_count else 0)

    return train_loss_history, test_loss_history


def train_models(train_loader, train_loader_dir, test_loader, test_loader_dir, input_size, epochs=EPOCHS):
    model = VanillaLSTM(input_size)
    direction_model = VanillaLSTMDirection(input_size)

    reg_loss_history, reg_test_loss_history = train_regression_model(model, train_loader, test_loader, epochs)
    dir_loss_history, dir_test_loss_history = train_direction_model(
        direction_model, train_loader_dir, test_loader_dir, epochs
    )

    return {
        "model": model,
        "direction_model": direction_model,
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
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(real_price, pred_price)
    mape = float(np.mean(np.abs((real_price - pred_price) / np.maximum(np.abs(real_price), 1e-8))) * 100)
    r2 = r2_score(real_price, pred_price)
    avg_price = real_price.mean()
    error_rate = mae / avg_price
    return {
        "rmse": rmse,
        "mae": float(mae),
        "mape": mape,
        "r2": float(r2),
        "avg_price": float(avg_price),
        "error_rate": float(error_rate),
    }


def predict_direction(direction_model, test_loader_dir):
    direction_model.eval()
    probabilities = []
    dir_reals = []
    with torch.no_grad():
        for x, y in test_loader_dir:
            out = direction_model(x)
            probability = torch.sigmoid(out)
            probabilities.append(probability.numpy())
            dir_reals.append(y.numpy())

    probabilities = np.concatenate(probabilities)
    dir_reals = np.concatenate(dir_reals)
    dir_preds = (probabilities >= 0.5).astype(int)
    return probabilities, dir_reals, dir_preds


def evaluate_direction(dir_reals, dir_preds, probabilities=None, trade_signal=None):
    direction_accuracy = float(np.mean(dir_reals == dir_preds)) if len(dir_reals) else 0.0
    avg_up_probability = float(np.mean(probabilities)) if probabilities is not None and len(probabilities) else 0.0
    positive_rate = float(np.mean(dir_preds)) if len(dir_preds) else 0.0
    signal_hit_rate = 0.0
    signal_count = 0
    if trade_signal is not None and len(trade_signal):
        min_len = min(len(trade_signal), len(dir_reals))
        buy_mask = np.asarray(trade_signal[:min_len]) == 1
        signal_count = int(buy_mask.sum())
        if signal_count:
            signal_hit_rate = float(np.mean(np.asarray(dir_reals[:min_len])[buy_mask] == 1))
    return {
        "direction_accuracy": direction_accuracy,
        "avg_up_probability": avg_up_probability,
        "positive_rate": positive_rate,
        "signal_hit_rate": signal_hit_rate,
        "signal_count": signal_count,
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
    """股票数据爬取 + Vanilla LSTM 收盘价预测 + 可视化完整流程。"""
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
    price_metrics = evaluate_prices(real_price, pred_price)
    print(f"数据截止日期: {pandas_df['交易日期'].max()}")
    print(f"模型: Vanilla LSTM 收盘价预测")
    print(f"RMSE: {price_metrics['rmse']:.4f}, MAE: {price_metrics['mae']:.4f}")
    print(f"MAPE: {price_metrics['mape']:.4f}%, R2: {price_metrics['r2']:.4f}")
    print(f"平均股价: {price_metrics['avg_price']:.2f}")
    print(f"MAE误差率: {price_metrics['error_rate']:.2%}")

    probabilities, dir_reals, dir_preds = predict_direction(direction_model, test_loader_dir)
    backtest = run_high_frequency_backtest(
        real_price,
        pred_price,
        probabilities,
        pandas_df,
        buy_threshold=BUY_THRESHOLD,
        sell_threshold=SELL_THRESHOLD,
        fee=TRANSACTION_FEE,
        initial_capital=INITIAL_CAPITAL,
        max_position_pct=MAX_POSITION_PCT,
        slippage=SLIPPAGE,
        lot_size=LOT_SIZE,
        min_expected_return=MIN_EXPECTED_RETURN,
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
    )
    cls_metrics = evaluate_direction(dir_reals, dir_preds, probabilities=probabilities, trade_signal=backtest.signal)
    print(f"信号模型: Enhanced Vanilla LSTM 涨跌概率")
    print(f"方向命中率: {cls_metrics['direction_accuracy']:.4f}")
    print(f"平均上涨概率: {cls_metrics['avg_up_probability']:.4f}")
    print(f"模型看涨比例: {cls_metrics['positive_rate']:.4f}")
    print(f"交易信号命中率: {cls_metrics['signal_hit_rate']:.4f}")
    print(f"买入信号次数: {cls_metrics['signal_count']}")
    print(
        f"高频策略总收益率: {backtest.total_return:.4f}, "
        f"最大回撤: {backtest.max_drawdown:.4f}, 年化夏普: {backtest.sharpe:.4f}"
    )
    print(
        f"期末资产: {backtest.final_capital:.2f}, 交易次数: {backtest.trade_count}, "
        f"胜率: {backtest.win_rate:.4f}, 平均仓位: {backtest.exposure:.4f}, "
        f"累计交易成本: {backtest.fees_paid:.2f}"
    )

    future_dates, future_prices = predict_future_prices(model, scaled_df, scaler, pandas_df)
    print("📢 未来10天预测价格：")
    for date_text, price in zip(future_dates, future_prices):
        print(f"{date_text}: {price:.2f}")

    plot_loss_curves(
        trained["reg_loss_history"],
        trained["reg_test_loss_history"],
        trained["dir_loss_history"],
        trained["dir_test_loss_history"],
        EPOCHS,
        stock_label=stock_label,
    )
    dates = pandas_df["交易日期"].iloc[-len(real_price):]
    plot_price_prediction(dates, real_price, pred_price, stock_label=stock_label)
    plot_high_frequency_backtest(
        backtest.dates_bt,
        backtest.cumulative_return,
        backtest.buy_hold,
        backtest.max_dd_idx,
        stock_label=stock_label,
    )

    return []
