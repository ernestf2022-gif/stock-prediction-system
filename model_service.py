#模型服务：训练、预测、输出结果。
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backtest_service import run_daily_signal_backtest
from config import (
    ADAPTIVE_SIGNAL_THRESHOLDS,
    BATCH_SIZE,
    BUY_THRESHOLD,
    BUY_SIGNAL_QUANTILE,
    DIRECTION_RETURN_THRESHOLD,
    END_DATE,
    EPOCHS,
    INITIAL_CAPITAL,
    LOT_SIZE,
    MAX_POSITION_PCT,
    MIN_COMMISSION,
    MIN_EXPECTED_RETURN,
    MIN_SIGNAL_THRESHOLD_GAP,
    RANDOM_SEED,
    SELL_THRESHOLD,
    SELL_SIGNAL_QUANTILE,
    SELL_TAX,
    SIGNAL_ADAPTIVE_WINDOW,
    SLIPPAGE,
    START_DATE,
    STOCK_CODE,
    STOP_LOSS,
    TAKE_PROFIT,
    TIME_STEP,
    TRANSACTION_FEE,
)
from data_service import format_stock_label, prepare_stock_dataset, resolve_stock_name
from plot_service import plot_daily_backtest, plot_direction_loss_curves, setup_matplotlib


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

DIRECTION_MODEL_NAME = "BiLSTM-Attention Direction"


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


class BiLSTMAttentionDirection(nn.Module):
    """双向 LSTM + LayerNorm + Attention：输出下一周期上涨概率，用于交易信号。"""

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
    for column in ["开盘价", "最高价", "最低价", "收盘价"]:
        train_df.attrs[f"raw_{column}"] = train_data_raw[column].to_numpy(dtype=float)
        test_df.attrs[f"raw_{column}"] = test_data_raw[column].to_numpy(dtype=float)
    if "交易日期" in pandas_df.columns:
        train_df.attrs["raw_dates"] = pandas_df["交易日期"].iloc[:train_size].to_numpy()
        test_df.attrs["raw_dates"] = pandas_df["交易日期"].iloc[train_size:].to_numpy()
    train_df.attrs["raw_close"] = train_data_raw["收盘价"].to_numpy(dtype=float)
    test_df.attrs["raw_close"] = test_data_raw["收盘价"].to_numpy(dtype=float)
    return train_df, test_df, scaler


def build_data_loaders(train_data, test_data, batch_size=BATCH_SIZE):
    X_train_dir, y_train_dir = create_direction_dataset(train_data)
    X_test_dir, y_test_dir = create_direction_dataset(test_data)

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
    return train_loader_dir, test_loader_dir


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


def train_models(train_loader_dir, test_loader_dir, input_size, epochs=EPOCHS):
    direction_model = BiLSTMAttentionDirection(input_size)

    dir_loss_history, dir_test_loss_history = train_direction_model(
        direction_model, train_loader_dir, test_loader_dir, epochs
    )

    return {
        "direction_model": direction_model,
        "dir_loss_history": dir_loss_history,
        "dir_test_loss_history": dir_test_loss_history,
    }


def predict_direction(direction_model, test_loader_dir, decision_threshold=0.5):
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
    dir_preds = (probabilities >= decision_threshold).astype(int)
    return probabilities, dir_reals, dir_preds


def calibrate_signal_thresholds(
    train_probabilities,
    base_buy_threshold=BUY_THRESHOLD,
    base_sell_threshold=SELL_THRESHOLD,
    buy_quantile=BUY_SIGNAL_QUANTILE,
    sell_quantile=SELL_SIGNAL_QUANTILE,
    min_gap=MIN_SIGNAL_THRESHOLD_GAP,
):
    probabilities = np.asarray(train_probabilities, dtype=float)
    probabilities = probabilities[np.isfinite(probabilities)]
    if not ADAPTIVE_SIGNAL_THRESHOLDS or len(probabilities) < 10:
        return base_buy_threshold, base_sell_threshold

    adaptive_buy = float(np.quantile(probabilities, buy_quantile))
    adaptive_sell = float(np.quantile(probabilities, sell_quantile))
    buy_threshold = min(base_buy_threshold, adaptive_buy)
    sell_threshold = max(base_sell_threshold, adaptive_sell)

    if sell_threshold >= buy_threshold - min_gap:
        sell_threshold = buy_threshold - min_gap

    buy_threshold = min(max(buy_threshold, 0.01), 0.99)
    sell_threshold = min(max(sell_threshold, 0.01), buy_threshold - 1e-6)
    return buy_threshold, sell_threshold


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
        positive_rate = float(signal_count / min_len) if min_len else 0.0
        if signal_count:
            signal_hit_rate = float(np.mean(np.asarray(dir_reals[:min_len])[buy_mask] == 1))
    return {
        "direction_accuracy": direction_accuracy,
        "avg_up_probability": avg_up_probability,
        "positive_rate": positive_rate,
        "signal_hit_rate": signal_hit_rate,
        "signal_count": signal_count,
    }


def inverse_test_market_data(test_data, prediction_len, scaler, features=FEATURES):
    start = TIME_STEP
    end = TIME_STEP + prediction_len
    raw_close = test_data.attrs.get("raw_收盘价")
    if raw_close is None:
        raw_close = test_data.attrs.get("raw_close")
    raw_map = {
        "open": test_data.attrs.get("raw_开盘价"),
        "high": test_data.attrs.get("raw_最高价"),
        "low": test_data.attrs.get("raw_最低价"),
        "close": raw_close,
    }
    raw_dates = test_data.attrs.get("raw_dates")

    if all(values is not None and len(values) >= end for values in raw_map.values()):
        market_data = pd.DataFrame({name: np.asarray(values[start:end], dtype=float) for name, values in raw_map.items()})
    else:
        market_data = pd.DataFrame()
        for output_name, feature_name in {
            "open": "开盘价",
            "high": "最高价",
            "low": "最低价",
            "close": "收盘价",
        }.items():
            values = test_data[feature_name].iloc[start:end].to_numpy(dtype=float)
            dummy = np.zeros((len(values), len(features)))
            dummy[:, features.index(feature_name)] = values
            market_data[output_name] = scaler.inverse_transform(dummy)[:, features.index(feature_name)]

    if raw_dates is not None and len(raw_dates) >= end:
        market_data["date"] = raw_dates[start:end]
    else:
        market_data["date"] = np.arange(len(market_data))
    return market_data


def run_pipeline(stock_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE, stock_name=None):
    """股票数据爬取 + 涨跌概率信号训练 + 日线策略回测流程。"""
    set_random_seed()
    stock_name = stock_name or resolve_stock_name(stock_code, allow_remote=False)
    stock_label = format_stock_label(stock_code, stock_name)
    lstm_df, _, _ = prepare_stock_dataset(stock_code, start_date, end_date)

    setup_matplotlib()
    pandas_df = lstm_df.copy().dropna()
    train_data, test_data, scaler = scale_train_test_data(pandas_df)
    train_loader_dir, test_loader_dir = build_data_loaders(train_data, test_data)

    trained = train_models(
        train_loader_dir,
        test_loader_dir,
        input_size=len(FEATURES),
        epochs=EPOCHS,
    )
    direction_model = trained["direction_model"]

    print(f"数据截止日期: {pandas_df['交易日期'].max()}")

    train_probabilities, _, _ = predict_direction(direction_model, train_loader_dir)
    buy_threshold, sell_threshold = calibrate_signal_thresholds(train_probabilities)
    probabilities, dir_reals, dir_preds = predict_direction(
        direction_model,
        test_loader_dir,
        decision_threshold=buy_threshold,
    )
    market_data = inverse_test_market_data(test_data, len(probabilities), scaler)
    backtest = run_daily_signal_backtest(
        market_data["close"].to_numpy(dtype=float),
        None,
        probabilities,
        pandas_df,
        market_data=market_data,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        fee=TRANSACTION_FEE,
        sell_tax=SELL_TAX,
        min_commission=MIN_COMMISSION,
        initial_capital=INITIAL_CAPITAL,
        max_position_pct=MAX_POSITION_PCT,
        slippage=SLIPPAGE,
        lot_size=LOT_SIZE,
        min_expected_return=MIN_EXPECTED_RETURN,
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
        reference_probabilities=train_probabilities,
        buy_quantile=BUY_SIGNAL_QUANTILE,
        sell_quantile=SELL_SIGNAL_QUANTILE,
        min_threshold_gap=MIN_SIGNAL_THRESHOLD_GAP,
        adaptive_window=SIGNAL_ADAPTIVE_WINDOW,
    )
    cls_metrics = evaluate_direction(dir_reals, dir_preds, probabilities=probabilities, trade_signal=backtest.signal)
    print(f"信号模型: {DIRECTION_MODEL_NAME} 涨跌概率")
    print(
        f"实际买入阈值: {buy_threshold:.4f}, 实际卖出阈值: {sell_threshold:.4f}, "
        f"滚动阈值窗口: {SIGNAL_ADAPTIVE_WINDOW}"
    )
    print(f"方向命中率: {cls_metrics['direction_accuracy']:.4f}")
    print(f"平均上涨概率: {cls_metrics['avg_up_probability']:.4f}")
    print(f"信号触发比例: {cls_metrics['positive_rate']:.4f}")
    print(f"交易信号命中率: {cls_metrics['signal_hit_rate']:.4f}")
    print(f"买入信号次数: {cls_metrics['signal_count']}")
    print(
        f"日线策略总收益率: {backtest.total_return:.4f}, "
        f"最大回撤: {backtest.max_drawdown:.4f}, 年化夏普: {backtest.sharpe:.4f}"
    )
    print(
        f"期末资产: {backtest.final_capital:.2f}, 交易次数: {backtest.trade_count}, "
        f"胜率: {backtest.win_rate:.4f}, 平均仓位: {backtest.exposure:.4f}, "
        f"累计交易成本: {backtest.fees_paid:.2f}"
    )

    plot_direction_loss_curves(
        trained["dir_loss_history"],
        trained["dir_test_loss_history"],
        EPOCHS,
        stock_label=stock_label,
    )
    plot_daily_backtest(
        backtest.dates_bt,
        backtest.cumulative_return,
        backtest.buy_hold,
        backtest.max_dd_idx,
        stock_label=stock_label,
    )

    return []
