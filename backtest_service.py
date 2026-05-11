#回测服务：用历史数据验证策略收益与风险
from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestResult:
    signal: np.ndarray
    signal_bt: np.ndarray
    real_price: np.ndarray
    pred_price: np.ndarray
    dates: object
    dates_bt: object
    buy_hold: np.ndarray
    daily_return: np.ndarray
    cumulative_return: np.ndarray
    drawdown: np.ndarray
    max_drawdown: float
    max_dd_idx: int
    sharpe: float
    total_return: float


def make_signal(probabilities, buy_threshold=0.6, sell_threshold=0.4):
    signal = []
    for probability in probabilities:
        if probability > buy_threshold:
            signal.append(1)
        elif probability < sell_threshold:
            signal.append(-1)
        else:
            signal.append(0)
    return np.array(signal)


def run_high_frequency_backtest(
    real_price,
    pred_price,
    probabilities,
    pandas_df,
    buy_threshold=0.55,
    sell_threshold=0.45,
    fee=0.001,
):
    """高频回测：每个测试周期都按涨跌概率重新调整仓位。"""
    signal = make_signal(probabilities, buy_threshold, sell_threshold)
    min_len = min(len(signal), len(real_price), len(pred_price))
    signal = signal[:min_len]
    real_price = real_price[:min_len]
    pred_price = pred_price[:min_len]
    dates = pandas_df["交易日期"].iloc[-min_len:]

    price_diff = (real_price[1:] - real_price[:-1]) / real_price[:-1]
    buy_hold = (real_price[1:] - real_price[0]) / real_price[0]

    # 高频口径：每个周期都重新判断下一周期是否持仓，允许频繁进出。
    signal_bt = np.where(signal[:-1] == 1, 1, 0)
    dates_bt = dates.iloc[1 : 1 + len(signal_bt)]
    trade_change = np.abs(np.diff(signal_bt))
    trade_change = np.insert(trade_change, 0, abs(signal_bt[0]) if len(signal_bt) else 0)

    min_len2 = min(len(price_diff), len(signal_bt), len(trade_change))
    price_diff = price_diff[:min_len2]
    signal_bt = signal_bt[:min_len2]
    trade_change = trade_change[:min_len2]
    dates_bt = dates_bt.iloc[:min_len2]
    buy_hold = buy_hold[:min_len2]

    daily_return = signal_bt * price_diff - fee * trade_change
    daily_return = np.nan_to_num(daily_return)
    cumulative_return = np.cumsum(daily_return)

    if len(cumulative_return) == 0:
        drawdown = np.zeros_like(cumulative_return)
        max_drawdown = 0.0
        max_dd_idx = -1
    else:
        cummax = np.maximum.accumulate(cumulative_return)
        drawdown = cumulative_return - cummax
        max_drawdown = float(drawdown.min())
        max_dd_idx = int(np.argmin(drawdown))

    sharpe = 0.0
    if len(daily_return) and np.std(daily_return) != 0:
        sharpe = float(np.mean(daily_return) / np.std(daily_return) * np.sqrt(252))

    total_return = float(cumulative_return[-1]) if len(cumulative_return) else 0.0

    return BacktestResult(
        signal=signal,
        signal_bt=signal_bt,
        real_price=real_price,
        pred_price=pred_price,
        dates=dates,
        dates_bt=dates_bt,
        buy_hold=buy_hold,
        daily_return=daily_return,
        cumulative_return=cumulative_return,
        drawdown=drawdown,
        max_drawdown=max_drawdown,
        max_dd_idx=max_dd_idx,
        sharpe=sharpe,
        total_return=total_return,
    )
