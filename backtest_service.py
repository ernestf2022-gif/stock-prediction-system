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
    trade_cycles: list
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


def make_holding_state(signal):
    position = 0
    signal_bt = []
    for item in signal:
        if item == 1:
            position = 1
        elif item == -1:
            position = 0
        signal_bt.append(position)
    return np.array(signal_bt)


def build_trade_cycles(signal_bt, real_price, dates_bt):
    trade_cycles = []
    position = 0
    entry_price = 0
    entry_date = None

    for i in range(len(signal_bt)):
        if signal_bt[i] == 1 and position == 0:
            position = 1
            entry_price = real_price[i]
            entry_date = dates_bt.iloc[i]
        elif signal_bt[i] == 0 and position == 1:
            exit_price = real_price[i]
            exit_date = dates_bt.iloc[i]
            ret = (exit_price - entry_price) / entry_price
            hold_days = (exit_date - entry_date).days
            trade_cycles.append(
                {
                    "entry_date": str(entry_date),
                    "exit_date": str(exit_date),
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "return": float(ret),
                    "hold_days": int(hold_days),
                }
            )
            position = 0

    return trade_cycles


def run_backtest(
    real_price,
    pred_price,
    probabilities,
    pandas_df,
    buy_threshold=0.6,
    sell_threshold=0.4,
    fee=0.001,
):
    signal = make_signal(probabilities, buy_threshold, sell_threshold)
    holding_state = make_holding_state(signal)

    min_len = min(len(holding_state), len(real_price), len(pred_price))
    holding_state = holding_state[:min_len]
    real_price = real_price[:min_len]
    pred_price = pred_price[:min_len]
    dates = pandas_df["交易日期"].iloc[-min_len:]

    price_diff = (real_price[1:] - real_price[:-1]) / real_price[:-1]
    buy_hold = (real_price[1:] - real_price[0]) / real_price[0]

    # 保留原脚本的策略口径：只在概率超过买入阈值的下一周期持仓。
    signal_bt = np.where(signal[:-1] == 1, 1, 0)
    dates_bt = dates.iloc[1 : 1 + len(signal_bt)]
    trade_cycles = build_trade_cycles(signal_bt, real_price, dates_bt)

    trade_change = np.abs(np.diff(signal_bt))
    trade_change = np.insert(trade_change, 0, 0)

    min_len2 = min(len(price_diff), len(signal_bt), len(trade_change))
    price_diff = price_diff[:min_len2]
    signal_bt = signal_bt[:min_len2]
    trade_change = trade_change[:min_len2]
    dates_bt = dates_bt.iloc[:min_len2]
    buy_hold = buy_hold[:min_len2]

    daily_return = signal_bt * price_diff - fee * trade_change
    daily_return = np.nan_to_num(daily_return)
    if len(daily_return):
        daily_return[0] = 0

    cumulative_return = np.cumsum(daily_return)
    if len(cumulative_return) == 0 or np.sum(signal_bt) == 0:
        drawdown = np.zeros_like(cumulative_return)
        max_drawdown = 0.0
        max_dd_idx = -1
    else:
        cummax = np.maximum.accumulate(cumulative_return)
        drawdown = cumulative_return - cummax
        drawdown[0] = 0
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
        trade_cycles=trade_cycles,
        total_return=total_return,
    )


def make_strategy_suggestion(probability, buy_threshold=0.6, sell_threshold=0.4):
    if probability > buy_threshold:
        return "建议买入"
    if probability < sell_threshold:
        return "建议卖出"
    return "建议观望"
