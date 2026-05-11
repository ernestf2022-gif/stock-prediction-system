#回测服务：用历史数据验证策略收益与风险
from dataclasses import dataclass

import numpy as np
import pandas as pd


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
    final_capital: float
    trade_count: int
    win_rate: float
    exposure: float
    fees_paid: float
    turnover: float


def make_signal(
    probabilities,
    pred_price=None,
    real_price=None,
    buy_threshold=0.6,
    sell_threshold=0.4,
    min_expected_return=0.002,
):
    probabilities = np.asarray(probabilities, dtype=float)
    signal = np.zeros(len(probabilities), dtype=int)

    if pred_price is not None and real_price is not None:
        pred_price = np.asarray(pred_price, dtype=float)
        real_price = np.asarray(real_price, dtype=float)
        min_len = min(len(probabilities), len(pred_price), len(real_price))
        probabilities = probabilities[:min_len]
        signal = signal[:min_len]
        expected_return = np.zeros(min_len)
        if min_len > 1:
            expected_return[1:] = pred_price[1:min_len] / np.maximum(real_price[: min_len - 1], 1e-8) - 1
    else:
        expected_return = np.zeros(len(probabilities))

    for idx, probability in enumerate(probabilities):
        if probability >= buy_threshold and expected_return[idx] >= min_expected_return:
            signal[idx] = 1
        elif probability <= sell_threshold or expected_return[idx] <= -min_expected_return:
            signal[idx] = -1
    return np.array(signal)


def run_high_frequency_backtest(
    real_price,
    pred_price,
    probabilities,
    pandas_df,
    buy_threshold=0.55,
    sell_threshold=0.45,
    fee=0.001,
    initial_capital=100000.0,
    max_position_pct=0.95,
    slippage=0.0005,
    lot_size=100,
    min_expected_return=0.002,
    stop_loss=0.035,
    take_profit=0.07,
):
    """高频回测：按预测概率和预期收益调仓，用现金/持仓账户模拟交易。"""
    signal = make_signal(
        probabilities,
        pred_price=pred_price,
        real_price=real_price,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        min_expected_return=min_expected_return,
    )
    min_len = min(len(signal), len(real_price), len(pred_price))
    signal = signal[:min_len]
    real_price = np.asarray(real_price[:min_len], dtype=float)
    pred_price = pred_price[:min_len]
    if "交易日期" in pandas_df.columns:
        dates = pandas_df["交易日期"].iloc[-min_len:]
    else:
        dates = pd.Series(np.arange(min_len))

    if min_len <= 1:
        empty = np.array([])
        return BacktestResult(
            signal=signal,
            signal_bt=empty,
            real_price=real_price,
            pred_price=pred_price,
            dates=dates,
            dates_bt=dates.iloc[:0],
            buy_hold=empty,
            daily_return=empty,
            cumulative_return=empty,
            drawdown=empty,
            max_drawdown=0.0,
            max_dd_idx=-1,
            sharpe=0.0,
            total_return=0.0,
            final_capital=float(initial_capital),
            trade_count=0,
            win_rate=0.0,
            exposure=0.0,
            fees_paid=0.0,
            turnover=0.0,
        )

    cash = float(initial_capital)
    shares = 0
    entry_price = 0.0
    fees_paid = 0.0
    turnover = 0.0
    trade_count = 0
    winning_trades = 0
    closed_trades = 0
    equity_curve = []
    signal_bt = []
    exposure_values = []

    for idx in range(1, min_len):
        previous_close = real_price[idx - 1]
        current_close = real_price[idx]
        equity_before = cash + shares * previous_close
        target_pct = max_position_pct if signal[idx] == 1 else 0.0

        if target_pct <= 0 and shares > 0:
            sell_price = previous_close * (1 - slippage)
            gross = shares * sell_price
            cost = gross * fee
            cash += gross - cost
            fees_paid += cost
            turnover += gross
            trade_count += 1
            closed_trades += 1
            if sell_price > entry_price:
                winning_trades += 1
            shares = 0
            entry_price = 0.0
        elif target_pct > 0:
            target_value = equity_before * target_pct
            buy_price = previous_close * (1 + slippage)
            target_shares = int(target_value / max(buy_price, 1e-8))
            if lot_size > 1:
                target_shares = (target_shares // lot_size) * lot_size
            shares_to_buy = max(target_shares - shares, 0)
            gross = shares_to_buy * buy_price
            cost = gross * fee
            if shares_to_buy > 0 and gross + cost <= cash:
                cash -= gross + cost
                fees_paid += cost
                turnover += gross
                trade_count += 1
                entry_price = buy_price if shares == 0 else entry_price
                shares += shares_to_buy

        if shares > 0 and entry_price > 0:
            holding_return = current_close / entry_price - 1
            if holding_return <= -stop_loss or holding_return >= take_profit:
                sell_price = current_close * (1 - slippage)
                gross = shares * sell_price
                cost = gross * fee
                cash += gross - cost
                fees_paid += cost
                turnover += gross
                trade_count += 1
                closed_trades += 1
                if sell_price > entry_price:
                    winning_trades += 1
                shares = 0
                entry_price = 0.0

        equity = cash + shares * current_close
        equity_curve.append(equity)
        signal_bt.append(1 if shares > 0 else 0)
        exposure_values.append((shares * current_close) / max(equity, 1e-8))

    equity_curve = np.asarray(equity_curve, dtype=float)
    signal_bt = np.asarray(signal_bt, dtype=int)
    dates_bt = dates.iloc[1 : 1 + len(equity_curve)]
    buy_hold = real_price[1 : 1 + len(equity_curve)] / real_price[0] - 1
    cumulative_return = equity_curve / initial_capital - 1
    daily_return = np.diff(np.insert(equity_curve, 0, initial_capital)) / np.maximum(
        np.insert(equity_curve, 0, initial_capital)[:-1],
        1e-8,
    )

    if len(cumulative_return) == 0:
        drawdown = np.zeros_like(cumulative_return)
        max_drawdown = 0.0
        max_dd_idx = -1
    else:
        equity_peak = np.maximum.accumulate(equity_curve)
        drawdown = equity_curve / np.maximum(equity_peak, 1e-8) - 1
        max_drawdown = float(drawdown.min())
        max_dd_idx = int(np.argmin(drawdown))

    sharpe = 0.0
    if len(daily_return) and np.std(daily_return) != 0:
        sharpe = float(np.mean(daily_return) / np.std(daily_return) * np.sqrt(252))

    total_return = float(cumulative_return[-1]) if len(cumulative_return) else 0.0
    final_capital = float(equity_curve[-1]) if len(equity_curve) else float(initial_capital)
    win_rate = float(winning_trades / closed_trades) if closed_trades else 0.0
    exposure = float(np.mean(exposure_values)) if exposure_values else 0.0

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
        final_capital=final_capital,
        trade_count=int(trade_count),
        win_rate=win_rate,
        exposure=exposure,
        fees_paid=float(fees_paid),
        turnover=float(turnover),
    )
