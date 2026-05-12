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


def _trade_cost(gross, fee=0.001, min_commission=5.0, sell_tax=0.0, is_sell=False):
    commission = max(float(gross) * fee, min_commission) if gross > 0 else 0.0
    tax = float(gross) * sell_tax if is_sell else 0.0
    return commission + tax


def make_signal(
    probabilities,
    pred_price=None,
    real_price=None,
    buy_threshold=0.6,
    sell_threshold=0.4,
    min_expected_return=0.002,
    reference_probabilities=None,
    buy_quantile=None,
    sell_quantile=None,
    min_threshold_gap=0.02,
    adaptive_window=20,
):
    probabilities = np.asarray(probabilities, dtype=float)
    signal = np.zeros(len(probabilities), dtype=int)
    reference_probabilities = np.asarray(
        [] if reference_probabilities is None else reference_probabilities,
        dtype=float,
    )
    reference_probabilities = reference_probabilities[np.isfinite(reference_probabilities)]
    if adaptive_window and len(reference_probabilities) > adaptive_window:
        reference_probabilities = reference_probabilities[-adaptive_window:]

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
        expected_return = None

    for idx, probability in enumerate(probabilities):
        current_buy_threshold = buy_threshold
        current_sell_threshold = sell_threshold
        if buy_quantile is not None and sell_quantile is not None and len(reference_probabilities):
            history = np.concatenate([reference_probabilities, probabilities[:idx]])
            if adaptive_window and len(history) > adaptive_window:
                history = history[-adaptive_window:]
            history = history[np.isfinite(history)]
            if len(history):
                current_buy_threshold = min(buy_threshold, float(np.quantile(history, buy_quantile)))
                current_sell_threshold = max(sell_threshold, float(np.quantile(history, sell_quantile)))
                if current_sell_threshold >= current_buy_threshold - min_threshold_gap:
                    current_sell_threshold = current_buy_threshold - min_threshold_gap

        expected_ok = expected_return is None or expected_return[idx] >= min_expected_return
        risk_off = expected_return is not None and expected_return[idx] <= -min_expected_return
        if probability >= current_buy_threshold and expected_ok:
            signal[idx] = 1
        elif probability <= current_sell_threshold or risk_off:
            signal[idx] = -1
    return np.array(signal)


def run_daily_signal_backtest(
    real_price,
    pred_price,
    probabilities,
    pandas_df,
    market_data=None,
    buy_threshold=0.55,
    sell_threshold=0.45,
    fee=0.001,
    sell_tax=0.0005,
    min_commission=5.0,
    initial_capital=100000.0,
    max_position_pct=0.95,
    slippage=0.0005,
    lot_size=100,
    min_expected_return=0.002,
    stop_loss=0.035,
    take_profit=0.07,
    reference_probabilities=None,
    buy_quantile=None,
    sell_quantile=None,
    min_threshold_gap=0.02,
    adaptive_window=20,
):
    """日线回测：收盘后产生信号，下一交易日开盘成交，按 OHLC 近似风控。"""
    if market_data is not None:
        market_data = market_data.reset_index(drop=True)
        real_price = market_data["close"].to_numpy(dtype=float)

    signal = make_signal(
        probabilities,
        pred_price=pred_price,
        real_price=real_price,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        min_expected_return=min_expected_return,
        reference_probabilities=reference_probabilities,
        buy_quantile=buy_quantile,
        sell_quantile=sell_quantile,
        min_threshold_gap=min_threshold_gap,
        adaptive_window=adaptive_window,
    )
    if pred_price is None:
        pred_price = np.full(len(real_price), np.nan)
    min_len = min(len(signal), len(real_price), len(pred_price))
    signal = signal[:min_len]
    real_price = np.asarray(real_price[:min_len], dtype=float)
    pred_price = pred_price[:min_len]

    if market_data is not None:
        market_data = market_data.iloc[:min_len].copy()
        open_price = market_data["open"].to_numpy(dtype=float)
        high_price = market_data["high"].to_numpy(dtype=float)
        low_price = market_data["low"].to_numpy(dtype=float)
        close_price = market_data["close"].to_numpy(dtype=float)
        dates = market_data["date"] if "date" in market_data.columns else pd.Series(np.arange(min_len))
    else:
        close_price = real_price
        open_price = real_price
        high_price = real_price
        low_price = real_price
        if "交易日期" in pandas_df.columns:
            dates = pandas_df["交易日期"].iloc[-min_len:].reset_index(drop=True)
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
    entry_idx = -1
    equity_curve = []
    signal_bt = []
    exposure_values = []

    for idx in range(min_len):
        day_open = open_price[idx]
        day_high = max(high_price[idx], day_open)
        day_low = min(low_price[idx], day_open)
        day_close = close_price[idx]
        equity_before = cash + shares * day_open
        target_pct = max_position_pct if signal[idx] == 1 else 0.0
        can_sell_today = shares > 0 and entry_idx >= 0 and entry_idx < idx

        if target_pct <= 0 and can_sell_today:
            sell_price = day_open * (1 - slippage)
            gross = shares * sell_price
            cost = _trade_cost(gross, fee=fee, min_commission=min_commission, sell_tax=sell_tax, is_sell=True)
            cash += gross - cost
            fees_paid += cost
            turnover += gross
            trade_count += 1
            closed_trades += 1
            if sell_price > entry_price:
                winning_trades += 1
            shares = 0
            entry_price = 0.0
            entry_idx = -1
        elif target_pct > 0:
            target_value = equity_before * target_pct
            buy_price = day_open * (1 + slippage)
            target_shares = int(target_value / max(buy_price, 1e-8))
            if lot_size > 1:
                target_shares = (target_shares // lot_size) * lot_size
            shares_to_buy = max(target_shares - shares, 0)
            gross = shares_to_buy * buy_price
            cost = _trade_cost(gross, fee=fee, min_commission=min_commission, is_sell=False)
            if shares_to_buy > 0 and gross + cost <= cash:
                cash -= gross + cost
                fees_paid += cost
                turnover += gross
                trade_count += 1
                entry_price = buy_price if shares == 0 else entry_price
                entry_idx = idx if shares == 0 else entry_idx
                shares += shares_to_buy

        can_sell_today = shares > 0 and entry_idx >= 0 and entry_idx < idx
        if can_sell_today and entry_price > 0:
            stop_price = entry_price * (1 - stop_loss)
            take_price = entry_price * (1 + take_profit)
            sell_price = None
            if day_low <= stop_price:
                sell_price = min(stop_price, day_open) * (1 - slippage)
            elif day_high >= take_price:
                sell_price = max(take_price, day_open) * (1 - slippage)

            if sell_price is not None:
                gross = shares * sell_price
                cost = _trade_cost(gross, fee=fee, min_commission=min_commission, sell_tax=sell_tax, is_sell=True)
                cash += gross - cost
                fees_paid += cost
                turnover += gross
                trade_count += 1
                closed_trades += 1
                if sell_price > entry_price:
                    winning_trades += 1
                shares = 0
                entry_price = 0.0
                entry_idx = -1

        equity = cash + shares * day_close
        equity_curve.append(equity)
        signal_bt.append(1 if shares > 0 else 0)
        exposure_values.append((shares * day_close) / max(equity, 1e-8))

    equity_curve = np.asarray(equity_curve, dtype=float)
    signal_bt = np.asarray(signal_bt, dtype=int)
    dates_bt = dates.iloc[: len(equity_curve)]
    buy_hold = close_price[: len(equity_curve)] / close_price[0] - 1
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
