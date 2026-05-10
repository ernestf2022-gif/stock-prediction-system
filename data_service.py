import os
import time

import numpy as np
import pandas as pd
import tushare as ts

from config import CSV_DIR, EXCEL_DIR, INDEX_CODE, TS_TOKEN, ensure_directories


DAILY_REQUIRED_COLS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "change", "vol", "amount"
]
INDEX_REQUIRED_COLS = ["trade_date", "close"]
OUTPUT_COLUMNS = [
    "股票代码", "交易日期", "开盘价", "最高价", "最低价", "收盘价",
    "昨收价", "涨跌额", "成交额(千元)", "MA5", "MA10", "MACD", "RSI",
    "VOLATILITY", "大盘指数", "RET", "LOG_RET", "INDEX_RET"
]


def normalize_trade_date(date_text):
    date = pd.to_datetime(str(date_text), format="%Y%m%d", errors="coerce")
    if pd.isna(date):
        raise ValueError(f"日期格式错误：{date_text}，请使用 YYYYMMDD，例如 20260506")

    today = pd.Timestamp.today().normalize()
    if date > today:
        date = today
    return date.strftime("%Y%m%d")


def fetch_tushare_data(fetch_func, dataset_name, required_cols, max_retries=3, **kwargs):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fetch_func(**kwargs)
            if result is not None and not result.empty:
                missing_cols = set(required_cols) - set(result.columns)
                if not missing_cols:
                    return result.copy()
                last_error = f"缺少字段: {sorted(missing_cols)}，实际字段: {list(result.columns)}"
            else:
                last_error = "返回空数据"
        except Exception as exc:
            last_error = str(exc)

        if attempt < max_retries:
            print(f"{dataset_name} 第{attempt}次获取失败：{last_error}，准备重试...")
            time.sleep(1)

    raise RuntimeError(f"❌ {dataset_name} 获取失败：{last_error}")


def fetch_tushare_data_with_enddate_fallback(
    fetch_func,
    dataset_name,
    required_cols,
    fallback_days=10,
    **kwargs,
):
    original_end_date = kwargs.get("end_date")
    last_error = None

    for offset in range(fallback_days + 1):
        try_end_date = (
            pd.to_datetime(original_end_date, format="%Y%m%d") - pd.offsets.BDay(offset)
        ).strftime("%Y%m%d")
        try:
            data = fetch_tushare_data(
                fetch_func,
                dataset_name,
                required_cols,
                **{**kwargs, "end_date": try_end_date},
            )
            if try_end_date != original_end_date:
                print(f"{dataset_name} 使用回退截止日期：{try_end_date}")
            return data
        except Exception as exc:
            last_error = str(exc)

    raise RuntimeError(f"❌ {dataset_name} 获取失败，已尝试回退 {fallback_days} 个交易日：{last_error}")


def fetch_market_data(stock_code, start_date, end_date, token=TS_TOKEN):
    ts.set_token(token)
    pro = ts.pro_api()

    start_date = normalize_trade_date(start_date)
    end_date = normalize_trade_date(end_date)

    stock_df = fetch_tushare_data_with_enddate_fallback(
        pro.daily,
        "股票日线数据",
        DAILY_REQUIRED_COLS,
        ts_code=stock_code,
        start_date=start_date,
        end_date=end_date,
    )
    index_df = fetch_tushare_data_with_enddate_fallback(
        pro.index_daily,
        "大盘指数数据",
        INDEX_REQUIRED_COLS,
        ts_code=INDEX_CODE,
        start_date=start_date,
        end_date=end_date,
    )
    return stock_df, index_df


def build_lstm_dataframe(stock_df, index_df):
    df = stock_df.rename(
        columns={
            "ts_code": "股票代码",
            "trade_date": "交易日期",
            "open": "开盘价",
            "high": "最高价",
            "low": "最低价",
            "close": "收盘价",
            "change": "涨跌额",
            "vol": "成交量",
            "amount": "成交额",
        }
    ).copy()
    df["交易日期"] = pd.to_datetime(df["交易日期"])

    index_df = index_df.rename(columns={"trade_date": "交易日期", "close": "大盘指数"}).copy()
    index_df["交易日期"] = pd.to_datetime(index_df["交易日期"])

    df = df.merge(index_df[["交易日期", "大盘指数"]], on="交易日期", how="left")
    df["交易日期"] = pd.to_datetime(df["交易日期"])
    df = df.sort_values("交易日期")
    df["大盘指数"] = df["大盘指数"].ffill()

    df["昨收价"] = df["收盘价"].shift(1)
    df["昨收价"] = df["昨收价"].fillna(df["收盘价"])
    df["涨跌额"] = df["涨跌额"].fillna(df["收盘价"] - df["昨收价"])
    df["成交额(千元)"] = df["成交额"] / 1000

    df["MA5"] = df["收盘价"].rolling(5).mean()
    df["MA10"] = df["收盘价"].rolling(10).mean()
    df["EMA12"] = df["收盘价"].ewm(span=12).mean()
    df["EMA26"] = df["收盘价"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]

    delta = df["收盘价"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VOLATILITY"] = df["收盘价"].pct_change().rolling(10).std()
    df["RET"] = df["收盘价"].pct_change()
    df["LOG_RET"] = np.log(df["收盘价"] / df["收盘价"].shift(1))
    df["INDEX_RET"] = df["大盘指数"].pct_change()

    return df.dropna()[OUTPUT_COLUMNS].copy()


def save_lstm_data(lstm_df, stock_code):
    ensure_directories()
    safe_stock_code = stock_code.replace(".", "_")
    csv_file = os.path.join(CSV_DIR, f"{safe_stock_code}_LSTM.csv")
    excel_file = os.path.join(EXCEL_DIR, f"{safe_stock_code}_LSTM.xlsx")
    lstm_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    lstm_df.to_excel(excel_file, index=False)
    return csv_file, excel_file


def prepare_stock_dataset(stock_code, start_date, end_date):
    stock_df, index_df = fetch_market_data(stock_code, start_date, end_date)
    lstm_df = build_lstm_dataframe(stock_df, index_df)
    csv_file, excel_file = save_lstm_data(lstm_df, stock_code)
    return lstm_df, csv_file, excel_file
