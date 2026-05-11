#数据服务：爬取 / 读取股票数据、清洗、格式化。
from contextlib import contextmanager
import json
import os
import threading
import time

import numpy as np
import pandas as pd
import tushare as ts

from config import CSV_DIR, EXCEL_DIR, INDEX_CODE, TS_TOKEN, ensure_directories


DAILY_REQUIRED_COLS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "change", "vol", "amount"
]
INDEX_REQUIRED_COLS = ["trade_date", "close"]
STOCK_BASIC_REQUIRED_COLS = ["ts_code", "name"]
OUTPUT_COLUMNS = [
    "股票代码", "交易日期", "开盘价", "最高价", "最低价", "收盘价",
    "昨收价", "涨跌额", "成交额(千元)", "MA5", "MA10", "MACD", "RSI",
    "VOLATILITY", "大盘指数", "RET", "LOG_RET", "INDEX_RET"
]
#系统对常用股票进行了本地预缓存，提高查询效率；对于未缓存股票，再通过 Tushare 接口动态获取，并写入本地缓存文件，减少重复请求。
STOCK_NAME_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_names_cache.json")
COMMON_STOCK_NAMES = {
    "000001.SZ": "平安银行",
    "000002.SZ": "万科A",
    "000568.SZ": "泸州老窖",
    "000858.SZ": "五粮液",
    "001979.SZ": "招商蛇口",
    "002230.SZ": "科大讯飞",
    "300750.SZ": "宁德时代",
    "600000.SH": "浦发银行",
    "600030.SH": "中信证券",
    "600036.SH": "招商银行",
    "600276.SH": "恒瑞医药",
    "600519.SH": "贵州茅台",
    "601318.SH": "中国平安",
    "601601.SH": "中国太保",
    "601628.SH": "中国人寿",
    "688981.SH": "中芯国际",
}
_stock_name_cache = None
_proxy_env_lock = threading.RLock()
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "FTP_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "ftp_proxy",
    "NO_PROXY",
    "no_proxy",
)


@contextmanager
def tushare_no_proxy_env():
    with _proxy_env_lock:
        old_env = {key: os.environ.get(key) for key in PROXY_ENV_KEYS}
        try:
            for key in PROXY_ENV_KEYS:
                os.environ.pop(key, None)
            os.environ["NO_PROXY"] = "*"
            os.environ["no_proxy"] = "*"
            yield
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def normalize_stock_code(stock_code):
    return str(stock_code or "").strip().upper()


def load_stock_name_cache():
    global _stock_name_cache
    if _stock_name_cache is not None:
        return _stock_name_cache

    cache = {}
    if os.path.exists(STOCK_NAME_CACHE_FILE):
        try:
            with open(STOCK_NAME_CACHE_FILE, "r", encoding="utf-8") as file:
                disk_cache = json.load(file)
            if isinstance(disk_cache, dict):
                cache.update(
                    {
                        normalize_stock_code(code): str(name).strip()
                        for code, name in disk_cache.items()
                        if normalize_stock_code(code) and str(name).strip()
                    }
                )
        except Exception as exc:
            print(f"读取股票名称缓存失败：{exc}")

    cache.update(COMMON_STOCK_NAMES)
    _stock_name_cache = cache
    return _stock_name_cache


def save_stock_name_cache(cache):
    tmp_file = STOCK_NAME_CACHE_FILE + ".tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as file:
            json.dump(cache, file, ensure_ascii=False, indent=2)
        os.replace(tmp_file, STOCK_NAME_CACHE_FILE)
    except Exception as exc:
        print(f"保存股票名称缓存失败：{exc}")


def fetch_stock_name_map(token=TS_TOKEN):
    stock_names = {}

    with tushare_no_proxy_env():
        ts.set_token(token)
        pro = ts.pro_api()

        for list_status in ("L", "P", "D"):
            try:
                stock_df = pro.stock_basic(exchange="", list_status=list_status, fields="ts_code,name")
            except Exception as exc:
                print(f"股票基础信息获取失败：list_status={list_status}, {exc}")
                continue

            if stock_df is None or stock_df.empty:
                continue
            if set(STOCK_BASIC_REQUIRED_COLS) - set(stock_df.columns):
                continue

            for _, row in stock_df[STOCK_BASIC_REQUIRED_COLS].iterrows():
                code = normalize_stock_code(row.get("ts_code"))
                name = str(row.get("name", "")).strip()
                if code and name:
                    stock_names[code] = name

    return stock_names


def resolve_stock_name(stock_code, allow_remote=True):
    code = normalize_stock_code(stock_code)
    if not code:
        return None

    cache = load_stock_name_cache()
    if code in cache:
        return cache[code]

    if not allow_remote:
        return None

    try:
        fetched_names = fetch_stock_name_map()
        if fetched_names:
            cache.update(fetched_names)
            save_stock_name_cache(cache)
    except Exception as exc:
        print(f"获取股票中文名称失败：{code}, {exc}")

    return cache.get(code)


def format_stock_label(stock_code, stock_name=None):
    code = normalize_stock_code(stock_code)
    name = str(stock_name or "").strip()
    if code and name:
        return f"{name}（{code}）"
    return code or name


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
            with tushare_no_proxy_env():
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
    with tushare_no_proxy_env():
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
