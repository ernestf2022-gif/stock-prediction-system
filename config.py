import os

# 全局配置：token、路径、默认股票代码、训练参数
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 优先使用环境变量，未设置时沿用原脚本里的 token。
TS_TOKEN = os.getenv("TS_TOKEN", "a7cebd950e2feebc158d3cb5ae85f7402c285c4f79abb0d962a6bfd2")
STOCK_CODE = "000005.SZ"
START_DATE = "20190101"
END_DATE = "20260506"
APP_VERSION = ""

CSV_DIR = os.path.join(BASE_DIR, "kyxl_csv")
EXCEL_DIR = os.path.join(BASE_DIR, "kyxl_excel")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")

INDEX_CODE = "000001.SH"
TIME_STEP = 20
EPOCHS = 100
BATCH_SIZE = 32

BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.4
TRANSACTION_FEE = 0.001
FUTURE_DAYS = 10


def ensure_directories():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(EXCEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


ensure_directories()
