#绘图服务：生成训练曲线、模型对比收盘价预测曲线和日线策略回测图片。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLOR_DIR_TRAIN = "#dc2626"
COLOR_DIR_TEST = "#9333ea"


def with_stock_title(title, stock_label=None):
    return f"{stock_label} - {title}" if stock_label else title


def setup_matplotlib():
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_direction_loss_curves(dir_loss_history, dir_test_loss_history, epochs, stock_label=None):
    epochs_axis = np.arange(1, epochs + 1)
    plt.figure(figsize=(12, 4.8))
    plt.plot(epochs_axis, dir_loss_history, label="训练集涨跌概率信号 Loss", color=COLOR_DIR_TRAIN)
    plt.plot(
        epochs_axis,
        dir_test_loss_history,
        label="测试集涨跌概率信号 Loss",
        color=COLOR_DIR_TEST,
        linestyle="--",
    )
    plt.title(with_stock_title("涨跌概率信号 Loss 曲线", stock_label))
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_price_prediction(dates, real_price, pred_price, stock_label=None, model_name=None, output_path=None):
    plt.figure(figsize=(14, 5))
    plt.plot(dates, real_price, label="真实收盘价")
    plt.plot(dates, pred_price, label="预测收盘价", linestyle="--")
    title = f"{model_name} 收盘价预测" if model_name else "收盘价预测"
    plt.title(with_stock_title(title, stock_label))
    plt.xlabel("时间")
    plt.ylabel("价格（元）")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_path:
        try:
            plt.savefig(output_path, bbox_inches="tight")
        except Exception:
            plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_daily_backtest(dates_bt, cumulative_return, buy_hold, max_dd_idx, stock_label=None):
    plt.figure(figsize=(14, 5))
    plt.plot(dates_bt, cumulative_return, label="日线策略累计收益")
    plt.plot(dates_bt, buy_hold, label="买入持有收益")

    if max_dd_idx > 0 and len(dates_bt) > max_dd_idx:
        plt.scatter(
            dates_bt.iloc[max_dd_idx],
            cumulative_return[max_dd_idx],
            marker="v",
            s=100,
            label="最大回撤点",
        )

    plt.title(with_stock_title("日线策略累计收益与持有收益", stock_label))
    plt.xlabel("时间")
    plt.ylabel("累计收益率")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
