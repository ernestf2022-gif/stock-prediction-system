import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


COLOR_REG_TRAIN = "#2563eb"
COLOR_REG_TEST = "#16a34a"
COLOR_DIR_TRAIN = "#dc2626"
COLOR_DIR_TEST = "#9333ea"
COLOR_MUTED = "#6b7280"


def with_stock_title(title, stock_label=None):
    return f"{stock_label} - {title}" if stock_label else title


def setup_matplotlib():
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_loss_curves(reg_loss_history, reg_test_loss_history, dir_loss_history, dir_test_loss_history, epochs, stock_label=None):
    epochs_axis = np.arange(1, epochs + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs_axis, reg_loss_history, label="训练集收盘价回归 Loss", color=COLOR_REG_TRAIN)
    plt.plot(
        epochs_axis,
        reg_test_loss_history,
        label="测试集收盘价回归 Loss",
        color=COLOR_REG_TEST,
        linestyle="--",
    )
    plt.title(with_stock_title("训练集 / 测试集 Loss 曲线", stock_label))
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(2, 1, 2)
    plt.plot(epochs_axis, dir_loss_history, label="训练集涨跌方向分类 Loss", color=COLOR_DIR_TRAIN)
    plt.plot(
        epochs_axis,
        dir_test_loss_history,
        label="测试集涨跌方向分类 Loss",
        color=COLOR_DIR_TEST,
        linestyle="--",
    )
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_roc_curve(dir_reals, probabilities, stock_label=None):
    plt.figure(figsize=(8, 6))
    if len(np.unique(dir_reals)) >= 2:
        fpr, tpr, _ = roc_curve(dir_reals, probabilities)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=COLOR_REG_TRAIN, lw=2, label=f"ROC曲线 (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color=COLOR_MUTED, lw=1.5, linestyle="--", label="随机猜测")
        plt.xlabel("假阳性率 FPR")
        plt.ylabel("真正率 TPR")
        plt.legend(loc="lower right")
    else:
        only_class = int(dir_reals[0]) if len(dir_reals) else "N/A"
        plt.text(
            0.5,
            0.5,
            f"测试集仅包含单一类别：{only_class}\n无法计算 ROC 曲线",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("假阳性率 FPR")
        plt.ylabel("真正率 TPR")

    plt.title(with_stock_title("涨跌方向分类 ROC 曲线", stock_label))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_prediction_and_backtest(
    dates,
    real_price,
    pred_price,
    signal_bt,
    dates_bt,
    cumulative_return,
    buy_hold,
    max_dd_idx,
    stock_label=None,
):
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(dates, real_price, label="真实收盘价")
    plt.plot(dates, pred_price, label="预测收盘价", linestyle="--")

    signal_plot = signal_bt.copy()
    position = 0
    buy_idx = []
    sell_idx = []

    for i in range(len(signal_plot)):
        if signal_plot[i] == 1 and position == 0:
            buy_idx.append(i)
            position = 1
        elif signal_plot[i] != 1 and position == 1:
            sell_idx.append(i)
            position = 0

    plt.scatter(dates.iloc[buy_idx], real_price[buy_idx], marker="^", label="买入", s=50)
    plt.scatter(dates.iloc[sell_idx], real_price[sell_idx], marker="v", label="卖出", s=50)

    plt.title(with_stock_title("收盘价预测与交易信号", stock_label))
    plt.ylabel("价格（元）")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(2, 1, 2)
    plt.plot(dates_bt, cumulative_return, label="策略累计收益")
    plt.plot(dates_bt, buy_hold, label="买入持有收益")

    if max_dd_idx > 0:
        plt.scatter(
            dates_bt.iloc[max_dd_idx],
            cumulative_return[max_dd_idx],
            marker="v",
            s=100,
            label="最大回撤点",
        )

    plt.title(with_stock_title("策略累计收益与最大回撤", stock_label))
    plt.xlabel("时间")
    plt.ylabel("累计收益率")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
