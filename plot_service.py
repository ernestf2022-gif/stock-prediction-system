# Plot helpers for price prediction experiment results.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def with_stock_title(title, stock_label=None):
    return f"{stock_label} - {title}" if stock_label else title


def setup_matplotlib():
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False


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
