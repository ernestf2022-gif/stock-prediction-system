import io
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback

from config import BASE_DIR, CSV_DIR, EXCEL_DIR, RESULT_DIR, ensure_directories
from model_service import run_pipeline
from plot_service import plt


jobs = {}
jobs_lock = threading.Lock()
JOBS_STATE_FILE = os.path.join(BASE_DIR, "jobs_state.json")


def save_jobs_unlocked():
    """调用方需先持有 jobs_lock。"""
    tmp_file = JOBS_STATE_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as file:
        json.dump(jobs, file, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp_file, JOBS_STATE_FILE)


def load_jobs_from_disk():
    if not os.path.exists(JOBS_STATE_FILE):
        return

    try:
        with open(JOBS_STATE_FILE, "r", encoding="utf-8") as file:
            disk_jobs = json.load(file)
    except Exception as exc:
        print(f"读取任务状态文件失败：{exc}")
        return

    if not isinstance(disk_jobs, dict):
        return

    for meta in disk_jobs.values():
        if isinstance(meta, dict) and meta.get("status") == "running":
            meta["status"] = "error"
            meta["error"] = "服务曾重启，后台任务已中断，请重新开始测试。"

    with jobs_lock:
        jobs.update(disk_jobs)


def parse_metrics_from_stdout(stdout_text):
    metrics = {}
    try:
        patterns = {
            "total_return": r"总收益率: *([-\d\.]+)",
            "max_drawdown": r"最大回撤: *([-\d\.]+)",
            "sharpe": r"年化夏普: *([-\d\.]+)",
            "accuracy": r"分类准确率: *([-\d\.]+)",
            "precision": r"分类精确率: *([-\d\.]+)",
            "recall": r"分类召回率: *([-\d\.]+)",
            "f1": r"分类F1: *([-\d\.]+)",
        }
        for key, pattern in patterns.items():
            matched = re.search(pattern, stdout_text)
            if matched:
                metrics[key] = float(matched.group(1))

        matched_cm = re.search(r"混淆矩阵: *TN=(\d+), *FP=(\d+), *FN=(\d+), *TP=(\d+)", stdout_text)
        if matched_cm:
            metrics["confusion_matrix"] = {
                "tn": int(matched_cm.group(1)),
                "fp": int(matched_cm.group(2)),
                "fn": int(matched_cm.group(3)),
                "tp": int(matched_cm.group(4)),
            }
    except Exception:
        pass
    return metrics


def run_pipeline_capture(stock_code, start_date, end_date, jobid):
    saved = []
    orig_show = plt.show

    def save_show():
        idx = len(saved)
        fname = f"{jobid}_{idx}.png"
        path = os.path.join(RESULT_DIR, fname)
        try:
            plt.savefig(path, bbox_inches="tight")
        except Exception:
            plt.savefig(path)
        plt.close()
        saved.append(fname)

    plt.show = save_show

    old_stdout = sys.stdout
    sio = io.StringIO()
    sys.stdout = sio

    start_time = time.time()
    try:
        trade_cycles = run_pipeline(stock_code=stock_code, start_date=start_date, end_date=end_date)
        stdout_text = sio.getvalue()
        duration = time.time() - start_time
        metrics = parse_metrics_from_stdout(stdout_text)

        with jobs_lock:
            jobs[jobid]["stdout"] = stdout_text
            jobs[jobid]["images"] = saved.copy()
            jobs[jobid]["finished_at"] = time.time()
            jobs[jobid]["duration"] = duration
            jobs[jobid]["trade_cycles"] = trade_cycles
            jobs[jobid]["metrics"] = metrics
            jobs[jobid]["status"] = "finished"
            save_jobs_unlocked()
    except Exception:
        with jobs_lock:
            jobs[jobid]["status"] = "error"
            jobs[jobid]["error"] = traceback.format_exc()
            jobs[jobid]["stdout"] = sio.getvalue()
            save_jobs_unlocked()
    finally:
        plt.show = orig_show
        sys.stdout = old_stdout


def expected_output_names(stock_code):
    safe_stock_code = stock_code.replace(".", "_")
    return f"{safe_stock_code}_LSTM.csv", f"{safe_stock_code}_LSTM.xlsx"


def attach_download_names(meta):
    meta = meta.copy()
    stock_code = meta.get("stock_code")
    if not stock_code:
        return meta

    real_csv_name, real_excel_name = expected_output_names(stock_code)
    csv_name = meta.get("csv_name")
    excel_name = meta.get("excel_name")

    if not csv_name or not os.path.exists(os.path.join(CSV_DIR, csv_name)):
        meta["csv_name"] = real_csv_name if os.path.exists(os.path.join(CSV_DIR, real_csv_name)) else None
    if not excel_name or not os.path.exists(os.path.join(EXCEL_DIR, excel_name)):
        meta["excel_name"] = real_excel_name if os.path.exists(os.path.join(EXCEL_DIR, real_excel_name)) else None

    return meta


def list_recent_jobs(limit=10):
    with jobs_lock:
        items = list(jobs.items())[-limit:]

    recent = {}
    for jobid, meta in items:
        item = attach_download_names(meta)
        item["created_at_human"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(meta.get("created_at", time.time()))
        )
        recent[jobid] = item
    return recent


def get_latest_finished_summary():
    latest_metrics = None
    latest_duration = None
    latest_images = None

    with jobs_lock:
        finished = [(jid, meta) for jid, meta in jobs.items() if meta.get("status") == "finished"]
        if finished:
            finished_sorted = sorted(finished, key=lambda item: item[1].get("finished_at", 0), reverse=True)
            _, meta = finished_sorted[0]
            latest_metrics = meta.get("metrics", {}) or None
            latest_duration = int(meta.get("duration")) if meta.get("duration") else None

        finished_with_imgs = [
            (jid, meta) for jid, meta in jobs.items() if meta.get("status") == "finished" and meta.get("images")
        ]
        if finished_with_imgs:
            finished_imgs_sorted = sorted(
                finished_with_imgs, key=lambda item: item[1].get("finished_at", 0), reverse=True
            )
            _, meta_with_img = finished_imgs_sorted[0]
            imgs = meta_with_img.get("images", []) or []
            latest_images = imgs if imgs else None

    return latest_metrics, latest_duration, latest_images


def create_job(stock_code, start_date, end_date):
    timestamp = time.strftime("%Y%m%d%H%M%S")
    jobid = f"{stock_code}_{timestamp}"
    csv_name, excel_name = expected_output_names(stock_code)

    with jobs_lock:
        jobs[jobid] = {
            "status": "running",
            "created_at": time.time(),
            "images": [],
            "stdout": "",
            "error": None,
            "metrics": {},
            "stock_code": stock_code,
            "csv_name": csv_name,
            "excel_name": excel_name,
            "duration": None,
        }
        save_jobs_unlocked()

    thread = threading.Thread(
        target=run_pipeline_capture,
        args=(stock_code, start_date, end_date, jobid),
        daemon=True,
    )
    thread.start()
    return jobid


def format_metric(value):
    return f"{value:.4f}" if value is not None else "N/A"


def get_job_page_context(jobid):
    with jobs_lock:
        meta = jobs.get(jobid)
        if not meta:
            return None
        meta_copy = meta.copy()

    meta_copy = attach_download_names(meta_copy)
    stdout_full = meta_copy.get("stdout", "") or ""
    snippet = stdout_full if len(stdout_full) <= 1200 else (stdout_full[:800] + "\n...\n" + stdout_full[-400:])

    raw_metrics = meta_copy.get("metrics", {}) or {}
    metrics = {}
    if raw_metrics:
        metrics = {
            "total_return": format_metric(raw_metrics.get("total_return")),
            "max_drawdown": format_metric(raw_metrics.get("max_drawdown")),
            "sharpe": format_metric(raw_metrics.get("sharpe")),
            "accuracy": format_metric(raw_metrics.get("accuracy")),
            "precision": format_metric(raw_metrics.get("precision")),
            "recall": format_metric(raw_metrics.get("recall")),
            "f1": format_metric(raw_metrics.get("f1")),
            "confusion_matrix": raw_metrics.get("confusion_matrix") or {},
        }

    return {
        "jobid": jobid,
        "status": meta_copy.get("status", "running"),
        "images": meta_copy.get("images", []),
        "metrics": metrics,
        "stdout_snippet": snippet,
        "error": meta_copy.get("error"),
        "csv_name": meta_copy.get("csv_name"),
        "excel_name": meta_copy.get("excel_name"),
        "duration": int(meta_copy.get("duration")) if meta_copy.get("duration") else None,
        "trade_cycles": meta_copy.get("trade_cycles", []),
    }


def get_job_status(jobid):
    with jobs_lock:
        meta = jobs.get(jobid)
    if not meta:
        load_jobs_from_disk()
        with jobs_lock:
            meta = jobs.get(jobid)
    return meta.get("status") if meta else None


def remove_all_files(dir_path):
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as exc:
                print(f"删除文件失败: {file_path}, {exc}")


def clear_all_jobs_and_files():
    ensure_directories()
    with jobs_lock:
        jobs.clear()
        save_jobs_unlocked()

    remove_all_files(RESULT_DIR)
    remove_all_files(CSV_DIR)
    remove_all_files(EXCEL_DIR)


load_jobs_from_disk()
