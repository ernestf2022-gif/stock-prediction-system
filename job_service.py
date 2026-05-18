# Experiment task scheduling for price prediction workflows.
import io
import json
import os
import shutil
import sys
import threading
import time
import traceback

from config import BASE_DIR, CSV_DIR, EXCEL_DIR, RESULT_DIR, ensure_directories
from data_service import format_stock_label, resolve_stock_name
from experiment_service import run_experiments, split_ablation_rows_for_display


jobs = {}
jobs_lock = threading.Lock()
JOBS_STATE_FILE = os.path.join(BASE_DIR, "jobs_state.json")


def save_jobs_unlocked():
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
        if not isinstance(meta, dict):
            continue
        if meta.get("job_type") != "experiment":
            meta["hidden"] = True
        if meta.get("status") == "running":
            meta["status"] = "error"
            meta["error"] = "服务曾重启，后台任务已中断，请重新运行实验。"

    with jobs_lock:
        jobs.update(disk_jobs)


def remove_named_file(dir_path, filename):
    if not filename:
        return

    base_dir = os.path.abspath(dir_path)
    file_path = os.path.abspath(os.path.join(base_dir, filename))
    try:
        if os.path.commonpath([base_dir, file_path]) != base_dir:
            return
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    except Exception as exc:
        print(f"删除文件失败: {file_path}, {exc}")


def get_download_names(meta):
    csv_names = set()
    excel_names = set()
    for key in ("csv_name", "model_csv", "ablation_csv", "sliding_window_csv"):
        filename = meta.get(key)
        if filename:
            csv_names.add(filename)
    excel_name = meta.get("excel_name")
    if excel_name:
        excel_names.add(excel_name)
    return csv_names, excel_names


def collect_download_names_unlocked():
    csv_names = set()
    excel_names = set()
    for meta in jobs.values():
        meta_csv_names, meta_excel_names = get_download_names(meta)
        csv_names.update(meta_csv_names)
        excel_names.update(meta_excel_names)
    return csv_names, excel_names


def remove_job_output_files(jobid, meta, remaining_csv_names=None, remaining_excel_names=None, remove_downloads=True):
    result_names = set(meta.get("images") or [])
    result_names.update(meta.get("result_files") or [])
    for key in ("model_csv", "ablation_csv", "sliding_window_csv"):
        filename = meta.get(key)
        if filename:
            result_names.add(filename)

    if jobid:
        try:
            for filename in os.listdir(RESULT_DIR):
                if filename.startswith(f"{jobid}_"):
                    result_names.add(filename)
        except FileNotFoundError:
            pass

    for filename in result_names:
        remove_named_file(RESULT_DIR, filename)

    if not remove_downloads:
        return

    csv_names, excel_names = get_download_names(meta)
    remaining_csv_names = remaining_csv_names or set()
    remaining_excel_names = remaining_excel_names or set()

    for filename in csv_names - remaining_csv_names:
        remove_named_file(CSV_DIR, filename)
    for filename in excel_names - remaining_excel_names:
        remove_named_file(EXCEL_DIR, filename)


def attach_display_fields(meta):
    meta = meta.copy()
    stock_code = meta.get("stock_code")
    stock_name = meta.get("stock_name") or resolve_stock_name(stock_code, allow_remote=False)
    meta["stock_name"] = stock_name
    meta["stock_label"] = meta.get("stock_label") or format_stock_label(stock_code, stock_name)
    meta["job_type"] = "experiment"
    meta["task_label"] = "价格预测实验"
    return meta


def list_recent_jobs(limit=10):
    with jobs_lock:
        visible_items = [(jobid, meta) for jobid, meta in jobs.items() if not meta.get("hidden")]
        items = visible_items[-limit:]

    recent = {}
    for jobid, meta in items:
        item = attach_display_fields(meta)
        item["created_at_human"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(meta.get("created_at", time.time()))
        )
        recent[jobid] = item
    return recent


def get_latest_finished_summary():
    with jobs_lock:
        finished = [
            (jid, meta)
            for jid, meta in jobs.items()
            if not meta.get("hidden") and meta.get("status") == "finished" and meta.get("job_type") == "experiment"
        ]
        if not finished:
            return None, None, None, None

        _, meta = sorted(finished, key=lambda item: item[1].get("finished_at", 0), reverse=True)[0]
        meta = attach_display_fields(meta)
        return (
            meta.get("model_rows") or None,
            int(meta.get("duration")) if meta.get("duration") else None,
            meta.get("model_prediction_images") or None,
            meta.get("stock_label"),
        )


def run_experiment_capture(stock_code, start_date, end_date, jobid):
    old_stdout = sys.stdout
    sio = io.StringIO()
    sys.stdout = sio
    start_time = time.time()

    try:
        result = run_experiments(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            output_prefix=jobid,
        )
        stdout_text = sio.getvalue()
        duration = time.time() - start_time

        with jobs_lock:
            meta = jobs.get(jobid)
            if meta is not None:
                meta.update(result)
                meta["stdout"] = stdout_text
                meta["finished_at"] = time.time()
                meta["duration"] = duration
                model_prediction_images = result.get("model_prediction_images", [])
                meta["result_files"] = [
                    result.get("model_csv"),
                    result.get("ablation_csv"),
                    result.get("sliding_window_csv"),
                    *model_prediction_images,
                ]
                meta["status"] = "finished"
                save_jobs_unlocked()
            else:
                remove_job_output_files(jobid, result)
    except Exception:
        error_text = traceback.format_exc()
        stdout_text = sio.getvalue()
        with jobs_lock:
            meta = jobs.get(jobid)
            if meta is not None:
                meta["status"] = "error"
                meta["error"] = error_text
                meta["stdout"] = stdout_text
                save_jobs_unlocked()
    finally:
        sys.stdout = old_stdout


def create_experiment_job(stock_code, start_date, end_date):
    timestamp = time.strftime("%Y%m%d%H%M%S")
    jobid = f"EXP_{stock_code}_{timestamp}"
    stock_name = resolve_stock_name(stock_code, allow_remote=False)

    with jobs_lock:
        jobs[jobid] = {
            "job_type": "experiment",
            "status": "running",
            "created_at": time.time(),
            "stdout": "",
            "error": None,
            "stock_code": stock_code,
            "stock_name": stock_name,
            "stock_label": format_stock_label(stock_code, stock_name),
            "start_date": start_date,
            "end_date": end_date,
            "duration": None,
            "model_rows": [],
            "ablation_rows": [],
            "market_index_model_rows": [],
            "sliding_window_rows": [],
            "model_csv": None,
            "ablation_csv": None,
            "sliding_window_csv": None,
            "model_prediction_images": [],
            "result_files": [],
        }
        save_jobs_unlocked()

    thread = threading.Thread(
        target=run_experiment_capture,
        args=(stock_code, start_date, end_date, jobid),
        daemon=True,
    )
    thread.start()
    return jobid


def get_experiment_page_context(jobid):
    with jobs_lock:
        meta = jobs.get(jobid)
        if not meta or meta.get("hidden"):
            return None
        meta_copy = meta.copy()

    if meta_copy.get("job_type") != "experiment":
        return None

    meta_copy = attach_display_fields(meta_copy)
    stdout_full = meta_copy.get("stdout", "") or ""
    snippet = stdout_full if len(stdout_full) <= 1200 else (stdout_full[:800] + "\n...\n" + stdout_full[-400:])
    ablation_rows = meta_copy.get("ablation_rows", [])
    market_index_model_rows = meta_copy.get("market_index_model_rows")
    if market_index_model_rows is None:
        ablation_rows, market_index_model_rows = split_ablation_rows_for_display(ablation_rows)

    return {
        "jobid": jobid,
        "status": meta_copy.get("status", "running"),
        "error": meta_copy.get("error"),
        "stdout_snippet": snippet,
        "stock_code": meta_copy.get("stock_code"),
        "stock_label": meta_copy.get("stock_label"),
        "start_date": meta_copy.get("start_date"),
        "end_date": meta_copy.get("end_date"),
        "epochs": meta_copy.get("epochs"),
        "duration": int(meta_copy.get("duration")) if meta_copy.get("duration") else None,
        "model_rows": meta_copy.get("model_rows", []),
        "ablation_rows": ablation_rows,
        "market_index_model_rows": market_index_model_rows,
        "sliding_window_rows": meta_copy.get("sliding_window_rows", []),
        "model_csv": meta_copy.get("model_csv"),
        "ablation_csv": meta_copy.get("ablation_csv"),
        "sliding_window_csv": meta_copy.get("sliding_window_csv"),
        "model_prediction_images": meta_copy.get("model_prediction_images", []),
    }


def get_job_status(jobid):
    with jobs_lock:
        meta = jobs.get(jobid)
    if not meta:
        load_jobs_from_disk()
        with jobs_lock:
            meta = jobs.get(jobid)
    return meta.get("status") if meta and not meta.get("hidden") else None


def delete_job(jobid):
    with jobs_lock:
        meta = jobs.pop(jobid, None)
        if meta is None:
            return False
        remaining_csv_names, remaining_excel_names = collect_download_names_unlocked()
        save_jobs_unlocked()

    remove_downloads = meta.get("status") == "finished"
    remove_job_output_files(jobid, meta, remaining_csv_names, remaining_excel_names, remove_downloads)
    return True


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
