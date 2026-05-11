# =====================================================
# 基于机器学习的股票交易数据分析与预测系统
# =====================================================
# 系统功能：
# 1. 股票数据获取
# 2. 数据清洗与特征工程
# 3. 模型训练
# 4. 股票涨跌预测
# 5. 量化回测分析
# 6. Web可视化展示
# 系统目前已经能够完成：
# 从股票数据获取到预测结果展示的完整流程
# =====================================================

#系统入口，Flask Web 服务，负责路由、页面跳转、接口调用。
import os

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for

from config import APP_VERSION, CSV_DIR, END_DATE, EXCEL_DIR, RESULT_DIR, START_DATE, STOCK_CODE
from experiment_service import run_experiments
from job_service import (
    clear_all_jobs_and_files,
    create_job,
    delete_job,
    get_job_page_context,
    get_job_status,
    get_latest_finished_summary,
    list_recent_jobs,
)


app = Flask(__name__)


def read_stock_params():
    if request.form:
        stock_code = request.form.get("stock_code")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
    else:
        payload = request.get_json() or {}
        stock_code = payload.get("stock_code")
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")

    if not stock_code or not start_date or not end_date:
        return None, None, None

    return stock_code.strip().upper(), start_date.strip(), end_date.strip()


@app.route("/", methods=["GET"])
def index():
    (
        latest_metrics,
        latest_duration,
        latest_images,
        latest_stock_label,
        latest_images_stock_label,
    ) = get_latest_finished_summary()
    return render_template(
        "index.html",
        default_code=STOCK_CODE,
        default_start=START_DATE,
        default_end=END_DATE,
        jobs=list_recent_jobs(limit=50),
        latest_metrics=latest_metrics,
        latest_duration=latest_duration,
        latest_images=latest_images,
        latest_stock_label=latest_stock_label,
        latest_images_stock_label=latest_images_stock_label,
        app_version=APP_VERSION,
    )


@app.route("/run", methods=["POST"])
def run_route():
    stock_code, start_date, end_date = read_stock_params()

    if not stock_code or not start_date or not end_date:
        return "参数不完整", 400

    jobid = create_job(stock_code, start_date, end_date)
    return redirect(url_for("job_page", jobid=jobid))


@app.route("/experiment", methods=["POST"])
def experiment_route():
    stock_code, start_date, end_date = read_stock_params()
    if not stock_code or not start_date or not end_date:
        return "参数不完整", 400

    try:
        context = run_experiments(stock_code, start_date, end_date)
    except Exception as exc:
        return render_template(
            "experiment.html",
            error=str(exc),
            stock_code=stock_code,
            stock_label=stock_code,
            start_date=start_date,
            end_date=end_date,
            model_rows=[],
            ablation_rows=[],
            app_version=APP_VERSION,
        ), 500

    return render_template("experiment.html", **context, app_version=APP_VERSION)


@app.route("/job/<jobid>", methods=["GET"])
def job_page(jobid):
    context = get_job_page_context(jobid)
    if context is None:
        abort(404)
    return render_template("job.html", **context, app_version=APP_VERSION)


@app.route("/status/<jobid>", methods=["GET"])
def status_route(jobid):
    status = get_job_status(jobid)
    if status is None:
        return jsonify({"status": "not_found"}), 404
    return jsonify({"status": status})


@app.route("/result/<path:fname>")
def result_file(fname):
    safe_path = os.path.join(RESULT_DIR, fname)
    if not os.path.exists(safe_path):
        abort(404)
    return send_from_directory(RESULT_DIR, fname)


@app.route("/download/csv/<path:fname>")
def download_csv(fname):
    path = os.path.join(CSV_DIR, fname)
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(CSV_DIR, fname, as_attachment=True)


@app.route("/download/excel/<path:fname>")
def download_excel(fname):
    path = os.path.join(EXCEL_DIR, fname)
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(EXCEL_DIR, fname, as_attachment=True)


@app.route("/clear_all", methods=["POST"])
def clear_all():
    clear_all_jobs_and_files()
    return jsonify({"message": "已清空所有任务和文件"}), 200


@app.route("/delete_job/<jobid>", methods=["POST"])
def delete_job_route(jobid):
    if not delete_job(jobid):
        return jsonify({"message": "任务不存在或已删除"}), 404
    return jsonify({"message": "已删除任务"}), 200


if __name__ == "__main__":
    print("启动 Flask 服务： http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000, use_reloader=False)
