# Stock price prediction experiment web service.
import os

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for

from config import APP_VERSION, CSV_DIR, END_DATE, EXCEL_DIR, RESULT_DIR, START_DATE, STOCK_CODE
from job_service import (
    clear_all_jobs_and_files,
    create_experiment_job,
    delete_job,
    get_experiment_page_context,
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
    latest_model_rows, latest_duration, latest_images, latest_stock_label = get_latest_finished_summary()
    return render_template(
        "index.html",
        default_code=STOCK_CODE,
        default_start=START_DATE,
        default_end=END_DATE,
        jobs=list_recent_jobs(limit=50),
        latest_model_rows=latest_model_rows,
        latest_duration=latest_duration,
        latest_images=latest_images,
        latest_stock_label=latest_stock_label,
        app_version=APP_VERSION,
    )


@app.route("/experiment", methods=["POST"])
def experiment_route():
    stock_code, start_date, end_date = read_stock_params()
    if not stock_code or not start_date or not end_date:
        return "参数不完整", 400

    jobid = create_experiment_job(stock_code, start_date, end_date)
    return redirect(url_for("experiment_page", jobid=jobid))


@app.route("/experiment/<jobid>", methods=["GET"])
def experiment_page(jobid):
    context = get_experiment_page_context(jobid)
    if context is None:
        abort(404)
    return render_template("experiment.html", **context, app_version=APP_VERSION)


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
    print("启动 Flask 服务：http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000, use_reloader=False)
