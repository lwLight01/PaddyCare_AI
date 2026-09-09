import os
import json
import logging
import uuid
from datetime import datetime
from flask import (
    Flask, request, jsonify, render_template, send_from_directory
)
from werkzeug.utils import secure_filename

from config import UPLOAD_DIR, LOG_DIR, API_LOG_FILE
from predict import predict as run_predict

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=API_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)


HISTORY_PATH = os.path.join(LOG_DIR, "history_records.json")


def load_history() -> list:
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            logging.warning("history_records.json was corrupted — resetting to empty.")
            save_history([])
    return []


def save_history(records: list):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    lang = request.form.get("lang", "bn")

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "কোনো ফাইল পাঠানো হয়নি।"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "ফাইল নির্বাচন করুন।"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": "শুধুমাত্র PNG, JPG, JPEG, WEBP ফাইল গ্রহণযোগ্য।"
        }), 400

    ext      = secure_filename(file.filename).rsplit(".", 1)[-1]
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    logging.info(f"Image uploaded: {filename}")

    result = run_predict(filepath, lang=lang)
    result["image_url"] = f"/static/uploads/{filename}"
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    records = load_history()
    records.insert(0, result)
    records = records[:50]
    save_history(records)

    logging.info(f"Prediction result: {result.get('status')}")
    return jsonify(result)


@app.route("/history", methods=["GET"])
def history_route():
    records = load_history()
    return jsonify(records)


@app.route("/history", methods=["DELETE"])
def clear_history():
    save_history([])
    return jsonify({"status": "ok", "message": "ইতিহাস মুছে ফেলা হয়েছে।"})


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
