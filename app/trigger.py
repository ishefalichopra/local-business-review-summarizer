from flask import Flask, jsonify
import subprocess
import threading
import datetime
app = Flask(__name__)

def run_ingest_background():
    with open("logs/ingest.log", "a") as log:
        log.write(f"\n[{datetime.datetime.now()}] Starting ingestion...\n")
    result = subprocess.run(
        ["python3", "app/ingest.py"],
        cwd="/home/shefalichopra/local-business-review-summarizer",
        capture_output=True,
        text=True
    )
    with open("logs/ingest.log", "a") as log:
        log.write(result.stdout)
        log.write(result.stderr)

@app.route("/ingest", methods=["POST"])
def run_ingest():
    try:
        # Run ingest in background thread so we respond immediately
        thread = threading.Thread(target=run_ingest_background)
        thread.daemon = True
        thread.start()
        return jsonify({"status": "started", "message": "Ingestion started in background"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8502)
