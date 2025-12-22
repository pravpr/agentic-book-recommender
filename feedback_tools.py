import csv
from datetime import datetime
from pathlib import Path

FEEDBACK_FILE = "feedback.csv"

def init_feedback_file():
    path = Path(FEEDBACK_FILE)
    if not path.exists():
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "user_id", "query_title", "recommended_title", "liked"])

def log_feedback(user_id, query_title, recommended_title, liked: bool):
    init_feedback_file()
    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            user_id,
            query_title,
            recommended_title,
            int(bool(liked))
        ])
