# app/core/memory.py

import json
import os
from datetime import datetime
from typing import Any, Dict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOG_PATH = os.path.join(DATA_DIR, "memory.log")


def append_interaction(user_id: str, text: str, recommendation: str, extra: Dict[str, Any] | None = None) -> None:
    """
    Append a single interaction to data/memory.log as JSONL.
    This is deliberately simple; good enough for debugging / demos.
    """
    record: Dict[str, Any] = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "text": text,
        "recommendation": recommendation,
    }
    if extra:
        record.update(extra)

    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # Don't crash the API if logging fails
        print("memory.append_interaction error:", e)