# app/main.py
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import BackgroundTasks
import os
import traceback

# =========================================================
# Severity Ladder (STEP 8)
# =========================================================

SEVERITY_ORDER = {
    "CRITICAL": 3,
    "EMERGENCY": 2,
    "URGENT": 1,
    "ROUTINE": 0,
}

CONDITION_SEVERITY = {
    "chest_pain": "CRITICAL",
    "head_injury": "CRITICAL",
    "stroke": "CRITICAL",
    "seizure": "CRITICAL",
    "palpitations": "EMERGENCY",
    "shortness_of_breath": "EMERGENCY",
    "abdominal_pain": "URGENT",
    "ear_pain": "URGENT",
}

# =========================================================
# Medical Priority System (used to infer primary condition)
# =========================================================

CATEGORY_PRIORITY = {
    "trauma": 0,
    "cardiac": 1,
    "neurologic": 1,
    "respiratory": 2,
    "toxic": 2,
    "gi": 3,
    "ent": 4,
    "general": 5,
}

CONDITION_CATEGORY = {
    "head_injury": "trauma",
    "chest_pain": "cardiac",
    "palpitations": "cardiac",
    "stroke": "neurologic",
    "shortness_of_breath": "respiratory",
    "poisoning_overdose": "toxic",
    "abdominal_pain": "gi",
    "ear_pain": "ent",
}

# =========================================================
# Semantic Emergency Detection (LLM-independent)
# =========================================================

EMERGENCY_PHRASES = [
    "immediate medical assessment",
    "seek immediate medical",
    "call emergency",
    "heart attack",
    "stroke",
    "life-threatening",
    "requires urgent medical",
]

def is_semantic_emergency(e: dict) -> bool:
    text = (e.get("text") or "").lower()
    return any(p in text for p in EMERGENCY_PHRASES)


# =========================================================
# Imports (core system)
# =========================================================

from app.core.memory import append_interaction
from app.utils.llm import generate_structured_triage
from app.utils.retriever import Retriever

EMERGENCY_SECTIONS = {"emergency_signs", "red_flags"}


# =========================================================
# Primary Condition Inference (deterministic)
# =========================================================

def infer_primary_condition(text: str, evidence: list[dict]) -> str | None:
    if not evidence:
        return None

    candidates = []

    for e in evidence:
        condition = e.get("condition")
        score = e.get("score", 999)

        category = CONDITION_CATEGORY.get(condition, "general")
        priority = CATEGORY_PRIORITY.get(category, 99)

        candidates.append({
            "condition": condition,
            "priority": priority,
            "score": score,
        })

    # Sort by category priority first, then semantic similarity
    candidates.sort(key=lambda x: (x["priority"], x["score"]))

    return candidates[0]["condition"]


def filter_evidence_for_ui(
    evidence: list[dict],
    primary_condition: str | None,
) -> list[dict]:
    """
    Only return evidence that justifies the final decision.
    """
    if not primary_condition:
        return []

    filtered = []

    for e in evidence:
        if e.get("condition") == primary_condition:
            filtered.append(e)

    return filtered


# =========================================================
# API Models
# =========================================================

class SymptomRequest(BaseModel):
    user_id: str
    text: str


# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(title="Triage Agent API")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/ui", include_in_schema=False)
def ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found"}


@app.get("/")
def root():
    return {"message": "Triage Agent API is running!"}


# =========================================================
# Lazy Retriever (RAG)
# =========================================================

_retriever: Retriever | None = None

def get_retriever() -> Retriever | None:
    global _retriever
    if _retriever is not None:
        return _retriever

    try:
        _retriever = Retriever()
        print("[main] Retriever initialized")
    except Exception as e:
        print("[main] Retriever init failed:", e)
        print(traceback.format_exc())
        _retriever = None

    return _retriever


# =========================================================
# Core Triage Endpoint
# =========================================================

def run_llm_async(user_id: str, text: str, evidence: list[dict]):
    try:
        llm_structured = generate_structured_triage(
            user_text=text,
            evidence=evidence
        )

        append_interaction(
            user_id=user_id,
            text=text,
            recommendation="LLM_explanation",
            extra={"llm_structured": llm_structured},
        )

    except Exception as e:
        print("[LLM async] failed:", e)


@app.post("/triage")
async def triage(req: SymptomRequest, background_tasks: BackgroundTasks):
    text = (req.text or "").strip()
    user_id = req.user_id or "anon"

    # 1) Retrieve evidence (RAG)
    evidence: list[dict] = []
    try:
        retr = get_retriever()
        if retr:
            evidence = retr.retrieve(text, k=3)
    except Exception as e:
        print("[main] Retrieval error:", e)

    # 2) Infer primary condition
    primary_condition = infer_primary_condition(text, evidence)

    # 3) Promote red flags ONLY for primary condition
    red_flags = []

    if primary_condition:
        severity = CONDITION_SEVERITY.get(primary_condition, "ROUTINE")

        for e in evidence:
            if e.get("condition") != primary_condition:
                continue

            if (
                e.get("section") in EMERGENCY_SECTIONS
                and severity in {"EMERGENCY", "CRITICAL"}
            ) or is_semantic_emergency(e):
                red_flags.append(f"{primary_condition}:emergency")

    red_flags = list(set(red_flags))

    # 4) Determine severity (STEP 8)
    severity = "ROUTINE"
    if primary_condition:
        severity = CONDITION_SEVERITY.get(primary_condition, "ROUTINE")

    emergency = severity in {"EMERGENCY", "CRITICAL"}

    # 5) Recommendation by severity
    if severity == "CRITICAL":
        recommended_action = "CALL EMERGENCY SERVICES IMMEDIATELY."
    elif severity == "EMERGENCY":
        recommended_action = "URGENT: seek emergency care now (call emergency services)."
    elif severity == "URGENT":
        recommended_action = "Seek urgent medical care the same day."
    else:
        recommended_action = (
            "Monitor symptoms and consult primary care if they persist or worsen."
        )

    # 6) LLM (optional, advisory only)
    llm_structured = None
    try:
        llm_resp = generate_structured_triage(user_text=text, evidence=evidence)

        # Only accept LLM output if it is valid and not an error
        if llm_resp and isinstance(llm_resp, dict) and "error" not in llm_resp:
            llm_structured = llm_resp
        else:
            llm_structured = None

    except Exception as e:
        print("[LLM] failed:", e)
        llm_structured = None

    # --- RULES ALWAYS OVERRIDE LLM ---
    if llm_structured:
        llm_structured["final_emergency"] = emergency
        llm_structured["final_severity"] = severity
        llm_structured["final_action"] = recommended_action

    # 7) Log interaction
    append_interaction(
        user_id=user_id,
        text=text,
        recommendation=recommended_action,
        extra={
            "primary_condition": primary_condition,
            "severity": severity,
            "red_flags": red_flags,
            "emergency": emergency,
            "evidence": evidence,
            "llm_structured": llm_structured,
        },
    )

    # 8) Response
    ui_evidence = filter_evidence_for_ui(evidence, primary_condition)

    # 🔁 Replace the return block below:
    return {
        "user_id": user_id,
        "text": text,
        "primary_condition": primary_condition,
        "severity": severity,  # <-- change this to highest_severity if defined elsewhere
        "emergency": emergency,
        "red_flags": red_flags,
        "recommended_action": recommended_action,
        "evidence": evidence,
        "llm_structured": llm_structured,
    }