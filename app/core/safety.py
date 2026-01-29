# app/core/safety.py

import re
from typing import List

# Simple keyword patterns for red-flag detection
RED_FLAG_PATTERNS = [
    r"chest pain",
    r"shortness of breath",
    r"difficulty breathing",
    r"can't breathe",
    r"cannot breathe",
    r"severe headache",
    r"worst headache",
    r"confusion",
    r"unconscious",
    r"loss of consciousness",
    r"fainting",
    r"passed out",
    r"sudden weakness",
    r"one side of the body",
    r"slurred speech",
    r"stroke",
    r"suicidal",
    r"suicide",
    r"self harm",
    r"self-harm",
]

# Precompile regex
_COMPILED = [re.compile(pat, re.IGNORECASE) for pat in RED_FLAG_PATTERNS]


def detect_red_flags(text: str) -> List[str]:
    """
    Return a list of red-flag phrases found in the text.
    """
    if not text:
        return []
    found = set()
    for pat, rx in zip(RED_FLAG_PATTERNS, _COMPILED):
        if rx.search(text):
            found.add(pat)
    return sorted(found)


def needs_emergency_escalation(text: str) -> bool:
    """
    True if we think this requires emergency care.
    Uses pattern-based rule engine for detection.
    """
    t = (text or "").lower()

    emergency_patterns = [
        # Cardiac / respiratory
        ("chest pain", "shortness of breath"),

        # Neurological / trauma
        ("head injury", "vomiting"),
        ("hit my head", "vomiting"),
        ("head injury", "loss of consciousness"),
        ("head injury", "seizure"),

        # Others (examples)
        ("suicidal",),
        ("cannot breathe",),
        ("severe bleeding",),
    ]

    for pattern in emergency_patterns:
        if all(p in t for p in pattern):
            return True

    return False