# app/utils/llm.py
import os
import typing
import traceback
import json

# Try to import Google GenAI SDK first (Gemini)
try:
    from google import genai
    _HAS_GENAI = True
except Exception:
    genai = None
    _HAS_GENAI = False

# Optional OpenAI fallback (will be unused if no OPENAI_API_KEY)
try:
    import openai
    _HAS_OPENAI = True
except Exception:
    openai = None
    _HAS_OPENAI = False

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Allow more room for the JSON so it is not truncated mid-response.
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

PROMPT_TEMPLATE = """You are a medical triage assistant. You MUST NOT provide diagnoses. You should:
1) Summarize the user's symptoms in one short sentence.
2) State urgency level: EMERGENCY / URGENT / ROUTINE.
3) Provide recommended next step(s) (short), and include explicit reasons.
4) Cite evidence from the provided sources (use the doc_id and the chunk text).
5) Provide a concise confidence estimate (low/medium/high).
6) Always include this safety disclaimer: "This is not medical advice. For emergencies call local emergency services."

User text:
\"\"\"{user_text}\"\"\"


Retrieved evidence (top to bottom):
{evidence_text}

Produce a JSON object only (no prose, no backticks) with keys:
- summary
- urgency
- recommendation
- reasons (list)
- sources (list of {{"doc_id":..., "text":...}})
- confidence
- disclaimer

FORMATTING RULES (IMPORTANT):
- Output MUST be a single valid JSON object.
- No extra text before or after the JSON.
- reasons: at most 3 items, each <= 30 words.
- sources: at most 3 items.
- Keep the entire JSON response compact (around 300 tokens or less).

Return VALID JSON ONLY.
"""


def _build_evidence_text(evidence: typing.List[dict]) -> str:
    lines = []
    for e in evidence:
        lines.append(f"{e.get('doc_id')} :: {e.get('chunk_id')}\n{e.get('text')}")
    return "\n\n---\n\n".join(lines) if lines else "No retrieved evidence available."


def _call_gemini(
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> typing.Optional[str]:
    """
    Call Google GenAI via google.genai.Client.models.generate_content(...)
    and return a string with the model's response, or None on failure.
    """
    if not _HAS_GENAI:
        return None

    try:
        # GEMINI_API_KEY or GOOGLE_API_KEY should already be in env
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
    except Exception as e:
        print("Gemini client init failed:", e)
        traceback.print_exc()
        return None

    cfg = {
        "max_output_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.9,
    }

    try:
        print("[LLM] Gemini called:", model)
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=cfg,
        )
    except Exception as e:
        print("Gemini generate_content failed:", e)
        traceback.print_exc()
        return None

    # Extract text in a robust way
    try:
        # 1) object with .text
        if hasattr(resp, "text") and resp.text:
            return resp.text if isinstance(resp.text, str) else str(resp.text)

        # 2) dict-like response
        if isinstance(resp, dict):
            for key in ("outputText", "text", "content", "output", "candidates", "results"):
                val = resp.get(key)
                if not val:
                    continue
                if isinstance(val, str):
                    return val
                if isinstance(val, list) and val:
                    first = val[0]
                    if isinstance(first, dict):
                        for subkey in ("text", "content", "output"):
                            if subkey in first:
                                subval = first[subkey]
                                if isinstance(subval, str):
                                    return subval
                                if isinstance(subval, list):
                                    return " ".join(
                                        p.get("text", str(p)) if isinstance(p, dict) else str(p)
                                        for p in subval
                                    )
                    else:
                        return str(first)
                if isinstance(val, dict):
                    for subkey in ("text", "outputText", "content"):
                        if subkey in val:
                            subval = val[subkey]
                            return subval if isinstance(subval, str) else str(subval)

        # 3) generic fallback
        return str(resp)
    except Exception as e:
        print("Gemini extraction failed:", e)
        traceback.print_exc()
        try:
            print("Raw resp repr:", repr(resp))
        except Exception:
            pass
        return None


def _call_openai(
    prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> typing.Optional[str]:
    """
    Optional OpenAI fallback. Will only run if OPENAI_API_KEY is set.
    """
    if not _HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a triage assistant. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        print("OpenAI call failed:", e)
        traceback.print_exc()
        return None


# -------- JSON cleanup helpers --------

def _strip_markdown_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` fences if present.
    """
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines).strip()
    return t


def _extract_json_block(text: str) -> typing.Any:
    """
    Try progressively harder to find a JSON (or JSON-like) object in the text.
    Handles slightly invalid or truncated output by trimming from the right
    until a valid JSON object is found.
    """
    cleaned = _strip_markdown_fences(text)

    # 1) First attempt: direct parse of the whole cleaned string
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2) Find the first '{' and the last '}' – assume JSON object lives there
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not extract JSON block from text")

    candidate = cleaned[start : end + 1]

    # 2a) Try full candidate as JSON
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # 2b) Try trimming from the right at brace/closing-bracket boundaries
    for cut in range(len(candidate) - 1, 0, -1):
        ch = candidate[cut]
        if ch not in ("}", "]"):
            continue
        snippet = candidate[: cut + 1]
        try:
            return json.loads(snippet)
        except Exception:
            continue

    # 2c) As a last resort, try interpreting as a Python literal
    try:
        import ast
        return ast.literal_eval(candidate)
    except Exception:
        pass

    # 3) Nothing worked
    raise ValueError("Could not extract JSON block from text")


# -------- Public function used by main.py --------

def generate_structured_triage(
    user_text: str,
    evidence: typing.List[dict],
) -> typing.Optional[dict]:
    """
    High-level helper:
      - builds the prompt
      - calls Gemini (or OpenAI fallback)
      - tries hard to parse JSON
      - on failure, returns {"error": ..., "raw": ...}
    """
    evidence_text = _build_evidence_text(evidence)
    prompt = PROMPT_TEMPLATE.format(
        user_text=user_text,
        evidence_text=evidence_text,
    )

    raw: typing.Optional[str] = None

    # Prefer Gemini if key is present
    if _HAS_GENAI and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raw = _call_gemini(prompt)

    # Optional OpenAI fallback (only if OpenAI key set)
    # if raw is None and _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
    #     raw = _call_openai(prompt)

    # If we never got any text back from an LLM
    if not raw:
        return {
            "error": "LLM unavailable or returned empty response.",
            "raw": None,
        }

    # Try to parse JSON (with cleanup)
    try:
        return _extract_json_block(raw)
    except Exception as e:
        # Last resort: return the raw string so the caller/UI can still inspect it
        return {
            "error": f"JSON parse failed: {e}",
            "raw": raw,
        }