# dev_check.py
# Run this to verify the Python environment and key packages for the triage project.
# Usage: python dev_check.py

import importlib
import sys
import traceback

def try_import(name, alias=None):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", None)
        print(f"[OK]  {name} (version: {ver})")
        return m
    except Exception as e:
        print(f"[MISSING] {name} -> {e.__class__.__name__}: {e}")
        return None

def print_heading(h):
    print("\n" + "="*8 + " " + h + " " + "="*8)

def check_torch():
    print_heading("Torch / CUDA")
    t = try_import("torch")
    if t:
        try:
            print("torch.__version__:", t.__version__)
            print("CUDA available:", t.cuda.is_available())
        except Exception as e:
            print("Error while querying torch:", e)
            traceback.print_exc()

def check_transformers_faiss():
    print_heading("Transformers / Sentence-Transformers / FAISS")
    try_import("transformers")
    try_import("sentence_transformers")
    try_import("faiss")  # faiss may be 'faiss' or 'faiss_cpu' installed as wheel, but import name is 'faiss'
    try_import("faiss_cpu")

def check_fastapi_uvicorn():
    print_heading("FastAPI / Uvicorn / HTTPX")
    try_import("fastapi")
    try_import("uvicorn")
    try_import("httpx")
    try_import("starlette")

def check_genai_openai():
    print_heading("GenAI (google-genai) / OpenAI SDK")
    g = try_import("google.genai")
    try_import("genai")  # in case package name differs
    try_import("openai")

def check_extra():
    print_heading("Other helpful packages")
    for pkg in ("pandas", "numpy", "scikit_learn", "sklearn", "faiss_cpu", "sqlalchemy", "redis", "celery"):
        try_import(pkg)

def check_fastapi_app_import():
    print_heading("Attempt to import app.main (if exists)")
    try:
        import app.main as am
        print("[OK] Imported app.main")
    except Exception as e:
        print(f"[ERROR] Importing app.main failed: {e.__class__.__name__}: {e}")
        traceback.print_exc()

def main():
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n"," "))
    check_torch()
    check_transformers_faiss()
    check_fastapi_uvicorn()
    check_genai_openai()
    check_extra()
    # Try fastapi app import only if package exists
    import os
    if os.path.isdir(os.path.join(os.getcwd(), "app")):
        check_fastapi_app_import()
    else:
        print("\nNote: no ./app folder found in current directory; skipping app.main import check.")

if __name__ == "__main__":
    main()