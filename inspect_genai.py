# inspect_genai.py
# Run this to inspect the installed google-genai client and its available methods.
import inspect, sys
try:
    import google.genai as genai
except Exception as e:
    print("Import google.genai failed:", type(e).__name__, e)
    sys.exit(1)

print("google.genai imported. Client constructor exists? ->", hasattr(genai, "Client"))
try:
    client = genai.Client()
except Exception as e:
    print("Client init error:", type(e).__name__, e)
    # still continue to inspect module objects if client couldn't init
    client = None

def inspect_obj(name, obj):
    print("\n--", name, "--")
    try:
        for attr in sorted(dir(obj)):
            if attr.startswith("_"):
                continue
            member = getattr(obj, attr)
            if callable(member):
                try:
                    sig = inspect.signature(member)
                    print(f"CALLABLE: {name}.{attr}{sig}")
                except Exception:
                    print(f"CALLABLE: {name}.{attr} (no signature)")
            else:
                print(f"ATTR: {name}.{attr}")
    except Exception as e:
        print("Inspect failed for", name, e)

# Inspect top-level client sub-objects if client created
if client:
    for name in ("models", "chats", "files", "operations", "vertexai"):
        if hasattr(client, name):
            inspect_obj(f"client.{name}", getattr(client, name))
else:
    # Inspect module-level objects
    for name in ("Client",):
        if hasattr(genai, name):
            inspect_obj(f"genai.{name}", getattr(genai, name))

print("\nDONE")