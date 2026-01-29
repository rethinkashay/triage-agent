# app/utils/dev_utils.py
# small helper utilities for development and testing

def pretty(obj):
    import json
    print(json.dumps(obj, indent=2, ensure_ascii=False))