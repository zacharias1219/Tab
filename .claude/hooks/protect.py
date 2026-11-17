"""PreToolUse hook: block edits to protected paths (secrets, the pre-registration, the frozen list, CI)."""
import json
import sys

PROTECTED = ("prereg/PREREG.md", "bench/selected_datasets.json", "bench/selected_datasets.sha256", ".github/workflows/", ".env")

data = json.load(sys.stdin)
path = (data.get("tool_input") or {}).get("file_path", "") or ""
if any(p in path.replace("\\", "/") for p in PROTECTED):
    print(f"Blocked: {path} is protected. Ask the user; amendments to PREREG.md are appended with a date.", file=sys.stderr)
    sys.exit(2)
sys.exit(0)
