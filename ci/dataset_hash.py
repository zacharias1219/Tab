"""HF-2: the frozen dataset list must hash to the committed value."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import datasets, paths  # noqa: E402


def main() -> int:
    sel = datasets.selected_list()
    try:
        frozen = datasets.check_frozen()
    except ValueError as e:
        print(f"FAIL {e}")
        return 1
    if not frozen:
        if sel.get("status") == "frozen":
            print("FAIL (HF-2): status is 'frozen' but bench/selected_datasets.sha256 is missing")
            return 1
        print("NOTICE: dataset list not frozen yet; the full benchmark refuses to run until it is")
        return 0
    print(f"OK: {paths.SELECTED.name} matches {paths.SELECTED_SHA.read_text().strip()[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
