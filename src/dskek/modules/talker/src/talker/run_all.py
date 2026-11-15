"""Run every test file. From the repo root (where the ``dskek`` package lives): python tests/run_all.py"""

import os
import subprocess
import sys
from pathlib import Path

TESTS = ["test_wake", "test_text", "test_tools", "test_llm_stream", "test_gateway", "test_mcp", "test_session"]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": f"{root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    failed: list[str] = []
    for name in TESTS:
        print(f"=== {name}", flush=True)
        if subprocess.run([sys.executable, str(root / "tests" / f"{name}.py")], env=env, check=False).returncode:
            failed.append(name)
    print("\nFAILED: " + ", ".join(failed) if failed else "\nall test files passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
