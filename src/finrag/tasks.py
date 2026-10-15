"""Cross-platform task runner (Windows has no `make`).

  uv run python -m finrag.tasks list
  uv run python -m finrag.tasks pipeline      # fetch -> parse -> chunk (all variants) -> load (all)
  uv run python -m finrag.tasks evals         # retrieval sweep, then full baseline/final on dev

Each task is a list of steps; a failing step stops the task. Steps run with
the current interpreter (the uv venv), so this works the same on Windows,
macOS, and Linux.
"""

from __future__ import annotations

import subprocess
import sys
import time

# Chunking variants the eval sweep compares (see finrag.evals.experiments).
CHUNK_VARIANTS = [
    ["--max-tokens", "512", "--overlap", "64"],  # fixed-512-64 + section-512
    ["--max-tokens", "256", "--overlap", "32"],  # section-256 (+ fixed-256-32)
    ["--max-tokens", "1024", "--overlap", "128"],  # section-1024 (+ fixed-1024-128)
]
LOADED_STRATEGIES = ["section-512", "fixed-512-64", "section-256", "section-1024"]


def py(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


TASKS: dict[str, list[list[str]]] = {
    "test": [py("pytest", "-q")],
    "lint": [
        py("ruff", "check", "src", "tests", "ui"),
        py("ruff", "format", "--check", "src", "tests", "ui"),
    ],
    "fetch": [py("finrag.ingest.fetch")],
    "parse": [py("finrag.ingest.parse")],
    "chunk": [py("finrag.chunking.run", *v) for v in CHUNK_VARIANTS],
    "load": [py("finrag.store.load", "--strategy", s) for s in LOADED_STRATEGIES],
    "audit": [py("finrag.evals.audit")],
    "sweep": [py("finrag.evals.experiments")],
    "api": [py("uvicorn", "finrag.api.main:app", "--port", "8000")],
    "ui": [py("streamlit", "run", "ui/Home.py")],
}
TASKS["pipeline"] = (
    TASKS["fetch"] + TASKS["parse"] + TASKS["chunk"] + TASKS["load"] + TASKS["audit"]
)
TASKS["evals"] = TASKS["sweep"] + [
    py("finrag.evals.run", "--name", "baseline", "--strategy", "fixed-512-64"),
    py(
        "finrag.evals.run", "--name", "final", "--method", "hybrid", "--filters", "auto", "--rerank"
    ),
    py("finrag.evals.report", "evals/results/baseline-dev.json", "evals/results/final-dev.json"),
]


def run(name: str) -> int:
    for step in TASKS[name]:
        label = " ".join(step[1:])
        print(f"\n>>> {label}", flush=True)
        started = time.time()
        code = subprocess.call(step)
        if code != 0:
            print(f"!!! failed ({code}): {label}", flush=True)
            return code
        print(f"--- ok in {time.time() - started:.0f}s", flush=True)
    return 0


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "list"
    if name not in TASKS:
        print("Tasks:")
        for task, steps in TASKS.items():
            print(f"  {task:9} {len(steps)} step(s)")
        sys.exit(0 if name == "list" else 2)
    sys.exit(run(name))


if __name__ == "__main__":
    main()
