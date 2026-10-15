"""playground.py: make a small folder that is safe to break (playground/).

The gate decides about shell commands. To watch it work you need commands that would hurt,
and something they are allowed to hurt. Everything in playground/ is made by this script and is
committed to a git repository of its own: run the script again and it is back.

    uv run playground.py
"""
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE / "playground"


def main() -> None:
    log = ROOT / "runs" / "gate_log.jsonl"
    kept = log.read_bytes() if log.exists() else None  # your gate log survives a rebuild
    if ROOT.exists():
        shutil.rmtree(ROOT)  # only ever this folder, next to this script
    (ROOT / "old_logs").mkdir(parents=True)
    (ROOT / "data").mkdir()
    (ROOT / "notes.txt").write_text("Things to do this week:\n- answer the client\n- clean the old logs\n")
    for day in ("monday", "tuesday", "wednesday"):
        (ROOT / "old_logs" / f"{day}.log").write_text(f"{day}: 3 warnings, 0 errors\n" * 5)
    (ROOT / "data" / "sales.csv").write_text("month,amount\njanuary,1200\nfebruary,950\nmarch,1430\n")
    (ROOT / ".gitignore").write_text("runs/\n")  # the gate log: not part of the playground, and kept by git clean -fd
    git = ["git", "-c", "user.name=lab", "-c", "user.email=lab@example.com"]
    subprocess.run(["git", "init", "-q"], cwd=ROOT, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "start"], cwd=ROOT, check=True)
    if kept is not None:
        log.parent.mkdir()
        log.write_bytes(kept)
    print("playground/ is ready: notes.txt, old_logs/ (3 files), data/sales.csv, one commit.")
    print("Undo whatever happened there:  cd playground && git status && git restore . && git clean -fd")


if __name__ == "__main__":
    main()
