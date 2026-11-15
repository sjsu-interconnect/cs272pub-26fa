"""Presentation helpers for the notebook; training and evaluation remain unchanged."""

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time


def run_process(script, arguments, run_dir=None):
    """Run the existing CLI in a clean process and stream its output."""
    command = [sys.executable, "-I", "-u", "-B", str(script), *map(str, arguments)]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, start_new_session=True,
    )
    lines = queue.Queue()

    def read_output():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    start = time.monotonic()
    closed = False
    try:
        while not closed:
            try:
                line = lines.get(timeout=10)
            except queue.Empty:
                step = "initializing / compiling"
                if run_dir and (Path(run_dir) / "metrics.jsonl").exists():
                    rows = read_metrics(Path(run_dir) / "metrics.jsonl")
                    if rows:
                        step = f"last logged environment step {rows[-1].get('step', '?')}"
                print(f"[{time.monotonic() - start:.0f}s] {step}", flush=True)
                continue
            if line is None:
                closed = True
            else:
                print(line, end="", flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        print("\nStopping the process group; the incomplete output is retained.")
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
        raise
    finally:
        process.stdout.close()
    if code:
        raise RuntimeError(f"{script.name} failed with exit code {code}; inspect the output directory.")
    print(f"Finished in {time.monotonic() - start:.1f} seconds.")


def read_metrics(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # The writer may be in the middle of appending the last line.
                continue
    return rows


def show_curves(path):
    import matplotlib.pyplot as plt

    rows = read_metrics(path)
    groups = [
        ("World Model losses", ["train/loss/obs", "train/loss/rew",
                                "train/loss/dyn", "train/loss/rep"]),
        ("Actor and Critic losses", ["train/loss/policy", "train/loss/value"]),
        ("Actor entropy", ["train/ent/action"]),
        ("Training episode return", ["episode/score"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (title, keys) in zip(axes.flat, groups):
        found = False
        for key in keys:
            values = [(r["step"], r[key]) for r in rows if "step" in r and key in r]
            if values:
                ax.plot(*zip(*values), marker=".", label=key.removeprefix("train/"))
                found = True
        ax.set(title=title, xlabel="Environment steps")
        ax.grid(alpha=0.25)
        if found:
            ax.legend(fontsize="small")
        else:
            ax.text(.5, .5, "Not logged for this run", ha="center", transform=ax.transAxes)
    plt.show()
