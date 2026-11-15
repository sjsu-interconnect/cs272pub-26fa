import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inference"))
import live_view


class FakeDrive:
    def __init__(self):
        self.frame = b"initial"
        self.phase = "observe"
        self.decisions = 0
        self.executions = 0

    def decide(self):
        self.decisions += 1
        self.phase = "decided"
        return {
            "decision": self.decisions,
            "actor_action_probabilities": [0.1, 0.1, 0.1, 0.6, 0.1],
            "selected_action": 3,
        }

    def execute(self):
        self.executions += 1
        last = self.executions == 2
        self.phase = "ended" if last else "observe"
        images = [f"{self.executions}-{i}".encode() for i in range(5)]
        self.frame = images[-1]
        return {
            "decision": self.executions,
            "actor_action_probabilities": [0.1, 0.1, 0.1, 0.6, 0.1],
            "selected_action": 3,
            "is_last": last,
            "collision": last,
        }, images


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for player")
        time.sleep(0.005)


def test_automatic_capture_replay_and_alignment(monkeypatch):
    monkeypatch.setattr(live_view, "FRAME_SECONDS", 0.005)
    monkeypatch.setattr("IPython.display.display", lambda _: None)
    drive = FakeDrive()
    view = live_view.LiveDriveView(drive)
    try:
        assert view.scene.value == b"initial"
        assert drive.decisions == 0
        view.show()
        wait_for(lambda: drive.phase == "ended" and not view._playing)
        assert (drive.decisions, drive.executions) == (2, 2)
        assert len(view._frames) == 13
        assert [f["decision"]["decision"] for f in view._frames[1:7]] == [1] * 6
        assert [f["decision"]["decision"] for f in view._frames[7:]] == [2] * 6
        assert all(f["ending"] is None for f in view._frames[:-1])
        assert view._frames[-1]["ending"]["collision"]
        view._command("previous")
        wait_for(lambda: view._index == 11 and view.scene.value == b"2-3")
        view._command("next")
        wait_for(lambda: view._index == 12 and view.scene.value == b"2-4")
        view._command("restart")
        wait_for(lambda: view._index == 0 and view.scene.value == b"initial")
        view.speed.value = 2.0
        view._command("play")
        wait_for(lambda: view._index == 12 and not view._playing)
        assert (drive.decisions, drive.executions) == (2, 2)
    finally:
        view.close()


def test_pause_prevents_new_decisions(monkeypatch):
    monkeypatch.setattr("IPython.display.display", lambda _: None)
    drive = FakeDrive()
    view = live_view.LiveDriveView(drive)
    try:
        view._command("pause")
        view.show()
        time.sleep(0.05)
        assert drive.decisions == 0
        assert view.scene.value == b"initial"
    finally:
        view.close()
