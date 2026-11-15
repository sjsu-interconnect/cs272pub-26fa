"""Progressively capture and replay one real LiveDrive episode in Jupyter."""

import html
import threading

LABELS = ("LEFT", "IDLE", "RIGHT", "FASTER", "SLOWER")
FRAME_SECONDS = 0.2
NAVY, MUTED, TRACK, HIGHLIGHT = "#0b2a5b", "#9aa7b4", "#e8ecf0", "#dce8f7"
RED, GRAY = "#c62828", "#5f6b76"


def probabilities_html(decision=None):
    """Render the probabilities and sampled action from the same policy call."""
    probabilities = decision["actor_action_probabilities"] if decision else [None] * 5
    sampled = decision["selected_action"] if decision else None
    rows = []
    for index, (label, probability) in enumerate(zip(LABELS, probabilities)):
        chosen = index == sampled
        width = 0.0 if probability is None else 100 * probability
        text = "–" if probability is None else f"{100 * probability:.1f}%"
        rows.append(
            "<div style='display:flex;align-items:center;gap:12px;padding:3px 8px;"
            f"border:2px solid {NAVY if chosen else 'transparent'};"
            f"background:{HIGHLIGHT if chosen else 'transparent'}'>"
            f"<span style='width:90px;font-weight:{700 if chosen else 400}'>{label}</span>"
            f"<span style='flex:1;height:22px;background:{TRACK};overflow:hidden'>"
            f"<span style='display:block;height:100%;width:{width:.2f}%;"
            f"background:{NAVY if chosen else MUTED}'></span></span>"
            f"<span style='width:65px;text-align:right'>{text}</span>"
            f"<span style='width:100px;color:{NAVY};font-weight:700'>"
            f"{'◀ sampled' if chosen else ''}</span></div>"
        )
    title = "Actor probabilities" if decision is None else (
        f"Decision {decision['decision']} · Actor probabilities"
    )
    return f"<div style='font-size:18px'><b>{title}</b>{''.join(rows)}</div>"


class LiveDriveView:
    """Automatic live capture with seekable, non-resampling playback."""

    _current = None

    def __init__(self, drive):
        import ipywidgets as widgets

        if LiveDriveView._current is not None:
            LiveDriveView._current.close()
        LiveDriveView._current = self
        self.drive = drive
        self._cond = threading.Condition()
        self._frames = [{"image": drive.frame, "decision": None, "ending": None}]
        self._index = 0
        self._playing = True
        self._closed = False
        self._dirty = True
        self._error = None
        self._thread = None

        self.play_button = widgets.Button(description="▶ Play")
        self.pause_button = widgets.Button(description="❚❚ Pause")
        self.previous_button = widgets.Button(description="◀ Previous")
        self.next_button = widgets.Button(description="Next ▶")
        self.restart_button = widgets.Button(description="↺ Restart")
        self.speed = widgets.Dropdown(
            options=[("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0)],
            value=1.0, description="Speed", layout=widgets.Layout(width="150px"),
        )
        self.scene = widgets.Image(
            format="png", layout=widgets.Layout(width="100%", max_width="1100px")
        )
        self.counter = widgets.HTML()
        self.status = widgets.HTML()
        self.bars = widgets.HTML()
        controls = widgets.HBox(
            [self.play_button, self.pause_button, self.previous_button,
             self.next_button, self.restart_button, self.speed],
            layout=widgets.Layout(flex_flow="row wrap", align_items="center"),
        )
        self.widget = widgets.VBox(
            [controls, self.counter, self.scene, self.status, self.bars],
            layout=widgets.Layout(max_width="1100px"),
        )
        for widget, name in (
            (self.widget, "lab3-live"), (self.scene, "lab3-scene"),
            (self.counter, "lab3-counter"), (self.status, "lab3-status"),
            (self.bars, "lab3-bars"),
        ):
            widget.add_class(name)
        self.play_button.on_click(lambda _: self._command("play"))
        self.pause_button.on_click(lambda _: self._command("pause"))
        self.previous_button.on_click(lambda _: self._command("previous"))
        self.next_button.on_click(lambda _: self._command("next"))
        self.restart_button.on_click(lambda _: self._command("restart"))
        self.speed.observe(lambda _: self._wake(), names="value")
        self._render()

    def show(self):
        from IPython.display import display

        display(self.widget)
        with self._cond:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="lab3-live-drive", daemon=True
                )
                self._thread.start()

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join()

    def _wake(self):
        with self._cond:
            self._dirty = True
            self._cond.notify_all()

    def _command(self, command):
        with self._cond:
            if command == "play":
                if self._error is None:
                    if self._index == len(self._frames) - 1 and self.drive.phase == "ended":
                        self._index = 0
                    self._playing = True
            elif command == "pause":
                self._playing = False
            elif command == "previous":
                self._playing = False
                self._index = max(0, self._index - 1)
            elif command == "next":
                self._playing = False
                self._index = min(len(self._frames) - 1, self._index + 1)
            elif command == "restart":
                self._playing = False
                self._index = 0
            self._dirty = True
            self._cond.notify_all()

    def _render(self):
        with self._cond:
            frame = self._frames[self._index]
            index, count, playing = self._index, len(self._frames), self._playing
            error = self._error
            self._dirty = False
        decision, ending = frame["decision"], frame["ending"]
        self.scene.value = frame["image"]
        self.bars.value = probabilities_html(decision)
        self.counter.value = (
            f"<span style='color:{GRAY}'>Frame {index + 1} of {count} captured"
            f" · {'playing' if playing else 'paused'}</span>"
        )
        if error is not None:
            message, color = f"Stopped: {html.escape(str(error))}", RED
        elif ending is not None:
            message = (
                "Collision" if ending["collision"] else "Episode ended"
            ) + f" after {ending['decision']} decisions. Replay the captured episode."
            color = RED if ending["collision"] else NAVY
        elif decision is None:
            message, color = "Initial scene · waiting for the first policy decision.", GRAY
        else:
            message = f"Sampled {LABELS[decision['selected_action']]} · real HighwayEnv"
            color = NAVY
        self.status.value = f"<div style='color:{color};font-weight:bold'>{message}</div>"
        self.play_button.disabled = playing or error is not None
        self.pause_button.disabled = not playing
        self.previous_button.disabled = index == 0
        self.next_button.disabled = index >= count - 1

    def _capture(self):
        if self.drive.phase == "observe":
            decision = self.drive.decide()
            return [{"image": self.drive.frame, "decision": decision, "ending": None}]
        if self.drive.phase == "decided":
            record, images = self.drive.execute()
            return [
                {"image": image, "decision": record,
                 "ending": record if record["is_last"] and i == len(images) - 1 else None}
                for i, image in enumerate(images)
            ]
        return []

    def _run(self):
        while True:
            with self._cond:
                if self._closed:
                    return
                dirty = self._dirty
            if dirty:
                self._render()
            with self._cond:
                if self._closed:
                    return
                if not self._playing:
                    self._cond.wait_for(lambda: self._closed or self._dirty or self._playing)
                    continue
                if self._index < len(self._frames) - 1:
                    changed = self._cond.wait_for(
                        lambda: self._closed or self._dirty or not self._playing,
                        timeout=FRAME_SECONDS / self.speed.value,
                    )
                    if not changed:
                        self._index += 1
                        self._dirty = True
                    continue
                if self.drive.phase == "ended":
                    self._playing = False
                    self._dirty = True
                    continue
            try:
                frames = self._capture()
                with self._cond:
                    self._frames.extend(frames)
                    self._dirty = True
                    self._cond.notify_all()
            except Exception as error:
                with self._cond:
                    self._error = error
                    self._playing = False
                    self._dirty = True
