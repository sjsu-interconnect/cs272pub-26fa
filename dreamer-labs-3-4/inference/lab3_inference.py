"""Drive HighwayEnv live with a trained DreamerV3 checkpoint. No training.

`LiveDrive` runs one real decision at a time:

- `decide()` passes the current 5x5 Kinematics observation through the trained
  Encoder + RSSM and Actor once, reads the Actor's five action probabilities from
  the state that same call produced, and keeps the action the call sampled.
- `execute()` runs exactly that action with one HighwayEnv step and returns the
  simulator frames rendered while it executes.

The two calls must alternate, so no decision is executed twice or skipped. The
checkpoint is loaded read-only: parameters are never updated and no optimizer
exists. The model loading, sampling seeds and environment setup follow the
historical C25K evaluator.
"""

import contextlib
from dataclasses import dataclass
import hashlib
import io
import math
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent
RUN = ROOT / "checkpoint"
CHECKPOINT = RUN / "ckpt" / "20261110T204617F528108"
CHECKPOINT_SHA256 = "a0903219ba173614a1e76e0e09a5f4b06674cefced81d26982cbd8f646c5cbb5"
OPTIMIZER_UPDATES = 999
TRAINING_STEPS = 24996
REWARD_STYLE = "C"  # the reward this checkpoint was trained with
ACTION_NAMES = ("LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER")
DEFAULT_SEED = 1000
MAX_DECISIONS = 31  # the evaluator's limit; the 30 s episode ends at 30
FRAMES_PER_DECISION = 5  # simulation_frequency 5 Hz / policy_frequency 1 Hz
# Display camera, as in Lab 2: follows the ego, keeps the road vertically
# centered, so a lane change shows as the ego moving between lanes.
DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 180  # the three-lane road plus a margin
DISPLAY_SCALING = 10.0  # px per metre: about 110 m of road across the view


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prepare_process():
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    # HighwayEnv skips all drawing under SDL's "dummy" driver (black frames);
    # "offscreen" also opens no window but draws the scene.
    os.environ["SDL_VIDEODRIVER"] = "offscreen"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    if str(RUNTIME) not in sys.path:
        sys.path.insert(0, str(RUNTIME))


@dataclass(frozen=True)
class _Run:
    config_path: Path
    checkpoint: Path
    checkpoint_hash: str
    updates: int
    reward_style: str
    reward_weights: dict | None
    config_hash: str | None = None


def _select_run(path):
    if path is None:
        return _Run(RUN / "config.yaml", CHECKPOINT, CHECKPOINT_SHA256,
                    OPTIMIZER_UPDATES, REWARD_STYLE, None)

    from optional_training.common import load_student_run

    run, manifest, completed, _, checkpoint = load_student_run(path)
    sources = manifest["provenance"]["source_sha256"]
    for name in (
        "dreamerv3/agent.py", "dreamerv3/rssm.py", "dreamerv3/configs.yaml",
        "embodied/jax/agent.py", "embodied/jax/transform.py", "embodied/jax/internal.py",
    ):
        if sources.get(name) != sha256(RUNTIME / name):
            raise ValueError(f"Incompatible training model source: {name}")
    return _Run(run / "config.yaml", checkpoint, completed["checkpoint_sha256"],
                completed["optimizer_updates"], manifest["reward_style"],
                manifest.get("reward_weights"), manifest["config_sha256"])


class _Policy:
    """The trained World Model and Actor, read-only and compiled once."""

    def __init__(self, run):
        import elements
        import jax
        import jax.numpy as jnp
        import ninjax as nj
        import numpy as np
        from dreamerv3.agent import Agent as Model

        self.jax, self.np = jax, np
        if run.config_hash is not None and sha256(run.config_path) != run.config_hash:
            raise ValueError("Run configuration changed after selection.")
        config = elements.Config.load(str(run.config_path))
        if config.task != "highway_fast-v0":
            raise ValueError("Only the verified Highway Kinematics configuration is supported.")
        source = run.checkpoint / "agent.pkl"
        if sha256(source) != run.checkpoint_hash or not (run.checkpoint / "done").is_file():
            raise ValueError("Checkpoint is incomplete or its hash does not match.")
        # Pickle is executable; a matching hash does not make an untrusted run safe.
        with source.open("rb") as stream:
            saved = pickle.load(stream)
        self.updates = int(saved["counters"]["updates"])
        if self.updates != run.updates:
            raise ValueError("Checkpoint update counter does not match its run.")
        self.config_seed = config.seed
        model_config = elements.Config(
            **config.agent,
            **{
                key: config[key]
                for key in (
                    "logdir", "seed", "jax", "batch_size", "batch_length",
                    "replay_context", "report_length", "replica", "replicas",
                )
            },
        )
        obs_space = {
            "obs": elements.Space(np.float32, (5, 5)),
            "reward": elements.Space(np.float32),
            **{key: elements.Space(bool) for key in ("is_first", "is_last", "is_terminal")},
        }
        act_space = {"action": elements.Space(np.int32, (), 0, 5)}
        # Policy-only: skip the full Agent's training initialization.
        model = object.__new__(Model)
        Model.__init__(model, obs_space, act_space, model_config)
        with jax.default_device(jax.devices("cpu")[0]), jax.transfer_guard("allow"):
            self.params = jax.tree.map(jnp.asarray, saved["params"])

        def readonly(fn):
            pure = nj.pure(fn)
            return jax.jit(
                lambda p, key, *args: pure(p, *args, seed=key, create=False, modify=False)[1]
            )

        self._init = readonly(lambda: model.init_policy(1))
        policy = readonly(lambda carry, obs: model.policy(carry, obs, mode="eval"))
        probabilities = readonly(
            lambda feat: jnp.stack(
                [
                    model.pol(model.feat2tensor(feat), 1)["action"].prob(
                        jnp.full(feat["deter"].shape[:-1], a, jnp.int32)
                    )
                    for a in range(5)
                ],
                -1,
            )
        )
        # Compile ahead of time (nothing is executed), so the first decision
        # in class is not delayed by compilation.
        started = time.perf_counter()
        carry = self.initial_carry()
        template = self.observation(np.zeros((5, 5), np.float32), 0.0, True)
        key = self.key(0)
        self._policy = policy.lower(self.params, key, carry, template).compile()
        self._probabilities = probabilities.lower(self.params, key, carry[1]).compile()
        self.compile_seconds = time.perf_counter() - started

    def key(self, counter):
        """The evaluator's counter-derived sampling seed."""
        rng = self.np.random.default_rng(seed=[self.config_seed, int(counter)])
        return self.jax.numpy.asarray(
            rng.integers(0, self.np.iinfo(self.np.uint32).max, (2,), self.np.uint32)
        )

    def initial_carry(self):
        return self._init(self.params, self.key(0))

    def observation(self, raw_obs, reward, is_first):
        np = self.np
        return {
            "obs": np.asarray(raw_obs, np.float32)[None],
            "reward": np.asarray([reward], np.float32),
            "is_first": np.asarray([is_first]),
            "is_last": np.asarray([False]),
            "is_terminal": np.asarray([False]),
        }

    def act(self, carry, obs, counter):
        """One policy call: (next carry, sampled action, its 5 probabilities)."""
        np = self.np
        carry, action, _ = self._policy(self.params, self.key(counter), carry, obs)
        # Read-only: the posterior state this same call inferred and passed on.
        probs = np.asarray(
            self._probabilities(self.params, self.jax.random.PRNGKey(0), carry[1])
        )[0].astype(float)
        if not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1, atol=1e-6):
            raise RuntimeError("Invalid Actor probabilities.")
        return carry, int(np.asarray(action["action"])[0]), probs.tolist()


def _make_camera(raw):
    import numpy as np
    from highway_env.envs.common.graphics import EnvViewer

    class RoadCamera(EnvViewer):
        """Display-only second viewer with its own config copy (from Lab 2).

        It draws the simulator's current state; it never steps, reseeds or
        reconfigures the environment, and the Actor never sees its pictures.
        """

        def window_position(self):
            return np.array([self.env.vehicle.position[0], self.lateral_center])

    config = dict(raw.config)
    config.update(
        {
            "offscreen_rendering": True,
            "screen_width": DISPLAY_WIDTH,
            "screen_height": DISPLAY_HEIGHT,
            "scaling": DISPLAY_SCALING,
            "centering_position": [0.5, 0.5],
            "render_agent": False,
        }
    )
    camera = RoadCamera(raw, config=config)
    lanes = raw.road.network.lanes_list()
    camera.lateral_center = float(np.mean([lane.position(0.0, 0.0)[1] for lane in lanes]))
    return camera


class LiveDrive:
    """One live HighwayEnv episode driven by the trained DreamerV3 Actor.

    Phases: "observe" (next: decide) -> "decided" (next: execute) -> "observe"
    ... -> "ended". `history` holds every executed decision in order.
    """

    def __init__(self, seed=DEFAULT_SEED, *, checkpoint=None, defer_policy=False):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32).")
        _prepare_process()
        self.seed = seed
        started = time.perf_counter()
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                from optional_training import rewards

                self._rewards = rewards
                self._run = _select_run(checkpoint)
                self._policy = None if defer_policy else _Policy(self._run)
        except Exception as error:
            tail = "\n".join(log.getvalue().splitlines()[-20:])
            hint = " Install inference/requirements.txt." if isinstance(error, ImportError) else ""
            raise RuntimeError(f"Could not load the trained agent: {error}.{hint}\n{tail}") from error
        self.load_seconds = time.perf_counter() - started
        self.compile_seconds = None if defer_policy else self._policy.compile_seconds
        self.updates = None if defer_policy else self._policy.updates
        self._adapter = None
        self.reset()

    def reset(self):
        """Start the episode again from the same traffic (env.reset(seed))."""
        from embodied.envs.highway import HighwayEnv

        self.close()
        with contextlib.redirect_stdout(io.StringIO()):
            adapter = HighwayEnv(
                "fast-v0", **self._rewards.settings(
                    self._run.reward_style, self._run.reward_weights))
        self._adapter = adapter
        try:
            self._raw = self._rewards.verify_environment(
                adapter, self._run.reward_style, self._run.reward_weights)
        except Exception:
            self.close()
            raise
        self._raw_obs, _ = adapter._env.reset(seed=self.seed)
        adapter._done = False
        self._camera = _make_camera(self._raw)
        self._counter = (2000 + self.seed) * 1000
        self._carry = None if self._policy is None else self._policy.initial_carry()
        self._reward = 0.0
        self.decision = 0
        self.phase = "observe"
        self.pending = None
        self.history = []
        self.frame = self._render()
        self.speed = float(self._raw.vehicle.speed)
        return self.frame

    def decide(self):
        """Run the trained model once on the current observation; sample an action."""
        if self.phase != "observe":
            raise RuntimeError(f"Cannot decide during phase {self.phase!r}.")
        if self._policy is None:
            started = time.perf_counter()
            self._policy = _Policy(self._run)
            self.load_seconds = time.perf_counter() - started
            self.compile_seconds = self._policy.compile_seconds
            self.updates = self._policy.updates
            self._carry = self._policy.initial_carry()
        step = self.decision + 1
        if step > MAX_DECISIONS:
            raise RuntimeError("Episode exceeded the unchanged 30-second limit.")
        started = time.perf_counter()
        observation = self._policy.observation(self._raw_obs, self._reward, step == 1)
        counter = self._counter
        self._carry, action, probabilities = self._policy.act(
            self._carry, observation, counter
        )
        self._counter += 1
        self.decision = step
        self.pending = {
            "decision": step,
            "observation": self._policy.np.asarray(self._raw_obs).tolist(),
            "actor_action_probabilities": probabilities,
            "selected_action": action,
            "selected_action_name": ACTION_NAMES[action],
            "rng_counter": counter,
            "inference_seconds": time.perf_counter() - started,
        }
        self.phase = "decided"
        return dict(self.pending)

    def execute(self):
        """Run the decided action with exactly one env.step; return its frames."""
        if self.phase != "decided":
            raise RuntimeError(f"Cannot execute during phase {self.phase!r}.")
        np = self._policy.np
        decision = self.pending
        action = decision["selected_action"]
        raw = self._raw
        frames = []
        road_step = raw.road.step

        def step_and_render(dt):
            road_step(dt)
            frames.append(self._render())

        started = time.perf_counter()
        raw.road.step = step_and_render
        try:
            result = self._adapter.step({"reset": False, "action": action})
        finally:
            del raw.road.step  # restore the class method
        step_seconds = time.perf_counter() - started
        if len(frames) != FRAMES_PER_DECISION:
            raise RuntimeError("Unexpected number of simulator frames for one step.")
        self._raw_obs, self._reward = result["obs"], float(result["reward"])
        expected = self._rewards.training_score(raw, action, self._run.reward_style)
        if not math.isclose(self._reward, expected, abs_tol=4e-6):
            raise RuntimeError("Reward does not match the checkpoint's training design.")
        if not np.isfinite(self._raw_obs).all():
            raise RuntimeError("Non-finite observation.")
        record = {
            **decision,
            "next_observation": np.asarray(self._raw_obs).tolist(),
            "reward": self._reward,
            "collision": bool(raw.vehicle.crashed),
            "is_last": bool(result["is_last"]),
            "is_terminal": bool(result["is_terminal"]),
            "speed_mps": float(raw.vehicle.speed),
            "position": [float(value) for value in raw.vehicle.position],
            "step_seconds": step_seconds,
        }
        self.history.append(record)
        self.pending = None
        self.frame = frames[-1]
        self.speed = record["speed_mps"]
        self.phase = "ended" if record["is_last"] else "observe"
        return dict(record), frames

    def checkpoint_unchanged(self):
        return sha256(self._run.checkpoint / "agent.pkl") == self._run.checkpoint_hash

    def close(self):
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None

    def _render(self):
        from PIL import Image

        self._camera.display()
        buffer = io.BytesIO()
        Image.fromarray(self._camera.get_image()).save(buffer, format="PNG", compress_level=1)
        return buffer.getvalue()
