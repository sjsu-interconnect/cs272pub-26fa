"""Seed namespaces for fresh teaching runs, never the historical adapter."""

import hashlib
import json

PROTOCOL = "teaching-seeded-v1"
ENV_DOMAIN = 0x48575931
REPLAY_DOMAIN = 0x52504C31


def reset_seed(base, worker, episode):
    import numpy as np

    return int(np.random.SeedSequence(
        [base, ENV_DOMAIN, worker, episode]).generate_state(1, dtype=np.uint32)[0])


def replay_seed(base):
    import numpy as np

    return int(np.random.SeedSequence(
        [base, REPLAY_DOMAIN]).generate_state(1, dtype=np.uint32)[0])


def seed_training_environment(adapter, base, worker, journal):
    import gymnasium as gym
    import numpy as np

    episode = 0

    class SeededResets(gym.Wrapper):
        def reset(self, *, seed=None, options=None):
            nonlocal episode
            if seed is not None:
                raise ValueError("Teaching resets derive seeds from base/worker/episode")
            derived = reset_seed(base, worker, episode)
            observation, info = self.env.reset(seed=derived, options=options)
            raw = self.unwrapped
            vehicles = np.asarray([
                [*vehicle.position, vehicle.speed, vehicle.heading]
                for vehicle in raw.road.vehicles], dtype=np.float64)
            journal.parent.mkdir(parents=True, exist_ok=True)
            with journal.open("x" if episode == 0 else "a") as stream:
                stream.write(json.dumps({
                    "protocol": PROTOCOL, "base_seed": base, "worker": worker,
                    "episode": episode, "reset_seed": derived,
                    "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                    "vehicle_state_sha256": hashlib.sha256(vehicles.tobytes()).hexdigest(),
                }) + "\n")
            episode += 1
            return observation, info

    adapter._env = SeededResets(adapter._env)
