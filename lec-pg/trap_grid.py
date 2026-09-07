"""TrapGrid: a tiny Gymnasium environment for the policy gradient lecture.

    .  .  G        G = goal   (+10, episode ends)
    .  X  .        X = trap   (-10, episode ends)
    .  .  .        any other step costs -0.1

The agent sees its own (row, col) position and picks one of four moves.
Moving off the grid leaves it where it was (and still costs a step).

Each episode starts from a random non-terminal cell.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

SIZE = 3
GOAL = (0, 2)
TRAP = (1, 1)

# action index -> (row change, col change)
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = ["up", "down", "left", "right"]

GOAL_REWARD = 10.0
TRAP_REWARD = -10.0
STEP_REWARD = -0.1


class TrapGridEnv(gym.Env):
    """Reach the goal, avoid the trap sitting in the middle of the grid."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, max_steps=20, render_mode=None):
        self.max_steps = max_steps
        self.render_mode = render_mode

        # an observation is the agent's (row, col); an action is one of 4 moves
        self.observation_space = spaces.MultiDiscrete([SIZE, SIZE])
        self.action_space = spaces.Discrete(4)

        self.free_cells = [
            (r, c)
            for r in range(SIZE)
            for c in range(SIZE)
            if (r, c) not in (GOAL, TRAP)
        ]
        self.pos = self.free_cells[0]
        self.t = 0

    def _obs(self):
        return np.array(self.pos, dtype=np.int64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.free_cells[int(self.np_random.integers(len(self.free_cells)))]
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        dr, dc = MOVES[int(action)]

        # walls: a move that would leave the grid keeps the agent in place
        row = min(max(self.pos[0] + dr, 0), SIZE - 1)
        col = min(max(self.pos[1] + dc, 0), SIZE - 1)
        self.pos = (row, col)
        self.t += 1

        if self.pos == GOAL:
            return self._obs(), GOAL_REWARD, True, False, {}
        if self.pos == TRAP:
            return self._obs(), TRAP_REWARD, True, False, {}

        truncated = self.t >= self.max_steps
        return self._obs(), STEP_REWARD, False, truncated, {}

    def render(self):
        rows = []
        for r in range(SIZE):
            cells = []
            for c in range(SIZE):
                if (r, c) == self.pos:
                    cells.append("A")
                elif (r, c) == GOAL:
                    cells.append("G")
                elif (r, c) == TRAP:
                    cells.append("X")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return "\n".join(rows)


gym.register(id="TrapGrid-v0", entry_point=TrapGridEnv)


# if __name__ == "__main__":
#     env = gym.make("TrapGrid-v0", render_mode="ansi")
#     obs, info = env.reset(seed=0)

#     print(f"observation space: {env.observation_space}")
#     print(f"action space:      {env.action_space}\n")
#     print(f"start: agent at row {obs[0]}, col {obs[1]}")
#     print(env.render())

#     rng = np.random.default_rng(0)
#     total = 0.0
#     while True:
#         action = int(rng.integers(4))
#         obs, reward, terminated, truncated, info = env.step(action)
#         total += reward
#         print(f"\n{ACTION_NAMES[action]:>5}  reward {reward:+.1f}")
#         print(env.render())
#         if terminated or truncated:
#             break

#     print(f"\nepisode return under a random policy: {total:+.1f}")
