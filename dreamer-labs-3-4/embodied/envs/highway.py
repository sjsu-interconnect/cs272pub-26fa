import gymnasium as gym
import numpy as np
import elements


TASK_MAP = {
    'fast-v0':          'highway-fast-v0',
    'highway-v0':       'highway-v0',
    'roundabout-v0':    'roundabout-v0',
    'intersection-v0':  'intersection-v0',
    'merge-v0':         'merge-v0',
    'racetrack-v0':     'racetrack-v0',
    'parking-v0':       'parking-v0',
    'exit-v0':          'exit-v0',
    'u-turn-v0':        'u-turn-v0',
}


class HighwayEnv:

    def __init__(self, task, seed=None, config=None, reward_mode='native'):
        import highway_env
        if reward_mode not in ('native', 'speed_collision'):
            raise ValueError(f'Unknown Highway reward mode: {reward_mode}')
        gym_task = TASK_MAP.get(task, task)
        self._env = gym.make(gym_task, config=config)
        self._reward_mode = reward_mode
        if reward_mode == 'speed_collision':
            expected = {
                'right_lane_reward': 0.0, 'lane_change_reward': 0.0,
                'normalize_reward': False, 'reward_speed_range': [20, 30],
            }
            active = {key: self._env.unwrapped.config[key] for key in expected}
            active['reward_speed_range'] = list(active['reward_speed_range'])
            speed = self._env.unwrapped.config['high_speed_reward']
            collision = self._env.unwrapped.config['collision_reward']
            if (active != expected or not np.isfinite(speed) or not np.isfinite(collision)
                    or speed < 0 or collision > 0):
                self._env.close()
                raise ValueError('Invalid speed_collision reward config')
        self._obs_shape = self._env.observation_space.shape
        self._act_n = int(self._env.action_space.n) \
            if hasattr(self._env.action_space, 'n') else None
        print(f'Highway env: {gym_task}, obs: {self._obs_shape}, '
              f'act: {self._act_n}')
        self._done = True

    @property
    def obs_space(self):
        return {
            "obs":         elements.Space(np.float32, self._obs_shape),
            "reward":      elements.Space(np.float32),
            "is_first":    elements.Space(bool),
            "is_last":     elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    @property
    def act_space(self):
        return {
            "action": elements.Space(np.int32, (), 0, self._act_n),
            "reset":  elements.Space(bool),
        }

    def step(self, action):
        if action["reset"] or self._done:
            obs, _ = self._env.reset()
            self._done = False
            return self._obs(obs, 0.0, is_first=True)
        obs, reward, terminated, truncated, _ = self._env.step(
            action["action"])
        if self._reward_mode == 'speed_collision':
            vehicle = self._env.unwrapped.vehicle
            forward_speed = vehicle.speed * np.cos(vehicle.heading)
            # Do not inherit the native reward's implicit on-road multiplier.
            config = self._env.unwrapped.config
            reward = config['high_speed_reward'] * np.clip((forward_speed - 20) / 10, 0, 1)
            reward += config['collision_reward'] * float(vehicle.crashed)
        self._done = terminated or truncated
        return self._obs(obs, reward,
                         is_last=self._done, is_terminal=terminated)

    def _obs(self, obs, reward,
             is_first=False, is_last=False, is_terminal=False):
        return {
            "obs":         obs.astype(np.float32),
            "reward":      np.float32(reward),
            "is_first":    is_first,
            "is_last":     is_last,
            "is_terminal": is_terminal,
        }

    def close(self):
        self._env.close()