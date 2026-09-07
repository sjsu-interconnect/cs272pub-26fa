"""REINFORCE (Monte-Carlo policy gradient) on TrapGrid.

The objective is the expected return, J(theta) = v_{pi_theta}(s_0), and we climb
it with sampled episodes. For every step t of an episode we apply

    theta <- theta + alpha * gamma^t * G_t * grad_theta ln pi_theta(A_t | S_t)

Read the update as a vote: G_t says how good the rest of the episode turned out,
and grad ln pi says which way to move theta to make A_t more likely. A large
positive G_t makes the action we took more probable; a negative one makes it
less probable. Nothing here ever consults a value function or a table of Q
values -- the policy itself is what improves.

Run this file to train and plot the learning curve:

    python reinforce.py
"""

import numpy as np
import gymnasium as gym

# importing trap_grid also registers "TrapGrid-v0" with gymnasium
from trap_grid import ACTION_NAMES, GOAL_REWARD, TRAP_REWARD
from policy import FEATURE_NAMES, N_ACTIONS, N_FEATURES, action_probs, grad_log_pi

ALPHA = 0.05
GAMMA = 0.9
N_EPISODES = 600


def run_episode(env, theta, rng):
    """Follow pi_theta for one episode; return [(obs, action, reward), ...]."""
    obs, _ = env.reset()
    trajectory = []
    while True:
        probs = action_probs(obs, theta)
        action = rng.choice(N_ACTIONS, p=probs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        trajectory.append((obs, action, reward))
        obs = next_obs
        if terminated or truncated:
            return trajectory


def discounted_returns(rewards, gamma):
    """G_t = R_{t+1} + gamma * G_{t+1}, computed backwards through the episode."""
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    return returns


def reinforce(n_episodes=N_EPISODES, alpha=ALPHA, gamma=GAMMA, seed=0):
    """Train a softmax policy with REINFORCE. Returns (theta, episode returns)."""
    env = gym.make("TrapGrid-v0")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    theta = np.zeros(N_FEATURES)
    history = []

    for _ in range(n_episodes):
        trajectory = run_episode(env, theta, rng)
        returns = discounted_returns([r for _, _, r in trajectory], gamma)

        # one gradient-ascent step per visited state
        for t, ((obs, action, _), g_t) in enumerate(zip(trajectory, returns)):
            theta = theta + alpha * (gamma ** t) * g_t * grad_log_pi(obs, action, theta)

        history.append(sum(r for _, _, r in trajectory))

    return theta, np.array(history)


def plot_learning_curve(history, window=25, path="learning_curve.png"):
    import matplotlib.pyplot as plt

    smoothed = np.convolve(history, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(8, 4.5))
    plt.scatter(np.arange(len(history)), history, s=4, alpha=0.3,
                color="tab:blue", edgecolors="none", label="episode return")
    plt.plot(np.arange(window - 1, len(history)), smoothed, lw=2,
             color="tab:blue", label=f"{window}-episode average")
    plt.axhline(GOAL_REWARD, ls="--", lw=1, color="gray", label="reached the goal (+10)")
    plt.axhline(TRAP_REWARD, ls=":", lw=1, color="gray", label="fell in the trap (-10)")

    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title("REINFORCE on TrapGrid: the policy learns to reach the goal")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nsaved learning curve to {path}")
    plt.show()


def greedy_rollout(theta, seed=3):
    """Follow the most likely action from each state and print the grid.
    Evaluation use only."""
    env = gym.make("TrapGrid-v0", render_mode="ansi")
    obs, _ = env.reset(seed=seed)
    print(env.render())

    while True:
        probs = action_probs(obs, theta)
        action = int(np.argmax(probs))
        print(f"\n  pi(.|{tuple(int(x) for x in obs)}) = {np.round(probs, 3)}"
              f"  -> {ACTION_NAMES[action]}")
        obs, reward, terminated, truncated, _ = env.step(action)
        print(env.render())
        if terminated or truncated:
            print(f"\n  final reward {reward:+.1f}")
            return


if __name__ == "__main__":
    theta, history = reinforce()

    print(f"first 50 episodes: average return {history[:50].mean():+.2f}")
    print(f"last  50 episodes: average return {history[-50:].mean():+.2f}")

    print("\nlearned theta:")
    for name, w in zip(FEATURE_NAMES, theta):
        print(f"  {name:>12}: {w:+.2f}")
    print("\n  positive 'toward goal', negative 'toward trap' and 'hits wall' --")
    print("  the agent learned what to chase and what to avoid, in three numbers.")

    print("\ngreedy rollout under the learned policy:")
    greedy_rollout(theta)

    plot_learning_curve(history)
