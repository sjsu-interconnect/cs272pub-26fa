"""A softmax policy that is *computed* from features, not looked up in a table.

The policy is the softmax over linear action preferences:

    pi_theta(a | s) = exp(phi(s,a)^T theta) / sum_b exp(phi(s,b)^T theta)

so theta decides how much the agent likes each action, and moving theta moves
every probability. The gradient we need for REINFORCE is

    grad_theta ln pi_theta(a | s) = phi(s,a) - sum_b pi_theta(b|s) phi(s,b)
                                  = phi(s,a) - E_pi[ phi(s, .) ]

which reads as: push theta toward the feature vector of the action we actually took, and away from the average action under the current policy.

The whole policy here is defined by THREE numbers (a theta vector).

Run this file directly to see how theta shapes the action probabilities.
"""

import numpy as np

from trap_grid import ACTION_NAMES, GOAL, MOVES, SIZE, TRAP

FEATURE_NAMES = ["toward goal", "toward trap", "hits wall"]
N_FEATURES = len(FEATURE_NAMES)
N_ACTIONS = len(MOVES)


def _unit(v):
    """v scaled to length 1 (the zero vector is left alone)."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def phi(obs, a):
    """Feature vector for taking action `a` at position `obs`.

    Three numbers, describing the action by what it does rather than by which cell we are standing in:

        0. "toward goal"  +1 if the move points straight at the goal, -1 if straight away
        1. "toward trap"  same idea, measured against the trap
        2. "hits wall"    1 if the move would bump into the edge of the grid, else 0

    Features 0 and 1 are dot products of two unit vectors (the direction the action moves us, and the direction the target lies in. So each one is the cosine of the angle between them, landing in [-1, +1].
    """
    pos = np.asarray(obs, dtype=float)
    direction = np.array(MOVES[int(a)], dtype=float)
    ahead = pos + direction
    hits_wall = not (0 <= ahead[0] < SIZE and 0 <= ahead[1] < SIZE)

    return np.array([
        np.dot(direction, _unit(np.array(GOAL, dtype=float) - pos)),
        np.dot(direction, _unit(np.array(TRAP, dtype=float) - pos)),
        1.0 if hits_wall else 0.0,
    ])


def action_probs(obs, theta):
    """pi_theta(. | obs), computed by softmax over the linear preferences."""
    prefs = np.array([np.dot(phi(obs, a), theta) for a in range(N_ACTIONS)])
    exp_prefs = np.exp(prefs - prefs.max())  # shift by the max for stability
    return exp_prefs / exp_prefs.sum()


def grad_log_pi(obs, a, theta):
    """grad_theta ln pi_theta(a | obs) = phi(s,a) - sum_b pi(b|s) phi(s,b)."""
    probs = action_probs(obs, theta)
    expected_phi = sum(probs[b] * phi(obs, b) for b in range(N_ACTIONS))
    return phi(obs, a) - expected_phi


def describe(obs, theta):
    """Print the features, preferences and probabilities at one position."""
    probs = action_probs(obs, theta)
    print(f"  position {tuple(int(x) for x in obs)}")
    header = "  ".join(f"{n:>11}" for n in FEATURE_NAMES)
    print(f"    {'action':>5}  {header}   {'pref':>6}  {'pi(a|s)':>7}")
    for a in range(N_ACTIONS):
        f = phi(obs, a)
        feats = "  ".join(f"{x:>11.2f}" for x in f)
        print(f"    {ACTION_NAMES[a]:>5}  {feats}   {np.dot(f, theta):>6.2f}  {probs[a]:>7.3f}")


if __name__ == "__main__":
    print(__doc__.split("Run this file")[0].rstrip())

    print("\n" + "=" * 70)
    print("theta = [0, 0, 0]  ->  every action equally likely")
    print("=" * 70)
    describe((2, 1), np.zeros(N_FEATURES))

    print("\n" + "=" * 70)
    print("theta = [2, 0, 0]  ->  'head toward the goal', ignoring the trap")
    print("=" * 70)
    describe((2, 1), np.array([2.0, 0.0, 0.0]))
    print("  note: 'up' from (2,1) walks straight into the trap.")

    print("\n" + "=" * 70)
    print("theta = [2, -2, -1]  ->  toward the goal, away from the trap, no walls")
    print("=" * 70)
    describe((2, 1), np.array([2.0, -2.0, -1.0]))
    print("  the same three numbers now prefer 'right', which is safe.")
