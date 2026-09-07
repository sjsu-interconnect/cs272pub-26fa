# Policy Gradient — REINFORCE on a 3×3 grid

A small, self-contained demo for the policy gradient lecture. The agent learns a
policy **directly**, with no value function and no table of Q values, on a custom
Gymnasium environment.

The whole policy is **three numbers** in $\theta$.

## Files

| file | what it is |
|---|---|
| `trap_grid.py` | the environment, registered with Gymnasium as `TrapGrid-v0` |
| `policy.py` | the features `phi(s,a)`, the softmax policy, and gradient calculation |
| `reinforce.py` | the REINFORCE training loop, the learning curve, a greedy rollout (for evaluation) |

Requires `numpy`, `gymnasium`, and `matplotlib`.

## Running it

```bash
python reinforce.py   # train, print the learned theta, plot learning_curve.png
```

`trap_grid.py` has a random-policy demo at the bottom that is commented out — uncomment it to watch an untrained agent wander.

To better understand the feature encoding, you can run `policy.py` independently.

```bash
python policy.py      # how theta shapes action probabilities (no learning yet)
```

## The environment

```
        col 0   col 1   col 2
row 0     .       .       G
row 1     .       X       .
row 2     .       .       .
```

| | |
|---|---|
| `G` goal at `(0, 2)` | reward **+10**, episode ends |
| `X` trap at `(1, 1)` | reward **−10**, episode ends |
| any other step | reward **−0.1** |
| step limit | 20 steps, then truncated |

A move that would leave the grid leaves the agent where it was and still costs a step.

**Episodes start from a random non-terminal cell** (7 of them). 

## State and action encoding

**State.** An observation is the agent's position as `np.array([row, col])`:

```python
observation_space = spaces.MultiDiscrete([3, 3])
```

Note that **row indices grow downward**, so "up" decreases the row. This is the
usual source of sign confusion when reading the code.

**Action.** `spaces.Discrete(4)`, with these move vectors (`MOVES` in
`trap_grid.py`):

| index | name | `(Δrow, Δcol)` |
|---|---|---|
| 0 | up | `(-1, 0)` |
| 1 | down | `(1, 0)` |
| 2 | left | `(0, -1)` |
| 3 | right | `(0, 1)` |

Each move vector already has length 1, which matters for the features below.

## The feature vector `phi(s, a)`

This is the heart of the demo. `phi(s, a)` describes an action **by what it
does**, not by which cell we are standing in. It returns three numbers:

| # | name | meaning |
|---|---|---|
| 0 | `toward goal` | +1 if the move points straight at the goal, −1 if straight away |
| 1 | `toward trap` | the same, measured against the trap |
| 2 | `hits wall` | 1 if the move would bump into the edge of the grid, else 0 |

Features 0 and 1 are computed as a dot product of two **unit** vectors — the
direction the action moves us, and the direction the target lies in:

```python
np.dot(direction, _unit(np.array(GOAL, dtype=float) - pos))
```

Since both have length 1, the dot product is exactly the **cosine of the angle**
between them, so the value always lands in [−1, +1]. `_unit` divides out the
distance so the feature says *which way* the goal is, never *how far* — without
it, the same "heading for the goal" situation would score twice as strongly in a
far cell as in a near one.


### Why only three numbers work

The features are **relative to the agent** — they are computed from where the
goal and trap lie *from here*, never from which cell this is. So whatever
`theta` learns in one cell applies in all of them. That is the point of function
approximation, and it is why a 3-number policy covers all 7 non-terminal cells.

The trade-off worth naming out loud: these features are **hand-designed**.
REINFORCE still has to discover the signs and magnitudes, but it is not
discovering the representation. Choosing `phi` is the job a neural network would
take over.

## The policy

Softmax over linear action preferences:

```
pi_theta(a|s) = exp(phi(s,a)ᵀ theta) / Σ_b exp(phi(s,b)ᵀ theta)
```

The policy is *computed* from `theta` every time it is queried — nothing is
stored per state. The gradient REINFORCE needs is

```
∇_theta ln pi_theta(a|s) = phi(s,a) − Σ_b pi(b|s) phi(s,b)
                         = phi(s,a) − E_pi[phi(s,·)]
```

which reads as: push `theta` toward the features of the action we actually took,
and away from the average action under the current policy.

## The algorithm

For every step `t` of a sampled episode:

```
theta ← theta + alpha · gamma^t · G_t · ∇_theta ln pi_theta(A_t | S_t)
```

`G_t` is the discounted return from step `t` onward. Read it as a vote: `G_t`
says how well the rest of the episode turned out, and `∇ ln pi` says which way to
move `theta` to make that action more likely. Positive return → make it more
probable; negative → less.

Defaults are `alpha = 0.05`, `gamma = 0.9`, 600 episodes.

<!-- ## What you should see

Average return climbs from about **+4.7** over the first 50 episodes to **+9.5**
over the last 50, and the learned weights come out roughly:

```
   toward goal: +5.50
   toward trap: -6.64
     hits wall: -7.77
```

which reads in plain English: **chase the goal, avoid the trap, don't walk into
walls.** The greedy rollout then routes around the trap to the goal.

In `learning_curve.png`, the dots sit in two bands — +10 for episodes that
reached the goal, −10 for episodes that hit the trap — and the −10 band thins out
as the policy improves. -->

