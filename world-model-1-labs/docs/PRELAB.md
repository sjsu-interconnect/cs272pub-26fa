# Before the lab: a ten-minute primer

You need the concepts below, **not a completed training run**. Lecture 1 has
two required labs: [Lab 1: Learn a World Model from Experience](../notebooks/1_learn_world_model.ipynb)
and [Lab 2: Imagine and Plan with MPC](../notebooks/2_imagine_and_plan.ipynb).
The local workflow uses one configuration and one generated dataset. Lab 1
trains and exports a compatible checkpoint. Lab 2 then freezes that model and
uses it for imagination and MPC. The Source Package contains neither the
generated dataset nor a checkpoint; follow `docs/FROM_SOURCE.md` before Run All.

**Vocabulary:** a *saved trajectory* is one saved simulator episode of observations and
actions; a *latent* is a learned vector of numbers representing an observation,
not an image or a hand-labeled simulator state. A *checkpoint* is saved network
weights, so you can use a model without training it. A *horizon* of six means
six future transitions, not six candidate plans. *Held-out* data was excluded
from fitting the model and is used here for inspection.

## 1. One saved transition

At time $t$, the agent receives observation $o_t$, takes action $a_t$,
and receives **post-action reward** $r_{t+1}$ and observation $o_{t+1}$.
The dataset stores

$$
(o_t,\ a_t,\ r_{t+1},\ o_{t+1},\ done_{t+1}).
$$

Actions are discrete driving commands. Reward is HighwayEnv's numerical
objective (mainly speed/right-lane preferences with collision/off-road effects),
not a safety score. `done` records that the episode ended, including a time
limit. Stored arrays may call the reward simply `reward`; it belongs to the
transition **after** the action, not the preceding observation.
Here, "real" means saved or newly stepped **HighwayEnv simulator** data,
not a physical car or an on-road safety experiment.

Each observation is **four stacked grayscale frames**, shape `(4, 64, 64)`,
normalized to `[0, 1]` for the network. A single still image has limited motion
information. Four recent frames give a feedforward network short-term motion
context: a substitute for memory, not a recurrent state or complete history.

## 2. Four small networks

The **encoder** compresses pixels into a 64-dimensional latent:

$$
z_t = E(o_t).
$$

The **dynamics** predicts the next latent from the current latent and action:

$$
\hat z_{t+1}=f(z_t,a_t).
$$

The **reward head** predicts the reward returned after that action:

$$
\hat r_{t+1}=R(z_t,a_t).
$$

A hat marks a **prediction**. For example, $\hat z_{t+1}$ is the predicted
next latent, whereas $E(o_{t+1})$ encodes the saved next observation.

**Lecture notation.** Lecture 1 writes the dynamics and reward heads together
as one world model, $M_\phi(z_t,a_t)\to(\hat z_{t+1},\hat r_{t+1})$; here $f$
and $R$ name its two heads. Lab 2 opens with a table connecting
each symbol to the code (`model.encode`, `model.predict_next`,
`model.predict_reward`).

The **decoder** $D$ maps a latent back to an image stack. It supports
reconstruction training and visual inspection. It does **not** select actions:
the planner uses dynamics and predicted rewards, not decoded pictures. Decoded
images are omitted from the required lab.
The encoder/decoder use convolutional networks (CNNs); dynamics/reward use
multilayer perceptrons (MLPs). They are feedforward networks: each call depends
on its supplied input, not a recurrent hidden memory.

## 3. What one-step training taught the model

Training uses real transitions from a fixed dataset, not model-generated
future targets. For each transition it compares:

| Loss | Prediction | Real-data target |
|---|---|---|
| Dynamics MSE | $f(E(o_t),a_t)$ | $E(o_{t+1})$, detached |
| Reconstruction MSE | $D(E(o_t))$ | $o_t$ |
| Reward MSE | $R(E(o_t),a_t)$ | $r_{t+1}$ |

MSE is **mean squared error**, the average squared difference between
prediction and target. MAE, used in a pixel diagnostic, is **mean absolute
error**. Lower means closer within the same metric; latent MSE and pixel MAE
have different units/scales and should not be compared numerically.

The weighted losses train the networks jointly. There is no policy/value
network or reinforcement-learning policy update in this training loop.

**Stop-gradient is not collapse prevention.**
`encoder(next_obs).detach()` prevents dynamics-loss gradients from flowing
through the target encoding. It prevents the target branch from immediately
moving to match the prediction; gradients still reach the encoder through the
prediction branch and other losses. A constant encoder can still be a
degenerate solution to the dynamics loss. Reconstruction and reward
supervision help anchor the representation. Detaching a target is not a
permanently frozen encoder.

The dataset stores `done`, but this model does **not** train a
done/continuation head. Keep that missing piece in mind when the lab turns to planning.

## 4. What changes during imagination

Lab 2 encodes **one real starting observation**. Then it reuses its own
predicted latent as the next input:

$$
\hat z_{t+2}=f(\hat z_{t+1},a_{t+1}),\qquad
\hat z_{t+3}=f(\hat z_{t+2},a_{t+2}),\ \ldots
$$

Actions are the saved trajectory actions of a held-out saved trajectory: actions
that were taken in the saved simulator trajectory, not generated by the world
model. Saved future observations are available to **measure errors**, but
never fed back to correct this rollout.
This is **open-loop** prediction. Re-encoding a real observation every step
would instead be a sequence of one-step predictions from each observed state
(sometimes called teacher forcing).
Question 1 reads saved trajectories, and the single candidate search in 2.2
does not step a live environment either; section 2.3 and Question 3 select
actions and step fresh HighwayEnv episodes.

The model was trained on encoded real observations, not all the imperfect
predicted inputs it may encounter recursively. Small errors can change later
inputs and propagate. **Recursive prediction error can accumulate with
horizon; inspect the measured trend and variability rather than expecting
monotonic growth.** Per-horizon error is not a running sum: a later prediction
may happen to lie closer to its target.

**Latent error:** latent MSE compares the predicted latent with the encoding
of the corresponding saved future observation, which is a reference in the
learned representation, not the physical ground-truth state. It is
decoder-independent but depends on the representation and its scale. A small
latent error does not by itself prove correct rewards or safe control, and an
evaluation with saved trajectory actions says nothing direct about other action
sequences the planner may try.

## 5. Which model does Lab 2 use?

Lab 2 uses the model-only checkpoint exported by Lab 1. Both notebooks read
`configs/lab.yaml` and `datasets/highway.npz`. The world-model parameters are
frozen during Lab 2; MPC searches for an action sequence at each real decision.

To change the experiment, edit `data.num_transitions` or `data.dataset_path` in
the one configuration before collecting. Change `EPOCHS` in Lab 1 to change
training time. Then rerun Lab 1 before Lab 2.

More data or training is not a safety certificate or a guaranteed control
improvement. Predicted return is not actual episode return.

After completing `docs/FROM_SOURCE.md`, start
[Lab 1](../notebooks/1_learn_world_model.ipynb), then continue to
[Lab 2](../notebooks/2_imagine_and_plan.ipynb).
