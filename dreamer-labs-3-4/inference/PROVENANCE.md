# Provenance and limited distribution

The historical Experiment C checkpoint is step 24,996 with 999 optimizer
updates; `agent.pkl` SHA-256:
`a0903219ba173614a1e76e0e09a5f4b06674cefced81d26982cbd8f646c5cbb5`.
Its saved configuration is unchanged except for replacing the original
personal `logdir` with `bundled-checkpoint`. The checkpoint files are copies;
the original research evidence is unchanged.

Runtime code is copied from the verified locally corrected DreamerV3 working
tree based on commit `1bf6d51aca5b0fc9ef3a7017838d5b330ac13c77`;
that commit alone omits local corrections. The historical C25K evaluator was
an earlier version of the shared `optional_training/evaluate.py`; the latter
also supports custom reward weights.

`lab3_inference.py` drives one live episode a decision at a time. It loads the
trained model policy-only (without the full Agent's dummy training
initialization), checks the checkpoint hash and update counter, and uses the
evaluator's environment settings, `env.reset(seed)`, and counter-derived
stochastic sampling seeds. Each decision makes one policy call; the five Actor
probabilities are read from the posterior state that same call returned, so
the Actor is not sampled again. Each executed action is exactly one
`env.step`. For seeds 1000, 1001 and 1002, observations, probabilities, sampled
actions, rewards, speeds, positions and termination are identical to the
previous whole-episode runner and to its verified recordings (macOS ARM64 CPU,
JAX 0.7.1, HighwayEnv 1.12.1).

The driving view is a second, display-only HighwayEnv viewer, as in Lab 2: it
follows the ego car and keeps the road centered, and draws the simulator state
after each of the five 5 Hz physics steps of a decision. It does not step,
reseed or reconfigure the simulator, and the Actor never sees it. Rendering
uses SDL's `offscreen` driver, because HighwayEnv skips drawing under `dummy`.
`live_view.py` captures those returned frames and their corresponding policy
decision once; playback and seeking never resample or step the environment.
The initial frame is displayed before deferred policy loading and compilation.

An optional completed Lab 4 run is checked against its recorded configuration,
model-source hashes, checkpoint hash and update count. Its recorded reward
preset and weights configure the live environment. The bundled reward adapter
supports the current Lab 4 custom weights; the model and training code are
unchanged. Checkpoint hashes establish integrity, not trust: pickle files must
come from a trusted source.

The previous decision viewer, recordings, policy menu and optional training
reference are kept unchanged in
`backups/lab3-dreamer-inference-before-live-only-2026-11-12/` of the course
repository; they are not part of this package.

The included upstream notice is in `LICENSE`; dependencies are installed
separately under their own licenses. Permission for local changes, teaching
materials, trained checkpoint, and simulator-rendered media requires human
authorization before redistribution. This is not a public release.
