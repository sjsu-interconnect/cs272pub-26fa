"""Non-destructive local training runs and concise notebook guidance."""

from pathlib import Path

import torch

from lab1.utils.model_handoff import save_personal_checkpoint

OBSERVATION_PREPROCESSING = {
    "type": "GrayscaleObservation",
    "weights": [0.2989, 0.5870, 0.1140],
    "normalization": "uint8 / 255.0",
    "frame_order": "oldest to newest",
    "scaling": 1.75,
    "actions": ["lane left", "idle", "lane right", "faster", "slower"],
}


class StudentWorkflowError(Exception):
    """Stop a notebook with an actionable message instead of a long traceback."""

    def _render_traceback_(self):
        return [str(self)]


def require_student_file(path, message):
    if not Path(path).is_file():
        raise StudentWorkflowError(message)


def new_training_attempt(run_directory):
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        attempt = directory / f"attempt_{number:02d}"
        try:
            attempt.mkdir()
        except FileExistsError:
            number += 1
        else:
            return attempt


def save_student_model(path, model, cfg, transitions, *, selected_epoch=None):
    destination = Path(path)
    if destination.exists():
        saved = torch.load(destination, map_location="cpu", weights_only=True)
        expected = model.state_dict()
        unchanged = (
            saved.get("training_transitions") == transitions
            and saved.get("epochs") == cfg.training.epochs
            and saved.get("seed") == cfg.seed
            and saved.get("model_state", {}).keys() == expected.keys()
            and all(
                torch.equal(saved["model_state"][name], value.detach().cpu())
                for name, value in expected.items()
            )
        )
        if not unchanged:
            raise StudentWorkflowError(
                "A different Lab 1 model already exists at this handoff path.\n"
                "It was not overwritten. Choose a new RUN_NAME in Lab 1 and set\n"
                "PERSONAL_MODEL_PATH to that run in Lab 2."
            )
        return destination
    save_personal_checkpoint(
        destination,
        model,
        cfg,
        training_transitions=transitions,
        epochs=cfg.training.epochs,
        seed=cfg.seed,
    )
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    payload["observation_preprocessing"] = OBSERVATION_PREPROCESSING.copy()
    payload["observation_preprocessing"]["scaling"] = cfg.env.scaling
    payload["training_config"] = cfg.to_dict()
    payload["selected_checkpoint_epoch"] = selected_epoch
    import hashlib

    dataset = Path(cfg.data.dataset_path)
    if dataset.is_file():
        payload["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    torch.save(payload, destination)
    return destination


def validate_student_preprocessing(path, cfg):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("This is not a Lab 1 model-only checkpoint.")
    preprocessing = payload.get("observation_preprocessing")
    if preprocessing is None:
        raise ValueError(
            "This older handoff lacks observation preprocessing/action metadata. "
            "Re-export with the updated Lab 1 notebook; the file was not changed."
        )
    if preprocessing != OBSERVATION_PREPROCESSING or cfg.env.scaling != 1.75:
        raise ValueError("Incompatible observation preprocessing or action mapping.")
    return payload
