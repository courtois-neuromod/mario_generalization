import torch
import skvideo.io
import numpy as np
from tqdm import tqdm

from src.models import PPO


BUTTON_NAMES = ["run", "up", "down", "left", "right", "jump"]


def save_video(path, frames):
    """Save the frames array to video file."""
    writer = skvideo.io.FFmpegWriter(
        path,
        inputdict={"-r": "60"},
        outputdict={"-r": "60"},
    )
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()


def get_predictions_and_targets(data_file, run_list, model, batch_size, device):
    """Return the predictions and targets for the given runs and model.

    Args:
        data_file: h5py object with frames and actions per run.
        run_list: list of strings with the run names.
        model: torch model to use to predict the actions.
        batch_size: int, batch size to use.
        device: string, torch device to use.

    Returns:
        tuple, predictions and targets as numpy arrays.
    """
    predictions = []
    target_actions = []
    if len(run_list) > 1:
        run_list = tqdm(run_list, desc="Get model predictions")
    for run in run_list:
        frames = data_file[run]["frames"][:]
        target_actions.append(data_file[run]["actions"][:])
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            with torch.no_grad():
                preds = model(
                    torch.from_numpy(batch_frames.astype(np.float32)).to(device)
                )
            if isinstance(model, PPO):
                preds = preds[0]
            preds = preds.detach().cpu().numpy()
            predictions.append(preds)

    predictions = np.concatenate(predictions)
    target_actions = np.concatenate(target_actions)
    assert predictions.shape[0] == target_actions.shape[0]
    return predictions, target_actions
