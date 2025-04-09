import retro
import retrowrapper
import numpy as np
import pickle as pk
from pathlib import Path


def add_unused_buttons(actions):
    """Add unused buttons to action vector."""
    return [actions[0], 0, 0, 0] + actions[1:]


def emulator_step(emulator, actions):
    """Step the emulator and returns the observation."""
    step_res = emulator.step(add_unused_buttons(actions))
    obs, _rew, term, trunc, info = step_res
    done = term or trunc
    if done:
        emulator.reset()
    return obs, done, info


def make_emulator(level):
    """Instantiates the emulator."""
    emulator = retrowrapper.RetroWrapper(
        "SuperMarioBros-Nes",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        state=level,
        render_mode=None,
    )
    return emulator


def replay_bk2(path, emulator, skip_first_step=True):
    """Replay a bk2 file and return the images as a numpy array
    of shape (n_frames, channels=3, width, height), actions a list of list of bool,
    done a list of bool.
    """
    movie = retro.Movie(path)
    emulator.initial_state = movie.get_state()
    emulator.reset()
    images = []
    done = []
    actions = []
    infos = []
    states = []

    if skip_first_step:
        movie.step()
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(emulator.num_buttons):
                keys.append(movie.get_key(i, p))
        actions.append(keys)
        obs, _rew, _term, _trunc, _info = emulator.step(keys)
        images.append(obs)
        done.append(_term or _trunc)
        infos.append(_info)
        states.append(np.fromstring(emulator.em.get_state(), np.int8))
    images = np.moveaxis(np.array(images, dtype=np.uint8), -1, 1)

    return images, actions, done, infos, states


def save_frames_actions(
    directory,
    frames,
    logits,
    actions,
    suffixe="",
):
    """Save the frames, logits and actions in a directory.

    Args:
        directory: str, the directory where to save the files
        frames: np.ndarray or list of np.ndarray, the frames to save
        logits: np.ndarray or list of np.ndarray, the logits to save
        actions: np.ndarray or list of np.ndarray, the actions to save
        suffixe: str, a suffixe to append to the filenames. Defaults to "".
    """
    if suffixe and suffixe[0] != "_":
        suffixe = "_" + suffixe
    frames = np.array(frames, dtype=np.uint8)
    actions = np.array(actions, dtype=bool)
    logits = np.array(logits, dtype=np.float16)
    directory = Path(directory)
    np.save(
        directory / f"generated_frames{suffixe}.npy",
        frames.astype(np.uint8),
    )
    np.save(
        directory / f"generated_actions{suffixe}.npy",
        actions.astype(bool),
    )
    np.save(
        directory / f"generated_logits{suffixe}.npy",
        logits,
    )


def save_score(directory, score_per_life, score_per_game, suffixe=""):
    """Save the scores in a directory.

    Args:
        directory: str, the directory where to save the files
        score_per_life: list of dict, the scores (max horizontal distance)
            per life
        score_per_game: list of dict, the scores per game (3 lives)
        suffixe: str, a suffixe to append to the filenames. Defaults to "".
    """
    directory = Path(directory)
    if suffixe and suffixe[0] != "_":
        suffixe = "_" + suffixe
    score_per_life_path = directory / f"generated_score_per_life{suffixe}.pkl"
    with open(score_per_life_path, "wb") as f:
        pk.dump(score_per_life, f)
    score_per_game_path = directory / f"generated_score_per_game{suffixe}.pkl"
    with open(score_per_game_path, "wb") as f:
        pk.dump(score_per_game, f)
