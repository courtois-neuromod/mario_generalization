import retro
import retrowrapper
import cv2
import numpy as np
import torch.multiprocessing as mp
from collections import deque

from src.ppo.emulation import add_unused_buttons


def complex_movement_to_button_presses(action):
    """Convert an action in COMPLEX_MOVEMENT moveset to a binary vector of button presses."""

    # Buttons : [run, up, down, left, right, jump]
    # COMPLEX_MOVEMENT : (A is jump, B is run)
    # 0    ['NOOP']
    # 1    ['right']
    # 2    ['right', 'A']
    # 3    ['right', 'B']
    # 4    ['right', 'A', 'B']
    # 5    ['A']
    # 6    ['left']
    # 7    ['left', 'A']
    # 8    ['left', 'B']
    # 9    ['left', 'A', 'B']
    # 10   ['down']
    # 11   ['up']

    buttons = np.zeros(6, dtype=bool)
    if action in (1, 2, 3, 4):
        buttons[4] = True  # right
    if action in (6, 7, 8, 9):
        buttons[3] = True  # left
    if action in (2, 4, 5, 7, 9):
        buttons[5] = True  # jump
    if action in (3, 4, 8, 9):
        buttons[0] = True  # run
    if action == 10:
        buttons[2] = True  # down
    if action == 11:
        buttons[1] = True  # up
    return buttons


def preprocess_frames(frame_list, n_frame, downsample):
    """Prepprocess the raw frames for the PPO.
    The max of the two last frames is kept for every 'dowsample' frames.
    The frames are resized to 84x84, grayscaled,
    and rescaled to floats in the range [0,1].
    """
    frame_list = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frame_list]
    frame_list = [cv2.resize(f, (84, 84)) for f in frame_list]
    frame_list = np.array(frame_list, dtype=np.float32) / 255.0
    out_frames = []
    for i in range(n_frame - 1, -1, -1):
        frame = np.stack(
            (frame_list[-i * downsample - 2], frame_list[-i * downsample - 1])
        ).max(axis=0)
        out_frames.append(frame)
    return out_frames


def create_train_env(level, int_path, player_actions):
    world = level[1]
    stage = level[3]
    retro.data.Integrations.add_custom_path(int_path)
    env = retrowrapper.RetroWrapper(
        "SuperMarioBros-Nes",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        state=f"Level{world}-{stage}",
        render_mode=None,
    )
    env = CustomWrapper(env, player_actions)
    return env


class CustomWrapper:
    """Wrapper around the retro environment to preprocess frames and compute reward."""

    def __init__(self, env, player_actions, n_frame=4, downsample=4, seed=2024):
        self.env = env
        self.n_frame = n_frame
        self.player_actions = player_actions
        self.downsample = downsample
        self.frame_stack = deque([], maxlen=n_frame * downsample)
        self.curr_score = 0.0
        self.curr_lives = 2
        self.last_x = 0
        self.last_time = None
        self.rng = np.random.default_rng(seed=seed)

    def step(self, action):
        action = complex_movement_to_button_presses(action)
        action = add_unused_buttons(list(action))
        total_reward = 0
        for _ in range(self.downsample):
            obs, reward, term, trunc, info = self.env.step(action)
            self.frame_stack.append(obs.copy())
            done = term or trunc
            reward = 0
            # time penalty
            reward += (
                min(info["time"] - self.last_time, 0)
                if self.last_time is not None
                else 0
            )
            self.las_time = info["time"]
            # movement reward
            x_pos = 256 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
            diff_x = x_pos - self.last_x
            reward += diff_x if -5 <= diff_x <= 5 else 0
            self.last_x = x_pos
            # death and game over penalty
            if info["lives"] < self.curr_lives:
                reward -= 15
                self.curr_lives = info["lives"]
            reward = max(min(reward, 15), -15)
            # score reward
            reward += min((info["score"] - self.curr_score) / 4.0, 50)
            self.curr_score = info["score"]
            if done:
                if info["lives"] < 0:
                    reward -= 50
                total_reward += reward
                obs = self.reset()
                return obs, total_reward / 10.0, done, info
            total_reward += reward
        obs = preprocess_frames(self.frame_stack, self.n_frame, self.downsample)
        return obs, total_reward / 10.0, done, info

    def reset(self):
        n_actions = self.rng.integers(int(0.8 * len(self.player_actions)))
        self.curr_lives = 2
        self.curr_score = 0
        self.last_x = 0
        self.last_time = None
        obs, _ = self.env.reset()
        for _ in range(self.n_frame * self.downsample):
            self.frame_stack.append(obs.copy())
        for i in range(n_actions):
            obs, rew, term, trunc, info = self.env.step(self.player_actions[i])
            self.frame_stack.append(obs)
        return preprocess_frames(self.frame_stack, self.n_frame, self.downsample)


class MultipleEnvironments:
    """Class to spawn multiple environments in parrallel."""

    def __init__(
        self, player_actions_dict, int_path=None, use_gym_super_mario_bros=False
    ):
        self.agent_conns, self.env_conns = zip(
            *[mp.Pipe() for _ in range(len(player_actions_dict))]
        )
        self.envs = [
            create_train_env(lvl, int_path, player_actions)
            for lvl, player_actions in player_actions_dict.items()
        ]
        self.num_states = 4  # self.envs[0].observation_space.shape[0]
        self.num_actions = 12
        for index in range(len(self.envs)):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    def run(self, index):
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(action))
            elif request == "reset":
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError
