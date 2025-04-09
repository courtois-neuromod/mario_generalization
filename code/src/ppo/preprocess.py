import cv2
import numpy as np

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
