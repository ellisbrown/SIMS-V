import os
from typing import Sequence, Optional, Dict, Any, Union

import imageio
import numpy as np


def save_frames_to_mp4(
    frames: Union[Sequence[np.ndarray], np.ndarray],
    file_path: str,
    fps: float,
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    if not isinstance(frames, np.ndarray):
        frames = np.array(frames)

    # Convert to list for imageio.mimwrite
    if isinstance(frames, np.ndarray):
        frames = list(frames)

    kwargs = {
        "fps": fps,
        "quality": 5,
        **(extra_kwargs if extra_kwargs is not None else {}),
    }
    imageio.mimwrite(file_path, frames, macro_block_size=1, **kwargs)  # type: ignore
