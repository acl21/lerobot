#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Port the CALVIN play dataset (npz frames + ep_start_end_ids.npy) to LeRobot format.

CALVIN stores one interaction timestep per npz file (e.g. ``episode_0001234.npz``), indexed by a
global frame id. Continuous teleoperation segments are defined in ``ep_start_end_ids.npy`` as
``[start_frame_id, end_frame_id]`` pairs (both inclusive).

Dataset schema: https://github.com/mees/calvin/blob/main/dataset/README.md
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds

CALVIN_FPS = 30
CALVIN_ROBOT_TYPE = "Franka"
CALVIN_DEFAULT_TASK = "play"

# CALVIN npz keys -> LeRobot feature keys
CALVIN_NPZ_KEYS = ("rgb_static", "rgb_gripper", "robot_obs", "scene_obs", "rel_actions")

# Dataset schema adapted from: https://github.com/mees/calvin/blob/main/dataset/README.md
CALVIN_FEATURES = {
    "observation.images.static": {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.gripper": {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (15,),
        "names": {
            "axes": [
                "ee_x",
                "ee_y",
                "ee_z",
                "ee_roll",
                "ee_pitch",
                "ee_yaw",
                "gripper_width",
                "joint_0",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper_action",
            ],
        },
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (24,),
        "names": {
            "axes": [
                "sliding_door",
                "drawer",
                "button",
                "switch",
                "lightbulb",
                "green_light",
                "red_block_x",
                "red_block_y",
                "red_block_z",
                "red_block_roll",
                "red_block_pitch",
                "red_block_yaw",
                "blue_block_x",
                "blue_block_y",
                "blue_block_z",
                "blue_block_roll",
                "blue_block_pitch",
                "blue_block_yaw",
                "pink_block_x",
                "pink_block_y",
                "pink_block_z",
                "pink_block_roll",
                "pink_block_pitch",
                "pink_block_yaw",
            ],
        },
    },
    # rel_actions from CALVIN: relative EE pose + gripper
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": [
                "delta_ee_x",
                "delta_ee_y",
                "delta_ee_z",
                "delta_ee_roll",
                "delta_ee_pitch",
                "delta_ee_yaw",
                "gripper_action",
            ],
        },
    },
}


def lookup_naming_pattern(dataset_dir: Path) -> tuple[tuple[Path, str], int]:
    """Return CALVIN npz naming pattern and zero-padding width."""
    for entry in os.scandir(dataset_dir):
        filename = Path(entry.path)
        if filename.suffix == ".npz" and "episode" in filename.stem:
            aux_naming_pattern = re.split(r"\d+", filename.stem)
            naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
            n_digits = len(re.findall(r"\d+", filename.stem)[0])
            return naming_pattern, n_digits

    raise FileNotFoundError(f"No CALVIN npz files found in {dataset_dir}")


def get_frame_path(naming_pattern: tuple[Path, str], n_digits: int, frame_id: int) -> Path:
    prefix, suffix = naming_pattern
    return Path(f"{prefix}{frame_id:0{n_digits}d}{suffix}")


def load_episode_boundaries(raw_dir: Path) -> np.ndarray:
    ep_path = raw_dir / "ep_start_end_ids.npy"
    if not ep_path.exists():
        raise FileNotFoundError(
            f"Missing {ep_path}. Expected CALVIN split folder with ep_start_end_ids.npy "
            "(e.g. task_D_D/training)."
        )
    boundaries = np.load(ep_path)
    if boundaries.ndim != 2 or boundaries.shape[1] != 2:
        raise ValueError(f"Expected ep_start_end_ids.npy with shape (N, 2), got {boundaries.shape}")
    return boundaries


def load_scene_info(raw_dir: Path) -> dict[str, list[int]] | None:
    scene_path = raw_dir / "scene_info.npy"
    if not scene_path.exists():
        logging.warning("scene_info.npy not found, skipping scene metadata.")
        return None
    return np.load(scene_path, allow_pickle=True).item()


def load_language_annotations(raw_dir: Path) -> dict | None:
    for lang_path in (
        raw_dir / "lang_annotations" / "auto_lang_ann.npy",
        raw_dir / "auto_lang_ann.npy",
    ):
        if lang_path.exists():
            return np.load(lang_path, allow_pickle=True).item()
    logging.warning("auto_lang_ann.npy not found, using default task for all episodes.")
    return None


def load_calvin_frame(frame_path: Path) -> dict[str, np.ndarray]:
    data = np.load(frame_path)
    missing = [key for key in CALVIN_NPZ_KEYS if key not in data]
    if missing:
        raise KeyError(f"Missing keys {missing} in {frame_path}")

    return {
        "observation.images.static": data["rgb_static"],
        "observation.images.gripper": data["rgb_gripper"],
        "observation.state": data["robot_obs"].astype(np.float32),
        "observation.environment_state": data["scene_obs"].astype(np.float32),
        "action": data["rel_actions"].astype(np.float32),
    }


def generate_lerobot_frames(
    raw_dir: Path,
    start_frame_id: int,
    end_frame_id: int,
    naming_pattern: tuple[Path, str],
    n_digits: int,
):
    for frame_id in range(start_frame_id, end_frame_id + 1):
        frame_path = get_frame_path(naming_pattern, n_digits, frame_id)
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing CALVIN frame file: {frame_path}")

        frame = load_calvin_frame(frame_path)
        frame["task"] = "play"
        yield frame


def save_source_episode_manifest(
    manifest_path: Path,
    episodes: list[dict],
    raw_dir: Path,
) -> None:
    manifest = {
        "source": "calvin",
        "raw_dir": str(raw_dir.resolve()),
        "episodes": episodes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)


def port_calvin(
    raw_dir: Path,
    repo_id: str,
    push_to_hub: bool = False,
    episode_start: int | None = None,
    episode_end: int | None = None,
):
    """Port CALVIN npz play data to LeRobot format.

    Args:
        raw_dir: Path to a CALVIN ``training/`` or ``validation/`` folder.
        repo_id: Hugging Face dataset repo id (e.g. ``user/calvin_d_d``).
        push_to_hub: Upload dataset to the Hub when porting completes.
        episode_start: First episode index from ``ep_start_end_ids.npy`` to port (inclusive).
        episode_end: Last episode index to port (exclusive). ``None`` ports until the end.
    """
    raw_dir = raw_dir.resolve()
    if not raw_dir.is_dir():
        raise NotADirectoryError(raw_dir)

    boundaries = load_episode_boundaries(raw_dir)
    naming_pattern, n_digits = lookup_naming_pattern(raw_dir)

    ep_start = 0 if episode_start is None else episode_start
    ep_end = len(boundaries) if episode_end is None else episode_end
    if ep_start < 0 or ep_start >= len(boundaries):
        raise ValueError(f"episode_start={ep_start} out of range [0, {len(boundaries)})")
    if ep_end <= ep_start or ep_end > len(boundaries):
        raise ValueError(f"episode_end={ep_end} out of range ({ep_start}, {len(boundaries)}]")

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=CALVIN_ROBOT_TYPE,
        fps=CALVIN_FPS,
        features=CALVIN_FEATURES,
    )

    source_episodes: list[dict] = []
    start_time = time.time()
    num_episodes = ep_end - ep_start

    logging.info(f"Porting {num_episodes} CALVIN episodes from {raw_dir}")

    for local_ep_idx, ep_idx in enumerate(range(ep_start, ep_end)):
        start_frame_id, end_frame_id = boundaries[ep_idx]
        start_frame_id = int(start_frame_id)
        end_frame_id = int(end_frame_id)

        elapsed_time = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)
        logging.info(
            f"{local_ep_idx} / {num_episodes} episodes processed "
            f"(CALVIN frames {start_frame_id}-{end_frame_id}, "
            f"after {d} days, {h} hours, {m} minutes, {s:.3f} seconds)"
        )

        for frame in generate_lerobot_frames(
            raw_dir,
            start_frame_id,
            end_frame_id,
            naming_pattern,
            n_digits,
        ):
            lerobot_dataset.add_frame(frame)

        lerobot_dataset.save_episode()

        source_episodes.append(
            {
                "lerobot_episode_index": lerobot_dataset.meta.total_episodes - 1,
                "calvin_episode_index": ep_idx,
                "calvin_start_frame_id": start_frame_id,
                "calvin_end_frame_id": end_frame_id,
                "calvin_scene": "d",
                "task": "play",
            }
        )

    lerobot_dataset.finalize()

    save_source_episode_manifest(
        lerobot_dataset.meta.root / "meta" / "calvin_source_episodes.json",
        source_episodes,
        raw_dir,
    )

    if push_to_hub:
        lerobot_dataset.push_to_hub(tags=["calvin"], private=False)


def validate_dataset(repo_id: str) -> None:
    """Sanity check that metadata can be loaded and all files are present."""
    meta = LeRobotDatasetMetadata(repo_id)

    if meta.total_episodes == 0:
        raise ValueError("Number of episodes is 0.")

    manifest_path = meta.root / "meta" / "calvin_source_episodes.json"
    if not manifest_path.exists():
        raise ValueError(f"CALVIN source manifest is missing: {manifest_path}")

    for ep_idx in range(meta.total_episodes):
        data_path = meta.root / meta.get_data_file_path(ep_idx)
        if not data_path.exists():
            raise ValueError(f"Parquet file is missing in: {data_path}")

        for vid_key in meta.video_keys:
            vid_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
            if not vid_path.exists():
                raise ValueError(f"Video file is missing in: {vid_path}")


def main():
    parser = argparse.ArgumentParser(description="Port CALVIN npz play data to LeRobot format.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="CALVIN split folder containing npz frames (e.g. path/to/task_D_D/training).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: user/dataset_name.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload dataset to the Hugging Face Hub after porting.",
    )
    parser.add_argument(
        "--episode-start",
        type=int,
        default=None,
        help="First episode index from ep_start_end_ids.npy to port (inclusive).",
    )
    parser.add_argument(
        "--episode-end",
        type=int,
        default=None,
        help="Last episode index to port (exclusive). Defaults to all episodes.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    port_calvin(**vars(args))


if __name__ == "__main__":
    main()

# python examples/port_datasets/port_calvin.py --raw-dir /home/lagandua/data/task_D_D_224/training --repo-id acl21/calvin_d_d_224_training