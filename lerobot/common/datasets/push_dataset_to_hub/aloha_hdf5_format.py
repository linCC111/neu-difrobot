#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
Contains utilities to process raw data format of HDF5 files like in: https://github.com/tonyzhaozh/act
"""

import gc
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    get_default_encoding,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def get_cameras(hdf5_data):
    # ignore depth channel, not currently handled
    # TODO(rcadene): add depth
    rgb_cameras = [key for key in hdf5_data["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118
    return rgb_cameras


def check_format(raw_dir) -> bool:
    # only frames from simulation are uncompressed
    compressed_images = "sim" not in raw_dir.name

    hdf5_paths = list(raw_dir.glob("episode_*.hdf5"))
    assert len(hdf5_paths) != 0
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as data:
            assert "/action" in data
            assert "/observations/qpos" in data

            assert data["/action"].ndim == 2
            assert data["/observations/qpos"].ndim == 2

            num_frames = data["/action"].shape[0]
            assert num_frames == data["/observations/qpos"].shape[0]

            for camera in get_cameras(data):
                assert num_frames == data[f"/observations/images/{camera}"].shape[0]

                if compressed_images:
                    assert data[f"/observations/images/{camera}"].ndim == 2
                else:
                    assert data[f"/observations/images/{camera}"].ndim == 4
                    b, h, w, c = data[f"/observations/images/{camera}"].shape
                    assert c < h and c < w, f"Expect (h,w,c) image format but ({h=},{w=},{c=}) provided."


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    # only frames from simulation are uncompressed
    compressed_images = "sim" not in raw_dir.name

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    num_episodes = len(hdf5_files)

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx in tqdm.tqdm(ep_ids):
        ep_path = hdf5_files[ep_idx]
        with h5py.File(ep_path, "r") as ep:
            num_frames = ep["/action"].shape[0]

            # last step of demonstration is considered done
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True

            state = torch.from_numpy(ep["/observations/qpos"][:])
            action = torch.from_numpy(ep["/action"][:])
            if "/observations/qvel" in ep:
                velocity = torch.from_numpy(ep["/observations/qvel"][:])
            if "/observations/effort" in ep:
                effort = torch.from_numpy(ep["/observations/effort"][:])

            ep_dict = {}

            for camera in get_cameras(ep):
                img_key = f"observation.images.{camera}"

                if compressed_images:
                    import cv2

                    # load one compressed image after the other in RAM and uncompress
                    imgs_array = []
                    for data in ep[f"/observations/images/{camera}"]:
                        imgs_array.append(cv2.imdecode(data, 1))
                    imgs_array = np.array(imgs_array)

                else:
                    # load all images in RAM
                    imgs_array = ep[f"/observations/images/{camera}"][:]

                if video:
                    # save png images in temporary directory
                    tmp_imgs_dir = videos_dir / "tmp_images"
                    save_images_concurrently(imgs_array, tmp_imgs_dir)

                    # encode images to a mp4 video
                    fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                    video_path = videos_dir / fname
                    encode_video_frames(tmp_imgs_dir, video_path, fps, **(encoding or {}))

                    # clean temporary images directory
                    shutil.rmtree(tmp_imgs_dir)

                    # store the reference to the video frame
                    ep_dict[img_key] = [
                        {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
                    ]
                else:
                    ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

            ep_dict["observation.state"] = state
            if "/observations/velocity" in ep:
                ep_dict["observation.velocity"] = velocity
            if "/observations/effort" in ep:
                ep_dict["observation.effort"] = effort
            ep_dict["action"] = action
            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
            ep_dict["next.done"] = done
            # TODO(rcadene): add reward and success by computing them in sim

            assert isinstance(ep_idx, int)
            ep_dicts.append(ep_dict)

        gc.collect()

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    if "observation.velocity" in data_dict:
        features["observation.velocity"] = Sequence(
            length=data_dict["observation.velocity"].shape[1], feature=Value(dtype="float32", id=None)
        )
    if "observation.effort" in data_dict:
        features["observation.effort"] = Sequence(
            length=data_dict["observation.effort"].shape[1], feature=Value(dtype="float32", id=None)
        )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 50

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    return hf_dataset, episode_data_index, info

def test():
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.compute_stats import compute_stats
    from lerobot.scripts.push_dataset_to_hub import save_meta_data
    #set data dir
    print("start dataset convert")
    repo_id = "aloha_sim_transfer_cube_scripted"
    root = "/home/lin/workspace/lerobot/dataset/sim_transfer_cube_scripted/"
    local_dir = root + repo_id
    raw_dir = Path(root)
    videos_dir = Path(local_dir + "/videos")
    #get hf dataset
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(raw_dir= raw_dir, videos_dir= videos_dir)
    #compute stats
    lerobot_dataset = LeRobotDataset.from_preloaded(repo_id= repo_id, hf_dataset= hf_dataset, episode_data_index= episode_data_index, info= info, videos_dir=videos_dir)
    stats = compute_stats(lerobot_dataset)
    #save to file
    hf_dataset = hf_dataset.with_format(None)
    hf_dataset.save_to_disk(local_dir + "/train")
    lerobot_dataset.stats = stats
    meta_data_dir = Path(local_dir + "/meta_data")
    save_meta_data(info=info, stats= stats, episode_data_index= episode_data_index, meta_data_dir= meta_data_dir)
    print("done")

if __name__ == "__main__":
    test()