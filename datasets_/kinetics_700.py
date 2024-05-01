import os
from pathlib import Path

import albumentations as A
import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class KineticsVideoDataset(Dataset):

    def __init__(self, meta, transform=None):
        print("Before:", meta.shape[0])
        for i, row in meta.iterrows():
            if not os.path.exists(row['video_path']):
                print(row['video_path'])
                meta.drop(i, inplace=True)
        meta.reset_index(drop=True, inplace=True)
        print("After:", meta.shape[0])

        self.meta = meta

        if transform is None:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5)
            ], additional_targets={
                f'image{i}': 'image' for i in range(1, 8)
            })

        self.transform = transform

        self.labels = self.meta["label"].unique()
        self.labels2id = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video = []
        while True:
            try:
                file_path = self.meta['video_path'].iloc[idx]
                container = av.open(file_path)

                indices = sample_frame_indices(clip_len=8, frame_sample_rate=5,
                                               seg_len=container.streams.video[0].frames)
                video = read_video_pyav(container, indices)
                while video.shape[0] < 8:
                    video = np.vstack([video, video[-1:]])

            except Exception as e:
                print("loop Error: ", e)
                continue
            break

        if self.transform:
            transformed = apply_video_augmentations(video, self.transform)
            video = transformed

        return video, self.labels2id[self.meta['label'].iloc[idx]]

    def validate_videos(self):
        for i, row in self.meta.iterrows():
            if not os.path.exists(row['video_path']):
                print(row['video_path'])
                self.meta.drop(i, inplace=True)
                continue

            self.__getitem__(i)
        self.meta.reset_index(drop=True, inplace=True)
        return self.meta


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    while converted_len >= seg_len and clip_len > 1:
        clip_len -= 1
        converted_len = int(clip_len * frame_sample_rate)
    end_idx = converted_len
    start_idx = 0
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def apply_video_augmentations(video, transform):
    targets = {'image': video[0]}
    for i in range(1, video.shape[0]):
        targets[f'image{i}'] = video[i]
    transformed = transform(**targets)
    transformed = np.concatenate(
        [np.expand_dims(transformed['image'], axis=0)]
        + [np.expand_dims(transformed[f'image{i}'], axis=0) for i in range(1, video.shape[0])]
    )
    return transformed


class KineticsKeypointsDataset(Dataset):
    def __init__(self, meta, path: Path):
        self.path = path

        video_tags = set()
        for filename in os.listdir(self.path):
            tag = filename[:-5]
            video_tags.add(tag)

        self.video_tags = video_tags
        self.meta = meta

        print("Before:", self.meta.shape[0])
        to_drop = []
        for i, row in self.meta.iterrows():
            if row['youtube_id'] not in video_tags:
                print("Skipping ", row['youtube_id'])
                to_drop.append(i)

        self.meta.drop(to_drop, inplace=True)
        self.meta.reset_index(drop=True, inplace=True)
        print("After:", self.meta.shape[0])

        self.labels = self.meta["label"].unique()
        self.labels2id = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        keypoints = None
        for i in range(8):
            if not (self.path / f"{self.meta.iloc[idx]['youtube_id']}_{i}.pt").exists():
                # print(f"Not found {self.meta.iloc[idx]['youtube_id']}_{i}.pt")
                continue
            if keypoints is None:
                keypoints = torch.load(self.path / f"{self.meta.iloc[idx]['youtube_id']}_{i}.pt")[0]["keypoints"]
                continue
            keypoints = torch.cat(
                (keypoints, torch.load(self.path / f"{self.meta.iloc[idx]['youtube_id']}_{i}.pt")[0]["keypoints"]))

        keypoints = keypoints.cpu()
        if keypoints.shape[0] < 8:
            keypoints = torch.cat(
                (keypoints, torch.zeros(8 - keypoints.shape[0], keypoints.shape[1], keypoints.shape[2]).cpu()))

        return keypoints[:, :, :2].flatten(), self.labels2id[self.meta['label'].iloc[idx]]

# train_dataset = KineticsKeypointsDataset(meta=pd.read_csv("../data/kinetics_700/dancing.csv"), path=Path("../data/keypoints"))
# train_dataset[0]
