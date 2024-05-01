import os
from pathlib import Path
from preprocessing.pose_estimation import PoseEstimator
from datasets_.kinetics_700 import KineticsVideoDataset
import pandas as pd
import torch

# Initialize PoseEstimator and KineticsVideoDataset
pose_estimator = PoseEstimator(device="cuda:0")
df = pd.read_csv("../data/kinetics_700/dancing.csv")
df["video_path"] = "../data/kinetics_700/videos/" + df["youtube_id"] + ".mp4"
dataset = KineticsVideoDataset(df)

keypoints_dir = Path("../data/keypoints")
keypoints_dir.mkdir(parents=True, exist_ok=True)

# Loop over the KineticsVideoDataset
for i in range(len(dataset)):
    video, label_id = dataset[i]
    if f"{dataset.meta.iloc[i]['youtube_id']}_0.mp4" in os.listdir("../data/keypoints"):
        print(f"Skipping video {i}")
        continue

    for frame_num, frame in enumerate(video):
        print(f"Processing frame {frame_num} of video {i}")
        keypoints = pose_estimator.predict([torch.from_numpy(frame).float()],
                                           save_path=keypoints_dir / f"{dataset.meta.iloc[i]['youtube_id']}_{frame_num}.pt")
