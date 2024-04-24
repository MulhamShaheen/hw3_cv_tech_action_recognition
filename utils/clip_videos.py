import os

import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def create_clips_from_vids(video_dir, dataset_df: pd.DataFrame, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, row in dataset_df.iterrows():
        video_path = video_dir + row['youtube_id'] + '.mp4'
        label = row['label']
        start_time = row['time_start']
        end_time = row['time_end']
        output_file = f"{output_dir}/{row['youtube_id']}.mp4"
        if os.path.exists(output_file):
            print(f"{video_path} already exists in the directory")
            continue
        if not os.path.exists(video_path):
            print(f"Skipping {video_path}")
            continue
        print("Video info: ", video_path, start_time, end_time, output_file)

        with VideoFileClip(video_path, audio=False) as video:
            if end_time >= video.duration:
                continue
            ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_file)


df = pd.read_csv("E:/AI Talent Hub/CV Tech/hw5_action_recognition/data/kinetics_700/dancing.csv")
create_clips_from_vids("../data/kinetics_700/videos_2/", df, "../data/kinetics_700/clips_2/")
