import os

import pandas as pd
import pytube
from pytube.exceptions import PytubeError
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

data = pd.read_csv("../data/kinetics_700/train.csv")
df = data[data.label.str.contains('dancing')]

new_df = pd.DataFrame()
label_values = df['label'].unique()
for label in label_values:
    label_data = df[df['label'] == label].sample(200)
    new_df = pd.concat([new_df, label_data])

os.makedirs("../data/kinetics_700/videos", exist_ok=True)


new_df = new_df.reset_index()
k = 0
length = new_df.shape[0]
lst_name_video = []
lst_target = []
for i, row in new_df.iterrows():
    try:
        tag = row['youtube_id']
        video_url = f"https://www.youtube.com/watch?v={tag}"
        yt = pytube.YouTube(video_url)
        stream = yt.streams.first()
        filename = "../video.mp4"
        stream.download(filename=filename)

        start_time = row.time_start
        end_time = row.time_end

        name_video = f"../data/kinetics_700/videos/video_{k:04d}.mp4"
        ffmpeg_extract_subclip(filename, start_time, end_time, targetname=name_video)
        k += 1
        lst_name_video.append(name_video)
        lst_target.append(row.label)
    except PytubeError as e:
        print(f"Error occurred while processing video {i + 1}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error occurred while processing video {i + 1}: {str(e)}")

    print(f"{i + 1} / {length} is ready")

data_n = pd.DataFrame({"name_video": lst_name_video,
                       "label": lst_target})

data_n.to_csv('data.csv')

# 696
