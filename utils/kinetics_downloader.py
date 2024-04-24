import os
import time

import pandas as pd
import pytube
from moviepy.video.io.VideoFileClip import VideoFileClip
from pytube.exceptions import PytubeError
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def download_kinetics_class(class_data: pd.DataFrame, download_path: str, log_path: str = None) -> pd.DataFrame:
    os.makedirs(download_path, exist_ok=True)
    class_data = class_data.reset_index()
    k = 0

    # get list of video names in the directory
    video_names = os.listdir(download_path)
    video_names = [name.split('.')[0] for name in video_names]
    video_names = set(video_names)

    length = class_data.shape[0]
    lst_name_video = []
    lst_target = []
    for i, row in class_data.iterrows():
        try:
            tag = row['youtube_id']
            if tag in video_names:
                print(f"Video {tag} already exists in the directory")
                continue
            video_url = f"https://www.youtube.com/watch?v={tag}"
            yt = pytube.YouTube(video_url)
            stream = yt.streams.first()
            filename = f"{download_path}/{tag}.mp4"
            stream.download(filename=filename)

            start_time = row.time_start
            end_time = row.time_end

            name_video = f"{download_path}/video_{k:04d}.mp4"
            if log_path is not None:
                with open(log_path, 'a') as f:
                    f.write(f"{tag}\n")

            time.sleep(2)
            # ffmpeg_extract_subclip(filename, start_time, end_time, targetname=name_video)
            # clip = VideoFileClip(filename).subclip(start_time, end_time)
            # clip = clip.volumex(2)
            # clip.write_videofile(filename)

            k += 1
            lst_name_video.append(name_video)
            lst_target.append(row.label)
        except PytubeError as e:
            print(f"Error occurred while processing video {i + 1}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error occurred while processing video {i + 1}: {str(e)}")

        print(f"{i + 1} / {length} is ready")

    class_data = pd.DataFrame({'video': lst_name_video, 'label': lst_target})
    return class_data


# class_data = pd.read_csv("../data/kinetics_700/dancing.csv")
# start_time = time.time()
# download_kinetics_class(class_data.iloc[:8], "../data/kinetics_700/videos", "log.txt")
# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds") # Time taken: .2661943435669 seconds