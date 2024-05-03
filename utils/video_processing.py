import cv2
import numpy as np
import albumentations as A


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
    return frames



def write_video(name, video, frame_sample_rate):
    video_writer = cv2.VideoWriter(
        name,
        cv2.VideoWriter_fourcc(*'MJPG'),
        int(25 / frame_sample_rate),
        (video.shape[2], video.shape[1])
    )
    for frame in video:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()


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


def apply_video_augmentations(video, transform):
    targets = {'image': video[0]}
    for i in range(1, video.shape[0]):
        targets[f'image{i}'] = video[i]
    transformed = transform(*video)
    transformed = np.concatenate(
        [np.expand_dims(transformed['image'], axis=0)]
        + [np.expand_dims(transformed[f'image{i}'], axis=0) for i in range(1, video.shape[0])]
    )
    return transformed
