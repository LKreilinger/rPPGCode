import os

from moviepy.editor import *

def frames_to_video(path, duration, fps):
    directory = sorted(os.listdir(path))
    clips = []
    for filename in directory:
        if filename.endswith(".jpg"):
            file_path = os.path.join(path + '\\' + filename)
            clips.append(ImageClip(file_path).set_duration(duration))

    print(clips)
    video = concatenate(clips, method="compose")
    video.write_videofile('test1.mp4', fps=fps)
