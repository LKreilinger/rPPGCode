## Example how to get live stream can be found in following URL:
## https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py

# Run both scripts simultaneously
#start python PACED_test.py "&" start python RealsenseGetFrames.py

import pyrealsense2 as rs
import numpy as np
import cv2
import keyboard
import time

##############
# ## Convert milisecons to time
# import datetime
# my_ms = 1658218108598
# datetime.datetime.fromtimestamp(my_ms / 1000)

##############
# Definitions and path
fps = 30
videoLength = 180  # in seconds
maxFrames = fps*videoLength
destination_path = 'C:/Users/Chaputa/Documents/Trier/Master/Masterarbeit/Realsense/Data/test/all/'

# Configure  color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Press 1 or 2 on keyboard to start saving frames")
print("Press c on keyboard to stop saving frames or stop the livestream")
# Start streaming
pipeline.start(config)
try:
    while True:
        # detect keypress (s)
        if keyboard.is_pressed("1") or keyboard.is_pressed("2"):
            print("Start saving frames")
            for iteratingFrames in range(maxFrames):

                # wait for frame save in color_image
                frame = pipeline.wait_for_frames()
                color_frame = frame.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())

                # save image wth name in milliseconds
                imageName = str(int(round(time.time() * 1000))) + '.jpg'
                cv2.imwrite(destination_path + imageName, color_image)
                if keyboard.is_pressed("c"):
                    break
            cv2.destroyAllWindows()
            break
        # else only show livestream
        else:
            # Wait for a frame
            frame = pipeline.wait_for_frames()
            color_frame = frame.get_color_frame()

            color_image = np.asanyarray(color_frame.get_data())

            # Show image
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(1)
            if keyboard.is_pressed("c"):
                cv2.destroyAllWindows()
                break

finally:
        # Stop streaming
        pipeline.stop()
