"""
Intel Realsense D435 camera dirver
Author: Wenhai Liu, Shanghai Jiao Tong University
"""

import pyrealsense2 as rs
import numpy as np
import cv2

import matplotlib.pyplot as plt

resolution_width = 640 # 640 # 1280 # pixels
resolution_height = 480 # 480 # 720 # pixels

class Camera(object):
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, 6)
        # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
        self.config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, 6)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.pipeline.start(self.config)
        for i in range(10):
            self.get_data(False)
        print('Camera is initialized')

    def get_data(self,get_depth=True):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        self.color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        color_image = np.asanyarray(color_frame.get_data())
        dis_none = np.zeros([resolution_height,resolution_width])

        if get_depth:
            depth_frame = aligned_frames.get_depth_frame()
            dis_image = np.asanyarray(depth_frame.get_data()).astype(float)
            for y in xrange(resolution_height):
                for x in xrange(resolution_width):
                    dis_image[y,x] = depth_frame.get_distance(x,y)

            dis_img = dis_image*1000
            return color_image,dis_img
        else:
            return color_image,dis_none

    def get_intrinsics(self):
        return self.color_intrinsics

if __name__ == '__main__':
    camera = Camera()
    while True:
        color_image, depth_image = camera.get_data(get_depth=True)
        cv2.imshow("rgb",color_image)
        cv2.imshow("depth",depth_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    color_image, depth_image = camera.get_data(get_depth=True)
    print color_image.shape,depth_image.shape
    print camera.color_intrinsics
    plt.subplot(211)
    plt.imshow(color_image)
    plt.subplot(212)
    plt.imshow(depth_image)
    plt.show()
    camera.pipeline.stop()


