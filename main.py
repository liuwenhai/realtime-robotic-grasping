#!/usr/bin/env python

from tf.transformations import *
import cv2
import time
import numpy as np
from camera import Camera

from moveit_commander import RobotCommander
import rospy
from std_msgs.msg import String

deg2reg = 3.1415926/180.



def run():
    while q != 27:
        color_image, depth_image = cam.get_data(get_depth=False)

        q = (cv2.waitKey(10)) & 0xFF

        target_H = np.idendity(4)  # target pose, pose estimation
        current_pose = get_frame('tool0')

        rvec, _ = cv2.Rodrigues(target_H[:3, :3])
        rvec = rvec.flatten(1)
        tvec = target_H[:3, 3]
        target_pose = tvec.tolist() + rvec.tolist()
        diff_pose = np.array(target_pose) - np.array(current_pose)
        vv_t, vv_r = 3, 0.5
        while np.sum((diff_pose[:3] * vv_t > 0.5).astype(np.int8)) > 0:
            vv_t -= 0.1

        diff_pose_vel = (diff_pose[:3] * vv_t).tolist() + (diff_pose[3:] * vv_r).tolist()

        diff_vel_msg = "speedl([%5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f], %5.2f,%5.2f )\n" % (
            diff_pose_vel[0], diff_pose_vel[1], diff_pose_vel[2],
            diff_pose_vel[3], diff_pose_vel[4], diff_pose_vel[5],
            1.2, 0.5
        )

        pub.publish(diff_vel_msg)
        print(time.time() - t_pub)
        t_pub = time.time()

def get_frame(link='tool0'):
    '''
    get transform from 'base' to link, in [x,y,z,Rodrigues]
    :param link: link can be 'tool0', ...
    :return:
    '''
    pose = robot.get_link(link).pose().pose
    x,y,z  = pose.position.x, pose.position.y, pose.position.z
    rx,ry,rz,rw = pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w
    pose_offset = [0,0,0,0,0,1,0]
    pose_offset = quaternion_matrix(np.array(pose_offset[3:]))
    pose = quaternion_matrix(np.array([rx,ry,rz,rw]))
    pose[:3,3] = np.array([x,y,z])
    pose = np.dot(pose_offset,pose)
    return pose

if __name__ == "__main__":
    rospy.init_node('ur_move', anonymous=True)
    robot = RobotCommander()
    cam = Camera()
    pub = rospy.Publisher('/ur_driver/URScript', String, queue_size=1)
    run()







