"""
visual servoing code for UR5
output motion command is transmitted to UR5 robot via ROS
Author: Wenhai Liu, Shanghai Jiao Tong University
"""


import pyrealsense2 as rs
import matplotlib.pyplot as plt
from realsense_device_manager import DeviceManager
from pose_estimation import detect_grid,pose_estimation
from utilty import *
import datetime,h5py

import rospy
from std_msgs.msg import String

deg2reg = 3.1415926/180.

# paramters for cameras
resolution_width = 640 # 1280 # pixels
resolution_height = 480 # 720 # pixels
frame_rate = 15  # fps
dispose_frames_for_stablisation = 15  # frames
serials = [u'819612070850', u'818312071299']
win_serial = {u'819612070850':'eye-to-hand',
              u'818312071299':'eye-in-hand'}



def run():
    try:
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
        rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
        device_manager = DeviceManager(rs.context(), rs_config)
        device_manager.enable_all_devices()
        # device_manager.enable_device(u'819612070850',False)

        # pdb.set_trace()
        print 'Cam Init...'
        for frame in range(dispose_frames_for_stablisation):
            frames = device_manager.poll_frames()

        intrinsics_devices = device_manager.get_device_intrinsics(frames)
        # pdb.set_trace()
        t0 = time.time()
        q = (cv2.waitKey(1)) & 0xFF
        imgs_fanuc,imgs_ur = [],[]
        times = []
        vels = []
        ur_poses = []
        i = 0
        t_pub = time.time()
        save_img = False
        while  q!= 27:
            frames = device_manager.poll_frames()
            if frames is {} or len(frames) !=2:
                continue
            # print device_manager._enabled_devices
            if q == ord('s'):
                save_img = True
                start_time = time.time()
            for serial in device_manager._enabled_devices:
                color_img = np.array(frames[serial][rs.stream.color].get_data())
                cv2.imshow(win_serial[serial],color_img)
                if serial == u'818312071299':
                    img_ur = color_img
                    time_ur = time.time()
                elif serial == u'819612070850':
                    img_fanuc = color_img
                    time_fanuc = time.time()
                    # print "fanuc_camera is coming......"
            # print '% d camera time : %f '%(len(device_manager._enabled_devices),time.time()-t0)
            q = (cv2.waitKey(10)) & 0xFF
            if save_img:
                imgs_fanuc.append(img_fanuc)
                imgs_ur.append(img_ur)
                times.append(time_fanuc - start_time)
                ur_poses.append(get_frame('tool0'))
            t0 = time.time()
            if time.time() - t_pub > 0.15:
                ret, img, corner = detect_grid(img_fanuc)
                if ret:
                    rvec, tvec, error = pose_estimation(corner)
                    tvec = tvec / 1000.
                    # print 'error:',error
                    Pose2_H = XYZRodrigues_to_Hmatrix(tvec.flatten(1).tolist() + rvec.flatten(1).tolist())
                    # print "Detection error is %f" % error
                else:
                    print "Detection of chess point is wrong."
                    continue

                temp = np.dot(ur2fanuc_H, fanuc_hand_to_eye_H)
                temp = np.dot(temp, Pose2_H)
                temp = np.dot(temp, Track_H)
                target_H = np.dot(temp, np.linalg.inv(ur5_hand_in_eye_H))
                target_Q = quaternion_from_matrix(target_H)
                # print "ur5 target pose: "
                # print "%f %f %f %f %f %f %f" % (target_H[0, 3], target_H[1, 3], target_H[2, 3],
                #                                 target_Q[0], target_Q[1], target_Q[2],
                #                                 target_Q[3])

                # cv2.destroyAllWindows()
                # pdb.set_trace()
                tt = time.time()
                # t_matrix,_ = getFrame('base','tool0')
                # current_pose = Hmatrix_to_XYZRodrigues(t_matrix)
                # print current_pose
                current_pose = get_frame('tool0')
                # print current_pose
                # pdb.set_trace()
                # print " get current pose cause : %f"%(time.time()-tt)

                rvec, _ = cv2.Rodrigues(target_H[:3, :3])
                rvec = rvec.flatten(1)
                tvec = target_H[:3,3]
                target_pose = tvec.tolist() + rvec.tolist()
                diff_pose = np.array(target_pose) - np.array(current_pose)
                vv_t, vv_r = 3,0.5
                while np.sum((diff_pose[:3]*vv_t>0.5).astype(np.int8))>0:
                    vv_t -= 0.1

                diff_pose_vel = (diff_pose[:3]*vv_t).tolist() + (diff_pose[3:]*vv_r).tolist()
                # pdb.set_trace()
                # print "-----  diff pose vel ----"
                # print diff_pose_vel
                # pdb.set_trace()
                # pose_msg = "movel(p[%5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f], %5.2f, %5.2f)\n" % (
                # target_H[0, 3], target_H[1, 3], target_H[2, 3],
                # rvec[0], rvec[1], rvec[2], acc, vel)

                diff_vel_msg = "speedl([%5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f], %5.2f,%5.2f )\n"%(
                    diff_pose_vel[0],diff_pose_vel[1],diff_pose_vel[2],
                    diff_pose_vel[3],diff_pose_vel[4],diff_pose_vel[5],
                    1.2,0.5
                )
                vels.append(diff_pose_vel)
                # pdb.set_trace()
                pub.publish(diff_vel_msg)
                print time.time()-t_pub
                i+=1
                t_pub = time.time()

        # pdb.set_trace()
        xx,yy,zz,rr,pp,qq =[],[],[],[],[],[]
        for vel in vels:
            x,y,z,r,p,q = vel
            xx.append(x)
            yy.append(y)
            zz.append(z)
            rr.append(r)
            pp.append(p)
            qq.append(q)
        plt.plot(xx,'r')
        plt.plot(yy,'b')
        plt.plot(zz, 'g')
        # plt.figure()
        plt.plot(rr, 'c')
        plt.plot(pp, 'k')
        plt.plot(qq, 'w')
        label=['x','y','z','rx','ry','rz']
        plt.ylim(-0.02,0.02)
        plt.legend(label)
        plt.show()
        outfile_name = datetime.datetime.now().strftime(("%y%m%d_%H%M%S"))
        with h5py.File(outfile_name + '_fanuc_ur_cam.hdf5', 'w') as f:
            f.create_dataset('fanuc', data=np.array(imgs_fanuc))
            f.create_dataset('ur', data=np.array(imgs_ur))
            f.create_dataset('time', data=np.array(times))
            f.create_dataset('pose',data = np.array(ur_poses))

        # pub.publish(pose_msg)
        # pdb.set_trace()
    except KeyboardInterrupt:
        print("The program was interupted by the user. Closing the program...")

    finally:
        device_manager.disable_streams()
        cv2.destroyAllWindows()

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
    return Hmatrix_to_XYZRodrigues(pose)

if __name__ == "__main__":
    rospy.init_node('ur_move', anonymous=True)
    robot = RobotCommander()
    pub = rospy.Publisher('/ur_driver/URScript', String, queue_size=1)

    # print get_frame()
    # pdb.set_trace()
    ur2fanuc_H, fanuc_hand_to_eye_H, ur5_hand_in_eye_H = transform()
    Track_H = euler_matrix(3.1415,0,-3.1415/2,'rxyz')
    Track_H[:3,3] = np.array([0.045,0.06,0.4])
    acc = 0.8
    vel = 0.25
    run()







