import cv2
import numpy as np
import tf2_ros
import rospy
import time
from tf.transformations import *
from moveit_commander import RobotCommander
import roslib

deg2reg = 3.1415926/180.


# def transform():
#     ur5_hand_in_eye_H = np.array(
#         [[-0.12616053315003012, 0.992008781073542, -0.0014484986316887858, -0.11173105023987219],
#          [-0.991603524147393, -0.1261505124245682, -0.028434118827382254, 0.04326014328038761],
#          [-0.02838962440348554, -0.0021509272430085824, 0.9995946192023168, -0.008639185400829547],
#          [0.0, 0.0, 0.0, 1.0]])
#     ur5_base2grid = np.array([[0.06201582599876852, -0.9967074237509491, -0.05223359805179729, 0.18409855748635115],
#                               [0.99631468280959, 0.05871457951047764, 0.06252720184794881, -0.557909958581177],
#                               [-0.05925445252229406, -0.05591877674497228, 0.9966754738949032, 0.008351384942893765],
#                               [0.0, 0.0, 0.0, 1.0]])
#     ur5_base2grid_H = quaternion_from_matrix(ur5_base2grid)
#
#     fanuc_hand_to_eye_H = np.array([[0.08120261, 0.98746563, -0.13534312, 0.73213321],
#                                     [0.99338612, -0.09124322, -0.06970433, 0.01903843],
#                                     [-0.08117977, -0.1287878, -0.98834384, 0.59451529],
#                                     [0., 0., 0., 1.]])
#     fanuc_hand_to_eye_Q = quaternion_from_matrix(fanuc_hand_to_eye_H)
#
#     fanuc_hand_grid = np.array([[-0.00285707, -0.09107573, 0.99583987, 0.13925434],
#                                 [0.01048259, -0.99579195, -0.09104128, 0.03880048],
#                                 [0.99994097, 0.01017886, 0.00379976, 0.02558422],
#                                 [0., 0., 0., 1.]])
#
#     fanuc_robot = euler_matrix(173.627 * deg2reg, -85.985 * deg2reg, -0.985 * deg2reg, 'sxyz')
#     fanuc_robot[:3, 3] = np.array([0.815133, -0.111093, -0.323657])
#
#     fanuc2hand_grid = np.dot(fanuc_robot, fanuc_hand_grid)
#     fanuc2hand_grid_Q = quaternion_from_matrix(fanuc2hand_grid)
#     ur2fanuc_H = np.dot(ur5_base2grid, np.linalg.inv(fanuc2hand_grid))
#     ur2fanuc_Q = quaternion_from_matrix(ur2fanuc_H)
#     print "ur5 to fanuc transform : "
#     print "%f %f %f %f %f %f %f" % (ur2fanuc_H[0, 3], ur2fanuc_H[1, 3], ur2fanuc_H[2, 3],
#                                     ur2fanuc_Q[0], ur2fanuc_Q[1], ur2fanuc_Q[2], ur2fanuc_Q[3])
#     print "fanuc to hand_grid transform : "
#     print "%f %f %f %f %f %f %f" % (fanuc2hand_grid[0, 3], fanuc2hand_grid[1, 3], fanuc2hand_grid[2, 3],
#                                     fanuc2hand_grid_Q[0], fanuc2hand_grid_Q[1], fanuc2hand_grid_Q[2],
#                                     fanuc2hand_grid_Q[3])
#
#     print "ur5 base to grid transform : "
#     print "%f %f %f %f %f %f %f" % (ur5_base2grid[0, 3], ur5_base2grid[1, 3], ur5_base2grid[2, 3],
#                                     ur5_base2grid_H[0], ur5_base2grid_H[1], ur5_base2grid_H[2],
#                                     ur5_base2grid_H[3])
#
#     print "fanuc base to hand_eye cam transform : "
#     print "%f %f %f %f %f %f %f" % (fanuc_hand_to_eye_H[0, 3], fanuc_hand_to_eye_H[1, 3], fanuc_hand_to_eye_H[2, 3],
#                                     fanuc_hand_to_eye_Q[0], fanuc_hand_to_eye_Q[1], fanuc_hand_to_eye_Q[2],
#                                     fanuc_hand_to_eye_Q[3])
#     return fanuc_hand_to_eye_H, ur2fanuc_H

def transform():
    ur5_hand_in_eye_H = np.array(
        [[-0.12616053315003012, 0.992008781073542, -0.0014484986316887858, -0.11173105023987219],
         [-0.991603524147393, -0.1261505124245682, -0.028434118827382254, 0.04326014328038761],
         [-0.02838962440348554, -0.0021509272430085824, 0.9995946192023168, -0.008639185400829547],
         [0.0, 0.0, 0.0, 1.0]])
    ur5_base2grid = np.array([[0.06201582599876852, -0.9967074237509491, -0.05223359805179729, 0.18409855748635115],
                              [0.99631468280959, 0.05871457951047764, 0.06252720184794881, -0.557909958581177],
                              [-0.05925445252229406, -0.05591877674497228, 0.9966754738949032, 0.008351384942893765],
                              [0.0, 0.0, 0.0, 1.0]])
    ur5_base2grid_H = quaternion_from_matrix(ur5_base2grid)

    fanuc_hand_to_eye_H = np.array([[ 0.08047114 , 0.9881277 , -0.13087418 , 0.73314894],
                                    [ 0.99373238 ,-0.08975325, -0.06663571 , 0.01739733],
                                    [-0.07759097, -0.12469166 ,-0.98915703 , 0.5918548 ],
                                    [ 0.        ,  0.       ,   0.         , 1.        ]])
    fanuc_hand_to_eye_Q = quaternion_from_matrix(fanuc_hand_to_eye_H)

    fanuc_hand_grid = np.array([[-0.00142523, -0.08221592 , 0.99661352 , 0.13804052],
                                [ 0.01253588, -0.99653769 ,-0.08219174 , 0.03865498],
                                [ 0.99992041,  0.01237629 , 0.00245094 , 0.02029621],
                                [ 0.        ,  0.         , 0.          ,1.        ]])

    fanuc_robot = euler_matrix(173.627 * deg2reg, -85.985 * deg2reg, -0.985 * deg2reg, 'sxyz')
    fanuc_robot[:3, 3] = np.array([0.815133, -0.111093, -0.323657])

    fanuc2hand_grid = np.dot(fanuc_robot, fanuc_hand_grid)
    fanuc2hand_grid_Q = quaternion_from_matrix(fanuc2hand_grid)
    ur2fanuc_H = np.dot(ur5_base2grid, np.linalg.inv(fanuc2hand_grid))
    ur2fanuc_Q = quaternion_from_matrix(ur2fanuc_H)
    print "ur5 to fanuc transform : "
    print "%f %f %f %f %f %f %f" % (ur2fanuc_H[0, 3], ur2fanuc_H[1, 3], ur2fanuc_H[2, 3],
                                    ur2fanuc_Q[0], ur2fanuc_Q[1], ur2fanuc_Q[2], ur2fanuc_Q[3])
    print "fanuc to hand_grid transform : "
    print "%f %f %f %f %f %f %f" % (fanuc2hand_grid[0, 3], fanuc2hand_grid[1, 3], fanuc2hand_grid[2, 3],
                                    fanuc2hand_grid_Q[0], fanuc2hand_grid_Q[1], fanuc2hand_grid_Q[2],
                                    fanuc2hand_grid_Q[3])

    print "ur5 base to grid transform : "
    print "%f %f %f %f %f %f %f" % (ur5_base2grid[0, 3], ur5_base2grid[1, 3], ur5_base2grid[2, 3],
                                    ur5_base2grid_H[0], ur5_base2grid_H[1], ur5_base2grid_H[2],
                                    ur5_base2grid_H[3])

    print "fanuc base to hand_eye cam transform : "
    print "%f %f %f %f %f %f %f" % (fanuc_hand_to_eye_H[0, 3], fanuc_hand_to_eye_H[1, 3], fanuc_hand_to_eye_H[2, 3],
                                    fanuc_hand_to_eye_Q[0], fanuc_hand_to_eye_Q[1], fanuc_hand_to_eye_Q[2],
                                    fanuc_hand_to_eye_Q[3])
    return ur2fanuc_H, fanuc_hand_to_eye_H, ur5_hand_in_eye_H


def fanuc_pose_to_Hmatrix(pose):
    '''
    transform panel reading pose to Hmatrix
    :param pose: [X,Y,Z,W,P,R],[mm,deg]
    :return:
    '''
    euler = np.array(pose[3:]) * deg2reg
    t_matrix = np.eye(4)
    t_matrix[:3, :3] = euler_matrix(euler[0], euler[1], euler[2], 'sxyz')[:3, :3]
    t_matrix[:3, 3] = np.array(pose[:3]) / 1000.
    return t_matrix


def XYZRodrigues_to_Hmatrix(pos):
    '''
    :param pos: [x,y,z,rodrigue]
    :return: np.array((4,4))
    '''
    xyz = pos[:3]
    rod = pos[3:6]
    rot_matrix,_ = cv2.Rodrigues(np.array(rod))
    Hmatrix = np.identity(4)
    Hmatrix[:3,:3] = rot_matrix
    Hmatrix[:3,3] = np.array(xyz).reshape(3)
    return Hmatrix

def Hmatrix_to_XYZRodrigues(Hmatrix):
    rot_matrix = Hmatrix[:3,:3]
    xyz = Hmatrix[:3,3]
    rod,_ = cv2.Rodrigues(rot_matrix)
    return xyz.tolist()+rod.flatten().tolist()

def getFrame(source ='base',target='tool0',flags = 'T_matrix'):
    '''
    print getFrame('base','tool0')
    print getFrame('base','tool0','matrix')
    print getFrame('base', 'tool0','euler')
    :param source:
    :param target:
    :param flags:
    :return:
    '''
    assert flags == 'quaternion' or flags == 'matrix' or flags == 'euler' or flags == 'T_matrix'
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    rate = rospy.Rate(1000.0)
    bool_get_tf = False
    time0 = time.time()
    while not rospy.is_shutdown() and not bool_get_tf:
        try:
            trans = tfBuffer.lookup_transform(source, target, rospy.Time())
            bool_get_tf = True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            # print 'sb'
            continue
        rate.sleep()
        if time.time()-time0>3:
            break
    if not bool_get_tf:
        return None,None
    print "get pose cause : %f"%(time.time()-time0)
    x,y,z = trans.transform.translation.x,trans.transform.translation.y,trans.transform.translation.z
    qx,qy,qz,qw = trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w
    if flags=='quaternion':
        return np.array([x,y,z]),np.array([qx,qy,qz,qw])
    elif flags=='matrix':
        return np.array([x,y,z]), quaternion_matrix(np.array([qx,qy,qz,qw]))
    elif flags=='euler':
        return np.array([x,y,z]),euler_from_matrix(quaternion_matrix(np.array([qx,qy,qz,qw])), 'sxyz')
    elif flags == 'T_matrix':
        t_matrix = np.zeros((4,4))
        t_matrix[:3,:3] = quaternion_matrix(np.array([qx,qy,qz,qw]))[:3,:3]
        t_matrix[3,3] = 1
        t_matrix[:3,3] = np.array([x,y,z])
        return t_matrix,None
        # print trans




if __name__ == "__main__":
    pass