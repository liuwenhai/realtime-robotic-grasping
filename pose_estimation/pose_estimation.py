"""
 pose estimation using SOLVEPNP_ITERATIVE in opencv
Author: Wenhai Liu, Shanghai Jiao Tong University
"""

import h5py
import cv2
from utilty import *
import numpy as np
import matplotlib.pyplot as plt
from camera import Camera

from tf.transformations import *

cameraMatrix_1 = np.array([[616.734, 0, 322.861], [0, 616.851, 234.728], [0, 0, 1]])
cameraMatrix_2 = np.array([[614.27, 0, 318.754], [0, 614.466, 237.351], [0, 0, 1]])
intrinsics = {'ur_camera':cameraMatrix_1,
              'fanuc_camera':cameraMatrix_2}

deg2reg = 3.1415926/180.
reg2deg = 180./3.1415926

def detect_grid(img, pattern_size=(8, 6)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(img, pattern_size, None,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH)
    # pdb.set_trace()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if ret == True:
        # corners2 = corners
        if (cv2.__version__).split('.')[0] == '2':
            # pdb.set_trace()
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            corners = corners
            # pdb.set_trace()
        else:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    if not ret:
        return ret,None,None
    for i in range(pattern_size[0] * pattern_size[1]):
        x, y = int(corners[i, 0, 0]), int(corners[i, 0, 1])
        if i == 0 or i == 7 or i == 40 or i == 47:
            cv2.circle(img,(x,y),5,(0,0,255),-1)
    return ret,img,corners

def pose_estimation(corner):
    # calibration board parameter
    square_size = 15
    pattern_size = (8, 6)
    # chess points coordinate in calibration board frame
    objp = np.zeros((pattern_size[0]*pattern_size[1],3),np.float32)
    objp[:, :2] = square_size * np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    for i in range(pattern_size[0] * pattern_size[1]):
        x, y = objp[i, 0], objp[i, 1]
        objp[i, 0], objp[i, 1] = y, x
    # pdb.set_trace()
    # solve pnp
    dist = (0.0, 0.0, 0.0, 0.0, 0.0)
    rvec = np.ones((3,1))
    tvec = np.ones((3,1))
    # pdb.set_trace()
    # cv2.solvePnP(objp[[0,7,40,47],:], corner[[0,7,40,47],:,:], intrinsics['fanuc_camera'],
    #              distCoeffs=dist, rvec=rvec, tvec=tvec, useExtrinsicGuess=False, flags=cv2.CV_P3P)
    cv2.solvePnP(objp, corner, intrinsics['fanuc_camera'],distCoeffs=dist, rvec=rvec, tvec=tvec, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE)  # CV_ITERATIVE
    # cv2.solvePnP(objp, corner, intrinsics['fanuc_camera'],
    #              distCoeffs=dist, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.CV_EPNP)
    imgpoints_project,_ = cv2.projectPoints(objp,rvec,tvec,intrinsics['fanuc_camera'],dist)
    error = cv2.norm(corner,imgpoints_project,cv2.NORM_L2)/len(imgpoints_project)
    return rvec,tvec,error


def float_list(start,stop,n):
    start = int(start*1000000)
    stop = int(stop*1000000)
    step = (stop-start)/n
    aa = [i/1000000. for i in range(start,stop,int(step))]
    return aa[:n]

def run_1():
    path = "190112_095745_fanuc_ur_cam.hdf5"
    f = h5py.File(path, 'r')
    imgs = np.array(f['fanuc'])
    imgs_ur = np.array(f['ur'])
    times = np.array(f['time'])
    t0 = 0
    tt = time.time()
    # camera = Camera()
    errors=[]
    tts = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img_fanuc=imgs[i].copy()
        img_ur = imgs_ur[i]
        # cv2.imshow('fanuc',img)
        # cv2.imshow('ur',img_ur)
        # pdb.set_trace()
        # img,_ = camera.get_data(False)
        t1 = times[i] * 1000
        t = t1 - t0
        t0 = t1
        ret, img, corners = detect_grid(img)
        # pdb.set_trace()
        if not ret:
            print 'no feature'
            continue
        # pdb.set_trace()
        cv2.imshow('fanuc', img_fanuc)
        cv2.imshow('ur', img_ur)
        q = (cv2.waitKey(int(t)))&0xFF
        if q == ord('s'):
            cv2.imwrite('ur_calib.jpg',img_ur)
            cv2.imwrite('fanuc_calib.jpg',img_fanuc)
        rvec, tvec, error = pose_estimation(corners)
        tvec = tvec / 1000.
        print 'error:',error
        H = XYZRodrigues_to_Hmatrix(tvec.flatten(1).tolist() + rvec.flatten(1).tolist())
        Q = quaternion_from_matrix(H)
        # print "%f %f %f %f %f %f %f" % (H[0, 3], H[1, 3], H[2, 3],
        #                                 Q[0], Q[1], Q[2], Q[3])
        du_t = time.time() - tt
        print 'time',du_t
        errors.append(error)
        tts.append(du_t)
        tt = time.time()
        # cv2.waitKey(int(t))
        # cv2.waitKey()
        # pdb.set_trace()

    plt.plot(errors,'r')
    plt.plot(tts,'b')
    plt.show()


def run_caltime():
    path = "190112_095745_fanuc_ur_cam.hdf5"
    f = h5py.File(path, 'r')
    imgs = np.array(f['fanuc'])
    imgs_ur = np.array(f['ur'])
    times = np.array(f['time'])
    t0 = 0
    tt = time.time()
    # camera = Camera()
    errors=[]
    tts = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img_fanuc=imgs[i].copy()
        img_ur = imgs_ur[i]
        # cv2.imshow('fanuc',img)
        # cv2.imshow('ur',img_ur)
        # pdb.set_trace()
        # img,_ = camera.get_data(False)
        t1 = times[i] * 1000
        t = t1 - t0
        t0 = t1
        tt = time.time()
        ret, img, corners = detect_grid(img)
        print 'feature time:',time.time()-tt
        # pdb.set_trace()
        if not ret:
            print 'no feature'
            continue
        # pdb.set_trace()
        cv2.imshow('fanuc', img_fanuc)
        cv2.imshow('ur', img_ur)
        q = (cv2.waitKey(int(t)))&0xFF
        if q == ord('s'):
            cv2.imwrite('ur_calib.jpg',img_ur)
            cv2.imwrite('fanuc_calib.jpg',img_fanuc)
        t0_pose = time.time()
        rvec, tvec, error = pose_estimation(corners)
        t_pose = time.time() - t0_pose
        print 'pose time:', t_pose
        tvec = tvec / 1000.
        print 'error:',error
        # H = XYZRodrigues_to_Hmatrix(tvec.flatten(1).tolist() + rvec.flatten(1).tolist())
        # Q = quaternion_from_matrix(H)
        # # print "%f %f %f %f %f %f %f" % (H[0, 3], H[1, 3], H[2, 3],
        # #                                 Q[0], Q[1], Q[2], Q[3])
        errors.append(error)
        tts.append(t_pose)

        # cv2.waitKey(int(t))
        # cv2.waitKey()
        # pdb.set_trace()

    plt.plot(errors,'r')
    plt.plot(tts,'b')
    plt.show()



if __name__ == "__main__":
    run_1()
    # run_caltime()