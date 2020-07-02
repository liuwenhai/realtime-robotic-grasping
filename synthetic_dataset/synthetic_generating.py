"""
Automatic generation of ocllusion aware object detection dataset
dataset format : VOC2007
Author: Wenhai Liu, Shanghai Jiao Tong University
"""

from fun_list import *
import time
import pdb
import os.path as osp


def run():
    '''
    the quality of synthetic image depends on the object image patch,
    to generate your own dataset,
    use grabcut to acquire fine object mask and save different object patch to path : ./model/
    :return:
    '''
    data_path = osp.split(osp.realpath(__file__))[0]

    objects = ['chips','fresh milk','guangming']

    object_num = range(1, len(objects) + 1)
    object_class = dict(zip(objects, object_num))

    img_id = 0
    images = []

    bg_dir = 'background'
    bg_point_range = {'bg.png': [(10, 10), (600, 400)]}

    bg_dir = osp.join(data_path, bg_dir)

    path = 'dataset/occ_aware_object_detection'
    mkdir(path)

    # number = [5, 10, 15, 15, 20, 25, 30, 40, 50]  # objects per image
    t0 = time.time()
    for num in range(10):
        tic = time.time()
        name = '%0.6d' % (num + 1)
        img_bg_path = object_random(os.listdir(bg_dir))
        p1, p2 = bg_point_range[img_bg_path]
        # range1,range2 = bg_range[img_bg_path]
        img_bg_path = os.path.join(bg_dir, img_bg_path)
        alpha = random.uniform(0.85, 1.3)
        scale = random.uniform(0.85, 1.3)
        # alpha = random.uniform(0.95, 1.1)
        # scale = random.uniform(0.46, 0.54)
        # scale = random.uniform(0.95, 1.05)
        if np.random.random()>0.5:
            scale = 1
        if np.random.random()>0.5:
            alpha = 1
        print("the %dth img: brightness alpha is %f, scale is %f"%(num+1,alpha,scale))
        image_bg = cv2.imread(img_bg_path)

        size = image_bg.shape
        y_,x_,_ = size
        image_bg = contrast_img(image_bg, alpha, 0)
        mask = np.zeros((y_, x_, 2), np.uint8)
        img_syn = np.zeros((y_, x_, 3), np.uint8)
        objects_added = []
        imgs_added = []
        points = []
        image = {}
        image['id'] = img_id
        img_id += 1
        image['width'] = x_
        image['height'] = y_
        image['file_name'] = name + '.jpg'
        images.append(image)
        num_per_pic = random.choice(range(1, 5))
        for i in range(num_per_pic):
            i += 1
            # object = object_random(objects_random)
            object = object_random(objects)
            object_path = os.path.join('model', 'obj', object)
            object_pose = object_random(os.listdir(object_path))
            img_file_path = os.path.join('model', 'obj', object, object_pose)

            objects_added.append(object)
            angle = random.randint(0, 360)
            if np.random.random() > 0.5:
                angle = 0
            img = cv2.imread(img_file_path)
            img = rotateMask(img,angle)
            img = contrast_img(img, alpha, 0)
            yy,xx = img.shape[:2]
            img = cv2.resize(img, (int(xx*scale),int(yy*scale)), interpolation=cv2.INTER_CUBIC)
            imgs_added.append(img)
            occlusion = occlusion_random()

            p2_adapt = (p2[0]-img.shape[1],p2[1]-img.shape[0]) # img adding point should in scope of bin
            [x, y] = point_random(p1, p2_adapt) # y is x, x is y
            # x, y = x*scale_, y*scale_
            # point = [x, y, object_class[object], i]
            point = [x, y, object_class[object], i]
            points.append(point)
            if img is None:
                pdb.set_trace()
            img_syn = img_overlay(img_syn, img, point, mask, occlusion)
            # pdb.set_trace()
        mask_bg = np.zeros((y_, x_, 2), np.uint8)
        img_syn = img_overlay(image_bg, img_syn, [0, 0, 0, 0], mask_bg, False)
        pascal_xml(img_syn, mask, imgs_added, objects_added, points, 0.5, path, name) # output pascal annotation
        print('..........one time: ', time.time() - tic)

    print("Num images: %s" % len(images))


    print("time is :",time.time()-t0)




if __name__ == "__main__":
    run()

