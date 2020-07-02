import os
import random
import numpy as np
import cv2
from lxml import etree

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def object_random(objects):
    """
    random choice the object
    :param objects: ['object1','object2',...]
    :return: 'object3'
    """
    return random.choice(objects)


def contrast_img(img1, c, b):
    '''
    :param img1: original image
    :param c:  > 1 brighter, < 1 darker
    :param b:  scalar added to each sum
    :return:   processed image
    '''
    rows, cols, chunnel = img1.shape
    blank = np.zeros([rows, cols, chunnel], img1.dtype)  # np.zeros(img1.shape, dtype=uint8)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    #cv2.imshow("process", dst)
    return dst


def rotateMask(mask,angle):
    h,w,c = mask.shape
    max_h = np.max(mask.shape)+100
    mask_ = np.zeros((max_h,max_h,c))
    h_, w_  = int(abs(max_h-h)/2), int(abs(max_h-w)/2)
    mask_[h_:(h_+h),w_:(w_+w),:]  = mask
    M = cv2.getRotationMatrix2D((max_h / 2, max_h / 2), angle, 1)
    mask_rot = cv2.warpAffine(mask_, M, mask_.shape[::-1][1:])
    mask = get_roi(mask_rot)
    mask = mask.astype(np.uint8)
    return mask


def get_roi(img):
    '''
    get rotation image
    :param img:
    :return:
    '''
    y_index, x_index = np.where((img != [0, 0, 0]).all(axis=2))
    y_min, y_max = np.min(y_index), np.max(y_index)
    x_min, x_max = np.min(x_index), np.max(x_index)
    img_roi = img[y_min:y_max, x_min:x_max, :]
    return img_roi

def occlusion_random():
    """
    random decide whether to occlude
    :return:
    """
    p=random.random()
    if p>0.5:
        return True
    else:
        return False


def point_random(p_left_up, p_right_bottom):
    '''

    :param p_left_up: (xmin,ymin)
    :param p_right_bottom: (xmax,ymax)
    :return: x,y is not normal
    '''
    if p_left_up[0]>=p_right_bottom[0]:
        y = p_left_up[0]
    else:
        y = random.randint(p_left_up[0], p_right_bottom[0])
    if p_left_up[1]>=p_right_bottom[1]:
        x = p_left_up[1]
    else:
        x = random.randint(p_left_up[1], p_right_bottom[1])
    return [x,y]

def img_overlay(image1,image2,point,mask,occlusion):
    """
    add image2 to image1 at (point[0],point[1]) with class point[2]
    :param image1: background image,(height,width,3)
    :param image2: sliding image adding to background image,(height,width,3)
    :param point: point[x,y,class,i] indicate where to add and the class of image2
    :param mask: creat the mask image with class value, (height,width,2),0 is object class, 1 is num of object
    :param occlusion: decide whether the sliding image is occluded by background image, bool value
    :return: added image,(height,width,3), and the mask is changed
    """
    img1=image1.copy()
    img2=image2
    height,width,rgb=img1.shape
    height_r,width_r,rgb_r=img2.shape
    # x is height, y is width, but generally x is width, y is height
    x=point[0]
    y=point[1]
    object=point[2]
    # print '...',point[3]
    if x+height_r>height or y+width_r>width:
        return img1
    for i in range(height_r):
        for j in range(width_r):
            if img2[i,j,0]<5 and img2[i,j,1]<5 and img2[i,j,2]<5:
                img1[x+i,y+j,:]=img1[x+i,y+j,:]
            else:
                if mask[x+i,y+j,0]!=0:
                    img1[x+i,y+j,:]= img1[x+i,y+j,:] if occlusion else img2[i,j,:]
                    mask[x + i, y + j, 0] =mask[x + i, y + j, 0] if occlusion else object
                    mask[x + i, y + j, 1] = mask[x + i, y + j, 1] if occlusion else point[3]
                else:
                    img1[x + i, y + j, :] = img2[i, j, :]
                    mask[x + i, y + j, 0] = object
                    mask[x + i, y + j, 1] = point[3]
    return img1

def occlusion_ratio(mask,image2,point):
    """
    compute the occlusion ration based on image1 and image2
    :param mask: mask of synthetic image with lots of objects,(height,width,2)
    :param image2: sliding image,(height,width,3)
    :param point: [x,y,class,i]
    :return:
    """
    height, width, rgb = mask.shape
    height_r, width_r, rgb_r = image2.shape
    x=point[0]
    y=point[1]
    object=point[2]
    if x+height_r>height or y+width_r>width:
        return 1
    total=0
    occlusion=0
    for i in range(height_r):
        for j in range(width_r):
            if image2[i,j,0]>4 or image2[i,j,1]>4 or image2[i,j,2]>4:
                total=total+1
                if mask[x + i, y + j, 0] != object or mask[x+i,y+j,1]!=point[3]:
                    occlusion = occlusion + 1
    # print '...occlusion,total',occlusion,total
    return float(occlusion)/float(total)

def pascal_xml(img_syn,mask,imgs_added,objects_added,points,ratio,path,name):
    """
    write synthetic images  to xml files like Pascal VOC2007
    :param img_syn:
    :param mask:
    :param imgs_added:
    :param objects_added:
    :param points: [num][x,y,class,i]
    :param ratio:
    :param path: '/home/robot/Downloads/segmentation/dataset/data_sr300/VOCdevkit'
    :param name: '000000'
    :return:
    """
    annotation_path=os.path.join(path,'VOC2007','Annotations',name+'.xml')
    img_path=os.path.join(path,'VOC2007','JPEGImages',name+'.jpg')

    if not os.path.exists(os.path.join(path,'VOC2007','JPEGImages')):
        os.makedirs(os.path.join(path,'VOC2007','JPEGImages'))
    if not os.path.exists(os.path.join(path,'VOC2007','Annotations')):
        os.makedirs(os.path.join(path,'VOC2007','Annotations'))

    cv2.imwrite(img_path,img_syn)
    annotation=etree.Element("annotation")
    etree.SubElement(annotation, "folder").text = "VOC2007"
    etree.SubElement(annotation, "filename").text = name+'.jpg'
    source = etree.SubElement(annotation, "source")
    etree.SubElement(source, "database").text = "The VOC2007 Database"
    etree.SubElement(source, "annotation").text = "PASCAL VOC2007"
    etree.SubElement(source, "image").text = "flickr"
    etree.SubElement(source, "flickrid").text = " "
    owner = etree.SubElement(annotation, "owner")
    etree.SubElement(owner, "flickrid").text = 'sjtu'
    etree.SubElement(owner, "name").text = 'Wenhai Liu'
    size = etree.SubElement(annotation, "size")
    etree.SubElement(size, "width").text = '640'
    etree.SubElement(size, "height").text = '480'
    etree.SubElement(size, "depth").text = '3'
    etree.SubElement(annotation, "segmented").text = '0'

    for i,img in enumerate(imgs_added):
        point=points[i]
        # print '....',i,point
        height,width,rgb=img.shape
        xmin=point[1]
        ymin=point[0]
        xmax=point[1]+width
        ymax=point[0]+height
        ratio_object=occlusion_ratio(mask,img,point)  # 1 is occlusion totally.
        if ratio_object<1 and ratio_object>ratio:
            key_object = etree.SubElement(annotation, "object")
            etree.SubElement(key_object, "name").text = objects_added[i]
            etree.SubElement(key_object, "difficult").text = '1'
            etree.SubElement(key_object, "occlusion").text = str(ratio_object)
            bndbox = etree.SubElement(key_object, "bndbox")
            etree.SubElement(bndbox, "xmin").text = str(xmin)
            etree.SubElement(bndbox, "ymin").text = str(ymin)
            etree.SubElement(bndbox, "xmax").text = str(xmax)
            etree.SubElement(bndbox, "ymax").text = str(ymax)
        elif ratio_object<=ratio:
            key_object = etree.SubElement(annotation, "object")
            etree.SubElement(key_object, "name").text = objects_added[i]
            etree.SubElement(key_object, "difficult").text = '0'
            etree.SubElement(key_object, "occlusion").text = str(ratio_object)
            bndbox = etree.SubElement(key_object, "bndbox")
            etree.SubElement(bndbox, "xmin").text = str(xmin)
            etree.SubElement(bndbox, "ymin").text = str(ymin)
            etree.SubElement(bndbox, "xmax").text = str(xmax)
            etree.SubElement(bndbox, "ymax").text = str(ymax)
    doc = etree.ElementTree(annotation)
    doc.write(open(annotation_path, "w"), pretty_print=True)