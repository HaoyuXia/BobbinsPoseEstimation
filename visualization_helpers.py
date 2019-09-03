from pycocotools.mask import decode
import numpy as np
import json
import cv2
from skimage import measure

def showBBox(path, img, img_id, threshold):
    bbox = []
    dst = img * (1/255)
    with open (path, 'r') as bbox_file:
        bboxes = json.load(bbox_file)
        for i, data1 in enumerate(bboxes):
            if ( (data1['image_id'] == img_id) and (data1['score'] > threshold) ):
                bbox.append(data1)
    for j, data2 in enumerate(bbox):
        x1, y1, w, h = map(int, data2['bbox'])
        x2 = x1 + w
        y2 = y1 + h
        color = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        cv2.rectangle(dst, (x1,y1), (x2,y2), color, 3)
    return dst, bbox

def getSegm(path, img, img_id, threshold):
    with open(path, 'r') as segm_file:
        segms = json.load(segm_file)
    segm = []
    mask = []
    for i, data in enumerate(segms):
        if ( (data['image_id'] == img_id) and (data['score'] > threshold) ):
            segm.append(data)
            mask.append(decode(data['segmentation']))
    return mask, segm

def showSegm(img, mask):
    dst = img * (1/255)
    for i in range(len(mask)):
        m = mask[i]
        index = np.argwhere(m == 1)
        for i in range(len(index)):
            dst[index[i][0]][index[i][1]] = [0,0,0]
        mm = np.ones( (m.shape[0], m.shape[1], 3) )
        color_mask = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        for k in range(3):
            mm[:,:,k] = color_mask[k]
        mm = cv2.bitwise_and(mm, mm, mask = m)
        dst = cv2.add(dst, mm)
    return dst

def getBestMask(img, segm):
    best_score = segm[0]['score']
    best_index = 0
    for i, data in enumerate(segm):
        if segm[i]['score'] > best_score:
            best_score = segm[i]['score']
            best_index = i
    best_mask = decode(segm[best_index]['segmentation'])
    return best_mask

def removeInvalidMask(mask):
    if len(mask) <= 0:
        return mask
    else:
        clean_mask = []
        for i, m in enumerate(mask):
            label, num = measure.label(m, neighbors=8, background=0, return_num=True)
            if num > 1:
                max_label = 0
                max_id = 0
                for j in range(num):
                    label_sum = np.sum(label==j+1)
                    if label_sum > max_label:
                        max_label = label_sum
                        max_id = j+1
                #m = np.array((label==max_id), dtype='uint8')
                mm = np.array((label==max_id), dtype='uint8')
                clean_mask.append(mm)
                print('mask[{:.0f}] has benn modified'.format(i))
            elif num == 1:
                clean_mask.append(m)
        return clean_mask