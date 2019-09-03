from __future__ import print_function
import sys
ros_packages = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_packages in sys.path:
    sys.path.remove(ros_packages)
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2
import time
import img2pc
import visualization_helpers
import argparse

if __name__ == '__main__':
    start = time.time()
    
    parser = argparse.ArgumentParser(description='path to annotations, images and inference')
    parser.add_argument('--ann-path', 
                        default='/home/xia/maskrcnn-benchmark/datasets/bobbins100/bobbins100_val.json', 
                        help='path to annotations', 
                        type=str)
    parser.add_argument('--image-path', 
                        default='/home/xia/bobbins/data/', 
                        help='path to images', 
                        type=str)
    parser.add_argument('--inf-path', 
                        default='/home/xia/maskrcnn-benchmark/output/r_max_iter_5200/inference/spindle100_val/', 
                        help='path to inference', 
                        type=str)
    args = parser.parse_args()
    
    pylab.rcParams['figure.figsize'] = (12.0, 9.0)
    # load image
    coco=COCO(args.ann_path)
    imgIds = coco.getImgIds()
    imgId=np.random.randint(0,len(imgIds))
    img = coco.loadImgs(imgIds[imgId])[0]
    colorDir = args.image_path + 'rgb'
    color = cv2.imread('%s/%s'%(colorDir,img['file_name']))
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depthDir = args.image_path + 'depth'
    depth_name = 'depth_0%03d.png'%(img['id']-1)
    depth = cv2.imread('%s/%s'%(depthDir,depth_name), cv2.IMREAD_ANYDEPTH)
    
    # show input image
    plt.axis('off')
    plt.title('Input Image', fontsize = 'xx-large')
    plt.imshow(color)
    plt.show()
    
    # get best segmentation
    segm_path = args.inf_path + 'segm.json'
    segm_img, segm = visualization_helpers.getSegm(segm_path, color, img['id'], 0.85)
    best_mask = visualization_helpers.getBestMask(color, segm)
    
    # show best segmentation
    plt.axis('off')
    plt.title('Best Segmentation', fontsize = 'xx-large')
    mask = []
    mask.append(best_mask)
    dst = visualization_helpers.showSegm(color, mask)
    plt.imshow(dst)
    plt.show()
    
    # get point cloud
    color_roi = cv2.bitwise_and(color, color, mask = best_mask)
    depth_roi = cv2.bitwise_and(depth, depth, mask = best_mask)
    pc_roi = img2pc.createPointCloud(color_roi, depth_roi, True)
        
    end = time.time()
    print('Running this code uses ', end - start, 's')