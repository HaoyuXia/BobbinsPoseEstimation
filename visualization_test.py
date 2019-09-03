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
import visualization_helpers
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='path to annotations, images and inference')
    parser.add_argument('--ann-path', 
                        default='/home/xia/maskrcnn-benchmark/datasets/bobbins300/bobbins300v3_val.json', 
                        help='path to annotations', 
                        type=str)
    parser.add_argument('--image-path', 
                        default='/home/xia/bobbins/data/rgb', 
                        help='path to images', 
                        type=str)
    parser.add_argument('--inf-path', 
                        default='/home/xia/maskrcnn-benchmark/output/r_300_7200_0010_480_norm_rgb/bobbins300v3_val/', 
                        help='path to inference', 
                        type=str)
    args = parser.parse_args()
    
    pylab.rcParams['figure.figsize'] = (12.0, 9.0)
    #annFile = '/home/xia/maskrcnn-benchmark/datasets/bobbins300/bobbins300v3_val.json'
    coco=COCO(args.ann_path)
    
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))
    
    # imgIds = coco.getImgIds(imgIds = [324158])
    imgIds = coco.getImgIds()
    imgId=np.random.randint(0,len(imgIds))
    img = coco.loadImgs(imgIds[imgId])[0]
    #dataDir = '/home/xia/bobbins/data/rgb'
    # dataType = 'val2017'
    # I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
    # I = io.imread('%s/%s'%(dataDir,img['file_name']))
    I = cv2.imread('%s/%s'%(args.image_path,img['file_name']))
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    
    plt.axis('off')
    plt.title('Input Image', fontsize = 'xx-large')
    plt.imshow(I)
    plt.show()
    
    catIds=[]
    for ann in coco.dataset['annotations']:
        if ann['image_id']==imgIds[imgId]:
            catIds.append(ann['category_id'])
    
    plt.axis('off')
    plt.title('Ground Truth', fontsize = 'xx-large')
    plt.imshow(I)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()
    print('Ground Truth: {} instances'.format(len(anns)))
    
    # show bbox
    bbox_path = args.inf_path + 'bbox.json'
    bbox_img, bbox = visualization_helpers.showBBox(bbox_path, I, img['id'], 0.85)
    plt.axis('off')
    plt.title('Object Detection', fontsize = 'xx-large')
    plt.imshow(bbox_img)
    plt.show()
    print('Detection: {} bounding boxes'.format(len(bbox)))
    
    # show segmentation
    segm_path = args.inf_path + 'segm.json'
    mask, segm = visualization_helpers.getSegm(segm_path, I, img['id'], 0.85)
    clean_mask = visualization_helpers.removeInvalidMask(mask)
    dst = visualization_helpers.showSegm(I, clean_mask)
    plt.axis('off')
    plt.title('Instance Segmentation', fontsize = 'xx-large')
    plt.imshow(dst)
    plt.show()
    print('Segmentation: {} instances'.format(len(segm)))