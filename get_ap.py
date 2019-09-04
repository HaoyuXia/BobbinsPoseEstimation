from __future__ import print_function
import sys
ros_packages = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_packages in sys.path:
    sys.path.remove(ros_packages)
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

if __name__ == '__main__':
    # load log
    log_file = '/home/xia/maskrcnn-benchmark/output/r_300_7200_0010_480_step3600_5400/test_log.txt'
    pylab.rcParams['figure.figsize'] = (12.0, 9.0)
    with open(log_file, 'r') as f:
        log = f.readlines()
        f.close()
    bbox = []
    segm = []
    for i in range(len(log)):
        if (re.search(r'(?=Task: bbox)', log[i])):
            bbox.append(log[i:i+3])
        elif (re.search(r'(?=Task: segm)', log[i])):
            segm.append(log[i:i+3])
    
    #get AP
    bbox_ap = []
    for j in range(len(bbox)):
        idx = re.search(r'(?=,)', bbox[j][2]).start()
        bbox_ap.append(float(bbox[j][2][0:idx]))
    bbox_ap_max = bbox_ap.index(max(bbox_ap))
    
    segm_ap = []        
    for k in range(len(segm)):
        idx = re.search(r'(?=,)', segm[k][2]).start()
        segm_ap.append(float(segm[k][2][0:idx]))
    segm_ap_max = segm_ap.index(max(segm_ap))
    
    num_iter = np.array(range(400,7201,400))
    
    # plot fugure
    plt.figure()
    plt.plot(num_iter, bbox_ap)
    plt.plot(num_iter, segm_ap)
    plt.xlabel('iteration', fontsize=25)
    plt.xticks(size=15)
    plt.ylabel('AP', fontsize=25)
    plt.yticks(size=15)
    plt.text(num_iter[bbox_ap_max], bbox_ap[bbox_ap_max]+0.001, 
             '[%.d, %.4f]'%(num_iter[bbox_ap_max], bbox_ap[bbox_ap_max]), 
             ha='center', va= 'bottom',fontsize=15)
    plt.scatter(num_iter[bbox_ap_max], bbox_ap[bbox_ap_max], marker='o', c='r')
    plt.text(num_iter[segm_ap_max], segm_ap[segm_ap_max]+0.001, 
             '[%.d, %.4f]'%(num_iter[segm_ap_max], segm_ap[segm_ap_max]), 
             ha='center', va= 'bottom',fontsize=15)
    plt.scatter(num_iter[segm_ap_max], segm_ap[segm_ap_max], marker='o', c='r')
    plt.legend(labels=['bbox ap', 'segm ap'], loc='lower right', fontsize=20)