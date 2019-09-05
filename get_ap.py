from __future__ import print_function
import sys
ros_packages = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_packages in sys.path:
    sys.path.remove(ros_packages)
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def get_ap(log_file):
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
    return bbox_ap, segm_ap, bbox_ap_max, segm_ap_max

def plot_figure(num_iter, val1, val2, val1_max, val2_max, val1_name, val2_name):
    plt.figure()
    plt.plot(num_iter, val1)
    plt.plot(num_iter, val2)
    plt.xlabel('iteration', fontsize=25)
    plt.xticks(size=15)
    plt.ylabel('AP', fontsize=25)
    plt.yticks(size=15)
    plt.text(num_iter[val1_max], val1[val1_max]+0.001, 
             '[%.d, %.4f]'%(num_iter[val1_max], val1[val1_max]), 
             ha='center', va= 'bottom',fontsize=15)
    plt.scatter(num_iter[val1_max], val1[val1_max], marker='o', c='r')
    plt.text(num_iter[val2_max], val2[val2_max]+0.001, 
             '[%.d, %.4f]'%(num_iter[val2_max], val2[val2_max]), 
             ha='center', va= 'bottom',fontsize=15)
    plt.scatter(num_iter[val2_max], val2[val2_max], marker='o', c='r')
    plt.legend(labels=[val1_name, val2_name], loc='lower right', fontsize=20)

if __name__ == '__main__':
    pylab.rcParams['figure.figsize'] = (12.0, 9.0)
    # load log and get ap
    train_log = '/home/xia/maskrcnn-benchmark/output/r_300_4800_0010/train_log.txt'
    test_log = '/home/xia/maskrcnn-benchmark/output/r_300_4800_0010/test_log.txt'
    train_bbox, train_segm, train_bbox_max, train_segm_max = get_ap(train_log)
    test_bbox, test_segm, test_bbox_max, test_segm_max = get_ap(test_log)
    num_iter = np.array(range(200,4801,200))
    
    # plot fugure
    plot_figure(num_iter, train_bbox, train_segm, train_bbox_max, train_segm_max, 'train bbox ap', 'train segm ap')
    plot_figure(num_iter, test_bbox, test_segm, test_bbox_max, test_segm_max, 'test bbox ap', 'test segm ap')
    plot_figure(num_iter, train_bbox, test_bbox, train_bbox_max, test_bbox_max, 'train bbox ap', 'test bbox ap')
    plot_figure(num_iter, train_segm, test_segm, train_segm_max, test_segm_max, 'train segm ap', 'test segm ap')