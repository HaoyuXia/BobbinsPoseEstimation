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
    log_file = '/home/xia/maskrcnn-benchmark/output/r_300_7200_0010_480_new/log.txt'
    pylab.rcParams['figure.figsize'] = (12.0, 9.0)
    with open(log_file, 'r') as f:
        log = f.readlines()
    loss = []
    lr = []
    num_iter = []
    for i in range(len(log)):
        if 'loss:' in log[i]:
            temp = re.sub(' ', '', log[i])
            loss_pos = re.search(r'(?<=loss_mask:)(.*?)(?=\()', temp)
            lr_pos = re.search(r'(?<=lr:)(.*?)(?=maxmem:)', temp)
            iter_pos = re.search(r'(?<=iter:)(.*?)(?=loss:)', temp)
            loss.append( float( temp[loss_pos.start() : loss_pos.end()] ) )
            lr.append( float( temp[lr_pos.start() : lr_pos.end()] ) )
            num_iter.append( int( temp[iter_pos.start() : iter_pos.end()] ) )
    f.close()
    plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('loss_mask')
    plt.plot(num_iter, loss)
    
    plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('learning rate')
    plt.plot(num_iter, lr)