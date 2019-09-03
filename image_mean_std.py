import sys
ros_packages = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_packages in sys.path:
    sys.path.remove(ros_packages)
import cv2
import numpy as np
import os

img_path = '/home/xia/bobbins/bobbins300/train/'
img_names = os.listdir(img_path)

pixel_sum = np.array([0,0,0])
#pixel_all = np.array([0,0,0])
for i in range(len(img_names)):
    img = cv2.imread(img_path + img_names[i]).astype(int)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(int)
    pixel = img.reshape((-1,3))
    #pixel_all = np.vstack((pixel_all, pixel))
    pixel_sum = pixel_sum + np.sum(pixel, axis = 0)
pixel_mean = pixel_sum * (1 / (720*1280*len(img_names)))

#pixel_all = np.delete(pixel_all, 0, axis=0)
#pixel_mean = np.mean(pixel_all, axis=0)
#pixel_std = np.std(pixel_all, axis=0)
print('pixel mean is: [{:.4f}, {:.4f}, {:.4f}]'
      .format(pixel_mean[0], pixel_mean[1], pixel_mean[2]))
#print('pixel std is: ', pixel_std)
