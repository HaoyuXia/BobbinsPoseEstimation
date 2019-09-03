import sys
ros_packages = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_packages in sys.path:
    sys.path.remove(ros_packages)
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from predictor_bobbin import BobbinsDemo
import time

def load(path):
    pil_image = Image.open(path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, img_name):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.title('Prediction of {name}'.format(name=img_name), fontsize = 'xx-large')
    plt.show()

if __name__ == '__main__':
    start = time.time()
    
    config_path = '/home/xia/maskrcnn-benchmark/configs/e2e_mask_rcnn_R_101_FPN_1x_predictor.yaml'
    img_path = '/home/xia/bobbins/data/test/rgb/'    
    pylab.rcParams['figure.figsize'] = (12.0, 9.0)
    
    # update the config options with the config file
    cfg.merge_from_file(config_path)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    bobbin_demo = BobbinsDemo(cfg,
                              min_image_size=720,
                              confidence_threshold=0.95,
                              )
    
    # load image
    #imgId = np.random.randint(0,200)
    #imgId = 400 + imgId 
    imgId = np.random.randint(0,20)
    img_name = 'rgb_' + '{:0>4d}'.format(imgId) + '.png'
    img_file = img_path + img_name
    image = load(img_file)
    
    # compute predictions
    predictions = bobbin_demo.run_on_opencv_image(image)
    imshow(predictions, img_name)
    end = time.time()
    print('Running this code uses ', end - start, 's')
