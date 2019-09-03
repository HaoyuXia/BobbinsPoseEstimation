import sys
ros_packages = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_packages in sys.path:
    sys.path.remove(ros_packages)
import argparse
import json
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw
import shutil

class via2coco(object):
    def __init__(self, image_path, json_path):
        self.image_path = image_path
        self.json_path = json_path
        self.images = []
        self.annotations = []
        self.categories = []
        self.length = 0
        self.width = 0
        self.height = 0
        self.cat_count = 1
        self.ann_count = 1
        self.labels = []
        self.index = 0
        self.train = {}
        self.val = {}
        
    def load_json(self):
        with open (self.json_path, 'r') as f:
            data = json.load(f)
            data = list(data['_via_img_metadata'].values())
            for n, ann in enumerate(data):
                self.images.append(self.get_image(data, n))
                for m, region in enumerate(data[n]['regions']):
                    if data[n]['regions'][m]['region_attributes']['classification'] not in self.labels:
                        self.labels.append(data[n]['regions'][m]['region_attributes']['classification'])
                        self.categories.append(self.get_category(data, n, m))
                    self.annotations.append(self.get_annotation(data, n, m))
        self.length = len(self.images)
        return self.images, self.labels, self.categories, self.annotations
    
    def get_image(self, data, n):
        image = {}
        image_name = self.image_path + data[n]['filename']
        img = cv2.imread(image_name)
        image['file_name'] = data[n]['filename']
        self.width = img.shape[1]
        self.height = img.shape[0]
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = n + 1
        return image
    
    def get_category(self, data, n, m):
        category = {}
        category['supercategory'] = 'bobbin'
        category['name'] = 'bobbin'
        category['id'] = self.cat_count
        self.cat = self.cat_count + 1
        return category
    
    def get_annotation(self, data, n, m):
        annotation = {}
        segmentation = []
        seg_list = []
        annotation['image_id'] = n + 1
        annotation['id'] = self.ann_count
        annotation['iscrowd'] = 0
        self.ann_count = self.ann_count + 1
        annotation['category_id'] = self.categories[self.labels == data[n]['regions'][m]['region_attributes']['classification']]['id']
        # get segmentation
        for i in range(0, len(data[n]['regions'][m]['shape_attributes']['all_points_x'])):
            segmentation.append(data[n]['regions'][m]['shape_attributes']['all_points_x'][i])
            segmentation.append(data[n]['regions'][m]['shape_attributes']['all_points_y'][i])
        segmentation = np.asanyarray(segmentation).flatten()
        seg_list.append(segmentation.tolist())
        annotation['segmentation'] = seg_list
        # get area
        temp = segmentation.reshape(-1, 2)
        x = temp[:, 0]
        y = temp[:, 1]
        area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        annotation['area'] = area
        # get bbox
        annotation['bbox'] = self.get_bbox(temp)
        return annotation
    
    def get_bbox(self, temp):
        mask = Image.new('L', (self.width, self.height), 0)
        polygons = temp
        polygons = list(map(tuple, polygons))
        ImageDraw.Draw(mask).polygon(polygons, outline=1, fill=1)
        mask = np.array(mask)
        index = np.argwhere(mask == 1)
        lt_row = np.min(index[:,0])
        lt_col = np.min(index[:,1])
        rb_row = np.max(index[:,0])
        rb_col = np.max(index[:,1])
        bbox = [lt_col, lt_row, rb_col-lt_col, rb_row-lt_row]
        #area = len(index[:, 0])
        return bbox
    
    def split_train_val(self, ratio, seed):
        train_idx = []
        train_images = []
        train_annotations = []
        val_idx = []
        val_images = []
        val_annotations = []
        # get index
        if (self.length % 10 != 0):
            print('invalid length!!!')
            return self.train, self.val 
        if (isinstance(ratio, int) == False):
            print('invalid ratio!!!')
            return self.train, self.val
        if (seed != 0):
            random.seed(seed)
        for i in range(0, int(self.length/10)):
            val_test = random.sample(range(10*i + 1, 10*i + 10 + 1), ratio)
            val_idx.append(val_test[0])
            val_idx.append(val_test[1])
            for j in range(10*i + 1, 10*i + 10 + 1, 1):
                if j not in val_test:    
                    train_idx.append(j)
        # split images            
        for i, img in enumerate(self.images):
            if self.images[i]['id'] in train_idx:
                train_images.append(self.images[i])
            elif self.images[i]['id'] in val_idx:
                val_images.append(self.images[i])
        # split annotations
        for i, ann in enumerate(self.annotations):
            if self.annotations[i]['image_id'] in train_idx:
                train_annotations.append(self.annotations[i])
            elif self.annotations[i]['image_id'] in val_idx:
                val_annotations.append(self.annotations[i])
        self.train['images'] = train_images
        self.train['categories'] = self.categories
        self.train['annotations'] = train_annotations
        self.val['images'] = val_images
        self.val['categories'] = self.categories
        self.val['annotations'] = val_annotations 
        return self.train, self.val
    
    def copy_image(self, image_path, train_path, val_path):
        if len(self.train)==0 or len(self.val)==0 :
            print('train or val is empty!!!')
        else:
            for i in range(len(self.train['images'])):
                shutil.copy(image_path+self.train['images'][i]['file_name'], train_path)
            for j in range(len(self.val['images'])):
                shutil.copy(image_path+self.val['images'][j]['file_name'], val_path)
    
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='path to image and json')
    parser.add_argument('--image-path', 
                        default='/home/xia/bobbins/bobbins300/rgb/', 
                        help='path to rgb image', 
                        type=str)
    parser.add_argument('--json-path', 
                        default='/home/xia/bobbins/bobbins300/bobbins300v3.json', 
                        help='path to json file', 
                        type=str)
    parser.add_argument('--save-path', 
                        default='/home/xia/bobbins/bobbins300/', 
                        help='path to save new json file', 
                        type=str)
    parser.add_argument('--train-path', 
                        default='/home/xia/bobbins/bobbins300/train/', 
                        help='path to train set', 
                        type=str)
    parser.add_argument('--val-path', 
                        default='/home/xia/bobbins/bobbins300/val/', 
                        help='path to val set', 
                        type=str)
    args = parser.parse_args()
    dataset = via2coco(args.image_path, args.json_path)
    images, labels, categories, annotations = dataset.load_json()
    train, val = dataset.split_train_val(2, 20)
    
    '''
    dataset.copy_image(args.image_path, args.train_path, args.val_path)
    json.dump(train, open(args.save_path + 'bobbins300_train.json', 'w'), cls=MyEncoder, indent=4)
    json.dump(val, open(args.save_path + 'bobbins300_val.json', 'w'), cls=MyEncoder, indent=4)
    '''
    
    '''
    save_path = '/home/xia/spindle100_coco.json'
    data_coco = {}
    data_coco['images'] = images
    data_coco['categories'] = categories
    data_coco['annotations'] = annotations
    json.dump(data_coco, open(save_path, 'w'), cls=MyEncoder, indent=4)
    '''