# coding: utf-8
 
# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 
 
# In[1]:
 
 
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
 
# from config import Config
# import utils
# import model as modellib
# import visualize
# from model import log

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR) 

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from PIL import Image
import yaml
import tensorflow
 
iter_num=0
 
# get_ipython().run_line_magic('matplotlib', 'inline')
 
# Root directory of the project
# ROOT_DIR = os.getcwd()
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(COCO_MODEL_PATH)
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
 
 
# ## Configurations
 
# In[2]:
 
 
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "exposed_wire"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8*3, 16*3, 32*3, 64*3, 128*3)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ShapesConfig()
config.display()
 
 
# ## Notebook Preferences
 
# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()
 
# In[3]:
 
 
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
 
 
# In[4]:
 
 
class BrokenDataset(utils.Dataset):
    #得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
            n = np.max(image)
            return n
        
    #解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self,image_id):
            info=self.image_info[image_id]
            with open(info['yaml_path']) as f:
                temp=yaml.load(f.read())
                labels=temp['label_names']
                del labels[0]
            return labels
    #重新写draw_mask
    def draw_mask(self, num_obj, mask, image):
            info = self.image_info[image_id]
            for index in range(num_obj):
                for i in range(info['width']):
                    for j in range(info['height']):
                        at_pixel = image.getpixel((i, j))
                        if at_pixel == index + 1:
                            mask[j, i, index] =1
            return mask      
 
     #重新写load_shapes，里面包含自己的自己的类别（我的是box、column、package、fruit四类）
     #并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_shapes(self, count, height, width, img_floder, mask_floder, imglist,dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "SlidingDoor")
        self.add_class("shapes", 2, "Wall")
        self.add_class("shapes", 3, "Shelf")

        for i in range(count):
            filestr = imglist[i].split(".")[0]
          # print(filestr)
           #filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + filestr + ".png"
           #yaml_path=dataset_root_path+"total/"+filestr+"_json/info.yaml"
            yaml_path=dataset_root_path+"labelme_json/"+filestr+"_json/info.yaml"
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=width, height=height, mask_path=mask_path,yaml_path=yaml_path)
            
    #重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
       #img = cv2.imread(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels=[]
        labels=self.from_yaml_get_class(image_id)
        labels_form=[]
        for i in range(len(labels)):
            if labels[i].find("SlidingDoor") != -1:
                # print "box"
                labels_form.append("SlidingDoor")
            elif labels[i].find("Wall") != -1:
                # print "box"
                labels_form.append("Wall")
            elif labels[i].find("Shelf") != -1:
                # print "box"
                labels_form.append("Shelf")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)
    
 
 
# In[5]:
 
 
#基础设置
import os
dataset_root_path="/home/enwhsaa/ros/train_model/Mask_RCNN-master/train_data/"
img_floder = dataset_root_path+"rgb"
mask_floder = dataset_root_path+"mask"
#yaml_floder = dataset_root_path
imglist = os.listdir(img_floder)
count = len(imglist)
width = 640
height = 480
 
 
# In[6]:
 
 
# Training dataset
dataset_train = BrokenDataset()
dataset_train.load_shapes(count, 480, 640, img_floder, mask_floder, imglist,dataset_root_path)
dataset_train.prepare()
 
# Validation dataset
dataset_val = BrokenDataset()
dataset_val.load_shapes(count, 480, 640, img_floder, mask_floder, imglist,dataset_root_path)
dataset_val.prepare()
 
 
# In[7]:
 
 
# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
 
 
# ## Ceate Model
 
# In[8]:
 
 
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
 
 
# In[9]:
 
 
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
 
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
 
 
# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.
 
# In[10]:
 
 
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')
 
 
# In[ ]:
 
 
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=20, 
            layers="all")
 
 
# In[ ]:
 
 
# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)
 
