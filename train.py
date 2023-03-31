from model import model_train, model_simple_unet_initializer
import skimage
from skimage.io import imread
import pandas as pd
import os 
import numpy as np
from utils import rle_decode, multi_rle_encode, rle_decode, masks_as_image

ship_dir = ''
masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations.csv'))
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')

train_images = os.listdir(train_image_dir)
train_temp = np.random.choice(train_images, 2000) # We choose 2000 random images every time

train_temp_img = []
train_temp_mask = []

for img in train_temp:
    train_temp_img.append(imread(train_image_dir + '/' + img).astype('uint8'))
    train_temp_mask.append(masks_as_image(masks.query('ImageId=="'+img+'"')['EncodedPixels']))
    
train_temp_img = np.array(train_temp_img)
train_temp_mask = np.array(train_temp_mask)

model = model_simple_unet_initializer(1, 4, 2, 5, 16, True, 3, True, False, 'l2', 0.001)

model, results1 = model_train(model, train_temp_img, train_temp_mask, 30, 50, 8, 6, 'model')
