from model import model_simple_unet_initializer
from skimage.io import imread,imsave
import numpy as np


if __name__ == '__main__':
    model = model_simple_unet_initializer(1, 4, 2, 5, 16, True, 3, True, False, 'l2', 0.001)

    model.load_weights('model_best_train.h5')

    print('Print name of image to process')
    name_of_image = input()

    image = imread(name_of_image)
    image = image.reshape(1,768,768,3)
    
    output = model(image)

    output = np.array(output)
    
    output = output.reshape(768,768,1)
    
    imsave(f'result_{name_of_image}.jpg',output)