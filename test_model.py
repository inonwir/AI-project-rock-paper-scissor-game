from keras.models import load_model
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
import os

# load trained model
model = load_model('powyingshoop-v3.h5')

rootDir = './powyingshoop'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    test_images = np.zeros([len(fileList),299,299,3])
    for i, fname in enumerate(fileList):
        print('\t%s' % fname)
        image = Image.open(dirName + '/' + fname) # read image
        image = image.resize((299,299)) # resize
        image = img_to_array(image) # convert to np array
        image /= 255.0 # scale
        test_images[i] = image # store in 4D array


# classify
result = model.predict(test_images)
for i in range(result.shape[0]):
    row = result[i,:]
    max_prob = np.argmax(row)
    if max_prob == 0:
        print(row,'paper')
    elif max_prob == 1:
        print(row,'rock')
    else:
        print(row,'scissors')



