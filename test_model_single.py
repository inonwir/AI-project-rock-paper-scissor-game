from keras.models import load_model
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
import os

# load trained model
model = load_model('powyingshoop-v3.h5')

# rootDir = './powyingshoop'
# for dirName, subdirList, fileList in os.walk(rootDir):
#     print('Found directory: %s' % dirName)
#     test_images = np.zeros([len(fileList),299,299,3])
#     for i, fname in enumerate(fileList):
#         print('\t%s' % fname)
#         image = Image.open(dirName + '/' + fname) # read image
#         image = image.resize((299,299)) # resize
#         image = img_to_array(image) # convert to np array
#         image /= 255.0 # scale
#         test_images[i] = image # store in 4D array

image = Image.open('faiisexy3.jpg') # read image
image = image.resize((299,299)) # resize
image = img_to_array(image) # convert to np array
image /= 255.0 # scale
#test_images[i] = image # store in 4D array
image = np.reshape(image,(1,299,299,3))

# classify
result = model.predict(image)
#for i in range(result.shape[0]):
#row = result[i,:]
max_prob = np.argmax(result)
if max_prob == 0:
    print(result,'paper')
elif max_prob == 1:
    print(result,'rock')
else:
    print(result,'scissors')

com1=np.random.randint(0,3)
#0paper
#1rock
#2scissor
if com1 == 0 and max_prob == 0:
	print('Com1 is paper you are paper Draw!')
elif com1 == 0 and max_prob == 1:
	print('Com1 is paper you are rock You Lose!')
elif com1 == 0 and max_prob == 2:
	print('Com1 is paper you are scissor You Win!')

elif com1 == 1 and max_prob == 0:
	print('Com1 is rock you are paper You Win!')
elif com1 == 1 and max_prob == 1:
	print('Com1 is rock you are rock Darw!')
elif com1 == 1 and max_prob == 2:
	print('Com1 is rock you are scissor You Lose!')

elif com1 == 2 and max_prob == 0:
	print('Com1 is scissor you are paper You Lose!')
elif com1 == 2 and max_prob == 1:
	print('Com1 is scissor you are rock You Win!')
elif com1 == 2 and max_prob == 2:
	print('Com1 is scissor you are scissor Draw!')

