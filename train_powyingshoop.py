from keras.applications.inception_v3 import InceptionV3 
from keras.models import Model #class for creating trained object
from keras.layers import Dense, GlobalAveragePooling2D, Dropout #Dense layer - full connection (all units) , GlobalAvgPooling - 3D arrays find avg of each array give out 1 pixel, Dropout - random purpose top layer not rely on lower too much incase that it was dropped out
from keras.preprocessing.image import ImageDataGenerator #load all file in floder - load all images only call file
from keras.optimizers import Adam #change weigh, to get the output --> good one is Adam

#parameter
FC_SIZE = 256 #top class (last two layers) that we inserted only 3 weight is too less, so we have 256x3
BATCH_SIZE = 20 #dataset 500 images, input as batch (1 batch 20). too much GPU cannot handle (RAM is not enough)
EPOCHS = 5 #train for 5 rounds

base_model = InceptionV3(weights='imagenet', include_top=False)#remove top layer

datagen_train = ImageDataGenerator(
	rescale=1./255, 
	rotation_range=180., #random rotate 180 any value 0-180
	width_shift_range=0.1, #re porportion imgae 0.1
	height_shift_range=0.1,
	zoom_range=0.5,
	horizontal_flip=True, 
	vertical_flip=True,

	#do all this for atificial increase dataset
	)

datagen_test = ImageDataGenerator(
	rescale=1./255, #adjust pixel as same as train
	)

train_datagen = datagen_train.flow_from_directory(
  './powyingshoop123_train',
  target_size=(299,299), #resize image 
  batch_size=BATCH_SIZE 
)

print(train_datagen.class_indices)

test_datagen = datagen_test.flow_from_directory(
  './powyingshoop123_test',
  target_size=(299,299),
  batch_size=BATCH_SIZE
)

def add_new_classifier(base_model, nb_classes):
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #activation function 'relu' faster train
  x = Dropout(0.5)(x)
  prediction = Dense(nb_classes, activation='softmax')(x) #last layer 'softmax' we need output as probbility between 0 to 1 and add all equals 1
  model = Model(inputs=base_model.input, outputs=prediction) #input and output
  return model

model = add_new_classifier(base_model, 3) # 3 classes

def setup_transfer_learn(model): #train only top 2 layers
  for i, layer in enumerate(model.layers):
    if i == len(model.layers)-1 or i == len(model.layers)-2: #-1 last, -2 second last
      layer.trainable = True #attribut tainrable - use for loop
    else:
      layer.trainable = False #false not train

  model.compile(
    optimizer=Adam(), #potimizer - adjust weight
    loss='categorical_crossentropy', #Loss adjust W to get less L, Classify problem called 'crossentopy' - we have 3 classes 'categorical_crossentropy' for 3 or more classes
    metrics=['accuracy'] #classify 'accuracy' how accurate it is?
  )

  return model

def setup_finetune(model): #train half lower layer 
  for i, layer in enumerate(model.layers):
    if i < 165: # 591 for InceptionResnet, # 165 for Inception #165 about half
      layer.trainable = True
    else:
      layer.trainable = False

  model.compile(
    optimizer=Adam(lr=0.0001), #specify Lr minimize learning rate 0.0001
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  return model  

model = setup_transfer_learn(model)
model.fit_generator(
  train_datagen,#object of data generator
  steps_per_epoch=20000//BATCH_SIZE,
  epochs=EPOCHS,#1 batch 1 step (20000 floor divide Batchsize // means round down)
  shuffle=True#shuffle images, random between classes
)

model = setup_finetune(model) #train half lower layer
model.fit_generator(
  train_datagen, 
  steps_per_epoch=20000//BATCH_SIZE, 
  epochs=EPOCHS,
  shuffle=True 
)

result = model.evaluate_generator(test_datagen) #test form test floder
print(result)

model.save('powyingshoop-v1.h5')
