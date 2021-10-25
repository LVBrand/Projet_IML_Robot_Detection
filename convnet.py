# Tensorflow & Keras
import tensorflow as tf
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Autres librairies
import numpy as np
import matplotlib.pyplot as plt
import os, shutil


###################################################################################
########## LOADING DATA ###########################################################
###################################################################################

# You need to specify where's your project directory
pdir = '/Users/Lucas/Desktop/IML/Projet_IML_Robot/'

# The directory where you uncompressed the dogs vs cats dataset
# Yours will be different
original_dataset_dir = '/Users/Lucas/Desktop/IML/kaggle_dataset_dogs_vs_cats_uncompressed/train'

###################################################################################
########## CREATING DIRECTORIES ###################################################
###################################################################################

# Directory where you'll store your smaller dataset
base_dir = pdir+'data'
if (os.path.exists(pdir+'data'))==False:
    os.mkdir(base_dir)

# Data / Train
train_dir = os.path.join(base_dir, 'train')
if (os.path.exists(pdir+'data/train'))==False:
    os.mkdir(train_dir)

# Data / Validation
validation_dir = os.path.join(base_dir, 'validation')
if (os.path.exists(pdir+'data/validation'))==False:
    os.mkdir(validation_dir)

# Data / Test
test_dir = os.path.join(base_dir, 'test')
if (os.path.exists(pdir+'data/test'))==False:
    os.mkdir(test_dir)

# Data / Train / Cats
train_cats_dir = os.path.join(train_dir, 'cats')
if (os.path.exists(pdir+'data/train/cats'))==False:
    os.mkdir(train_cats_dir)

# # Data / Train / Dogs
train_dogs_dir = os.path.join(train_dir, 'dogs')
if (os.path.exists(pdir+'data/train/dogs'))==False:
    os.mkdir(train_dogs_dir)

# Data / Validation / Cats
validation_cats_dir = os.path.join(validation_dir, 'cats')
if (os.path.exists(pdir+'data/validation/cats'))==False:
    os.mkdir(validation_cats_dir)

# Data / Validation / Dogs  
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if (os.path.exists(pdir+'data/validation/dogs'))==False:
    os.mkdir(validation_dogs_dir)

# Data / Test / Cats    
test_cats_dir = os.path.join(test_dir, 'cats')
if (os.path.exists(pdir+'data/test/cats'))==False:
    os.mkdir(test_cats_dir)

# Data / Test / Dogs   
test_dogs_dir = os.path.join(test_dir, 'dogs')
if (os.path.exists(pdir+'data/test/dogs'))==False:
    os.mkdir(test_dogs_dir)


###################################################################################
########## COPYING IMAGES TO TRAINING, VAL AND TEST DIRECTORIES ###################
###################################################################################

# Copy the first 1000 cat images to train_cats_dir
if (os.path.exists(pdir+'data/train/cats/cat.1.jpg'))==False:
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
print('total training cat images :', len(os.listdir(train_cats_dir)))


# Copy the next 500 cat images to validation_cats_dir
if (os.path.exists(pdir+'data/validation/cats/cat.1000.jpg'))==False:
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
print('total validation cat images :', len(os.listdir(validation_cats_dir)))


# Copy the next 500 cat images to test_cats_dir
if (os.path.exists(pdir+'data/test/cats/cat.1500.jpg'))==False:
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
print('total test cat images :', len(os.listdir(test_cats_dir)))


# Copy the first 1000 dog images to train_dogs_dir
if (os.path.exists(pdir+'data/train/dogs/cat.1.jpg'))==False:
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
print('total training dog images :', len(os.listdir(train_dogs_dir)))


# Copy the next 500 dog images to validation_dogs_dir
if (os.path.exists(pdir+'data/validation/dogs/cat.1000.jpg'))==False:
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
print('total validation dog images :', len(os.listdir(validation_dogs_dir)))

    
# Copy the next 500 dog images to test_dogs_dir
if (os.path.exists(pdir+'data/test/dogs/cat.1500.jpg'))==False:
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)
print('total test dog images :', len(os.listdir(test_dogs_dir)))

###################################################################################
########## DATA PREPROCESSING #####################################################
###################################################################################

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

###################################################################################
########## INSTANTIATING THE CONVNET ##############################################
###################################################################################

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

###################################################################################
########## FITTING THE MODEL ######################################################
###################################################################################

# Fit
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)

# Saving the model
model.save('cats_and_dogs_small_2.h5')

###################################################################################
########## PREDICTION #############################################################
###################################################################################

def predict_animal(img_path, model):    
    categories = ['Cat', 'Dog']
    img = image.load_img(img_path, target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = categories[int(model.predict(x)[0][0])]
    return pred