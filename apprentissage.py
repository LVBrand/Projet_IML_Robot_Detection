# Tensorflow & Keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import Model

# Autres librairies
import numpy as np
import matplotlib.pyplot as plt



# Loading Dataset
train_data = image_dataset_from_directory(
    directory='.\data'
    

)