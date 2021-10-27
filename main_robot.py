# installer : qibullet, opencv-contrib-python (avec pip)
from qibullet import SimulationManager
from qibullet import PepperVirtual
import pybullet as p
import cv2
import time
import keyboard

import os, shutil
import tensorflow as tf
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np


if __name__ == "__main__":
    simulation_manager = SimulationManager()
    client_id = simulation_manager.launchSimulation(gui=True)

# Pepper, le gentil robot
pepper = simulation_manager.spawnPepper(
    client_id,
    spawn_ground_plane=True)


# Création d'une camera
handle = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_TOP)

# Laser
pepper.showLaser(True)
pepper.subscribeLaser()


# Choisir aléatoirement une image parmi les données de test
# Mettre l'image sur un cube
# L'image change aléatoirement quand on appuie sur un bouton
# Le robot se déplace jusqu'à devant le cube portant l'image
# On capture ce que voit le robot par sa camera
# On utilise la fonction predict_animal() sur l'image (avec un boutton)
# Suivant la prédiction, on entend un son de chat ou de chien

# Load our train CNN model
model = models.load_model("cats_and_dogs_full.h5")

# Prediction
def predict_animal(img_path, model):    
    categories = ['Cat', 'Dog']
    img = image.load_img(img_path, target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = categories[int(model.predict(x)[0][0])]
    return pred



# ##############################################################################################
# #### Creating Cat and Dog objects ############################################################
# ##############################################################################################


# Creating a cube

cubeTextureId = p.loadTexture('./obj/dog_2_object/Australian_Cattle_Dog_dif.jpg')

visualShapeId = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents = [1,1,2],
    rgbaColor=None)

multiBodyId = p.createMultiBody(
    baseVisualShapeIndex = visualShapeId,
    basePosition=[0,5,0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))


p.changeVisualShape(multiBodyId, -1, textureUniqueId=cubeTextureId)



# # Visual Shape
# cat1_visualShapeId = p.createVisualShape(
#     shapeType = p.GEOM_MESH,
#     fileName = './obj/cat_1_object/12222_Cat_v1_l3.obj',
#     rgbaColor=None,
#     meshScale=[0.01, 0.01, 0.01])

# cat2_visualShapeId = p.createVisualShape(
#     shapeType = p.GEOM_MESH,
#     fileName = './obj/cat_2_object/12221_Cat_v1_l3.obj',
#     rgbaColor=None,
#     meshScale=[0.01, 0.01, 0.01])

# dog1_visualShapeId = p.createVisualShape(
#     shapeType = p.GEOM_MESH,
#     fileName = './obj/dog_1_object/13466_Canaan_Dog_v1_L3.obj',
#     rgbaColor=None,
#     meshScale=[0.01, 0.01, 0.01])

# dog2_visualShapeId = p.createVisualShape(
#     shapeType = p.GEOM_MESH,
#     fileName = './obj/dog_2_object/13463_Australian_Cattle_Dog_v3.obj',
#     rgbaColor=None,
#     meshScale=[0.01, 0.01, 0.01])



# # Collision Shape
# cat1_collisionShapeId = p.createCollisionShape(
#     shapeType = p.GEOM_MESH,
#     fileName = './obj/cat_1_object/12222_Cat_v1_l3.obj',
#     meshScale=[0.01, 0.01, 0.01])

# cat2_collisionShapeId = p.createCollisionShape(
#     shapeType = p.GEOM_MESH,
#     fileName = './obj/cat_2_object/12221_Cat_v1_l3.obj',
#     meshScale=[0.01, 0.01, 0.01])

# dog1_collisionShapeId = p.createCollisionShape(
#     shapeType = p.GEOM_MESH,
#     fileName = './obj/dog_1_object/13466_Canaan_Dog_v1_L3.obj',
#     meshScale=[0.01, 0.01, 0.01])

# dog2_collisionShapeId = p.createCollisionShape(
#     shapeType = p.GEOM_MESH,
#     fileName = './obj/dog_2_object/13463_Australian_Cattle_Dog_v3.obj',
#     meshScale=[0.01, 0.01, 0.01])



# # Multi Body


# cat1_multiBodyId = p.createMultiBody(
#     baseMass=1.0,
#     baseCollisionShapeIndex=cat1_collisionShapeId,
#     baseVisualShapeIndex=cat1_visualShapeId,
#     basePosition=[0,0,1],
#     baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

# cat2_multiBodyId = p.createMultiBody(
#     baseMass=1.0,
#     baseCollisionShapeIndex=cat2_collisionShapeId,
#     baseVisualShapeIndex=cat2_visualShapeId,
#     basePosition=[0,0,3],
#     baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

# dog1_multiBodyId = p.createMultiBody(
#     baseMass=1.0,
#     baseCollisionShapeIndex=dog1_collisionShapeId,
#     baseVisualShapeIndex=dog1_visualShapeId,
#     basePosition=[0,0,5],
#     baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

# dog2_multiBodyId = p.createMultiBody(
#     baseMass=1.0,
#     baseCollisionShapeIndex=dog2_collisionShapeId,
#     baseVisualShapeIndex=dog2_visualShapeId,
#     basePosition=[0,0,7],
#     baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))



# # Textures
# cat1_textureId = p.loadTexture('./obj/cat_1_object/Cat_diffuse.jpg')
# cat2_textureId = p.loadTexture('./obj/cat_2_object/Cat_diffuse.jpg')
# dog1_textureId = p.loadTexture('./obj/dog_1_object/13466_Canaan_Dog_diff.jpg')
# dog2_textureId = p.loadTexture('./obj/dog_2_object/Australian_Cattle_Dog_dif.jpg')

# p.changeVisualShape(cat1_visualShapeId, -1, textureUniqueId=cat1_textureId)
# p.changeVisualShape(cat2_visualShapeId, -1, textureUniqueId=cat2_textureId)
# p.changeVisualShape(dog1_visualShapeId, -1, textureUniqueId=dog1_textureId)
# p.changeVisualShape(dog2_visualShapeId, -1, textureUniqueId=dog2_textureId)

p.setGravity(0, 0, -9.8)

p.setRealTimeSimulation(1)



# try:
#     while True:
#         img = pepper.getCameraFrame(handle)
#         cv.imshow('top camera', img)

#         # if keyboard.is_pressed('p'):
#         #     print('Predicting animal')
#         #     predict_animal(img_path, model)


        
