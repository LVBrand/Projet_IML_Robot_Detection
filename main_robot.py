# installer : qibullet, opencv-contrib-python (avec pip)
from qibullet import SimulationManager
from qibullet import PepperVirtual
import pybullet as p
import cv2
import time
import keyboard
import os, shutil, random
from keras import models
from keras.preprocessing import image
import numpy as np

# Chemin vers le dataset de Test de kaggle (à changer suivant l'utilisateur)
kaggleTestDatasetPath = '/Users/Lucas/Documents/Cours/S9 - SIIA/IML - Interactive Machine Learning/Projet_IML_Robot/kaggle_dataset_dogs_vs_cats_uncompressed/test'

if __name__ == "__main__":
    simulation_manager = SimulationManager()
    client_id = simulation_manager.launchSimulation(gui=True)

    # Pepper, le gentil robot
    pepper = simulation_manager.spawnPepper(
        client_id,
        spawn_ground_plane=True)

    # On se connecte
    p.connect(p.DIRECT)

    # Création d'une camera
    handle = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_TOP)

    # Laser
    pepper.showLaser(True)
    pepper.subscribeLaser()

    # On charge notre modèle de convnet entraîné sur 25000 images
    model = models.load_model("cats_and_dogs_full.h5")



    # Fonction de prédiction
    def predict_animal(img_path, model):    
        categories = ['Cat', 'Dog']
        img = image.load_img(img_path, target_size=(150,150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = categories[int(model.predict(x)[0][0])]
        return pred

    # On définit une fonction qui instancie un objet affichant une image à faire prédire
    def create_picture(position, orientation, new_img_path):
        # On instancie un mesh à partir de l'objet quad.obj
        visualShapeId = p.createVisualShape(p.GEOM_MESH, fileName='./obj/quad.obj')
        # On charge la texture (ici, l'image qu'on veut faire prédire)
        textureUniqueId = p.loadTexture(new_img_path)
        # On crée l'objet en lui donnant une position, une orientation et une forme
        multiBodyId = p.createMultiBody(
            baseVisualShapeIndex=visualShapeId,
            basePosition=position,
            baseOrientation=p.getQuaternionFromEuler(orientation))
        # On applique la texture à l'objet (On affiche l'image sur l'objet qui sert ici de tableau)
        p.changeVisualShape(multiBodyId, -1, textureUniqueId=textureUniqueId)

    """ 
    L'objet quad.obj a été créé par Tristan Guichaoua qui l'a gentiment mit à disposition de la classe.
    Merci à lui, il m'a fait économiser pas mal de temps ;-)
    """

    try:
        while True:
            capture = pepper.getCameraFrame(handle)
            cv2.imshow("Top camera", capture)

            # On appuie sur la touche C pour changer d'image aléatoirement
            if keyboard.is_pressed('c'):
                # L'image met quelques secondes à charger
                print("Vous avez demandé une nouvelle image. Veuillez patienter.")
                random_file = random.choice(os.listdir(kaggleTestDatasetPath))
                newTextureId = kaggleTestDatasetPath+'/'+random_file
                create_picture(position=[3,0,1], orientation=[0,0,0], new_img_path=newTextureId)
            
            #if keyboard.is_pressed('v'):
            
    except KeyboardInterrupt:
        simulation_manager.stopSimulation(client_id)



        
