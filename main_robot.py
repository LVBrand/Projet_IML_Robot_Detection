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
predState = False
imgIsThere = False

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
    def predict_animal(img, model):    
        categories = ['Chat', 'Chien']
        #img = image.load_img(img_path, target_size=(150,150))
        x = cv2.resize(img, (150,150), interpolation = cv2.INTER_AREA)
        #x = np.asarray(img)
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        #x = np.concatenate((x, 3), axis=None)
        #x = x * (1./255)
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
        return multiBodyId

    """ 
    L'objet quad.obj a été créé par Tristan Guichaoua qui l'a gentiment mit à disposition de la classe.
    Merci à lui, il m'a fait économiser pas mal de temps ;-)
    """

    try:
        while True:
            capture = pepper.getCameraFrame(handle)
            cv2.imshow("Top camera", capture)
            cv2.waitKey(1)

            # On appuie sur la touche C pour changer d'image aléatoirement
            if imgIsThere==False :
                if keyboard.is_pressed('c'):
                    if predState==False:
                        # L'image met quelques secondes à charger
                        print("Vous avez demandé une image. Veuillez patienter.")
                        random_file = random.choice(os.listdir(kaggleTestDatasetPath))
                        newTextureId = kaggleTestDatasetPath+'/'+random_file
                        img = create_picture(position=[2,0,1], orientation=[0,0,0], new_img_path=newTextureId)
                        print("L'image est arrivée saine et sauve. Bisous.")
                        imgIsThere = True
            else:
                if keyboard.is_pressed('c'):
                    if predState==False:
                        print("Suppression de l'image précédente.")
                        p.removeBody(img)
                        imgIsThere = False
                        print("Vous avez demandé une nouvelle image. Veuillez patienter.")
                        random_file = random.choice(os.listdir(kaggleTestDatasetPath))
                        newTextureId = kaggleTestDatasetPath+'/'+random_file
                        img = create_picture(position=[2,0,1], orientation=[0,0,0], new_img_path=newTextureId)
                        print("La nouvelle image est arrivée saine et sauve. Re-bisous.")
                        imgIsThere = True

                    
            if keyboard.is_pressed('x'):
                predState = True
                print("Début de la prédiction")
                a = predict_animal(capture, model)
                print("Pepper a prédit : ", a)
                # p.removeBody(img)
                # print("img removed")
                predState = False
                print("Arrêt de la prédiction")
                
            
    except KeyboardInterrupt:
        simulation_manager.stopSimulation(client_id)