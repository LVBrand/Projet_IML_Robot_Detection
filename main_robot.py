# installer : qibullet, opencv-contrib-python, playsound (avec pip)
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

# On déclare ces deux booléens
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

    # On charge notre modèle de convnet entraîné sur 25 000 images
    model = models.load_model("cats_and_dogs_full.h5")



    # Fonction de prédiction
    # On resize l'image et on rajoute une dimension pour avoir une donnée conforme au layer
    # d'entrée : [1, 150, 150, 3]
    def predict_animal(img, model):    
        categories = ['Chat', 'Chien']
        x = cv2.resize(img, (150,150), interpolation = cv2.INTER_AREA)
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        pred = categories[int(model.predict(x)[0][0])]
        return pred

    # On définit une fonction qui instancie un objet affichant une image à faire prédire
    # On commence par instancier un mesh à partir de l'objet quad.obj
    # Ensuite on charge la texture (ici, l'image qu'on veut faire prédire)
    # Puis, on crée l'objet en lui donnant une position, une orientation et une forme.
    # Enfin, on applique la texture à l'objet (On affiche l'image sur l'objet qui sert ici de tableau)
    def create_picture(position, orientation, new_img_path):
        visualShapeId = p.createVisualShape(p.GEOM_MESH, fileName='./obj/quad.obj')
        textureUniqueId = p.loadTexture(new_img_path)
        multiBodyId = p.createMultiBody(
            baseVisualShapeIndex=visualShapeId,
            basePosition=position,
            baseOrientation=p.getQuaternionFromEuler(orientation))
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
            # Pour ne pas avoir plusieurs images superposées, on vérifie qu'il n'existe pas déjà
            # Un objet de créé, sinon on le supprime avant d'en créer un nouveau.
            if imgIsThere==False :
                if keyboard.is_pressed('c'):
                    if predState==False:
                        # L'image met quelques secondes à charger
                        print("Vous avez demandé une image. Veuillez patienter.")
                        random_file = random.choice(os.listdir(kaggleTestDatasetPath))
                        newTextureId = kaggleTestDatasetPath+'/'+random_file
                        img = create_picture(position=[2.5,0,1], orientation=[0,0,0], new_img_path=newTextureId)
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
                        img = create_picture(position=[2.5,0,1], orientation=[0,0,0], new_img_path=newTextureId)
                        print("La nouvelle image est arrivée saine et sauve. Re-bisous.")
                        imgIsThere = True

            # On appuie sur X pour débuter la prédiction       
            if keyboard.is_pressed('x'):
                predState = True
                print("Début de la prédiction...")
                a = predict_animal(capture, model)
                print("Pepper a prédit : ", a)
                # Pepper miaule 
                if a == 'Chat':
                    print('Miaou :3')
                    pepper.goToPosture("Crouch", 0.6)
                    time.sleep(1)
                    pepper.goToPosture("Stand", 0.6)
                # Pepper aboie
                else :
                    print('BARK! BARK! BARK!')
                    pepper.goToPosture("StandZero", 0.6)
                    time.sleep(1)
                    pepper.goToPosture("Stand", 0.6)
                predState = False
                print("Fin de la prédiction.")
                
            
    except KeyboardInterrupt:
        simulation_manager.stopSimulation(client_id)