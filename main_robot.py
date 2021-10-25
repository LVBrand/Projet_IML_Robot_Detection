# installer : qibullet, opencv-contrib-python (avec pip)
from qibullet import SimulationManager
from qibullet import PepperVirtual
import pybullet as p
import cv2
import time
import convnet

if __name__ == "__main__":
    simulation_manager = SimulationManager()
    client_id = simulation_manager.launchSimulation(gui=True)

    # Pepper, le robot raciste
    pepper = simulation_manager.spawnPepper(
        client_id,
        spawn_ground_plane=True)


    # Création d'une camera
    handle = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_TOP)

    # Laser
    pepper.showLaser(True)
    pepper.subscribeLaser()

    