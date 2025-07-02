#
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
#
from controller import Robot
import numpy as np


class CameraMount(Robot):
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        
        # get devices
        self.cam = self.getDevice("external_camera")
        self.emitter = self.getDevice("ext_cam_emitter")
        
        # activate / init devices
        self.cam.enable(self.timestep)
        
        # send camera specs once
        self.emitter.send([self.cam.getHeight(), self.cam.getWidth()])
        
    def send_depth_image(self):
        """
        gets the depth image and sends it using the emitter
        """
        image = self.cam.getRangeImage()  # as list
        self.emitter.send(image)
        
    def run(self):
        """
        starts the robot controller, continuously taking depth images and sending them
        """
        while self.step(self.timestep) != -1:
            self.send_depth_image()
        
        
        
if __name__ == '__main__':
    ctrl = CameraMount()
    ctrl.run()
 