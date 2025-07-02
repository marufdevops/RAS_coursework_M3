#
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
#
from controller import Supervisor
import numpy as np

from util import display_image, normalize_depth

TIME_STEP = 32


class RASRobot:
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #
    def __init__(self):
        self.__sup = Supervisor()
        self.__sup.getDevice("depth").enable(TIME_STEP)
        
        # set up receiver
        self.__receiver = self.__sup.getDevice("ext_cam_receiver")
        self.__receiver.enable(TIME_STEP)
        self.__ext_cam_w = None
        self.__ext_cam_h = None
        self.__external_depth_image_cache = None
        
        
        self.__total_cubes = 5
        
        # set motor velocity, initialise sensors
        self.motors = [
            self.__sup.getDevice("shoulder_pan_joint"),
            self.__sup.getDevice("shoulder_lift_joint"),
            self.__sup.getDevice("elbow_joint"),
            self.__sup.getDevice("wrist_1_joint"),
            self.__sup.getDevice("wrist_2_joint"),
            self.__sup.getDevice("wrist_3_joint"),
        ]
        for m in self.motors:
            m.getPositionSensor().enable(TIME_STEP)
            m.setVelocity(0.8)
        
        # initialise fingers
        self.__fingers = [
            self.__sup.getDevice('ROBOTIQ 2F-85 Gripper::left finger joint')
            # right finger mimics the left finger, so we only need to control one of them
        ]
        for finger in self.__fingers:
            finger.setVelocity(0.8)
            
        # shuffle the cubes
        self.__reset_scene()
        
        
    def __receive_data(self):
        """
        this function is used to receive data from the externally mounted camera
        it fills up the cache member variables, from which the image can be read
        """
        
        rec = self.__receiver  # shorthand
        
        while rec.getQueueLength() > 0:
            
            # first message is camera specs
            if not self.__ext_cam_h:
                data = rec.getInts()
                self.__ext_cam_h, self.__ext_cam_w = data
                print(f'camera specs received: ({self.__ext_cam_h}, {self.__ext_cam_w})')
                
            # all other messages are images
            else:
                # image comes as a list of floats
                data = rec.getFloats()
                img = np.array(data, copy=True).reshape(self.__ext_cam_h, self.__ext_cam_w)
                # store in cache
                self.__external_depth_image_cache = img
            
            # read in all available messages to have the latest depth image
            self.__receiver.nextPacket()


    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #   
    def __reset_scene(self):
        rng = np.random.default_rng()
        
        for i in range(self.__total_cubes):
            box = self.__sup.getFromDef(f'BOX{i+1}')
            rotation_field = box.getField('rotation')
            quaternion = rng.standard_normal(4)
            quaternion = quaternion / np.linalg.norm(list(quaternion))
            rotation_field.setSFRotation(quaternion.tolist())
     
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #       
    def close_gripper(self, timeout=1.2):
        """
        blocking behaviour that will close the gripper
        """
        for finger in self.__fingers:
            finger.setTorque(finger.getAvailableTorque()/2)
            
        for step in range(int(timeout * 1000) // TIME_STEP):
            self.step()
    
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #        
    def open_gripper(self, timeout=1.2):
        """
        blocking behaviour that will open the gripper
        """
        for finger in self.__fingers:
            finger.setPosition(0)
            
        for step in range(int(timeout * 1000) // TIME_STEP):
            self.step()
    
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #        
    def step(self):
        """
        step function of the simulation
        """
        self.__sup.step()
        self.__receive_data()  # this needs to be done every step
        
        # this is only for debug visualization purposes
        depth = self.get_wrist_depth_image()
        display_image(normalize_depth(depth), 'wrist depth')
        depth2 = self.get_external_depth_image()
        if depth2 is not None:
            # display_image(normalize_depth(depth2), 'external depth')
            display_image(depth2, 'external depth')
    
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #        
    def get_wrist_depth_image(self):
        """ 
        This method returns a 2-dimensional array containing the depth of each 
        pixel. This is from the robot's wrist.
        
        :returns: (64, 128) ndarray
        """
        device = self.__sup.getDevice("depth")
        ret = device.getRangeImage(data_type="buffer")
        ret = np.ctypeslib.as_array(ret, (device.getHeight(), device.getWidth()))
        return ret
                
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #        
    def get_external_depth_image(self):
        """ 
        This method returns a 2-dimensional array containing the depth of each 
        pixel. This is from the external mount.
        It returns None if no image is avaialble yet, this should only happen 
        at the start of the simulation.
        
        :returns: (64, 128) ndarray or None if no image available yet.
        """
        if self.__external_depth_image_cache is not None:
            return self.__external_depth_image_cache.copy()
        else:
            return None
        