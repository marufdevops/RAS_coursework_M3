import numpy as np
import kinpy as kp
from scipy.spatial.transform import Rotation as R

from util import display_image, normalize_depth
from UR5e import UR5e


"""
MISSION 2: Control of a Robot Manipulator Arm for a Pick and Place Task
Learning outcomes: 1, 2, 3

Scenario:
A manipulator arm is equipped with a depth camera at its end-effector. 
There is a second camera on the top of the frame, looking at the scene
from above.
The existing controller already provides behaviours to move
the robot in joint space and to open/close the gripper.
Forward and inverse kinematics are provided.
There are five cubes randomly distributed in the robot's work space,
and there is a crate, which is always in the same spot.

Task: 
Use the depth camera to detect the cubes. Implement a controller for the 
manipulator arm that clears the objects from the table and drops 
them into the crate.

Hints:
1) INSTALLATION
The project requires the kinpy library, which you should install into
your environment. It is an open source library, so you can directly 
look at the code on github.
Installation with conda & pip:
conda create -n M3 python=3.10
conda activate M3
pip install numpy scipy kinpy opencv-python

2) COORDINATE TRANSFORMS
You can move the robot in workspace by computing the inverse kinematics.
The kinpy.Transform object describes the pose in workspace. 
See more details here: 
https://github.com/neka-nat/kinpy/blob/master/kinpy/transform.py
It consists of a pos (position as [x, y, z]) and rot (rotation). 
The rotation is displayed as a quaternion in the [w, x, y, z] format. 
As mentioned in the tutorial, you can also display it as RPY angles.
You only need to adjust the Y angle (yaw).

3) GRASPING
Simulating contact-rich tasks is very difficult. You might notice that
the cubes sometimes act unexpectedly when being grasped.
It is generally a good idea to align the gripper as best as possible to the 
parallel surfaces of the cube, and to only move with low velocity when
picking up an object. Still, it sometimes falls. Don't worry!
"""

def main():
    # initialise robot
    robot = UR5e()
    
    # define configuration init_pos as intermediate waypoint
    # avoids collisions with the frame
    init_pos = [0, -1.57, 1.57, -1.57, -1.57, 0.0]
    robot.move_to_joint_pos(init_pos)
    
    # now we can move to the robot's home position
    # this demonstrates how you can adjust speed and timeout
    # let's move a little bit slower for up to 10 seconds
    # note: it's just a demonstration how you can change this,
    #       the robot can go fast here
    robot.move_to_joint_pos(robot.home_pos, velocity=0.3, timeout=10)
    
    # this is how you close and open the gripper
    robot.close_gripper()
    robot.open_gripper()
    
    # there are two depth cameras avaialble
    # one on the wrist of the robot
    depth = robot.get_wrist_depth_image()
    
    # and an external one in the middle of the frame
    external_depth = robot.get_external_depth_image()
    
    # you can use either, or both, as you like.
    
    # for displaying images, some functions from utils may be helpful (see below)
    # the robot automatically shows the (normalized) depth images
    # normalization is just so it is better visible - you should work with the raw data
    display_image(normalize_depth(depth), 'depth image')
    display_image(normalize_depth(external_depth), 'external depth image')
    
    # now it's your turn! you need to:
    # detect cubes
    # move robot to a cube and align gripper orientation
    # grasp cube
    # move above the basket and drop cube
    # until all cubes are gone (hopefully :-))
        
        
if __name__ == '__main__':
    main()
 