import numpy as np
import kinpy as kp
from scipy.spatial.transform import Rotation as R
from UR5e import UR5e



def main():
    # initialise robot and move to home position
    robot = UR5e()
    robot.move_to_joint_pos(robot.home_pos)  # this is synchronised PTP with a timeout
    
    # use FK to get end-effector pose
    tf_ee = robot.forward_kinematics()  # check UR5e.py, this is preset to be the TCP (Tool Center Point)
    
    # print out pos and rotation in different representations
    print('pos', tf_ee.pos)
    print('rot quaternion', tf_ee.rot)  # quaternion [w, x, y, z]
    print('rot mat', tf_ee.matrix()[:3, :3])  # 3x3 rotation matrix

    # Convert quaternion to Euler angles (RPY)
    quat = tf_ee.rot  # [w, x, y, z] format
    # scipy expects [x, y, z, w] format
    scipy_quat = [quat[1], quat[2], quat[3], quat[0]]
    r = R.from_quat(scipy_quat)
    euler_angles = r.as_euler('xyz', degrees=False)  # roll, pitch, yaw
    print('rot rpy', euler_angles)
    
    # to go and grasp the cube, we need to use inverse kinematics
    # the cube's position is [0, 0.5, 0.03] (from scene tree)
    # the orientation is 45 degree = pi/4
    # in practice, you will have to detect position and orientation from the camera
    tf_target = kp.Transform(
        pos = np.array([0, 0.5, 0.03]),
        rot = np.array([-np.pi/2, 0, -np.pi/4])  # as rpy
    )
    # calculate IK
    joint_pos = robot.inverse_kinematics(tf_target)
    # move robot
    robot.move_to_joint_pos(joint_pos)
    
    # close gripper (hopefully grasped!)
    robot.close_gripper()
    
    # move back to home configuration - note that we use very slow velocity
    robot.move_to_joint_pos(robot.home_pos, velocity=0.1, timeout=10)
    
    # that's all for this tutorial.
        
        
if __name__ == '__main__':
    main()
 