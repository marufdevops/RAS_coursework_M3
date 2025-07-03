from controller import Robot
import numpy as np
import kinpy as kp  # pip install kinpy

TIME_STEP = 32
URDF_FN = '../../resources/ur5e_2f85_camera.urdf'


class UR5e(Robot):
    def __init__(self):
        """
        initialise robot, kinematic chain, motors, and sensors
        """
        super().__init__()
        
        # set up kinematic chain using wrist_3_link as end-effector link
        end_effector_link = 'wrist_3_link'
        with open(URDF_FN, 'r') as f:
            urdf_content = f.read()
        self.chain = kp.build_serial_chain_from_urdf(urdf_content, end_effector_link)
        print('kinematic chain:')
        print(self.chain)
        
        # initialise motors and their position sensors
        self.motors = [
            self.getDevice("shoulder_pan_joint"),
            self.getDevice("shoulder_lift_joint"),
            self.getDevice("elbow_joint"),
            self.getDevice("wrist_1_joint"),
            self.getDevice("wrist_2_joint"),
            self.getDevice("wrist_3_joint"),
        ]
        for m in self.motors:
            m.getPositionSensor().enable(TIME_STEP)
            m.setVelocity(0.8)
        self.getDevice("camera").enable(TIME_STEP)
            
        print('robot initialised.')

    def joint_pos(self):
        """
        returns the current joint position of the robot
        :return: (6,) nd-array
        """
        joint_pos = np.asarray([m.getPositionSensor().getValue() for m in self.motors])
        return joint_pos
        
    def forward_kinematics(self, joint_pos=None):
        """
        computes the end-effector pose for given joint position.
        if joint position is None, uses the robot's current joint position.
        
        :param joint_pos: list of joint positions, optional
        :return: kinpy.Transform object
        """
        if joint_pos is None:
            joint_pos = self.joint_pos()

        ee_pose = self.chain.forward_kinematics(joint_pos.tolist())
        return ee_pose
        
    def inverse_kinematics(self, target_pose):
        """
        for a given target pose as kinpy.Transform, returns a corresponding joint position
        
        :param target_pose: kinpy.Transform, target end-effector pose
        :return: nd-array, joint position that might (!) reach the end-effector pose
        """
        ik_result = self.chain.inverse_kinematics(target_pose, self.joint_pos().tolist())
        return ik_result
        
    def move_PTP(self, target_joint_pos, max_speed=0.8):
        """
        blocking behaviour, moves the robot to the desired joint position.
        PTP motion, non-synchronised. Each joint travels at maximum speed.
        
        :param target_joint_pos: list of joint positions
        :param max_speed: float, maximum joint velocity (default 0.8)
        """
        if len(target_joint_pos) != len(self.motors):
            raise ValueError('target joint configuration has unexpected length')
            
        # set target position for each motor, with maximum speed
        for pos, m in zip(target_joint_pos, self.motors):
            m.setPosition(pos)
            m.setVelocity(max_speed)
            
        # step through simulation for 5 seconds
        for i in range(5000 // TIME_STEP):
            self.step()
            # todo:
            # need to exit when robot has reached the target position
            
            
    def move_sync_PTP(self, target_joint_pos, max_speed=0.8):
        """
        blocking behaviour, moves the robot to the desired joint position.
        PTP motion, synchronised.
        All joints arrive at the target position at the same time.
        
        :param target_joint_pos: list of joint positions
        :param max_speed: float, maximum joint velocity (default 0.8)
        """
        if len(target_joint_pos) != len(self.motors):
            raise ValueError('target joint configuration has unexpected length')
            
        # calculate velocity gains for synchronised movement
        abs_distances = np.abs(target_joint_pos - self.joint_pos())
        max_dist = np.max(abs_distances)
        velocity_gains = abs_distances / max_dist
            
        # set target positions and individual target velocities
        for pos, m, gain in zip(target_joint_pos, self.motors, velocity_gains):
            m.setPosition(pos)
            m.setVelocity(gain * max_speed)
            
        # step through simulation for five seconds (this is a blocking behaviour)
        for i in range(5000 // TIME_STEP):
            self.step()
            # todo:
            # return early if target position reached before 5 seconds have passed
        
    
        
def tutorial_PTP():
    robot = UR5e()
    
    # define some robot poses as joint configurations
    home_pos = [1.57, -1.57, 1.57, -1.57, -1.57, 0.0]
    pos2 = [3.14, -1.2, 1.57, -1.57, -1.57, 2.57]
    
    # move to home pose
    robot.move_PTP(home_pos)
    
    # keep moving between the two poses
    while True:
        print('move PTP')
        robot.move_PTP(pos2)
        robot.move_PTP(home_pos)
        
        print('move sync PTP')
        robot.move_sync_PTP(pos2)    
        robot.move_sync_PTP(home_pos)
    

if __name__ == '__main__':
    tutorial_PTP()
