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
        # end_effector_link = 'wrist_3_link'
        end_effector_link = 'TCP'
        print(f"Loading URDF from: {URDF_FN}")
        try:
            with open(URDF_FN, 'r') as f:
                urdf_content = f.read()
            print(f"URDF content length: {len(urdf_content)}")
            print(f"URDF content type: {type(urdf_content)}")
            self.chain = kp.build_serial_chain_from_urdf(urdf_content, end_effector_link)
        except Exception as e:
            print(f"Error loading URDF: {e}")
            raise
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
        
        :param target_joint_pos: list of joint positions
        :param max_speed: float, maximum joint velocity (default 0.8)
        """
        if len(target_joint_pos) != len(self.motors):
            raise ValueError('target joint configuration has unexpected length')
            
        for pos, m in zip(target_joint_pos, self.motors):
            m.setPosition(pos)
            m.setVelocity(max_speed)
            
        # step through simulation until timeout or position reached
        timeout = 5  # seconds
        for step in range(int(timeout * 1000) // TIME_STEP):
            self.step()

            # check if the robot is close enough to the target position
            if all(abs(target_joint_pos - self.joint_pos()) < 0.001):
                return True
                
        print('Timeout. Robot has not reached the desired target position.')
        return False
            
            
    def move_sync_PTP(self, target_joint_pos, max_speed=0.8):
        """
        blocking behaviour, moves the robot to the desired joint position with synchronised PTP.
        
        :param target_joint_pos: list of joint positions
        :param max_speed: float, maximum joint velocity (default 0.8)
        """
        if len(target_joint_pos) != len(self.motors):
            raise ValueError('target joint configuration has unexpected length')
            
        # todo: calculate velocity gains
        # i forgot to use the absolute values in the tutorial - added here
        abs_distances = np.abs(target_joint_pos - self.joint_pos())
        max_dist = np.max(abs_distances)
        velocity_gains = abs_distances / max_dist
            
        for pos, m, gain in zip(target_joint_pos, self.motors, velocity_gains):
            m.setPosition(pos)
            m.setVelocity(gain * max_speed)
            
        # step through simulation until timeout or position reached
        timeout = 5  # seconds
        for step in range(int(timeout * 1000) // TIME_STEP):
            self.step()

            # check if the robot is close enough to the target position
            if all(abs(target_joint_pos - self.joint_pos()) < 0.001):
                return True
                
        print('Timeout. Robot has not reached the desired target position.')
        return False
        
    
    
        
def tutorial_PTP():
    robot = UR5e()
    
    # define some robot poses as joint configurations
    home_pos = [1.57, -1.57, 1.57, -1.57, -1.57, 0.0]
    pos2 = [3.14, -1.2, 1.57, -1.57, -1.57, 2.57]
    # pos2 = [0, -1.57, 1.57, -1.57, -1.57, 0.0]
        
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
    
    
def tutorial_FK_IK():
    # init robot
    robot = UR5e()
    home_pos = [1.57, -1.57, 1.57, -1.57, -1.57, 0.0]
    robot.move_sync_PTP(home_pos)
    
    # use forward kinematics to get the pose of the end effector (the end of kinematic chain)
    tf = robot.forward_kinematics()
    print('forward kinematics:')
    print(tf)  # shows current rotation and position

    # we want to keep the orientation (tf.rot), but change the position to move to the can
    tf.pos = np.array([-0.2, 0.55, 0.1])
    
    # use inverse kinematics to find corresponding joint position and go there
    joint_target = robot.inverse_kinematics(tf)
    robot.move_sync_PTP(joint_target)


if __name__ == '__main__':
    # tutorial_PTP()
    tutorial_FK_IK()
