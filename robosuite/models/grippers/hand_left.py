import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class HandLeft(GripperModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/hand_left.xml"), idn=idn)

    def format_action(self, action):
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        # self.current_action[:4] = np.clip(self.current_action[:4] + self.speed * np.sign(action[0]), -1.0, 1.0)
        # for i in range(1, 5):
        #     self.current_action[i * 2 + 2:i * 2 + 4] = np.clip(self.current_action[i * 2 + 2:i * 2 + 4] + self.speed * np.sign(action[i]), -1.0, 1.0)

        return self.current_action

    @property
    def init_qpos(self):
        """
        :return: init pos of each joint
        """
        return np.zeros(9)

    @property
    def _important_geoms(self):
        return {
            "left_fingerpad": ["finger_14_left_collision"],
            "right_fingerpad": ["finger_23_left_collision"],
        }

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        """
        :return: number of actuators
        """
        return 5

    @property
    def control_dim(self):
        return 1