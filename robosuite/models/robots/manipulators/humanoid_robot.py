import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Humanoid(ManipulatorModel):
    """
    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        # super().__init__(xml_path_completion("robots/humanoid/robot.xml"), idn=idn)
        super().__init__(xml_path_completion("robots/humanoid/body_single_arm.xml"), idn=idn)

    @property
    def default_mount(self):
        # return "RethinkMount"
        return "BoxMount"

    @property
    def default_gripper(self):
        # return "Robotiq85Gripper"
        # return None
        return "HandLeft"

    @property
    def default_controller_config(self):
        return "default_ur5e"

    @property
    def init_qpos(self):
        return np.array([0., 0, 0, 0, -0.22, 0, 0], dtype=np.float64)

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"

    @property
    def _eef_name(self):
        """
        XML eef name for this robot to which grippers can be attached. Note that these should be the raw
        string names directly pulled from a robot's corresponding XML file, NOT the adjusted name with an
        auto-generated naming prefix

        Returns:
            str: Raw XML eef name for this robot (default is "right_hand")
        """
        return "left_hand"
