"""
Defines GripperTester that is used to test the physical properties of various grippers
"""
import xml.etree.ElementTree as ET

import numpy as np

import robosuite.macros as macros
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.world import MujocoWorldBase
from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
from robosuite.utils.mjcf_utils import array_to_string, new_actuator, new_joint
import robosuite.utils.transform_utils as T


class RobotTwoArmTester:
    """
    A class that is used to test gripper

    Args:
        gripper (GripperModel): A gripper instance to be tested
        pos (str): (x y z) position to place the gripper in string form, e.g. '0 0 0.3'
        quat (str): rotation to apply to gripper in string form, e.g. '0 0 1 0' to flip z axis
        gripper_low_pos (float): controls the gipper y position, larger -> higher
        gripper_high_pos (float): controls the gipper y high position larger -> higher,
            must be larger than gripper_low_pos
        box_size (None or 3-tuple of int): the size of the box to grasp, None defaults to [0.02, 0.02, 0.02]
        box_density (int): the density of the box to grasp
        step_time (int): the interval between two gripper actions
        render (bool): if True, show rendering
    """

    def __init__(
        self,
        robot,
        pos,
        quat,
        gripper,
        gripper_low_pos,
        gripper_high_pos,
        box_size=None,
        box_density=10000,
        step_time=400,
        render=True,
    ):
        # define viewer
        self.viewer = None

        world = MujocoWorldBase()
        # Add a table
        arena = TableArena(table_full_size=(0.4, 0.4, 0.1), table_offset=(0, 0, 0.1), has_legs=False)
        world.merge(arena)

        # Add a gripper
        self.robot_model = robot
        self.gripper = {}
        self.has_gripper = {}
        self.eef_rot_offset = {}
        for arm in self.arms:
            self.gripper[arm] = gripper[arm]
            self.has_gripper[arm] = True
            self.robot_model.add_gripper(self.gripper[arm], self.robot_model.eef_name[arm])
            self.eef_rot_offset[arm] = T.quat_multiply(self.robot_model.hand_rotation_offset[arm], self.gripper[arm].rotation_offset)
        # Create another body with a slider joint to which we'll add this gripper
        robot_body = ET.Element("body")
        robot_body.set("pos", pos)
        robot_body.set("quat", quat)  # flip z
        robot_body.append(new_joint(name="gripper_z_joint", type="slide", axis="0 0 -1", damping="50"))
        # Add all gripper bodies to this higher level body
        for body in robot.worldbody:
            robot_body.append(body)
        # Merge the all of the gripper tags except its bodies
        world.merge(robot, merge_body=None)
        # Manually add the higher level body we created
        world.worldbody.append(robot_body)
        # Create a new actuator to control our slider joint
        world.actuator.append(new_actuator(joint="gripper_z_joint", act_type="position", name="gripper_z", kp="500"))

        # Add an object for grasping
        # density is in units kg / m3
        TABLE_TOP = [0, 0, 0.09]
        if box_size is None:
            box_size = [0.02, 0.02, 0.02]
        box_size = np.array(box_size)
        self.cube = BoxObject(
            name="object", size=box_size, rgba=[1, 0, 0, 1], friction=[1, 0.005, 0.0001], density=box_density
        )
        object_pos = np.array(TABLE_TOP + box_size * [0, 0, 1])
        mujoco_object = self.cube.get_obj()
        # Set the position of this object
        mujoco_object.set("pos", array_to_string(object_pos))
        # Add our object to the world body
        world.worldbody.append(mujoco_object)

        # add reference objects for x and y axes
        x_ref = BoxObject(
            name="x_ref", size=[0.01, 0.01, 0.01], rgba=[0, 1, 0, 1], obj_type="visual", joints=None
        ).get_obj()
        x_ref.set("pos", "0.2 0 0.105")
        world.worldbody.append(x_ref)
        y_ref = BoxObject(
            name="y_ref", size=[0.01, 0.01, 0.01], rgba=[0, 0, 1, 1], obj_type="visual", joints=None
        ).get_obj()
        y_ref.set("pos", "0 0.2 0.105")
        world.worldbody.append(y_ref)
        self.world = world
        self.is_render = render
        self.simulation_ready = False
        self.step_time = step_time
        self.cur_step = 0
        if gripper_low_pos > gripper_high_pos:
            raise ValueError(
                "gripper_low_pos {} is larger " "than gripper_high_pos {}".format(gripper_low_pos, gripper_high_pos)
            )
        self.gripper_low_pos = gripper_low_pos
        self.gripper_high_pos = gripper_high_pos
        self.controller = {}
        self.controller_config = {}
        self.control_timestep = 1 / 20
        self.model_timestep = 0.002

    def start_simulation(self):
        """
        Starts simulation of the test world
        """
        model_xml = self.world.get_xml()
        self.sim = MjSim.from_xml_string(model_xml)

        if self.is_render:
            self.viewer = OpenCVRenderer(self.sim)
            # We also need to add the offscreen context
            if self.sim._render_context_offscreen is None:
                render_context = MjRenderContextOffscreen(self.sim, device_id=-1)
                self.sim.add_render_context(render_context)
        self.sim_state = self.sim.get_state()

        # For gravity correction
        gravity_corrected = ["gripper_z_joint"]
        self._gravity_corrected_qvels = [self.sim.model.get_joint_qvel_addr(x) for x in gravity_corrected]

        self.gripper_z_id = self.sim.model.actuator_name2id("gripper_z")
        self.gripper_z_is_low = False
        self.gripper_is_closed = False

        self.object_id = self.sim.model.body_name2id(self.cube.root_body)
        object_default_pos = self.sim.data.body_xpos[self.object_id]
        self.object_default_pos = np.array(object_default_pos, copy=True)

        self.reset()
        self.simulation_ready = True

    def reset(self):
        """
        Resets the simulation to the initial state
        """
        self._load_controller()
        self.base_pos = self.sim.data.get_body_xpos(self.robot_model.root_body)
        self.base_ori = T.mat2quat(self.sim.data.get_body_xmat(self.robot_model.root_body).reshape((3, 3)))
        for arm in self.arms:
            self.controller[arm].update_base_pose(self.base_pos, self.base_ori)
        self.sim.set_state(self.sim_state)
        self.cur_step = 0

    def _load_controller(self):
        from robosuite.controllers import controller_factory, load_controller_config
        for arm in self.arms:
            self.controller_config[arm] = load_controller_config(default_controller="OSC_POSE")

            self.controller_config[arm]["robot_name"] = self.robot_model
            self.controller_config[arm]["sim"] = self.sim
            self.controller_config[arm]["eef_name"] = self.gripper[arm].important_sites["grip_site"]
            self.controller_config[arm]["eef_rot_offset"] = self.eef_rot_offset[arm]
            self.controller_config[arm]["ndim"] = self.single_dof
            (start, end) = (None, self.single_dof) if arm == "right" else (self.single_dof, None)
            self.controller_config[arm]["joint_indexes"] = {
                "joints": self.joint_indexes[start:end],
                "qpos": self.joint_pos_indexes[start:end],
                "qvel": self.joint_vel_indexes[start:end],
            }
            self.controller_config[arm]["actuator_range"] = (
                self.torque_limits[0][start:end],
                self.torque_limits[1][start:end],
            )
            self.controller_config[arm]["policy_freq"] = 20
            # Instantiate the relevant controller
            self.controller[arm] = controller_factory(self.controller_config[arm]["type"], self.controller_config[arm])

    @property
    def torque_limits(self):
        """
        Torque lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) torque values
                - (np.array) maximum (high) torque values
        """
        # Torque limit values pulled from relevant robot.xml file
        low = self.sim.model.actuator_ctrlrange[self.actuator_ids, 0]
        high = self.sim.model.actuator_ctrlrange[self.actuator_ids, 1]

        return low, high

    def close(self):
        """
        Close the viewer if it exists
        """
        if self.viewer is not None:
            self.viewer.close()

    def render(self):
        if self.is_render:
            self.viewer.render()

    # def step(self, action):
    #     """
    #     Forward the simulation by one timestep
    #
    #     Raises:
    #         RuntimeError: if start_simulation is not yet called.
    #     """
    #     if not self.simulation_ready:
    #         raise RuntimeError("Call start_simulation before calling step")
    #     # if self.gripper_z_is_low:
    #     #     self.sim.data.ctrl[self.gripper_z_id] = self.gripper_low_pos
    #     # else:
    #     #     self.sim.data.ctrl[self.gripper_z_id] = self.gripper_high_pos
    #     arm_action = action[:self.robot_model.dof]
    #     gripper_action = action[self.robot_model.dof:]
    #     self._apply_arm_action(arm_action)
    #     self._apply_gripper_action(gripper_action)
    #     self._apply_gravity_compensation()
    #     self.sim.step()
    #     if self.render:
    #         self.viewer.render()
    #     self.cur_step += 1

    def step(self, action):
        if not self.simulation_ready:
            raise RuntimeError("Call start_simulation before calling step")
        policy_step = True
        for i in range(int(self.control_timestep / self.model_timestep)):   # 0.05 / 0.002 = 25
            self.sim.forward()
            self._pre_action(action, policy_step)
            self.sim.step()
            policy_step = False

    def _pre_action(self, action, policy_step=False):
        arm_action = {"right": action["right_arm"], "left": action["left_arm"]}
        gripper_action = {"right": action["right_hand"], "left": action["left_hand"]}
        self._apply_arm_action(arm_action, policy_step)
        self._apply_gripper_action(gripper_action)
        self._apply_gravity_compensation()

    def _apply_arm_action(self, action, policy_step=False):
        torques = np.array([])
        for arm in self.arms:
            if policy_step:
                self.controller[arm].set_goal(action[arm])
            torques = np.concatenate((torques, self.controller[arm].run_controller()))
        low, high = self.torque_limits
        torques = np.clip(torques, low, high)
        self.sim.data.ctrl[self.actuator_ids] = torques
        # print(self.sim.data.ctrl[self.actuator_ids])
        # print(self.sim.data.qpos[self.joint_pos_indexes])

    def _apply_gripper_action(self, action):
        """
        Applies binary gripper action

        Args:
            action (np.array): Action to apply. Should be -1 (open) or 1 (closed)
        """
        for arm in self.arms:
            gripper_action_actual = self.gripper[arm].format_action(action[arm])
            # rescale normalized gripper action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange[self.gripper_actuator_ids[arm]]
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            applied_gripper_action = bias + weight * gripper_action_actual
            self.sim.data.ctrl[self.gripper_actuator_ids[arm]] = applied_gripper_action
            # print(self.sim.data.ctrl[self.gripper_actuator_ids])

    def _apply_gravity_compensation(self):
        """
        Applies gravity compensation to the simulation
        """
        self.sim.data.qfrc_applied[self._gravity_corrected_qvels] = self.sim.data.qfrc_bias[
            self._gravity_corrected_qvels
        ]

    # def loop(self, total_iters=1, test_y=False, y_baseline=0.01):
    #     """
    #     Performs lower, grip, raise and release actions of a gripper,
    #             each separated with T timesteps
    #
    #     Args:
    #         total_iters (int): Iterations to perform before exiting
    #         test_y (bool): test if object is lifted
    #         y_baseline (float): threshold for determining that object is lifted
    #     """
    #     seq = [(False, False), (True, False), (True, True), (False, True)]
    #     for cur_iter in range(total_iters):
    #         for cur_plan in seq:
    #             self.gripper_z_is_low, self.gripper_is_closed = cur_plan
    #             for step in range(self.step_time):
    #                 self.step()
    #         if test_y:
    #             if not self.object_height > y_baseline:
    #                 raise ValueError(
    #                     "object is lifed by {}, ".format(self.object_height)
    #                     + "not reaching the requirement {}".format(y_baseline)
    #                 )

    @property
    def arms(self):
        return "right", "left"

    @property
    def object_height(self):
        """
        Queries the height (z) of the object compared to on the ground

        Returns:
            float: Object height relative to default (ground) object position
        """
        return self.sim.data.body_xpos[self.object_id][2] - self.object_default_pos[2]

    @property
    def actuator_ids(self):
        return [self.sim.model.actuator_name2id(x) for x in self.robot_model.actuators]

    @property
    def gripper_actuator_ids(self):
        return {
            "right": [self.sim.model.actuator_name2id(x) for x in self.gripper["right"].actuators],
            "left": [self.sim.model.actuator_name2id(x) for x in self.gripper["left"].actuators],
        }

    @property
    def ee_pos(self):
        return {
            "right": np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.eef_name["right"])]),
            "left": np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.eef_name["left"])]),
        }

    @property
    def eef_name(self):
        return {
            "right": self.gripper["right"].important_sites["grip_site"],
            "left": self.gripper["left"].important_sites["grip_site"],
        }

    @property
    def joint_indexes(self):
        return [self.sim.model.joint_name2id(joint) for joint in self.robot_model.joints]

    @property
    def joint_pos_indexes(self):
        return [self.sim.model.get_joint_qpos_addr(jnt) for jnt in self.robot_model.joints]

    @property
    def joint_vel_indexes(self):
        return [self.sim.model.get_joint_qvel_addr(jnt) for jnt in self.robot_model.joints]

    @property
    def single_dof(self):
        """
        Returns:
            int: the index that correctly splits the right arm from the left arm joints
        """
        return int(len(self.robot_model.joints) / 2)

    @property
    def robot_dof(self):
        return len(self.robot_model.joints)


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    from robosuite.models.robots.robot_model import create_robot
    robot = create_robot("HumanoidTwoArm")
    # robot = create_robot("Panda")
    from robosuite.models.grippers import gripper_factory, NullGripper
    gripper = {}

    gripper["right"] = gripper_factory("HandRight", idn="_0_right")
    gripper["left"] = gripper_factory("HandLeft", idn="_0_left")

    tester = RobotTwoArmTester(robot, "0 0 0.2", "1 0 0 0", gripper, 0, 0.1)

    str = tester.world.get_xml()
    tester.start_simulation()

    # robot
    print(tester.single_dof)
    print(tester.robot_model.dof)
    print(tester.robot_model.joints)
    print(tester.joint_indexes)
    print(tester.joint_pos_indexes)
    print(tester.joint_vel_indexes)
    print(tester.robot_model.actuators)
    print(tester.actuator_ids)
    # print(tester.controller.control_limits)

    # gripper
    for arm in tester.arms:
        print(tester.gripper[arm].dof)
        print(tester.gripper[arm].joints)
        print([tester.sim.model.joint_name2id(jnt) for jnt in tester.gripper[arm].joints])
        print(tester.gripper[arm].actuators)
        print(tester.gripper_actuator_ids[arm])

    tester.viewer.set_camera(0)

    action = {}
    action["right_arm"] = np.array([0., 0, -0.1, 0, 0, 0])
    action["right_hand"] = np.array([-1])
    action["left_arm"] = np.array([0., 0.1, 0, 0, 0, 0])
    action["left_hand"] = np.array([1])

    import time
    start_time = time.time()
    while time.time() - start_time < 3:
        # if 1 < time.time() - start_time:
        #     action[0] = 0.05
        # else:
        #     action[0] = 0
        tester.step(action)
        tester.render()
        # print(tester.ee_pos["right"])
    tester.close()

