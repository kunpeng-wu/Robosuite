import argparse
import time

import numpy as np
import robosuite as suite
import os
import json
from robosuite.wrappers import VisualizationWrapper
from robosuite.utils.input_utils import input2action, input2action_Bimanual
from robosuite.devices import Keyboard, KeyboardBimanual, KeyboardSelf


def control_bimanual(args):
    controller_name = ''
    if args.controller == 'osc':
        controller_name = 'OSC_POSE'

    controller_config = suite.load_controller_config(default_controller=controller_name)
    config = {
        "env_name": 'TwoArmLift',
        "robots": 'HumanoidTwoArm',
        "controller_configs": controller_config,
    }

    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera='agentview',
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    str = env.model.get_xml()

    env = VisualizationWrapper(env, indicator_configs=None)

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    device = KeyboardBimanual(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    env.viewer.add_keypress_callback(device.on_press)

    env.reset()
    cam_id = 0
    num_cam = len(env.sim.model.camera_names)
    env.render()

    last_grasp = 0

    device.start_control()

    while True:
        # for _ in range(100):
        # active_robot = env.robots[args.arm == 'left']
        action = input2action_Bimanual(
            device=device, robot=env.robots[0]
        )
        # print(action)
        # if action is not None and action[0] != 0:
        #     print(action)
        if action is None:
            break

        obs, reward, done, info = env.step(action)
        env.render()
        # press 'q' to exit
    env.close()


def control_single(args):
    controller_name = ''
    if args.controller == 'osc':
        controller_name = 'OSC_POSE'

    controller_config = suite.load_controller_config(default_controller=controller_name)
    config = {
        "env_name": args.environment,
        # "env_name": "PickPlaceCan",
        # "robots": args.robots,
        # "robots": 'UR5e',
        # "robots": 'Panda',
        "robots": 'Humanoid',
        "controller_configs": controller_config,
    }

    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera='robot0_bodyview',
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    str = env.model.get_xml()

    env = VisualizationWrapper(env, indicator_configs=None)

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    device = KeyboardSelf(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    env.viewer.add_keypress_callback(device.on_press)

    env.reset()
    cam_id = 0
    num_cam = len(env.sim.model.camera_names)
    env.render()

    last_grasp = 0

    device.start_control()

    while True:
    # for _ in range(100):
        active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config
        )
        if action is None:
            break
        # action = np.clip(action, -1, 1)
        # if action[:6].any() != 0:
        #     print(action[:6])
        obs, reward, done, info = env.step(action)
        env.render()
        # press 'q' to exit
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=0.05, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    control_single(args)
    # control_bimanual(args)

