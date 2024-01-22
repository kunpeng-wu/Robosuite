import numpy as np
import robosuite as suite
import robosuite.controllers
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import os
import json
import argparse


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # options["env_name"] = choose_environment()
    options["env_name"] = 'Lift'
    # options["robots"] = choose_robots(exclude_bimanual=True)
    # options["robots"] = 'UR5e'
    options["robots"] = 'Humanoid'
    # joint_dim = 6 if options["robots"] == "UR5e" else 7
    joint_dim = 6
    # controller_name = choose_controller()
    controller_name = 'OSC_POSE'
    # print(options)

    options["controller_configs"] = load_controller_config(default_controller=controller_name)
    # print(options["controller_configs"])

    controller_settings = {
        "OSC_POSE": [6, 6, 0.1],
    }
    action_dim = controller_settings[controller_name][0]
    # num_test_steps = controller_settings[controller_name][1]
    num_test_steps = 3
    test_value = controller_settings[controller_name][2]

    steps_per_action = 75
    steps_per_rest = 75

    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        horizon=(steps_per_action + steps_per_rest) * num_test_steps,
        control_freq=20,
    )
    print(env.robots[0].robot_joints)



    env.reset()
    env.viewer.set_camera(camera_id=0)

    n = 0
    gripper_dim = 0
    # print(env.robots)
    for robot in env.robots:
        gripper_dim = robot.gripper.dof
        n += int(robot.action_dim / (action_dim + gripper_dim))

    neutral = np.zeros(action_dim + gripper_dim)    # 7

    count = 0
    # print(n)
    # print(gripper_dim)
    # print(np.tile(neutral, n))
    print(env.action_spec)
    str = env.robots[0].robot_model.get_xml()

    while count < num_test_steps:
        action = neutral.copy()

        for i in range(steps_per_action):
            if controller_name in {"IK_POSE", "OSC_POSE"} and count > 1:
                # vec = np.zeros(3)
                # vec[count - 3] = test_value
                action[1] = 0.1
                action[6:] = 1

            # total_action = np.tile(action, n)   # np.tile(action, 1) == action
            env.step(action)
            env.render()
            print(env.robots[0]._hand_pos)
        for i in range(steps_per_rest):
            total_action = np.tile(neutral, n)
            env.step(total_action)
            env.render()
        count += 1

    env.close()