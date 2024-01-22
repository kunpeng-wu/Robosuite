from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    # options["env_name"] = choose_environment()
    # options["env_name"] = 'Lift'
    options["env_name"] = 'TwoArmLift'

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        # options["env_configuration"] = choose_multi_arm_config()
        options["env_configuration"] = "bimanual"

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            # options["robots"] = "Baxter"
            options["robots"] = "HumanoidTwoArm"
        # else:
        #     options["robots"] = []
        #     # Have user choose two robots
        #     print("A multiple single-arm configuration was chosen.\n")
        #     for i in range(2):
        #         print("Please choose Robot {}...\n".format(i))
        #         options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        # options["robots"] = choose_robots(exclude_bimanual=True)
        # options["robots"] = 'UR5e'
        options["robots"] = 'Humanoid'

    # Choose controller
    # controller_name = choose_controller()
    controller_name = 'OSC_POSE'

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)
    # str = env.model.get_xml()
    # Get action limits
    low, high = env.action_spec
    print(low, high)
    print(env.robots[0].robot_joints)
    print([env.sim.model.joint_name2id(jnt) for jnt in env.robots[0].robot_model.joints])
    print(env.robots[0].dof)
    print(env.robots[0].action_dim)
    print(env.robots[0].robot_model.actuators)
    print([env.sim.model.actuator_name2id(x) for x in env.robots[0].robot_model.actuators])
    print("")

    for arm in env.robots[0].arms:
        print(arm + ":")
        print(env.robots[0].gripper[arm].dof)
        print(env.robots[0].gripper[arm].joints)
        print([env.sim.model.joint_name2id(jnt) for jnt in env.robots[0].gripper[arm].joints])
        print(env.robots[0].gripper[arm].actuators)
        print([env.sim.model.actuator_name2id(x) for x in env.robots[0].gripper[arm].actuators])
        print("")

    # print(env.sim.model.actuator_name2id(x) for x in env.robot_model.actuators)
    # do visualization
    action = np.zeros_like(low)
    for i in range(100):
        # action = np.random.uniform(low, high)
        # action = np.zeros(36)
        # action[18:24] = [0, 0, 0, 0, 0, 0]
        action[5] = -0.1
        action[9] = 0.1
        action[6] = 1  # right
        action[-1] = -1 # left
        obs, reward, done, _ = env.step(action)
        env.render()
    env.close()
