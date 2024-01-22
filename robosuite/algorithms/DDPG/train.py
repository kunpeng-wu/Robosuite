import numpy as np
import torch
import gym
import argparse
import os
import time
from datetime import datetime
import utils
import TD3
import DDPG
import robosuite as suite
from robosuite.wrappers import GymWrapper


# def eval_policy(policy, env_name, seed, eval_episodes=10):
#     """
#     Runs policy for X episodes and returns average reward
#     A fixed seed is used for the eval environment
#     """
#     eval_env = gym.make(env_name)
#
#     avg_reward = 0.
#     for _ in range(eval_episodes):
#         state, _ = eval_env.reset(seed=seed + 100)
#         truncated = False
#         timesteps = 0
#         while not truncated:
#             action = policy.select_action(np.array(state))
#             state, reward, done, truncated, info = eval_env.step(action)
#             avg_reward += reward
#             timesteps += 1
#
#     avg_reward /= eval_episodes
#
#     print("---------------------------------------")
#     print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
#     print("---------------------------------------")
#     return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG")  # Policy name (TD3, DDPG or OurDDPG)
    # parser.add_argument("--env_name", default="HalfCheetah-v4")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=5e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    env_name = 'Lift'
    robot = 'UR5e'

    file_name = f"{args.policy}_{env_name}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {env_name}, Seed: {args.seed}")
    print("---------------------------------------")

    # log_dir = "DDPG_logs/" + env_name + '/' + robot + '/'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # t_now = time.time()
    # time_str = datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')
    # log_f_name = log_dir + 'DDPG_' + env_name + "_log_" + time_str + ".csv"
    # print("logging at : " + log_f_name)
    # log_f = open(log_f_name, "w+")
    # log_f.write('episode,timestep,reward\n')

    # checkpoint_dir = "DDPG_preTrained/" + env_name + '/' + robot + '/'
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)

    # env = gym.make(args.env_name)
    env = GymWrapper(
        suite.make(
            env_name,
            robots=robot,
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        )
    )

    # Set seeds
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    # if args.load_model != "":
    #     policy_file = file_name if args.load_model == "default" else args.load_model
    #     policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, env_name, args.seed)]

    state, info = env.reset(seed=args.seed)
    truncated = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # for t in range(int(args.max_timesteps)):
    #
    #     episode_timesteps += 1
    #
    #     # Select action randomly or according to policy
    #     if t < args.start_timesteps:
    #         action = env.action_space.sample()
    #     else:
    #         action = (
    #                 policy.select_action(np.array(state))
    #                 + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
    #         ).clip(-max_action, max_action)
    #
    #     # Perform action
    #     next_state, reward, done, _, info = env.step(action)
    #     # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
    #
    #     # Store data in replay buffer
    #     replay_buffer.add(state, action, next_state, reward, float(truncated))
    #
    #     state = next_state
    #     episode_reward += reward
    #
    #     # Train agent after collecting sufficient data
    #     if t >= args.start_timesteps:
    #         policy.train(replay_buffer, args.batch_size)
    #
    #     if done:
    #         # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
    #         print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
    #
    #         # log average reward till last episode
    #         log_f.write('{},{},{}\n'.format(episode_num + 1, t + 1, round(episode_reward, 3)))
    #         log_f.flush()
    #
    #         # Reset environment
    #         state, _ = env.reset()
    #         done = False
    #         episode_reward = 0
    #         episode_timesteps = 0
    #         episode_num += 1
    #
    #     # Evaluate episode
    #     if (t + 1) % args.eval_freq == 0:
    #         # evaluations.append(eval_policy(policy, env_name, args.seed))
    #         # np.save(f"./results/{file_name}", evaluations)
    #         if args.save_model:
    #             policy.save(f"./models/{file_name}")
    time_step = 0
    max_epoch = 200  # 200 --> 1e5
    max_step = 500
    save_model_n_epoch = 50
    update_epoch = 4
    log_freq = max_step * 2
    print_freq = max_step * 10

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    for i_episode in range(1, max_epoch + 1):

        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, max_step + 1):

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            next_state, reward, done, _, _ = env.step(action)

            # saving reward and is_terminals
            replay_buffer.add(state, action, next_state, reward, float(done))

            time_step += 1
            state = next_state
            current_ep_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)

            # log in logging file
            # if time_step % log_freq == 0:
            #     # log average reward till last episode
            #     log_avg_reward = log_running_reward / log_running_episodes
            #     log_avg_reward = round(log_avg_reward, 4)
            #
            #     log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            #     log_f.flush()
            #
            #     log_running_reward = 0
            #     log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            # if time_step % save_model_freq == 0:
            #     print("--------------------------------------------------------------------------------------------")
            #     checkpoint_path = checkpoint_dir + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
            #     print("saving model at : " + checkpoint_path)
            #     ppo_agent.save(checkpoint_path)
            #     print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            #     print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print("epoch: {}, Timestep: {}, reward: {}".format(i_episode, time_step, current_ep_reward))

        # if i_episode % save_model_n_epoch == 0 and current_ep_reward > best_reward:
        #     print("--------------------------------------------------------------------------------------------")
        #     t_now = time.time()
        #     time_str = datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')
        #     checkpoint_path = checkpoint_dir + "PPO_{}_{}_{}_{}.pth".format(env_name, random_seed, time_str, round(current_ep_reward, 2))
        #     print("saving model at : " + checkpoint_path)
        #     ppo_agent.save(checkpoint_path)
        #     print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
        #     best_reward = current_ep_reward
        #     print("--------------------------------------------------------------------------------------------")

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

    # log_f.close()
    env.env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")
