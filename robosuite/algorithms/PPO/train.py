import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
from PPO import PPO
import robosuite as suite
from robosuite.wrappers import GymWrapper


################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    # env_name = "LunarLander-v2"
    env_name = "Lift"
    robot = 'UR5e'

    has_continuous_action_space = True  # continuous action space; else discrete

    max_step = 500  # max timesteps in one episode
    max_training_timesteps = int(1e4)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_step * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_step * 2  # log avg reward in the interval (in num timesteps)
    # save_model_freq = int(1e5)  # save model frequency (in num timesteps)
    # save_model_freq = int(max_training_timesteps / 5)  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_step * 4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    # env = gym.make(env_name)
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

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################
    # log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs/" + env_name + '/' + robot + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get number of log files in log directory
    # run_num = 0
    # current_num_files = next(os.walk(log_dir))[2]
    # run_num = len(current_num_files)

    # create new log file for each run
    t_now = time.time()
    time_str = datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')
    log_f_name = log_dir + 'PPO_' + env_name + "_log_" + time_str + ".csv"
    # print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    ################### checkpointing ###################
    run_num_pretrained = 2  # change this to prevent overwriting weights in same env_name folder
    checkpoint_dir = "PPO_preTrained/" + env_name + '/' + robot + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # checkpoint_path = checkpoint_dir + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    # print("save checkpoint path : " + checkpoint_path)

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_step)
    # print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    # i_episode = 0
    best_reward = -np.inf
    max_epoch = 1000     # 200 --> 1e5
    save_model_n_epoch = 50
    update_epoch = 4
    # training loop
    # while time_step <= max_training_timesteps:
    for i_episode in range(1, max_epoch + 1):

        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, max_step + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

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
        print("epoch {}, reward {}".format(i_episode, current_ep_reward))

        # update PPO agent
        if i_episode % update_epoch == 0:
            ppo_agent.update()

        # if continuous action space; then decay action std of ouput action distribution
        # if has_continuous_action_space and time_step % action_std_decay_freq == 0:
        #     ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        if i_episode % save_model_n_epoch == 0 and current_ep_reward > best_reward:
            print("--------------------------------------------------------------------------------------------")
            t_now = time.time()
            time_str = datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')
            checkpoint_path = checkpoint_dir + "PPO_{}_{}_{}_{}.pth".format(env_name, random_seed, time_str, round(current_ep_reward, 2))
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            best_reward = current_ep_reward
            print("--------------------------------------------------------------------------------------------")

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

    log_f.close()
    env.env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()