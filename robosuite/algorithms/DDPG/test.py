import os
import glob
import time
from datetime import datetime
import numpy as np

import gym

import DDPG


def func():
    env_name = "HalfCheetah-v4"
    total_test_episodes = 5  # total num of testing episodes
    max_ep_len = 1000
    render = False  # render environment on screen
    frame_delay = 0  # if required; add delay b/w frames

    env = gym.make(env_name, render_mode="human")
    # env = gym.make(env_name)

    # initialize a DDPG agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": 0.99,
        "tau": 0.005,
    }
    policy = DDPG.DDPG(**kwargs)
    checkpoint_path = "models/DDPG_{}_{}".format(env_name, 0)
    print("loading network from : " + checkpoint_path)

    policy.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state, _ = env.reset()

        for t in range(1, max_ep_len+1):
            action = policy.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if terminated or truncated:
                break

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    func()
