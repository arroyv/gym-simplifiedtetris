"""Contains a function that trains a Q-learning agent.
"""

import gym
import numpy as np
from gym_simplifiedtetris.agents.q_learning import QLearningAgent
from tqdm import tqdm


def train_q_learning(
    env: gym.Env,
    agent: QLearningAgent,
    num_eval_timesteps: int = 1,
    render: bool = False,
) -> QLearningAgent:
    """Train and evaluate a Q-learning agent. Returns the trained agent.

    :param env: Q-learning agent will be evaluated on this env.
    :param agent: Q-learning agent.
    :param num_eval_timesteps: agent will be evaluated for this number of timesteps.
    :param render: whether to render the env.

    :return: trained Q-learning agent.
    """
    ep_return = 0
    ep_returns = np.array([], dtype=int)
    done = False

    obs = env.reset()

    for _ in tqdm(range(num_eval_timesteps), desc="No. of time steps completed"):
        if render:
            env.render()
        action = agent.predict(obs)
        next_obs, reward, done, info = env.step(action)
        agent.learn(reward=reward, obs=obs, next_obs=next_obs, action=action)
        ep_return += info["num_rows_cleared"]

        # Anneal epsilon so that it is zero by the end of training.
        agent.epsilon -= 1 / num_eval_timesteps

        if done:
            obs = env.reset()
            ep_returns = np.append(ep_returns, ep_return)
            done = False
            ep_return = 0
        else:
            obs = next_obs

    agent.epsilon = 0
    return agent
