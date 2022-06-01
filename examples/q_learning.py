"""A script for running training and evaluating a Q-learning agent."""

import os
import sys

sys.path.append(os.getcwd())

from typing import Tuple

import gym
import numpy as np
from gym_simplifiedtetris.agents import QLearningAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
from tqdm import tqdm


def train_q_learning_agent(
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

    env.close()
    agent.epsilon = 0
    return agent


def eval_q_learning_agent(
    agent: QLearningAgent,
    env: gym.Env,
    num_episodes: int,
    render: bool = True,
) -> Tuple[float, float]:
    """Evaluate the agent's performance and return the mean score and standard deviation.

    :param agent: agent to evaluate on the env.
    :param env: agent will be evaluated on this env.
    :param num_episodes: number of games to evaluate the trained agent.
    :param render: renders the agent playing SimplifiedTetris after training.
    :return: mean score and standard deviation obtained from letting the agent play |num_episodes| games.
    """
    ep_returns = np.zeros(num_episodes, dtype=int)
    env._engine._final_scores = np.array([], dtype=int)

    for episode_id in tqdm(range(num_episodes), desc="No. of episodes completed"):
        obs = env.reset()
        done = False
        while not done:

            if render:
                env.render()

            action = agent.predict(obs)
            obs, _, done, info = env.step(action)
            ep_returns[episode_id] += info["num_rows_cleared"]

    env.close()

    mean_score = np.mean(ep_returns)
    std_score = np.std(ep_returns)

    print(
        f"""\nScore obtained from averaging over {num_episodes} games:\nMean = {np.mean(ep_returns):.1f}\nStandard deviation = {np.std(ep_returns):.1f}"""
    )
    return mean_score, std_score


def main() -> None:
    """Train and evaluate a Q-learning agent."""
    grid_dims = (7, 4)
    piece_size = 3

    env = Tetris(
        grid_dims=grid_dims,
        piece_size=piece_size,
    )
    agent = QLearningAgent(
        grid_dims=grid_dims,
        num_pieces=env.num_pieces,
        num_actions=env.num_actions,
    )

    num_eval_timesteps = 10000

    agent = train_q_learning_agent(
        env=env, agent=agent, num_eval_timesteps=num_eval_timesteps, render=True
    )

    del env
    env = Tetris(
        grid_dims=grid_dims,
        piece_size=piece_size,
    )

    num_episodes = 30

    eval_q_learning_agent(
        agent=agent,
        env=env,
        num_episodes=num_episodes,
    )


if __name__ == "__main__":
    main()
