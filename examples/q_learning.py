"""Q-learning agent training and evaluation."""

import os
import sys

sys.path.append(os.getcwd())

import gym
import numpy as np
from gym_simplifiedtetris.agents import QLearningAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
from tqdm import tqdm

from examples.evaluation import eval_agent


def train_q_learning_agent(
    env: gym.Env,
    agent: QLearningAgent,
    num_eval_steps: int = 1,
    render: bool = False,
) -> QLearningAgent:
    """Train and evaluate a Q-learning agent. Returns the trained agent.

    :param env: Q-learning agent that will be evaluated on this env.
    :param agent: Q-learning agent.
    :param num_eval_steps: agent that will be evaluated for this number of timesteps.
    :param render: whether to render the env.
    :return: trained Q-learning agent.
    """
    ep_return = 0
    ep_returns = np.empty((0,), dtype=int)
    done = False

    obs = env.reset()

    for _ in tqdm(range(num_eval_steps), desc="No. of time steps completed"):

        if render:
            env.render()

        action = agent.predict(obs)
        next_obs, reward, done, info = env.step(action)

        agent.learn(reward=reward, obs=obs, next_obs=next_obs, action=action)
        ep_return += info["num_rows_cleared"]

        # Anneal epsilon to zero over the training period.
        agent.epsilon -= 1 / num_eval_steps

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


def main() -> None:
    """Train and evaluate a Q-learning agent."""
    grid_dims = (7, 4)
    piece_size = 3

    env = Tetris(grid_dims=grid_dims, piece_size=piece_size)
    agent = QLearningAgent(
        grid_dims=grid_dims,
        num_pieces=env.num_pieces,
        num_actions=env.num_actions,
    )

    num_eval_steps = 10000

    trained_agent = train_q_learning_agent(
        env=env, agent=agent, num_eval_steps=num_eval_steps, render=False
    )

    del env

    env = Tetris(grid_dims=grid_dims, piece_size=piece_size)
    eval_agent(agent=trained_agent, env=env, num_episodes=30, render=False)


if __name__ == "__main__":
    main()
