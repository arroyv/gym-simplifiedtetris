"""Agent evaluation."""

import gym
import numpy as np
from gym_simplifiedtetris.agents.base import BaseAgent


def eval_agent(
    agent: BaseAgent, env: gym.Env, num_episodes: int, render: bool = True
) -> None:
    """Evaluate an agent in the env for |num_episodes| games.

    :param agent: The agent to evaluate.
    :param env: The environment to evaluate the agent in.
    :param num_episodes: The number of episodes to evaluate the agent for.
    :param render: Whether to render the environment.
    """
    ep_returns = np.zeros(num_episodes)

    obs = env.reset()

    episode_num = 0
    while episode_num < num_episodes:

        if render:
            env.render()

        action = agent.predict(obs=obs, env=env)

        obs, reward, done, info = env.step(action)
        ep_returns[episode_num] += info["num_rows_cleared"]

        if done:
            print(f"Episode {episode_num + 1} has terminated.")
            episode_num += 1
            obs = env.reset()

    env.close()

    print(
        f"""\nScore obtained from averaging over {num_episodes} games:\nMean = {np.mean(ep_returns):.1f}\nStandard deviation = {np.std(ep_returns):.1f}"""
    )
