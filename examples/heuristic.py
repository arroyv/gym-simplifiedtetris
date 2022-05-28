"""A script for running a heuristic agent."""

import os
import sys

import gym

sys.path.append(os.getcwd())

import numpy as np
from gym_simplifiedtetris.agents import HeuristicAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


def eval_heuristic_agent(
    agent: HeuristicAgent, env: gym.Env, num_episodes: int
) -> None:
    """Evaluate an agent that selects action according to a heuristic."""
    episode_num = 0
    ep_returns = np.zeros(num_episodes)

    obs = env.reset()

    while episode_num < num_episodes:
        env.render()

        action = agent.predict(env)

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


def main() -> None:
    """Evaluate an Heuristic agent."""
    grid_dims = (10, 10)

    agent = HeuristicAgent()
    env = Tetris(grid_dims=grid_dims, piece_size=4)

    num_episodes = 100

    eval_heuristic_agent(
        agent=agent,
        env=env,
        num_episodes=num_episodes,
    )


if __name__ == "__main__":
    main()
