"""Uniform agent evaluation."""

import os
import sys

sys.path.append(os.getcwd())

from gym_simplifiedtetris.agents import UniformAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris

from examples.evaluation import eval_agent


def main() -> None:
    """Evaluate a Uniform agent."""
    env = Tetris(grid_dims=(20, 10), piece_size=4)
    agent = UniformAgent(env.action_space.n)
    eval_agent(agent=agent, env=env, num_episodes=1000, render=False)


if __name__ == "__main__":
    main()
