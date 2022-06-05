"""Dellacherie agent evaluation."""

import os
import sys

sys.path.append(os.getcwd())

from gym_simplifiedtetris.agents import DellacherieAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris

from examples.evaluation import eval_agent


def main() -> None:
    """Evaluate a Dellacherie agent."""
    env = Tetris(grid_dims=(8, 6), piece_size=4)
    eval_agent(
        agent=DellacherieAgent(),
        env=env,
        num_episodes=10,
        render=False,
    )


if __name__ == "__main__":
    main()
