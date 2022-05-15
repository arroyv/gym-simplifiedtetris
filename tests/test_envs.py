"""A script for running some tests on the envs."""


import gym
import pytest
from gym_simplifiedtetris.register import env_list
from stable_baselines3.common.env_checker import check_env


@pytest.mark.parametrize("env_name", env_list)
def test_envs(env_name: str) -> None:
    """Check if each env created is compliant with the OpenAI Gym API.

    Plays ten games using an agent that selects actions uniformly at random. In every game, validate the reward received, and render the env for visual inspection.
    """
    env = gym.make(env_name)
    check_env(env=env, skip_render_check=True)

    _ = env.reset()

    num_episodes = 0
    is_first_move = True
    while num_episodes < 3:
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)

        assert (
            env.reward_range[0] <= reward <= env.reward_range[1]
        ), f"Reward seen: {reward}"

        if num_episodes == 0 and is_first_move:
            is_first_move = False

        if done:
            num_episodes += 1
            _ = env.reset()

    env.close()
