"""DQN agent.
"""

import itertools
import os
import sys
from pathlib import Path
from typing import Any, Tuple

import gym
import numpy as np
import torch as th

sys.path.append(os.getcwd())

import gym_simplifiedtetris
from gym_simplifiedtetris.agents.base import BaseAgent
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn import MlpPolicy
from tqdm import tqdm

N_STEPS = 128 * 8

DQN_PARAMS = {
    "num_iterations": [
        100,
    ],
    "num_steps": [
        N_STEPS,
    ],
    "learning_rate": [
        2.5e-4,
    ],
    "buffer_size": [
        20000,
    ],
    "learning_starts": [
        0,
    ],
    "batch_size": [
        256,
    ],
    "tau": [
        1.0,
    ],
    "gamma": [
        0.99,
    ],
    "train_freq": [
        4,
    ],
    "gradient_steps": [
        1,
    ],
    "target_update_interval": [
        100,
    ],
    "exploration_fraction": [
        1.0,
    ],
    "exploration_initial_eps": [
        0.15,
    ],
    "exploration_final_eps": [
        0.15,
    ],
    "max_grad_norm": [
        5,
    ],
    "net_arch": [
        [64, 64],
    ],
}


def train_dqn(
    model_log_dir: str,
    env: VecEnv,
    grid_dims: str,
    model_path: str,
    agent_name: str,
    num_iterations: int,
    num_steps: int,
    net_arch: list,
    learning_rate: float,
    buffer_size: int,
    learning_starts: int,
    batch_size: int,
    tau: float,
    gamma: float,
    train_freq: int,
    gradient_steps: int,
    target_update_interval: int,
    exploration_fraction: float,
    exploration_initial_eps: float,
    exploration_final_eps: float,
    max_grad_norm: float,
) -> Tuple[np.array, np.array]:
    """
    Trains and evaluates a DQN agent on the env.

    :param model_log_dir: the model log directory string.
    :param env: the environment to train the DQN agent on.
    :param grid_dims: the grid dims.
    :param model_path: the path to save the model with.
    :param agent_name: the name of the DQN agent.
    :param num_iterations: the number of times to let the actors go away and interact with
        the env.
    :param num_steps: the number of steps to run for each environment per update.
    :param net_arch: the shared network architecture.
    :param learning_rate: the NN learning rate.
    :param buffer_size: the size of the replay buffer.
    :param learning_starts: how many steps of the model to collect transitions for
        before learning starts.
    :param batch_size: the size of the minibatches being used to train the model.
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1
        for hard update.
    :param gamma: the discount factor.
    :param train_freq: the frequency to train the NN.
    :param gradient_steps: the number of gradient steps to perform after each rollout
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the
        exploration rate is reduced.
    :param exploration_initial_eps: initial value of random action probability.
    :param exploration_final_eps: final value of random action probability.
    :param max_grad_norm: the maximum value for the gradient clipping.
    :return: the time steps and episode lengths.
    """

    env.reset()

    # Instantiate a new agent if not already been trained.
    if not Path(model_path).is_file():
        agent = DQN(
            policy=MlpPolicy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=f"runs/tensorboard/dqn_tetris/{grid_dims}/",
            create_eval_env=True,
            policy_kwargs=dict(activation_fn=th.nn.Tanh, net_arch=net_arch),
            verbose=0,
        )
        agent.save(model_path)
        del agent

    agent = DQN.load(env=env, path=model_path)

    print(f"\nTraining {agent_name} agent now...")
    agent.learn(
        total_timesteps=num_iterations * num_steps,
        tb_log_name=agent_name,
        reset_num_timesteps=False,
    )

    # Retrieve the episode lengths.
    results = load_results(model_log_dir)
    time_steps = np.cumsum(results.l.values)
    episode_lengths = results.l.values

    agent.save(model_path)
    del agent

    env.close()

    return time_steps, episode_lengths


def eval_agent(
    env: gym.Env,
    agent: DQN,
    num_eps: int,
    render: bool,
    obs_space: str,
) -> Tuple[float, float, float]:
    """
    Evaluate a DQN agent on a SimplifiedTetris env.

    :param env: the env to evaluate the agent on.
    :param agent: the DQN agent to be evaluated.
    :param num_eps: the number of evaluation episodes.
    :param render: if True renders the agent interacting w/ the env.
    :param obs_space: which obs space being used.
    :return: the mean, std, max score obtained from evaluation.
    """
    episode_rewards = np.zeros(num_eps)
    episode_lengths = np.zeros(num_eps)

    current_rewards = np.zeros(1)
    current_lengths = np.zeros(1, dtype=int)

    _ = env.reset()

    for episode_id in tqdm(range(num_eps), desc="No. of episodes completed"):

        while not (done := False):
            action = env.env._engine.get_best_action(agent, obs_space)

            _, _, done, info = env.step(action)
            current_rewards += info["num_rows_cleared"]
            current_lengths += 1

            if done:
                episode_rewards[episode_id] = current_rewards
                episode_lengths[episode_id] = current_lengths
                current_rewards = 0
                current_lengths = 0
                _ = env.reset()

            if render:
                env.render()

    env.close()

    return np.mean(episode_rewards), np.std(episode_rewards), np.max(episode_rewards)


def main():
    eval_scores = {}

    hyperparams = list(DQN_PARAMS.values())
    hp_combos = list(itertools.product(*hyperparams))

    for hp_combo in hp_combos:
        dqn_params = {
            list(DQN_PARAMS.keys())[i]: hp_combo[i] for i in range(len(hp_combo))
        }

        height, width = 20, 10
        grid_dims = f"{height}x{width}"
        piece_size = 4
        incomplete_id = "simplifiedtetris-heights-shaped"
        env_id = incomplete_id + f"-{grid_dims}-{piece_size}-v0"

        env = gym.make(env_id)
        model_log_dir = f"runs/models/dqn/{grid_dims}/"
        os.makedirs(model_log_dir, exist_ok=True)

        agent_name = f"DQN-{env_id}"
        model_path = os.path.join(model_log_dir, agent_name)

        env = make_vec_env(env_id=env_id, n_envs=1, monitor_dir=model_log_dir)

        train_dqn(
            model_log_dir=model_log_dir,
            env=env,
            grid_dims=grid_dims,
            model_path=model_path,
            agent_name=agent_name,
            **dqn_params,
        )

        agent = DQN.load(env=env, path=model_path)

        num_episodes = 10

        print(f"\nEvaluating {agent_name} agent now...")

        eval_scores[agent_name] = eval_agent(
            env=gym.make(env_id),
            agent=agent,
            num_eps=num_episodes,
            render=True,
            obs_space="Heights",
        )

        print(
            f"\nScore obtained from averaging over {num_episodes}"
            f" games: {eval_scores[agent_name][0]:.1f} +/-"
            f" {eval_scores[agent_name][1]:.1f}"
        )


if __name__ == "__main__":
    main()
