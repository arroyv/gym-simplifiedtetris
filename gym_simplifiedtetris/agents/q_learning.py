"""Contains a Q-learning agent class.
"""

from typing import Optional, Sequence

import numpy as np


class QLearningAgent(object):
    """An agent that learns a Q-value for each state-action pair.

    :attr epsilon: exploration rate.
    :attr alpha: learning rate parameter.
    :attr gamma: discount rate parameter.
    :attr _q_table: table of state-action values.
    :attr _num_actions: number of actions available from each state.
    """

    def __init__(
        self,
        *,
        grid_dims: Sequence[int],
        num_pieces: int,
        num_actions: int,
        alpha: Optional[float] = 0.2,
        gamma: Optional[float] = 0.99,
        epsilon: Optional[float] = 1.0,
    ):
        """Constructor.

        :param grid_dims: grid dimensions.
        :param num_pieces: number of pieces in use.
        :param num_actions: number of actions available in each state.
        :param alpha: learning rate parameter.
        :param gamma: discount rate parameter.
        :param epsilon: exploration rate of the epsilon-greedy policy.
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        q_table_dims = [2 for _ in range(grid_dims[0] * grid_dims[1])]
        q_table_dims += [num_pieces] + [num_actions]
        self._q_table = np.zeros((q_table_dims), dtype="double")

        self._num_actions = num_actions

    def predict(self, obs: np.ndarray, /) -> int:
        """Returns an action whilst following an epsilon-greedy policy.

        :param obs: observations given to the agent by the env.

        :return: action chosen by the Q-learning agent.
        """
        # Choose an action at random with probability epsilon.
        if np.random.rand(1)[0] <= self.epsilon:
            return np.random.choice(self._num_actions)

        # Choose greedily from the set of all actions.
        return np.argmax(self._q_table[tuple(obs)])

    def learn(
        self, reward: float, obs: np.ndarray, next_obs: np.ndarray, action: int
    ) -> None:
        """Update the Q-learning agent's Q-table.

        :param reward: reward given to the agent by the env after taking action.
        :param obs: old observation given to the agent by the env.
        :param next_obs: next observation given to the agent by the env having taken action.
        :param action: action taken that generated next_obs.
        """
        _obs_action = tuple(list(obs) + [action])
        max_q_value = np.max(self._q_table[tuple(next_obs)])

        # Update the Q-table using the stored Q-value.
        self._q_table[_obs_action] += self.alpha * (
            reward + self.gamma * max_q_value - self._q_table[_obs_action]
        )
