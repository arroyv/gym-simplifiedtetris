"""Contains a class that represents a potential-based shaping reward.
"""

from typing import Tuple

import numpy as np


class _PotentialBasedShapingReward:
    """A potential-based shaping reward based on the feature, holes.

    :attr _heuristic_range: min and max heuristic values seen so far.
    :attr _old_potential: previous potential.
    """

    # The number of lines cleared is in the range [0, 4]. Every potential is in
    # the range [0, 1]. Therefore, the difference between the new potential
    # and the old potential is in the range [-1, 1]. Hence, the shaping reward
    # range is [-1, 5].
    reward_range = (-1, 5)

    # The initial potential is 1 because there are no holes at the start of a
    # game.
    _INITIAL_POTENTIAL = 1

    def __init__(self) -> None:
        """Initialise the shaping reward object."""
        # Setting the range in this way ensures that the min and max are definitely updated the first time the method "_update_range" is called.
        self._heuristic_range = {"min": 1000, "max": -1}

        self._old_potential = self._INITIAL_POTENTIAL

    def _get_reward(self) -> Tuple[float, int]:
        """Compute and return the potential-based shaping reward.

        :return: potential-based shaping reward and the number of lines cleared.
        """
        num_lines_cleared = self._engine._clear_rows()
        heuristic_value = self._engine._get_holes()
        self._update_range(heuristic_value)
        new_potential = np.clip(
            1
            - (heuristic_value - self._heuristic_range["min"])
            / (self._heuristic_range["max"] + 1e-9),
            0,
            1,
        )
        shaping_reward = (new_potential - self._old_potential) + num_lines_cleared
        self._old_potential = new_potential
        return (shaping_reward, num_lines_cleared)

    def _get_terminal_reward(self) -> float:
        """Compute and return the terminal potential-based shaping reward.

        :return: terminal potential-based shaping reward.
        """
        terminal_shaping_reward = -self._old_potential
        self._old_potential = self._INITIAL_POTENTIAL
        return terminal_shaping_reward

    def _update_range(self, heuristic_value: int) -> None:
        """Update the heuristic range.

        :param heuristic_value: computed heuristic value.
        """

        if heuristic_value > self._heuristic_range["max"]:
            self._heuristic_range["max"] = heuristic_value

        if heuristic_value < self._heuristic_range["min"]:
            self._heuristic_range["min"] = heuristic_value
