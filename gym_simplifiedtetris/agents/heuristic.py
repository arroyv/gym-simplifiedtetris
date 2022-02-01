"""Contains a heuristic agent class.
"""

import numpy as np


class HeuristicAgent(object):
    """An agent that selects actions according to a heuristic."""

    @staticmethod
    def predict(scores: np.ndarray) -> int:
        """Return the action yielding the largest heuristic score.

        Ties are separated using a priority rating, which is based on the translation and rotation.

        :param scores: scores for all available actions.

        :return: action with the largest rating; ties are separated based on the priority.
        """
        return np.argmax(scores)
