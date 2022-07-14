"""Simplified Tetris env, which has a binary obs space and a shaped reward function.
"""
import numpy as np
from gym import spaces
from typing import Any
from typing import Tuple
from gym_simplifiedtetris.envs.simplified_tetris_binary_env import (SimplifiedTetrisBinaryEnv,)
from gym_simplifiedtetris.register import register_env
from gym_simplifiedtetris.envs.reward_shaping._potential_based_shaping_reward import _PotentialBasedShapingReward

class SimplifiedTetrisBinaryShapednewEnv(_PotentialBasedShapingReward, SimplifiedTetrisBinaryEnv):
    """A simplified Tetris environment.

    The reward function is a potential-based shaping reward and the observation space is the grid's binary representation plus the current piece's id.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the object."""
        super().__init__()
        SimplifiedTetrisBinaryEnv.__init__(self, **kwargs)

    def _get_reward(self) -> Tuple[float, int]:
        """Compute and return the potential-based shaping reward.

        :return: potential-based shaping reward and the number of lines cleared.
        """
        num_lines_cleared = self._engine._clear_rows()

        # I chose the potential function to be a function of the well-known holes feature because the number of holes in a given state is (loosely speaking) inversely proportional to the potential of a state.
        heuristic_value = np.count_nonzero((self._engine._grid).cumsum(axis=1) * ~self._engine._grid)
        print('number or holes',np.count_nonzero((self._engine._grid).cumsum(axis=1) * ~self._engine._grid))
        print()
        print('(self._engine._grid).cumsum(axis=1) * ~self._engine._grid',(self._engine._grid).cumsum(axis=1) * ~self._engine._grid)
        print()
        print('(self._engine._grid).cumsum(axis=1)',(self._engine._grid).cumsum(axis=1))
        print()
        print('self._engine._grid', self._engine._grid)
        print()
        print('self._engine._grid.shape', self._engine._grid.shape)
        print()
        print('~self._engine._grid',~self._engine._grid)
        print()
        self._update_range(heuristic_value)

        # I wanted the difference in potentials to be in [-1, 1] to improve the stability of neural network convergence. I also wanted the agent to frequently receive non-zero rewards (since bad-performing agents in the standard game of Tetris rarely receive non-zero rewards). Hence, the value of holes was scaled by using the smallest and largest values of holes seen thus far to obtain a value in [0, 1). The result of this was then subtracted from 1 (to obtain a value in (0, 1]) because a state with a larger value of holes has a smaller potential (generally speaking). The function numpy.clip is redundant here.
        new_potential = np.clip(1 - (heuristic_value - self._heuristic_range["min"]) / (self._heuristic_range["max"] + 1e-9), 0, 1,)

        # Notice that gamma was set to 1, which isn't strictly allowed since it should be less than 1 according to Theorem 1 in this paper. I found that the agent rarely received positive rewards using this reward function because the agent was frequently transitioning to states with a lower potential (since it was rarely clearing lines).
        # HACK: Added 0.3.
        shaping_reward = (new_potential - self._old_potential) + num_lines_cleared + 0.3

        self._old_potential = new_potential
        
        return (shaping_reward, num_lines_cleared)            
    
    def landing_height(field, turn):
        """
        landing height: the height of the current piece after falling
        turn : current number of turns that have been played
        position : position of next piece given the action array
        """
        l_height = env.n_rows
        for i in range(len(field))[::-1]: # i in range of field, starting at the end of field
            for j in range(len(field[i])):
                if field[i][j] == turn and i < l_height:
                    l_height = i
        return l_height

    def eroded_peices(cleared_current_turn, field , turn):
        """
        eroded peices: (Number of cells in the last piece that cleared lines) x (the number of cleared lines)
        cleared_current_turn: The number of lines cleared on this turn
        field: The current tetris board after the action
        turn:  The current number of turns in the game
        """
        contribution = 4
        if cleared_current_turn > 0:
            for i in range(env.n_rows):
                for j in range(env.n_cols):
                    if field[i][j] == turn:
                        contribution -= 1
        return cleared_current_turn * contribution

    def row_transitions(field):
        """
        Row transition: The number of horizontal cell transitions
        field : The current state board
        """
        transitions = 0
        for i in range(env.n_rows):
            if field[i][0] == 0:
                transitions += 1
            if field[i][-1] == 0:
                transitions += 1
            for j in range(env.n_cols):
                if field[i][j] != 0:
                    if (j - 1 > - 1 and field[i][j - 1] == 0):
                        transitions += 1
                    if (j + 1 != env.n_cols and field[i][j + 1] == 0):
                        transitions += 1
        return transitions

    def column_transitions(field):
        """
        column_transitions: The number of vertical cell transtions
        field : The current state board
        """
        transitions = 0
        for i in range(env.n_cols):
            if field[0][i] == 0:
                transitions += 1
            if field[-1][i] == 0:
                transitions += 1
            for j in range(env.n_rows):
                if field[j][i] != 0:
                    if (j - 1 > - 1 and field[j-1][i] == 0):
                        transitions += 1
                    if (j + 1 != env.n_rows and field[j+1][i] == 0):
                        transitions += 1

        return transitions

    def num_holes(field):
        """
        num_holes: Number of holes on the board and the depth of the hole
        returns:
            holes: # of empty cells with at least one filled cell above
            depth: # of filled cells above holes summed over all columns
        parameters:
            field : current state board
        """
        holes = 0
        depth = 0
        for i in range(env.n_rows):
            for j in range(env.n_cols):
                if field[i][j] == 0:
                    if i + 1 != env.n_rows and field[i+1][j] != 0:
                        holes += 1
                        k = i + 1
                        while field[k][j] != 0:
                            depth += 1
                            k += 1
        return holes, depth

    def cum_wells(field):
        """
        cum_wells: The sum of the accumulated depths of the wells
        field: The current state board
        """
        cummulative_depth = 0
        for i in range(env.n_cols):
            for j in range(env.n_rows)[::-1]: 
                if field[j][i] != 0:
                    break
                elif (i == 0 or (i - 1 > -1 and field[j][i - 1] != 0)) and ((i + 1 < env.n_cols and field[j][i+1] !=0) or i == env.n_cols - 1):
                    k = j
                    temp = 0
                    while field[k][i] == 0 and ((i == 0 or (i - 1 > -1 and field[k][i - 1] != 0)) and ((i + 1 < env.n_cols and field[k][i+1] !=0) or i == env.n_cols - 1)):
                        temp += 1
                        k -= 1
                    if (field[k][i] != 0 or k == env.n_rows):
                        cummulative_depth += temp
        return cummulative_depth

    def row_hole(field):
        """
        row_hole: The number of rows that contain at least one hole
        field: The current state board
        """
        row_holes = 0
        for i in range(env.n_rows):
            for j in range(env.n_cols):
                if field[i][j] == 0:
                    if i + 1 != env.n_rows and field[i + 1][j] != 0:
                        row_holes += 1
                        break
        return row_holes

register_env(
    incomplete_id=f"simplifiedtetris-binary-shapednew", 
    entry_point=f"gym_simplifiedtetris.envs:SimplifiedTetrisBinaryShapednewEnv",
)