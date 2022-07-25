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

        # I chose the potential function to be a function of the well-known holes 
        # feature because the number of holes in a given state is (loosely speaking) inversely proportional to the potential of a state.
        heuristic_value = np.count_nonzero((self._engine._grid).cumsum(axis=1) * ~self._engine._grid)
        # print('self._engine._grid.shape')
        # print(self._engine._grid.shape)
        # print()
        # print('self._engine._grid')
        # print(self._engine._grid.astype(int))
        # print()
        # print('~self._engine._grid')
        # print((~self._engine._grid).astype(int))
        # print()
        # print('(self._engine._grid).cumsum(axis=1)')
        # print((self._engine._grid).cumsum(axis=1))
        # print()
        # print('number or holes')
        # print(np.count_nonzero((self._engine._grid).cumsum(axis=1) * ~self._engine._grid))
        # print()
        # print('(self._engine._grid).cumsum(axis=1) * ~self._engine._grid')
        # print((self._engine._grid).cumsum(axis=1) * ~self._engine._grid)
        # print() 

        # print('number of holes: ', num_holes(self._engine._grid))      
        # print('holes depth: ', depths(self._engine._grid))
        # print('row transitions: ', row_transitions(self._engine._grid)) 
        # print('column_transitions: ', column_transitions(self._engine._grid))
        # print('cum_wells: ', cum_wells(self._engine._grid))
        # print('row_hole: ', row_hole(self._engine._grid))

        # num_holes = num_holes(self._engine._grid)
        # n_depths = depths(self._engine._grid)
        n_row_transitions = row_transitions(self._engine._grid)
        n_column_transitions = column_transitions(self._engine._grid)
        n_cum_wells = cum_wells(self._engine._grid)
        n_row_hole = row_hole(self._engine._grid)
        landing_height = get_landing_height(self)
        eroded_cells = get_eroded_cells(self)


        self._update_range(heuristic_value)

        # I wanted the difference in potentials to be in [-1, 1] to improve the stability of neural network convergence. 
        # I also wanted the agent to frequently receive non-zero rewards 
        # (since bad-performing agents in the standard game of Tetris rarely receive non-zero rewards). 
        # Hence, the value of holes was scaled by using the smallest and largest values of holes seen thus far to obtain a value in [0, 1). 
        # The result of this was then subtracted from 1 (to obtain a value in (0, 1]) because a state with a larger value 
        # of holes has a smaller potential (generally speaking). The function numpy.clip is redundant here.
        new_potential = np.clip(1 - (heuristic_value - self._heuristic_range["min"]) / (self._heuristic_range["max"] + 1e-9), 0, 1,)

        # Notice that gamma was set to 1, which isn't strictly allowed since it should be less than 1 according to Theorem 1 in this paper. 
        # I found that the agent rarely received positive rewards using this reward function because the agent was frequently transitioning 
        # to states with a lower potential (since it was rarely clearing lines).
        # HACK: Added 0.3.
        # shaping_reward = (new_potential - self._old_potential) + num_lines_cleared + 0.3 
        # shaping_reward = (new_potential - self._old_potential) + num_lines_cleared + 0.3 - n_row_transitions - n_column_transitions - n_cum_wells - n_row_hole - landing_height + eroded_cells #- n_depths
        shaping_reward = -4 * heuristic_value - n_cum_wells - n_row_transitions - n_column_transitions - 4*landing_height + eroded_cells  # −4 × holes − cumulative wells − row transitions − column transitions − landing height + eroded cells

        # - row transitions - column transitions -4 x holes - cumulative wells
        # −4 × holes − cumulative wells − row transitions − column transitions − landing height + eroded cells
        self._old_potential = new_potential
        
        return (shaping_reward, num_lines_cleared, n_row_transitions, n_column_transitions, n_cum_wells, n_row_hole, landing_height, eroded_cells) # n_depths
# DONE
def num_holes(field):
    """
    num_holes: Number of holes on the board and the depth of the hole
    returns:
        holes: # of empty cells with at least one filled cell above
        depth: # of filled cells above holes summed over all columns
    parameters:
        field : current state board
    """
    fieldShape = field.shape
    holes = 0
    for i in range(fieldShape[0]):
        for j in range(fieldShape[1]):
            if field[i][j] == 0:
                if j > 0 and j < fieldShape[1] and field[i][j-1] != 0:
                    k = j
                    while k < fieldShape[1] and field[i][k] == 0:
                        holes += 1
                        k += 1
    return holes

def depths(field):
    '''
    depths: Depth of the hole
    returns:
    depth: # of filled cells above holes summed over all columns
    parameters:
    field : current state board
    '''
    fieldShape = field.shape
    depth = 0
    for i in range(fieldShape[0]):
        for j in range(fieldShape[1]):
            if field[i][j] == 0:
                if j > 0 and j < fieldShape[1] and field[i][j-1] != 0:
                    k = j - 1
                    while k >= 0 and k < fieldShape[1] and field[i][k] != 0 :
                        depth += 1
                        k -= 1
    return depth

def row_transitions(field):
    """
    Row transition: The number of horizontal cell transitions
    field : The current state board
    """
    fieldShape = field.shape
    num_transitions = 0
    for j in range(fieldShape[1]):
        for i in range(fieldShape[0]):
            if i + 1 < fieldShape[0]:
                if field[i][j] == 0 and field[i + 1][j] == 1:
                    num_transitions += 1
                elif field[i][j] == 1 and field[i + 1][j] == 0:
                    num_transitions += 1
    return num_transitions

def column_transitions(field):
    """
    column_transitions: The number of vertical cell transitions
    field : The current state board
    """
    fieldShape = field.shape
    num_transitions = 0
    for i in range(fieldShape[0]):
        for j in range(fieldShape[1]):
            if j+1 < fieldShape[1]:
                if field[i][j] == 0 and field[i][j+1] == 1:
                    num_transitions += 1
                elif field[i][j] == 1 and field[i][j+1] == 0:
                    num_transitions += 1
    return num_transitions

# TODO
def row_hole(field):
    """
    row_hole: The number of rows that contain at least one hole
    field: The current state board
    """
    fieldShape = field.shape
    row_holes = 0
    i = 0
    j = 19
    while i  >= 0 and i < fieldShape[0]:
        # print('l2',j,i)
        if field[i][j] == 0:
            k = j
            while k  > 0 and k < fieldShape[1]:
                k -= 1
                if field[i][k] == 1:
                    # print('l3',j,i,k)
                    row_holes += 1
                    j -= 1
                    i = 0
                    # print('l3',j,i,k)
                    break
        i += 1
    return row_holes

def cum_wells(field):
    """
    cum_wells: The sum of the accumulated depths of the wells
    field: The current state board
    """
    fieldShape = field.shape
    # print(fieldShape)
    cummulative_depth = 0
    for i in range(fieldShape[0]):
        temp = 0
        for j in range(fieldShape[1]):
            if j - 1 >= 0 and  field[i][j-1] == 1:
                break
            elif field[i][j] == 0 and i == 0 and field[i + 1][j] == 1 and j + 1 <= fieldShape[1] and j - 1 >= 0 and  field[i][j-1] == 0:
                temp += 1
                # print('s1',i,j,i+1, i-1)
                if j + 1 == fieldShape[1]:
                    cummulative_depth += temp
                elif field[i][j+1] == 1:
                    # print('s1',i,j,i+1, i-1, temp)
                    cummulative_depth += temp
            elif field[i][j] == 0 and i == fieldShape[0] - 1 and field[i - 1][j] == 1 and j + 1 <= fieldShape[1] and j - 1 >= 0 and  field[i][j-1] == 0:
                temp += 1
                # print('s2',i,j,i+1, i-1)
                if j + 1 == fieldShape[1]:
                    cummulative_depth += temp
                elif field[i][j+1] == 1:
                    # print('s2',i,j,i+1, i-1, temp)
                    cummulative_depth += temp
            elif field[i][j] == 0 and i - 1 >= 0 and i + 1 < fieldShape[0] and field[i - 1][j] == 1 and field[i + 1][j] == 1 and j + 1 <= fieldShape[1] and j - 1 >= 0 and  field[i][j-1] == 0:
                temp += 1
                # print('s3',i,j,i+1, i-1)
                if j + 1 == fieldShape[1]:
                    cummulative_depth += temp
                elif field[i][j+1] == 1:
                    # print('s3',i,j,i+1, i-1, temp)
                    cummulative_depth += temp
    return cummulative_depth

def get_landing_height(self):
    """Compute the landing height and return it.

    Landing height = the midpoint of the last piece to be placed.

    :param env: environment that the agent resides in.
    :return: landing height.
    """
    return (
        self._engine._last_move_info["landing_height"]
        if "landing_height" in self._engine._last_move_info
        else 0
    )

def get_eroded_cells(self):
    """Return the eroded cells value.

    Eroded cells = number of rows cleared x number of blocks removed that were added to the grid by the last action.

    :param env: environment that the agent resides in.
    :return: eroded cells.
    """
    return (
        self._engine._last_move_info["num_rows_cleared"]* self._engine._last_move_info["eliminated_num_blocks"]
        if "num_rows_cleared" in self._engine._last_move_info
        else 0
    )



# def _get_landing_height(self, env: SimplifiedTetrisEngine) -> int:
#         """Compute the landing height and return it.

#         Landing height = the midpoint of the last piece to be placed.

#         :param env: environment that the agent resides in.
#         :return: landing height.
#         """
#         return (
#             env._engine._last_move_info["landing_height"]
#             if "landing_height" in env._engine._last_move_info
#             else 0
#         )

#     def _get_eroded_cells(self, env: SimplifiedTetrisEngine) -> int:
#         """Return the eroded cells value.

#         Eroded cells = number of rows cleared x number of blocks removed that were added to the grid by the last action.

#         :param env: environment that the agent resides in.
#         :return: eroded cells.
#         """
#         return (
#             env._engine._last_move_info["num_rows_cleared"]
#             * env._engine._last_move_info["eliminated_num_blocks"]
#             if "num_rows_cleared" in env._engine._last_move_info
#             else 0
#         )

#     def _get_row_transitions(self, env: SimplifiedTetrisEngine) -> int:
#         """Return the row transitions value.

#         Row transitions = Number of transitions from empty to full cells (or vice versa), examining each row one at a time.

#         Author: Ben Schofield
#         Source: https://github.com/Benjscho/gym-mdptetris/blob/1a47edc33330deb638a03275e484c3e26932d802/gym_mdptetris/envs/feature_functions.py#L45

#         :param env: environment that the agent resides in.
#         :return: row transitions.
#         """
#         # Adds a column either side.
#         grid = np.ones((env._engine._width + 2, env._engine._height), dtype="bool")

#         grid[1:-1, :] = env._engine._grid.copy()
#         return int(np.diff(grid.T).sum())

#     def _get_col_transitions(self, env: SimplifiedTetrisEngine) -> int:
#         """Return the column transitions value.

#         Column transitions = Number of transitions from empty to full (or vice versa), examining each column one at a time.

#         Author: Ben Schofield
#         Source: https://github.com/Benjscho/gym-mdptetris/blob/1a47edc33330deb638a03275e484c3e26932d802/gym_mdptetris/envs/feature_functions.py#L60

#         :param env: environment that the agent resides in.
#         :return: column transitions.
#         """
#         # Adds a full row to the bottom.
#         grid = np.ones((env._engine._width, env._engine._height + 1), dtype="bool")

#         grid[:, :-1] = env._engine._grid.copy()
#         return int(np.diff(grid).sum())

#     def _get_holes(self, env: SimplifiedTetrisEngine) -> int:
#         """Compute the number of holes present in the current grid and return it.

#         A hole is an empty cell with at least one full cell above it in the same column.

#         :param env: environment that the agent resides in.
#         :return: value of the feature holes.
#         """
#         return np.count_nonzero((env._engine._grid).cumsum(axis=1) * ~env._engine._grid)

#     def _get_cum_wells(self, env: SimplifiedTetrisEngine) -> int:
#         """Compute the cumulative wells value and return it.

#         Cumulative wells is defined here:
#         https://arxiv.org/abs/1905.01652.  For each well, find the depth of
#         the well, d(w), then calculate the sum of i from i=1 to d(w).  Lastly,
#         sum the well sums.  A block is part of a well if the cells directly on
#         either side are full and the block can be reached from above (i.e., there are no full cells directly above it).

#         Attribution: Ben Schofield

#         :param env: environment that the agent resides in.
#         :return: cumulative wells value.
#         """
#         grid_ext = np.ones(
#             (env._engine._width + 2, env._engine._height + 1), dtype="bool"
#         )
#         grid_ext[1:-1, 1:] = env._engine._grid[:, : env._engine._height]

#         # This includes some cells that cannot be reached from above.
#         potential_wells = (
#             np.roll(grid_ext, 1, axis=0) & np.roll(grid_ext, -1, axis=0) & ~grid_ext
#         )

#         col_heights = np.zeros(env._engine._width + 2)
#         col_heights[1:-1] = env._engine._height - np.argmax(env._engine._grid, axis=1)
#         col_heights = np.where(col_heights == env._engine._height, 0, col_heights)

#         x = np.linspace(1, env._engine._width + 2, env._engine._width + 2)
#         y = np.linspace(env._engine._height + 1, 1, env._engine._height + 1)
#         _, yv = np.meshgrid(x, y)

#         # A cell that is part of a well must be above the playfield's outline, which consists of the highest full cells in each column.
#         above_outline = (col_heights.reshape(-1, 1) < yv.T).astype(int)

#         # Exclude the cells that cannot be reached from above by multiplying by 'above_outline'.
#         cumulative_wells = np.sum(
#             np.cumsum(potential_wells, axis=1) * above_outline,
#         )

#         return cumulative_wells


register_env(
    incomplete_id=f"simplifiedtetris-binary-shapednew", 
    entry_point=f"gym_simplifiedtetris.envs:SimplifiedTetrisBinaryShapednewEnv",
)