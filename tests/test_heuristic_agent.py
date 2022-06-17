"""Testing the Dellacherie agent's methods."""

import unittest

import numpy as np
import pytest
from gym_simplifiedtetris.agents import DellacherieAgent
from gym_simplifiedtetris.auxiliary import Polymino
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


@pytest.fixture(scope="function")
def agent():
    return DellacherieAgent()


@pytest.fixture(scope="function")
def env():
    env = Tetris()
    env.reset()
    return env


def test_get_dellacherie_scores_empty_grid(agent, env):
    env._engine._piece = Polymino(env.piece_size, 0)
    array_to_compare = np.NINF * np.ones(env._engine._num_actions, dtype="double")
    array_to_compare[10] = abs(0 - 6) * 100 + 10 - (90 / 90)
    array_to_compare[16] = abs(6 - 6) * 100 + 0 - (90 / 90)
    array_to_compare[27] = abs(3 - 6) * 100 + 10 - (270 / 90)
    array_to_compare[33] = abs(9 - 6) * 100 + 0 - (270 / 90)
    np.testing.assert_array_equal(
        agent._compute_dell_scores(env),
        array_to_compare,
    )


def test_get_dellacherie_funcs_populated_grid(agent, env):
    env._engine._grid[:, -5:] = True
    env._engine._grid[1, env._engine._height - 5 : env._engine._height - 1] = False
    env._engine._grid[env._engine._width - 1, env._engine._height - 2] = False
    env._engine._grid[env._engine._width - 2, env._engine._height - 1] = False
    env._engine._grid[env._engine._width - 3, env._engine._height - 3] = False
    env._engine._grid[env._engine._width - 1, env._engine._height - 6] = True
    env._engine._grid[3, env._engine._height - 3 : env._engine._height - 1] = False
    env._engine._piece = Polymino(env.piece_size, 0)
    env._engine._anchor = [0, 0]
    env._engine._hard_drop()
    env._engine._update_grid(True)
    env._engine._clear_rows()
    array_to_compare = np.array([func(env) for func in agent._get_dell_funcs()])
    np.testing.assert_array_equal(
        array_to_compare,
        np.array([5.5 + 0.5 * env.piece_size, 0, 48, 18, 5, 10], dtype="double"),
    )


def test_get_landing_height_I_piece_(agent, env):
    env._engine._piece = Polymino(env.piece_size, 0)
    env._engine._anchor = [0, env._engine._height - 1]
    env._engine._update_grid(True)

    assert agent._get_landing_height(env) == 0.5 * (1 + env.piece_size)


def test_get_eroded_cells_empty(agent, env):
    assert agent._get_eroded_cells(env) == 0


def test_get_eroded_cells_single(agent, env):
    env._engine._grid[:, env._engine._height - 1 :] = True
    env._engine._grid[0, env._engine._height - 1] = False
    env._engine._piece = Polymino(env.piece_size, 0)
    env._engine._anchor = [0, 0]
    env._engine._hard_drop()
    env._engine._update_grid(True)
    env._engine._clear_rows()

    assert agent._get_eroded_cells(env) == 1


def test_get_row_transitions_empty(agent, env):
    assert agent._get_row_transitions(env) == 40


def test_get_row_transitions_populated(agent, env):
    env._engine._grid[:, -2:] = True
    env._engine._grid[0, env._engine._height - 1] = False
    env._engine._grid[2, env._engine._height - 1] = False
    env._engine._grid[1, env._engine._height - 2] = False
    assert agent._get_row_transitions(env) == 42


def test_get_row_transitions_populated_more_row_transitions(agent, env):
    env._engine._grid[:, -2:] = True
    env._engine._grid[0, env._engine._height - 2 :] = False
    env._engine._grid[2, env._engine._height - 2 :] = False
    env._engine._grid[4, env._engine._height - 1] = False
    np.testing.assert_array_equal(agent._get_row_transitions(env), 46)


def test_get_column_transitions_empty(agent, env):
    assert agent._get_col_transitions(env) == 10


def test_get_column_transitions_populated(agent, env):
    env._engine._grid[:, -2:] = True
    env._engine._grid[0, env._engine._height - 1] = False
    env._engine._grid[2, env._engine._height - 1] = False
    env._engine._grid[1, env._engine._height - 2] = False
    assert agent._get_col_transitions(env) == 14


def test_get_column_transitions_populated_less_column_transitions(agent, env):
    env._engine._grid[:, -2:] = True
    env._engine._grid[0, env._engine._height - 2 :] = False
    env._engine._grid[2, env._engine._height - 2 :] = False
    env._engine._grid[4, env._engine._height - 1] = False
    np.testing.assert_array_equal(agent._get_col_transitions(env), 12)


def test_get_holes_empty(agent, env):
    assert agent._get_holes(env) == 0


def test_get_holes_populated_two_holes(agent, env):
    env._engine._grid[:, -2:] = True
    env._engine._grid[0, env._engine._height - 1] = False
    env._engine._grid[2, env._engine._height - 1] = False
    assert agent._get_holes(env) == 2


def test_get_holes_populated_no_holes(agent, env):
    env._engine._grid[:, -2:] = True
    env._engine._grid[0, env._engine._height - 2 :] = False
    assert agent._get_holes(env) == 0


def test_get_holes_populated_one_hole(agent, env):
    env._engine._grid[:, -2:] = True
    env._engine._grid[0, env._engine._height - 2 :] = False
    env._engine._grid[2, env._engine._height - 2 :] = False
    env._engine._grid[4, env._engine._height - 1] = False
    np.testing.assert_array_equal(agent._get_holes(env), 1)


def test_get_cumulative_wells_empty(agent, env):
    np.testing.assert_array_equal(agent._get_cum_wells(env), 0)


def test_get_cumulative_wells_populated(agent, env):
    env._engine._grid[:, -2:] = True
    env._engine._grid[0, env._engine._height - 2 :] = False
    np.testing.assert_array_equal(agent._get_cum_wells(env), 3)


def test_get_cumulative_wells_populated_deeper_well(agent, env):
    env._engine._grid[:, -2:] = True
    env._engine._grid[0, env._engine._height - 2 :] = False
    env._engine._grid[2, env._engine._height - 2 :] = False
    env._engine._grid[4, env._engine._height - 1] = False
    np.testing.assert_array_equal(agent._get_cum_wells(env), 6)
