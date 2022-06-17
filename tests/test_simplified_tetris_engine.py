"""Testing the class SimplifiedTetrisEngine."""

import numpy as np
import pytest
from gym_simplifiedtetris.auxiliary import Polymino
from gym_simplifiedtetris.envs import SimplifiedTetrisEngine as Engine


@pytest.fixture(scope="session")
def grid_dims():
    return (20, 10)


@pytest.fixture(scope="session")
def piece_size():
    return 4


@pytest.fixture(scope="function")
def engine(grid_dims, piece_size):
    num_actions, num_pieces = {
        1: (grid_dims[1], 1),
        2: (2 * grid_dims[1] - 1, 1),
        3: (4 * grid_dims[1] - 4, 2),
        4: (4 * grid_dims[1] - 6, 7),
    }[piece_size]

    engine = Engine(
        grid_dims=grid_dims,
        piece_size=piece_size,
        num_pieces=num_pieces,
        num_actions=num_actions,
    )

    engine._reset()

    return engine


def test_is_illegal_non_empty_overlapping(engine, piece_size):
    engine._piece = Polymino(piece_size, 0)
    engine._anchor = [0, engine._height - 1]
    engine._grid[0, engine._height - 1] = 1

    assert engine._is_illegal()


def test_hard_drop_empty_grid(engine, piece_size):
    engine._piece = Polymino(piece_size, 0)
    engine._anchor = [0, 0]
    engine._hard_drop()

    assert engine._anchor, [0, engine._height - 1]


def test_hard_drop_non_empty_grid(engine, piece_size):
    engine._piece = Polymino(piece_size, 0)
    engine._anchor = [0, 0]
    engine._grid[0, engine._height - 1] = 1
    engine._hard_drop()

    assert engine._anchor == [0, engine._height - 2]


def test_clear_rows_output_with_empty_grid(engine, piece_size):
    assert engine._clear_rows() == 0


def test_clear_rows_empty_grid_after(engine, piece_size):
    engine._clear_rows()
    grid_after = np.zeros((engine._width, engine._height), dtype="bool")

    np.testing.assert_array_equal(engine._grid, grid_after)


def test_clear_rows_output_one_full_row(engine, piece_size):
    engine._grid[:, engine._height - 1 :] = 1

    assert engine._clear_rows() == 1


def test_clear_rows_one_full_row_grid_after(engine, piece_size):
    engine._grid[:, engine._height - 1 :] = 1
    engine._clear_rows()
    grid_after = np.zeros((engine._width, engine._height), dtype="bool")

    np.testing.assert_array_equal(engine._grid, grid_after)


def test_clear_rows_output_two_full_rows(engine, piece_size):
    engine._grid[:, engine._height - 2 :] = 1

    assert engine._clear_rows() == 2


def test_clear_rows_two_full_rows_grid_after(engine, piece_size):
    engine._grid[:, engine._height - 2 :] = 1
    engine._clear_rows()
    grid_after = np.zeros((engine._width, engine._height), dtype="bool")

    np.testing.assert_array_equal(engine._grid, grid_after)


def test_clear_rows_output_two_full_rows_full_cell_above(engine, piece_size):
    engine._grid[:, engine._height - 2 :] = 1
    engine._grid[3, engine._height - 3] = 1

    assert engine._clear_rows() == 2


def test_clear_rows_two_full_rows_full_cell_above_grid_after(engine, piece_size):
    engine._grid[:, engine._height - 2 :] = 1
    engine._grid[3, engine._height - 3] = 1
    engine._clear_rows()
    grid_after = np.zeros((engine._width, engine._height), dtype="bool")
    grid_after[3, engine._height - 1] = 1

    np.testing.assert_array_equal(engine._grid, grid_after)


def test_update_grid_simple(engine, piece_size):
    engine._piece = Polymino(piece_size, 0)
    engine._anchor = [0, engine._height - 1]
    grid_to_compare = np.zeros((engine._width, engine._height), dtype="bool")
    grid_to_compare[0, engine._height - piece_size :] = 1
    engine._update_grid(True)

    np.testing.assert_array_equal(engine._grid, grid_to_compare)


def test_update_grid_empty(engine, piece_size):
    engine._piece = Polymino(piece_size, 0)
    engine._anchor = [0, engine._height - 1]
    grid_to_compare = np.zeros((engine._width, engine._height), dtype="bool")
    engine._update_grid(False)

    np.testing.assert_array_equal(engine._grid, grid_to_compare)


def test_update_grid_populated(engine, piece_size):
    engine._piece = Polymino(piece_size, 0)
    engine._grid[0, engine._height - piece_size :] = 1
    engine._anchor = [0, engine._height - 1]
    grid_to_compare = np.zeros((engine._width, engine._height), dtype="bool")
    engine._update_grid(False)

    np.testing.assert_array_equal(engine._grid, grid_to_compare)


def test_compute_all_available_actions(engine, piece_size):
    engine._compute_all_available_actions()

    for value in engine._all_available_actions.values():
        assert engine._num_actions == len(value)
