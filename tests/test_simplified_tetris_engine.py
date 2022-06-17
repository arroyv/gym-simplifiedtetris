"""Testing the class SimplifiedTetrisEngine."""

import unittest

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


def test_hard_drop_empty_grid(engine, piece_size) -> None:
    engine._piece = Polymino(piece_size, 0)
    engine._anchor = [0, 0]
    engine._hard_drop()

    assert engine._anchor, [0, engine._height - 1]


def test_hard_drop_non_empty_grid(engine, piece_size) -> None:
    engine._piece = Polymino(piece_size, 0)
    engine._anchor = [0, 0]
    engine._grid[0, engine._height - 1] = 1
    engine._hard_drop()

    assert engine._anchor == [0, engine._height - 2]


def test_clear_rows_output_with_empty_grid(engine, piece_size) -> None:
    assert engine._clear_rows() == 0


def test_clear_rows_empty_grid_after(engine, piece_size) -> None:
    engine._clear_rows()
    grid_after = np.zeros((engine._width, engine._height), dtype="bool")

    np.testing.assert_array_equal(engine._grid, grid_after)


def test_clear_rows_output_one_full_row(engine, piece_size) -> None:
    engine._grid[:, engine._height - 1 :] = 1

    assert engine._clear_rows() == 1


def test_clear_rows_one_full_row_grid_after(engine, piece_size) -> None:
    engine._grid[:, engine._height - 1 :] = 1
    engine._clear_rows()
    grid_after = np.zeros((engine._width, engine._height), dtype="bool")

    np.testing.assert_array_equal(engine._grid, grid_after)


def test_clear_rows_output_two_full_rows(engine, piece_size) -> None:
    engine._grid[:, engine._height - 2 :] = 1

    assert engine._clear_rows() == 2


def test_clear_rows_two_full_rows_grid_after(engine, piece_size) -> None:
    engine._grid[:, engine._height - 2 :] = 1
    engine._clear_rows()
    grid_after = np.zeros((engine._width, engine._height), dtype="bool")

    np.testing.assert_array_equal(engine._grid, grid_after)


def test_clear_rows_output_two_full_rows_full_cell_above(engine, piece_size) -> None:
    engine._grid[:, engine._height - 2 :] = 1
    engine._grid[3, engine._height - 3] = 1

    assert engine._clear_rows() == 2


def test_clear_rows_two_full_rows_full_cell_above_grid_after(
    engine, piece_size
) -> None:
    engine._grid[:, engine._height - 2 :] = 1
    engine._grid[3, engine._height - 3] = 1
    engine._clear_rows()
    grid_after = np.zeros((engine._width, engine._height), dtype="bool")
    grid_after[3, engine._height - 1] = 1

    np.testing.assert_array_equal(engine._grid, grid_after)


def test_update_grid_simple(engine, piece_size) -> None:
    engine._piece = Polymino(piece_size, 0)
    engine._anchor = [0, engine._height - 1]
    grid_to_compare = np.zeros((engine._width, engine._height), dtype="bool")
    grid_to_compare[0, engine._height - piece_size :] = 1
    engine._update_grid(True)

    np.testing.assert_array_equal(engine._grid, grid_to_compare)


def test_update_grid_empty(engine, piece_size) -> None:
    engine._piece = Polymino(piece_size, 0)
    engine._anchor = [0, engine._height - 1]
    grid_to_compare = np.zeros((engine._width, engine._height), dtype="bool")
    engine._update_grid(False)

    np.testing.assert_array_equal(engine._grid, grid_to_compare)


def test_update_grid_populated(engine, piece_size) -> None:
    engine._piece = Polymino(piece_size, 0)
    engine._grid[0, engine._height - piece_size :] = 1
    engine._anchor = [0, engine._height - 1]
    grid_to_compare = np.zeros((engine._width, engine._height), dtype="bool")
    engine._update_grid(False)

    np.testing.assert_array_equal(engine._grid, grid_to_compare)


def test_compute_all_available_actions(engine, piece_size) -> None:
    engine._compute_all_available_actions()

    for value in engine._all_available_actions.values():
        assert engine._num_actions == len(value)


class SimplifiedTetrisEngineStandardTetrisTest(unittest.TestCase):
    def setUp(self) -> None:
        height = 20
        width = 10
        self.piece_size = 4

        num_actions, num_pieces = {
            1: (width, 1),
            2: (2 * width - 1, 1),
            3: (4 * width - 4, 2),
            4: (4 * width - 6, 7),
        }[self.piece_size]

        self.engine = Engine(
            grid_dims=(height, width),
            piece_size=self.piece_size,
            num_pieces=num_pieces,
            num_actions=num_actions,
        )

        self.engine._reset()

    def tearDown(self) -> None:
        self.engine._close()
        del self.engine

    def test_is_illegal_non_empty_overlapping(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._anchor = [0, self.engine._height - 1]
        self.engine._grid[0, self.engine._height - 1] = 1

        self.assertEqual(self.engine._is_illegal(), True)

    def test_hard_drop_empty_grid(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._anchor = [0, 0]
        self.engine._hard_drop()

        self.assertEqual(self.engine._anchor, [0, self.engine._height - 1])

    def test_hard_drop_non_empty_grid(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._anchor = [0, 0]
        self.engine._grid[0, self.engine._height - 1] = 1
        self.engine._hard_drop()

        self.assertEqual(self.engine._anchor, [0, self.engine._height - 2])

    def test_clear_rows_output_with_empty_grid(self) -> None:

        self.assertEqual(self.engine._clear_rows(), 0)

    def test_clear_rows_empty_grid_after(self) -> None:
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")

        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test_clear_rows_output_one_full_row(self) -> None:
        self.engine._grid[:, self.engine._height - 1 :] = 1

        self.assertEqual(self.engine._clear_rows(), 1)

    def test_clear_rows_one_full_row_grid_after(self) -> None:
        self.engine._grid[:, self.engine._height - 1 :] = 1
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")

        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test_clear_rows_output_two_full_rows(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1

        self.assertEqual(self.engine._clear_rows(), 2)

    def test_clear_rows_two_full_rows_grid_after(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")

        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test_clear_rows_output_two_full_rows_full_cell_above(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1
        self.engine._grid[3, self.engine._height - 3] = 1

        self.assertEqual(self.engine._clear_rows(), 2)

    def test_clear_rows_two_full_rows_full_cell_above_grid_after(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1
        self.engine._grid[3, self.engine._height - 3] = 1
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")
        grid_after[3, self.engine._height - 1] = 1

        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test_update_grid_simple(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._anchor = [0, self.engine._height - 1]
        grid_to_compare = np.zeros(
            (self.engine._width, self.engine._height), dtype="bool"
        )
        grid_to_compare[0, self.engine._height - self.piece_size :] = 1
        self.engine._update_grid(True)

        np.testing.assert_array_equal(self.engine._grid, grid_to_compare)

    def test_update_grid_empty(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._anchor = [0, self.engine._height - 1]
        grid_to_compare = np.zeros(
            (self.engine._width, self.engine._height), dtype="bool"
        )
        self.engine._update_grid(False)

        np.testing.assert_array_equal(self.engine._grid, grid_to_compare)

    def test_update_grid_populated(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._grid[0, self.engine._height - self.piece_size :] = 1
        self.engine._anchor = [0, self.engine._height - 1]
        grid_to_compare = np.zeros(
            (self.engine._width, self.engine._height), dtype="bool"
        )
        self.engine._update_grid(False)

        np.testing.assert_array_equal(self.engine._grid, grid_to_compare)

    def test_compute_all_available_actions(self) -> None:
        self.engine._compute_all_available_actions()

        for value in self.engine._all_available_actions.values():
            self.assertEqual(self.engine._num_actions, len(value))


if __name__ == "__main__":
    unittest.main()
