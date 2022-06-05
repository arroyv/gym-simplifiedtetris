"""Testing SimplifiedTetrisEngine."""

import unittest

import numpy as np
from gym_simplifiedtetris.auxiliary import Polymino
from gym_simplifiedtetris.envs import SimplifiedTetrisEngine as Engine


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

    def test__is_illegal_non_empty_overlapping(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._anchor = [0, self.engine._height - 1]
        self.engine._grid[0, self.engine._height - 1] = 1
        self.assertEqual(self.engine._is_illegal(), True)

    def test__hard_drop_empty_grid(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._anchor = [0, 0]
        self.engine._hard_drop()
        self.assertEqual(self.engine._anchor, [0, self.engine._height - 1])

    def test__hard_drop_non_empty_grid(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._anchor = [0, 0]
        self.engine._grid[0, self.engine._height - 1] = 1
        self.engine._hard_drop()
        self.assertEqual(self.engine._anchor, [0, self.engine._height - 2])

    def test__clear_rows_output_with_empty_grid(self) -> None:
        self.assertEqual(self.engine._clear_rows(), 0)

    def test__clear_rows_empty_grid_after(self) -> None:
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")
        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test__clear_rows_output_one_full_row(self) -> None:
        self.engine._grid[:, self.engine._height - 1 :] = 1
        self.assertEqual(self.engine._clear_rows(), 1)

    def test__clear_rows_one_full_row_grid_after(self) -> None:
        self.engine._grid[:, self.engine._height - 1 :] = 1
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")
        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test__clear_rows_output_two_full_rows(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1
        self.assertEqual(self.engine._clear_rows(), 2)

    def test__clear_rows_two_full_rows_grid_after(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")
        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test__clear_rows_output_two_full_rows_full_cell_above(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1
        self.engine._grid[3, self.engine._height - 3] = 1
        self.assertEqual(self.engine._clear_rows(), 2)

    def test__clear_rows_two_full_rows_full_cell_above_grid_after(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1
        self.engine._grid[3, self.engine._height - 3] = 1
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")
        grid_after[3, self.engine._height - 1] = 1
        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test__update_grid_simple(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._anchor = [0, self.engine._height - 1]
        grid_to_compare = np.zeros(
            (self.engine._width, self.engine._height), dtype="bool"
        )
        grid_to_compare[0, self.engine._height - self.piece_size :] = 1
        self.engine._update_grid(True)
        np.testing.assert_array_equal(self.engine._grid, grid_to_compare)

    def test__update_grid_empty(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._anchor = [0, self.engine._height - 1]
        grid_to_compare = np.zeros(
            (self.engine._width, self.engine._height), dtype="bool"
        )
        self.engine._update_grid(False)
        np.testing.assert_array_equal(self.engine._grid, grid_to_compare)

    def test__update_grid_populated(self) -> None:
        self.engine._piece = Polymino(self.piece_size, 0)
        self.engine._grid[0, self.engine._height - self.piece_size :] = 1
        self.engine._anchor = [0, self.engine._height - 1]
        grid_to_compare = np.zeros(
            (self.engine._width, self.engine._height), dtype="bool"
        )
        self.engine._update_grid(False)
        np.testing.assert_array_equal(self.engine._grid, grid_to_compare)

    def test__compute_all_available_actions(self) -> None:
        self.engine._compute_all_available_actions()
        for value in self.engine._all_available_actions.values():
            self.assertEqual(self.engine._num_actions, len(value))


if __name__ == "__main__":
    unittest.main()
