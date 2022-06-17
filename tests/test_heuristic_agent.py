"""Testing the Dellacherie agent's methods."""

import unittest

import numpy as np
from gym_simplifiedtetris.agents import DellacherieAgent
from gym_simplifiedtetris.auxiliary import Polymino
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


class DellacherieAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DellacherieAgent()
        self.env = Tetris(grid_dims=(20, 10), piece_size=4)

        self.env.reset()

    def tearDown(self) -> None:
        self.env.close()
        del self.env
        del self.agent

    def test_get_dellacherie_scores_empty_grid(self) -> None:
        self.env._engine._piece = Polymino(self.env.piece_size, 0)
        # array_to_compare = np.array(
        #     [
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         614.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         4.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         312.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         302.0,
        #     ]
        # )
        array_to_compare = np.NINF * np.ones(
            self.env._engine._num_actions, dtype="double"
        )
        array_to_compare[10] = abs(0 - 6) * 100 + 10 - (90 / 90)
        array_to_compare[16] = abs(6 - 6) * 100 + 0 - (90 / 90)
        array_to_compare[27] = abs(3 - 6) * 100 + 10 - (270 / 90)
        array_to_compare[33] = abs(9 - 6) * 100 + 0 - (270 / 90)
        np.testing.assert_array_equal(
            self.agent._compute_dell_scores(self.env),
            array_to_compare,
        )

    def test_get_dellacherie_funcs_populated_grid(self) -> None:
        self.env._engine._grid[:, -5:] = True
        self.env._engine._grid[
            1, self.env._engine._height - 5 : self.env._engine._height - 1
        ] = False
        self.env._engine._grid[
            self.env._engine._width - 1, self.env._engine._height - 2
        ] = False
        self.env._engine._grid[
            self.env._engine._width - 2, self.env._engine._height - 1
        ] = False
        self.env._engine._grid[
            self.env._engine._width - 3, self.env._engine._height - 3
        ] = False
        self.env._engine._grid[
            self.env._engine._width - 1, self.env._engine._height - 6
        ] = True
        self.env._engine._grid[
            3, self.env._engine._height - 3 : self.env._engine._height - 1
        ] = False
        self.env._engine._piece = Polymino(self.env.piece_size, 0)
        self.env._engine._anchor = [0, 0]
        self.env._engine._hard_drop()
        self.env._engine._update_grid(True)
        self.env._engine._clear_rows()
        array_to_compare = np.array(
            [func(self.env) for func in self.agent._get_dell_funcs()]
        )
        np.testing.assert_array_equal(
            array_to_compare,
            np.array(
                [5.5 + 0.5 * self.env.piece_size, 0, 48, 18, 5, 10], dtype="double"
            ),
        )

    def test_get_landing_height_I_piece_(self) -> None:
        self.env._engine._piece = Polymino(self.env.piece_size, 0)
        self.env._engine._anchor = [0, self.env._engine._height - 1]
        self.env._engine._update_grid(True)
        self.assertEqual(
            self.agent._get_landing_height(self.env), 0.5 * (1 + self.env.piece_size)
        )

    def test_get_eroded_cells_empty(self) -> None:
        self.assertEqual(self.agent._get_eroded_cells(self.env), 0)

    def test_get_eroded_cells_single(self) -> None:
        self.env._engine._grid[:, self.env._engine._height - 1 :] = True
        self.env._engine._grid[0, self.env._engine._height - 1] = False
        self.env._engine._piece = Polymino(self.env.piece_size, 0)
        self.env._engine._anchor = [0, 0]
        self.env._engine._hard_drop()
        self.env._engine._update_grid(True)
        self.env._engine._clear_rows()
        self.assertEqual(self.agent._get_eroded_cells(self.env), 1)

    def test_get_row_transitions_empty(self) -> None:
        self.assertEqual(self.agent._get_row_transitions(self.env), 40)

    def test_get_row_transitions_populated(self) -> None:
        self.env._engine._grid[:, -2:] = True
        self.env._engine._grid[0, self.env._engine._height - 1] = False
        self.env._engine._grid[2, self.env._engine._height - 1] = False
        self.env._engine._grid[1, self.env._engine._height - 2] = False
        self.assertEqual(self.agent._get_row_transitions(self.env), 42)

    def test_get_row_transitions_populated_more_row_transitions(self) -> None:
        self.env._engine._grid[:, -2:] = True
        self.env._engine._grid[0, self.env._engine._height - 2 :] = False
        self.env._engine._grid[2, self.env._engine._height - 2 :] = False
        self.env._engine._grid[4, self.env._engine._height - 1] = False
        np.testing.assert_array_equal(self.agent._get_row_transitions(self.env), 46)

    def test_get_column_transitions_empty(self) -> None:
        self.assertEqual(self.agent._get_col_transitions(self.env), 10)

    def test_get_column_transitions_populated(self) -> None:
        self.env._engine._grid[:, -2:] = True
        self.env._engine._grid[0, self.env._engine._height - 1] = False
        self.env._engine._grid[2, self.env._engine._height - 1] = False
        self.env._engine._grid[1, self.env._engine._height - 2] = False
        self.assertEqual(self.agent._get_col_transitions(self.env), 14)

    def test_get_column_transitions_populated_less_column_transitions(self) -> None:
        self.env._engine._grid[:, -2:] = True
        self.env._engine._grid[0, self.env._engine._height - 2 :] = False
        self.env._engine._grid[2, self.env._engine._height - 2 :] = False
        self.env._engine._grid[4, self.env._engine._height - 1] = False
        np.testing.assert_array_equal(self.agent._get_col_transitions(self.env), 12)

    def test_get_holes_empty(self) -> None:
        self.assertEqual(self.agent._get_holes(self.env), 0)

    def test_get_holes_populated_two_holes(self) -> None:
        self.env._engine._grid[:, -2:] = True
        self.env._engine._grid[0, self.env._engine._height - 1] = False
        self.env._engine._grid[2, self.env._engine._height - 1] = False
        self.assertEqual(self.agent._get_holes(self.env), 2)

    def test_get_holes_populated_no_holes(self) -> None:
        self.env._engine._grid[:, -2:] = True
        self.env._engine._grid[0, self.env._engine._height - 2 :] = False
        self.assertEqual(self.agent._get_holes(self.env), 0)

    def test_get_holes_populated_one_hole(self) -> None:
        self.env._engine._grid[:, -2:] = True
        self.env._engine._grid[0, self.env._engine._height - 2 :] = False
        self.env._engine._grid[2, self.env._engine._height - 2 :] = False
        self.env._engine._grid[4, self.env._engine._height - 1] = False
        np.testing.assert_array_equal(self.agent._get_holes(self.env), 1)

    def test_get_cumulative_wells_empty(self) -> None:
        np.testing.assert_array_equal(self.agent._get_cum_wells(self.env), 0)

    def test_get_cumulative_wells_populated(self) -> None:
        self.env._engine._grid[:, -2:] = True
        self.env._engine._grid[0, self.env._engine._height - 2 :] = False
        np.testing.assert_array_equal(self.agent._get_cum_wells(self.env), 3)

    def test_get_cumulative_wells_populated_deeper_well(self) -> None:
        self.env._engine._grid[:, -2:] = True
        self.env._engine._grid[0, self.env._engine._height - 2 :] = False
        self.env._engine._grid[2, self.env._engine._height - 2 :] = False
        self.env._engine._grid[4, self.env._engine._height - 1] = False
        np.testing.assert_array_equal(self.agent._get_cum_wells(self.env), 6)


if __name__ == "__main__":
    unittest.main()
