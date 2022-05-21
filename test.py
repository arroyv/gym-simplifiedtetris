import numpy as np

height = 20
width = 10

board = np.zeros((height, width))

board[-4:, 0] = 1
board[-5:, 1] = 1
board[-3:, 2] = 1
board[-1:, 3] = 1
board[-7:, 4] = 1
board[-1, 4] = 0
board[-4:, 5] = 1
board[-4:, 6] = 1
board[-1, 6] = 0
board[-4:, 7] = 1
board[-3:, 8] = 1
board[-1:, 9] = 1

grid_ext = np.ones((height + 1, width + 2), dtype="bool")
grid_ext[:1, 1:-1] = False
grid_ext[1:, 1:-1] = board[:height]

potential_wells = (
    np.roll(grid_ext, 1, axis=1) & np.roll(grid_ext, -1, axis=1) & ~grid_ext
)

col_heights = np.zeros(12)
col_heights[1:-1] = height - np.argmax(board, axis=0)
col_heights = np.where(col_heights == 20, 0, col_heights)

x = np.linspace(1, width + 2, width + 2)
y = np.linspace(height + 1, 1, height + 1)
_, yv = np.meshgrid(x, y)

above_outline = (col_heights < yv).astype(int)

cumulative_wells = np.sum(
    np.cumsum(potential_wells, axis=0) * above_outline,
)

print(cumulative_wells)
