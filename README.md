<h1 align="center"> Gym-SimplifiedTetris </h1> <br>
<p align="center">
    <img alt="Tetris" title="Tetris" src="assets/20x10_4.gif" width="450">
  </a>
</p>

## Table of Contents <!-- omit in toc -->

- [1. Introduction](#1-introduction)
- [2. Setup](#2-setup)
- [3. Features](#3-features)
  - [3.1. Environments](#31-environments)
  - [3.2. `reset()` method](#32-reset-method)
  - [3.3. `step(action: int)` method](#33-stepaction-int-method)
  - [3.4. `render()` method](#34-render-method)
  - [3.5. Variable dimensions and piece size](#35-variable-dimensions-and-piece-size)
  - [3.6. Action and observation spaces](#36-action-and-observation-spaces)
  - [3.7. Game ending](#37-game-ending)
- [4. Examples](#4-examples)
- [6. Coming soon](#6-coming-soon)
- [7. Suggestions](#7-suggestions)
- [8. Inspiration](#8-inspiration)
- [9. License](#9-license)

## 1. Introduction

<p align="left">
  <a href="https://img.shields.io/github/license/OliverOverend/gym-simplifiedtetristemp">
    <img src="https://img.shields.io/github/license/OliverOverend/gym-simplifiedtetristemp?style=flat-square">
  </a>
  <a href="http://makeapullrequest.com">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square">
  </a>
</p>

Gym-SimplifiedTetris is a Python package that can create ***simplified*** reinforcement learning (RL) environments for Tetris that conform to the [OpenAI Gym](https://github.com/openai/gym) API.

This README summarises the package's functionality, describes how to build more custom Gym environments, and provides a couple of examples showing how to use an environment.

The environments are simplified because the artificial agent must select the column and piece's rotation before the piece is dropped vertically downwards. To the best of the author's knowledge, this is the first open-source package to create RL Gym environments that use the simplified setting, commonly used by previous approaches to the game of Tetris.

## 2. Setup

To clone the repository:
```bash
git clone https://github.com/OliverOverend/gym-simplifiedtetristemp
```

Here is a list of the versions used during development:

- Python 3.7.4
- NumPy 1.16.5
- Gym 0.18.0
- OpenCV-Python 4.5.1
- Matplotlib 3.4.2
- Pillow 6.2.0
- Stable-Baselines3 1.1.0

## 3. Features

### 3.1. Environments

There are currently two environments provided:
- `simplifiedtetris-binary-v0`: The observation space is a flattened NumPy array containing a binary representation of the grid, plus the current piece's ID
- `simplifiedtetris-partbinary-v0`: The observation space is a flattened NumPy array containing a binary representation of the grid excluding the top `piece_size` rows, plus the current piece's ID

More custom Gym environments with different observation spaces and reward functions can be implemented easily. To add more environments to `gym_simplifiedtetris.register.env_list`, ensure that they inherit from `SimplifiedTetrisBinaryEnv` and are registered using:
```python
register(
    idx='INSERT_ENV_NAME_HERE',
    entry_point='gym_simplifiedtetris.envs:INSERT_ENV_CLASS_NAME_HERE',
)
```

### 3.2. `reset()` method

The `reset()` method returns a 1D array that contains some binary representation of the grid, plus the current piece's ID.

```python
obs = env.reset()
```

### 3.3. `step(action: int)` method

Each environment's step method returns four values:
- `observation` (**NumPy array**): a 1D array that contains some binary representation of the grid, plus the current piece's ID
- `reward` (**float**): the amount of reward received from the previous action
- `done` (**bool**): a game termination flag
- `info` (**dict**): only contains the `num_rows_cleared` due to taking the previous action

```python
obs, rwd, done, info = env.step(action)
```

### 3.4. `render()` method

The user has access to the following controls during rendering:
- Pause (**SPACEBAR**)
- Speed up (**RIGHT key**)
- Slow down (**LEFT key**)
- Quit (**ESC**)

```python
env.render()
```

### 3.5. Variable dimensions and piece size

The user can choose to deviate from the standard grid dimensions and Tetriminos by editing the `gym_register` kwargs. The user can choose from four different sets of pieces: monominos, dominos, trominoes & Tetriminos. The user can select a height in the interval $[$`piece_size`$+1, 20]$ and a width in the interval $[$`piece_size`$, 10]$. Below is a GIF showing games being played on a $8 \times 6$ grid with trominoes as the pieces.

<p align="center">
    <img src="assets/8x6_3.gif" width="400">
</p>

### 3.6. Action and observation spaces

Each environment comes with an `observation_space` that is a `Box` space and an `action_space` that is a `Discrete` space. At each time step, the artificial agent must choose an action (an integer from a particular range). Each action maps to a translation/rotation tuple that specifies the column to drop the piece and its rotation. The ranges for the four different piece sizes are:
- Monominos: $[0, w - 1]$
- Dominos: $[0, 2w - 2]$
- Trominoes: $[0, 4w - 5]$
- Tetriminos: $[0, 4w  - 7]$

where $w$ is the grid width.

### 3.7. Game ending

Each game of Tetris terminates if the following condition is satisfied: any of the dropped piece's square blocks enter into the top `piece_size` rows before any full rows are cleared. This definition ensures that scores achieved are lower bounds on the score that could have been achieved on a standard game of Tetris, as laid out in Colin Fahey's ['Standard Tetris' specification](https://www.colinfahey.com/tetris/tetris.html#:~:text=5.%20%22Standard%20Tetris%22%20specification).

## 4. Examples

Here is an example of using an instance of the `simplifiedtetris-binary-v0` environment for ten games:

```python
import gym

import gym_simplifiedtetris

env = gym.make('simplifiedtetris-binary-v0')
obs = env.reset()

# Run 10 games of Tetris, selecting actions uniformly at random.
num_episodes = 0
while num_episodes < 10:
    env.render()
    action = env.action_space.sample()
    obs, rwd, done, info = env.step(action)

    if done:
        print(f"Episode {num_episodes + 1} has terminated.")
        num_episodes += 1
        obs = env.reset()

env.close()
```

The env can instead be imported directly:
```python
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris

env = Tetris(
    grid_dims=(20, 10),
    piece_size=4,
)
```

## 6. Coming soon

- Unit tests

## 7. Suggestions

Please feel free to provide any suggestions or file any issues [here](https://github.com/OliverOverend/gym-simplifiedtetristemp/issues/new).

## 8. Inspiration

This package utilises several methods from the [codebase](https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning) developed by Lay (2020). The class hierarchy design was inspired by a [codebase](https://github.com/Hewiiitt/Gym-Circuitboard) developed by Matt Hewitt.

## 9. License

This project is licensed under the terms of the [MIT license](https://github.com/OliverOverend/gym-simplifiedtetristemp/blob/master/LICENSE.md).