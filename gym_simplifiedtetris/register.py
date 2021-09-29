from gym.envs.registration import register as gym_register

env_list: list = []


def register(
        idx: str,
        entry_point: str,
):
    """
    This function performs some checks on the arguments provided, and then
    registers the custom environments in Gym.
    """

    assert idx.startswith(
        "simplifiedtetris-"), 'Env ID should start with "simplifiedtetris-".'
    assert entry_point.startswith(
        "gym_simplifiedtetris.envs:SimplifiedTetris"), 'Entry point should\
            start with "gym_simplifiedtetris.envs:SimplifiedTetris".'
    assert entry_point.endswith("Env"), 'Entry point should end with "Env".'
    assert idx not in env_list, 'Incorrect env ID provided.'

    gym_register(
        id=idx,
        entry_point=entry_point,
        kwargs={
            'grid_dims': (8, 6),
            'piece_size': 3,
        },
    )
    env_list.append(idx)
