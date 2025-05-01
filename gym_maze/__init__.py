from gym.envs.registration import register
from . import envs


register(
    id='maze-v0',
    entry_point='gym_maze.envs:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='maze-sample-5x5-v0',
    entry_point='gym_maze.envs:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='maze-random-5x5-v0',
    entry_point='gym_maze.envs:MazeEnvRandom5x5',
    max_episode_steps=2000,
    nondeterministic=True,
)

register(
    id='maze-sample-10x10-v0',
    entry_point='gym_maze.envs:MazeEnvSample10x10',
    max_episode_steps=10000,
)

register(
    id='maze-random-10x10-v0',
    entry_point='gym_maze.envs:MazeEnvRandom10x10',
    max_episode_steps=10000,
    nondeterministic=True,
)

register(
    id='maze-sample-3x3-v0',
    entry_point='gym_maze.envs:MazeEnvSample3x3',
    max_episode_steps=1000,
)

register(
    id='maze-random-3x3-v0',
    entry_point='gym_maze.envs:MazeEnvRandom3x3',
    max_episode_steps=1000,
    nondeterministic=True,
)


register(
    id='maze-sample-100x100-v0',
    entry_point='gym_maze.envs:MazeEnvSample100x100',
    max_episode_steps=1000000,
)

register(
    id='maze-random-100x100-v0',
    entry_point='gym_maze.envs:MazeEnvRandom100x100',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-10x10-plus-v0',
    entry_point='gym_maze.envs:MazeEnvRandom10x10Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-20x20-plus-v0',
    entry_point='gym_maze.envs:MazeEnvRandom20x20Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-30x30-plus-v0',
    entry_point='gym_maze.envs:MazeEnvRandom30x30Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-sample-10x10-plus-v0',
    entry_point='gym_maze.envs:MazeEnvSample10x10Plus',
    max_episode_steps=1000000,
)

register(
    id='maze-sample-20x20-v0',
    entry_point='gym_maze.envs:MazeEnvSample20x20',
    max_episode_steps=1000000,
)

register(
    id='maze-random-20x20-v0',
    entry_point='gym_maze.envs:MazeEnvRandom20x20',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-sample-30x30-v0',
    entry_point='gym_maze.envs:MazeEnvSample30x30',
    max_episode_steps=1000000,
)

register(
    id='maze-random-30x30-v0',
    entry_point='gym_maze.envs:MazeEnvRandom30x30',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-sample-50x50-v0',
    entry_point='gym_maze.envs:MazeEnvSample50x50',
    max_episode_steps=1000000,
)

register(
    id='maze-random-50x50-v0',
    entry_point='gym_maze.envs:MazeEnvRandom50x50',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-sample-200x200-v0',
    entry_point='gym_maze.envs:MazeEnvSample200x200',
    max_episode_steps=1000000,
)

register(
    id='maze-sample-500x500-v0',
    entry_point='gym_maze.envs:MazeEnvSample500x500',
    max_episode_steps=1000000,
)