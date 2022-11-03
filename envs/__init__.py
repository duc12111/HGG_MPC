import gym
from .fetch.test_env import TestEnv
from .fetch.pick_dyn_sqr_obstacle import FetchPickDynSqrObstacleEnv
from .fetch.pick_dyn_labyrinth import FetchPickDynLabyrinthEnv
from .fetch.pick_dyn_obstacles import FetchPickDynObstaclesEnv
from .fetch.pick_dyn_obstacles2 import FetchPickDynObstaclesEnv2
from .fetch.pick_dyn_obstacles_max import FetchPickDynObstaclesMaxEnv
from .fetch.pick_dyn_lifted_obstacles import FetchPickDynLiftedObstaclesEnv
from .fetch.pick_dyn_obstacles2_rstop import FetchPickDynObstaclesRstopEnv
from .fetch.pick_dyn_obstackles2_sin import FetchPickDynObstaclesSinEnv

def register_custom_envs():
    gym.envs.register(
        id='FetchPickDynSqrObstacle-v1',
        entry_point='envs:FetchPickDynSqrObstacleEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynLabyrinthEnv-v1',
        entry_point='envs:FetchPickDynLabyrinthEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynObstaclesEnv-v1',
        entry_point='envs:FetchPickDynObstaclesEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynObstaclesEnv-v2',
        entry_point='envs:FetchPickDynObstaclesEnv2',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynLiftedObstaclesEnv-v1',
        entry_point='envs:FetchPickDynLiftedObstaclesEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynObstaclesMaxEnv-v1',
        entry_point='envs:FetchPickDynObstaclesMaxEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='TestEnv-v1',
        entry_point='envs:TestEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynObstaclesRstopEnv-v1',
        entry_point='envs:FetchPickDynObstaclesRstopEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynObstaclesSinEnv-v1',
        entry_point='envs:FetchPickDynObstaclesSinEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
