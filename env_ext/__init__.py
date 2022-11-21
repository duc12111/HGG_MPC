import gym
import env_ext.fetch as fetch_env
from .utils import goal_distance, goal_distance_obs

Robotics_envs_id = [
    'FetchPickDynSqrObstacle-v1',
    # 'FetchPickDynLabyrinthEnv-v1', 	# was easy for the agent and not symmetric
    'FetchPickDynObstaclesEnv-v1',
    'FetchPickDynObstaclesEnv-v2',
    'FetchPickDynLiftedObstaclesEnv-v1',
    'FetchPickDynObstaclesMaxEnv-v1',
    'TestEnv-v1',
    'FetchPickDynObstaclesRstopEnv-v1',
    'FetchPickDynObstaclesRstopEnv-v2',
    'FetchPickDynObstaclesSinEnv-v1',
    'FetchPickDynLabyrinthEnv-v1',
    'FetchPickSafeRiskyLaneEnv-v1',
]


def make_env(args):
    assert args.env in Robotics_envs_id
    if args.env[:5] == 'Fetch' or args.env[:7] == 'TestEnv':
        return fetch_env.make_env(args)
    else:
        return None


def clip_return_range(args):
    gamma_sum_min = args.reward_min / (1.0 - args.gamma)
    gamma_sum_max = args.reward_max / (1.0 - args.gamma)
    return {
        'FetchPickDynSqrObstacle-v1': (gamma_sum_min, gamma_sum_max),
        'FetchPickDynLabyrinthEnv-v1': (gamma_sum_min, gamma_sum_max),
        'FetchPickDynObstaclesEnv-v1': (gamma_sum_min, gamma_sum_max),
        'FetchPickDynObstaclesEnv-v2': (gamma_sum_min, gamma_sum_max),
        'FetchPickDynLiftedObstaclesEnv-v1': (gamma_sum_min, gamma_sum_max),
        'FetchPickDynObstaclesMaxEnv-v1': (gamma_sum_min, gamma_sum_max),
        'TestEnv-v1': (gamma_sum_min, gamma_sum_max),
        'FetchPickDynObstaclesRstopEnv-v1': (gamma_sum_min, gamma_sum_max),
        'FetchPickDynObstaclesRstopEnv-v2': (gamma_sum_min, gamma_sum_max),
        'FetchPickDynObstaclesSinEnv-v1': (gamma_sum_min, gamma_sum_max),
        'FetchPickDynLabyrinthEnv-v1': (gamma_sum_min, gamma_sum_max),
        'FetchPickSafeRiskyLaneEnv-v1': (gamma_sum_min, gamma_sum_max),

    }[args.env]
