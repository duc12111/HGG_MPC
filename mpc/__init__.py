import forcespro
from .plot import MPCDebugPlot
from .pick_dyn_sqr_obstacles import generate_pathplanner
from .pick_dyn_obstacles import generate_pathplanner
from .pick_dyn_lifted_obstacles import generate_pathplanner
from .pick_dyn_obstacles_max import generate_pathplanner
from .pick_dyn_labyrinth import generate_pathplanner
from .pick_safe_risky_lane import generate_pathplanner
from .mpc_common import extract_parameters, make_obs, get_args


def make_mpc(args):
    a = {
        'FetchPickDynSqrObstacle-v1': pick_dyn_sqr_obstacles,
        'FetchPickDynObstaclesEnv-v1': pick_dyn_obstacles,
        'FetchPickDynObstaclesEnv-v2': pick_dyn_obstacles,
        'FetchPickDynObstaclesRstopEnv-v1': pick_dyn_obstacles,
        'FetchPickDynObstaclesSinEnv-v1': pick_dyn_obstacles,
        'FetchPickDynLiftedObstaclesEnv-v1': pick_dyn_lifted_obstacles,
        # 'FetchPickDynObstaclesMaxEnv-v1': pick_dyn_obstacles_max,
        'FetchPickDynObstaclesMaxEnv-v1': pick_dyn_obstacles,
        'FetchPickDynLabyrinthEnv-v1': pick_dyn_labyrinth,
        'FetchPickSafeRiskyLaneEnv-v1' : pick_safe_risky_lane,
    }

    return a[args.env].generate_pathplanner(create=args.mpc_gen, path=args.mpc_path, n_substep=args.env_n_substeps)
