from .col_test import ColTestGoalEnv
from .ghgg_custom import GHGGCustomGoalEnv
from .vanilla import VanillaGoalEnv
from .fixobj import FixedObjectGoalEnv
from .interval import IntervalGoalEnv
from .mpc_control import MPCControlGoalEnv
from .mpc_no_collision import MPCControlGoalNoCollisionEnv


def make_env(args):
    return {
        'vanilla': VanillaGoalEnv,
        'fixobj': FixedObjectGoalEnv,
        'interval': IntervalGoalEnv,
        'mpc': MPCControlGoalEnv,
        'col_test': ColTestGoalEnv,
        'ghgg_custom': GHGGCustomGoalEnv,
        'mpc_no_collision': MPCControlGoalNoCollisionEnv
    }[args.goal](args)
