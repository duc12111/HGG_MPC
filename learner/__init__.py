from .normal import NormalLearner
from .hgg import HGGLearner
from .ghgg import GHGGLearner
from .hgg_mpc import HGGMPCLearner

learner_collection = {
    'normal': NormalLearner,
    'hgg': HGGLearner,
    'ghgg': GHGGLearner,
    'hggmpc': HGGMPCLearner
}


def create_learner(args):
    return learner_collection[args.learn](args)
