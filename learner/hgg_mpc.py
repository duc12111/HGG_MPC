import copy
import math

import numpy as np
from env_ext import make_env
from env_ext.utils import goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils2.gcc_utils import gcc_load_lib, c_double, c_int
from learner.hgg import MatchSampler, TrajectoryPool
from policies.mpc_policy import MPCPolicy


class HGGMPCLearner:
    def __init__(self, args):
        self.args = args
        self.env = make_env(args)
        self.env_test = make_env(args)

        self.env_List = []
        for i in range(args.episodes):
            self.env_List.append(make_env(args))

        self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
        self.sampler = MatchSampler(args, self.achieved_trajectory_pool)
        self.mpc_policy = MPCPolicy(args)
        self.mpc_policy.set_envs(envs=self.env_List)
        for env in self.env_List:
            env.disable_action_limit()

    def learn(self, args, env, env_test, agent, buffer):
        initial_goals = []
        desired_goals = []
        for i in range(args.episodes):
            obs = self.env_List[i].reset()
            goal_a = obs['achieved_goal'].copy()
            goal_d = obs['desired_goal'].copy()
            initial_goals.append(goal_a.copy())
            desired_goals.append(goal_d.copy())

        self.sampler.update(initial_goals, desired_goals)

        achieved_trajectories = []
        achieved_init_states = []
        for i in range(args.episodes):
            obs = self.env_List[i].get_obs()
            init_state = obs['observation'].copy()
            explore_goal = self.sampler.sample(i)
            self.env_List[i].goal = explore_goal.copy()
            obs = self.env_List[i].get_obs()
            current = Trajectory(obs)
            trajectory = [obs['achieved_goal'].copy()]
            self.mpc_policy.reset()
            for timestep in range(args.timesteps):
                rl_action = agent.step(obs, explore=True)

                sub_goal = self.env.subgoal(rl_action)
                self.mpc_policy.set_sub_goals([sub_goal])
                mpc_actions, _ = self.mpc_policy.predict(obs=[obs])
                mpc_actions[0][3] = rl_action[3]
                assert len(mpc_actions) == 1
                final_action = mpc_actions[0]
                obs, reward, done, info = self.env_List[i].step(final_action)

                trajectory.append(obs['achieved_goal'].copy())
                if timestep == args.timesteps - 1: done = True
                current.store_step(rl_action, obs, reward, done)
                if done: break
            achieved_trajectories.append(np.array(trajectory))
            achieved_init_states.append(init_state)
            buffer.store_trajectory(current)
            agent.normalizer_update(buffer.sample_batch())

            if buffer.steps_counter >= args.warmup:
                for _ in range(args.train_batches):
                    info = agent.train(buffer.sample_batch())
                    args.logger.add_dict(info)
                agent.target_update()

        selection_trajectory_idx = {}
        for i in range(self.args.episodes):
            if goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1]) > 0.01:
                selection_trajectory_idx[i] = True
        for idx in selection_trajectory_idx.keys():
            self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())
