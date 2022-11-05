import numpy as np
import tensorflow as tf
from env_ext import goal_distance_obs
from utils2.tf_utils import get_vars, Normalizer
from algorithm.replay_buffer import goal_based_process
import os


class DDPG:

    def __init__(self, args):
        self.graph = None
        self.args = args
        self.saver: object = None
        if self.args.model_path:
            self.model_path = args.model_path
            self.model_epoch = args.model_epoch
            self.load_model()
            print(f"Loaded Model {self.model_path}")

        else:
            self.create_model()

        self.train_info_pi = {
            'Pi_q_loss': self.pi_q_loss,
            'Pi_l2_loss': self.pi_l2_loss
        }
        self.train_info_q = {
            'Q_loss': self.q_loss
        }
        self.train_info = {**self.train_info_pi, **self.train_info_q}

        self.step_info = {
            'Q_average': self.q_pi
        }

    def load_model(self):
        def load_session():
            self.meta_path = os.path.join(self.model_path, "saved_policy-{}.meta".format(self.model_epoch))
            self.checkpoint_path = os.path.join(self.model_path, "saved_policy-{}".format(self.model_epoch))
            self.sess = tf.compat.v1.Session()
            self.saver = tf.compat.v1.train.import_meta_graph(self.meta_path)
            self.saver.restore(self.sess, self.checkpoint_path)

        def load_input():
            self.raw_obs_ph = self.graph.get_tensor_by_name("raw_obs_ph:0")
            self.raw_obs_next_ph = self.graph.get_tensor_by_name("raw_obs_next_ph:0")
            self.acts_ph = self.graph.get_tensor_by_name("acts_ph:0")
            self.rews_ph = self.graph.get_tensor_by_name("rews_ph:0")

        def load_normalizer():
            self.obs_normalizer = Normalizer(self.args.obs_dims, self.sess)
            self.obs_normalizer.sum = self.graph.get_tensor_by_name("normalizer/normalizer_variables/sum:0")
            self.obs_normalizer.sum_sqr = self.graph.get_tensor_by_name("normalizer/normalizer_variables/sum_sqr:0")
            self.obs_normalizer.cnt = self.graph.get_tensor_by_name("normalizer/normalizer_variables/cnt:0")
            self.obs_normalizer.mean = self.graph.get_tensor_by_name("normalizer/normalizer_variables/mean:0")
            self.obs_normalizer.std = self.graph.get_tensor_by_name("normalizer/normalizer_variables/std:0")
            self.obs_normalizer.add_sum = self.graph.get_tensor_by_name("normalizer/Placeholder:0")
            self.obs_normalizer.add_sum_sqr = self.graph.get_tensor_by_name("normalizer/Placeholder_1:0")
            self.obs_normalizer.add_cnt = self.graph.get_tensor_by_name("normalizer/Placeholder_2:0")
            self.obs_normalizer.update_array_op = self.graph.get_operation_by_name("normalizer/group_deps")
            self.obs_normalizer.update_scalar_op = self.graph.get_operation_by_name("normalizer/group_deps_1")

            self.obs_ph = self.graph.get_tensor_by_name("clip_by_value:0")
            self.obs_next_ph = self.graph.get_tensor_by_name("clip_by_value_1:0")

        def load_network():
            self.pi = self.graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")
            self.q = self.graph.get_tensor_by_name("main/value/net/q/BiasAdd:0")
            self.q_pi = self.graph.get_tensor_by_name("main/value_1/net/q/BiasAdd:0")
            self.pi_t = self.graph.get_tensor_by_name("target/policy/net/pi/Tanh:0")
            self.q_t = self.graph.get_tensor_by_name("target/value/net/q/BiasAdd:0")

            self.pi_q_loss = self.graph.get_tensor_by_name("Neg:0")
            self.pi_l2_loss = self.graph.get_tensor_by_name("mul:0")
            self.pi_optimizer = tf.compat.v1.train.AdamOptimizer(self.args.pi_lr)
            self.pi_train_op = self.graph.get_operation_by_name("Adam")

            self.q_loss = self.graph.get_tensor_by_name("Mean_2:0")
            self.q_optimizer = tf.compat.v1.train.AdamOptimizer(self.args.q_lr)
            self.q_train_op = self.graph.get_operation_by_name("Adam_1")
            self.target_update_op = self.graph.get_operation_by_name("group_deps")

        self.graph = tf.Graph()
        with self.graph.as_default():
            load_session()
            load_input()
            load_normalizer()
            load_network()

    def create_model(self):
        def create_session():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=config)

        def create_inputs():
            self.raw_obs_ph = tf.compat.v1.placeholder(tf.float32, [None] + self.args.obs_dims, name='raw_obs_ph')
            self.raw_obs_next_ph = tf.compat.v1.placeholder(tf.float32, [None] + self.args.obs_dims,
                                                            name='raw_obs_next_ph')
            self.acts_ph = tf.compat.v1.placeholder(tf.float32, [None] + self.args.acts_dims, name='acts_ph')
            # self.rews_ph = tf.placeholder(tf.float32, [None, 1], name='rews_ph')
            self.rews_ph = tf.compat.v1.placeholder(tf.float32, [None, self.args.reward_dims], name='rews_ph')

        def create_normalizer():
            with tf.compat.v1.variable_scope('normalizer'):
                self.obs_normalizer = Normalizer(self.args.obs_dims, self.sess)
            self.obs_ph = self.obs_normalizer.normalize(self.raw_obs_ph)
            self.obs_next_ph = self.obs_normalizer.normalize(self.raw_obs_next_ph)

        def create_network():
            def mlp_policy(obs_ph):
                with tf.compat.v1.variable_scope('net',
                                                 initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                             mode="fan_avg",
                                                                                                             distribution="uniform")):
                    pi_dense1 = tf.compat.v1.layers.dense(obs_ph, 256, activation=tf.nn.relu, name='pi_dense1')
                    pi_dense2 = tf.compat.v1.layers.dense(pi_dense1, 256, activation=tf.nn.relu, name='pi_dense2')
                    pi_dense3 = tf.compat.v1.layers.dense(pi_dense2, 256, activation=tf.nn.relu, name='pi_dense3')
                    pi = tf.compat.v1.layers.dense(pi_dense3, self.args.acts_dims[0], activation=tf.nn.tanh, name='pi')
                return pi

            def mlp_value(obs_ph, acts_ph):
                state_ph = tf.concat([obs_ph, acts_ph], axis=1)
                with tf.compat.v1.variable_scope('net',
                                                 initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                             mode="fan_avg",
                                                                                                             distribution="uniform")):
                    q_dense1 = tf.compat.v1.layers.dense(state_ph, 256, activation=tf.nn.relu, name='q_dense1')
                    q_dense2 = tf.compat.v1.layers.dense(q_dense1, 256, activation=tf.nn.relu, name='q_dense2')
                    q_dense3 = tf.compat.v1.layers.dense(q_dense2, 256, activation=tf.nn.relu, name='q_dense3')
                    q = tf.compat.v1.layers.dense(q_dense3, 1, name='q')
                return q

            with tf.compat.v1.variable_scope('main'):
                with tf.compat.v1.variable_scope('policy'):
                    self.pi = mlp_policy(self.obs_ph)
                with tf.compat.v1.variable_scope('value'):
                    self.q = mlp_value(self.obs_ph, self.acts_ph)
                with tf.compat.v1.variable_scope('value', reuse=True):
                    self.q_pi = mlp_value(self.obs_ph, self.pi)

            with tf.compat.v1.variable_scope('target'):
                with tf.compat.v1.variable_scope('policy'):
                    self.pi_t = mlp_policy(self.obs_next_ph)

                with tf.compat.v1.variable_scope('value'):
                    self.q_t = mlp_value(self.obs_next_ph, self.pi_t)

        def create_operators():
            self.pi_q_loss = -tf.reduce_mean(input_tensor=self.q_pi)
            self.pi_l2_loss = self.args.act_l2 * tf.reduce_mean(input_tensor=tf.square(self.pi))
            self.pi_optimizer = tf.compat.v1.train.AdamOptimizer(self.args.pi_lr)
            self.pi_train_op = self.pi_optimizer.minimize(self.pi_q_loss + self.pi_l2_loss,
                                                          var_list=get_vars('main/policy'))

            if self.args.clip_return:
                return_value = tf.clip_by_value(self.q_t, self.args.clip_return_l, self.args.clip_return_r)
            else:
                return_value = self.q_t
            target = tf.stop_gradient(self.rews_ph + self.args.gamma * return_value)
            self.q_loss = tf.reduce_mean(input_tensor=tf.square(self.q - target))
            self.q_optimizer = tf.compat.v1.train.AdamOptimizer(self.args.q_lr)
            self.q_train_op = self.q_optimizer.minimize(self.q_loss, var_list=get_vars('main/value'))

            self.target_update_op = tf.group([
                v_t.assign(self.args.polyak * v_t + (1.0 - self.args.polyak) * v)
                for v, v_t in zip(get_vars('main'), get_vars('target'))
            ])
            self.saver = tf.compat.v1.train.Saver(max_to_keep=25)
            self.init_op = tf.compat.v1.global_variables_initializer()

            self.target_init_op = tf.group([
                v_t.assign(v)
                for v, v_t in zip(get_vars('main'), get_vars('target'))
            ])

        self.graph = tf.Graph()
        with self.graph.as_default():
            create_session()
            create_inputs()
            create_normalizer()
            create_network()
            create_operators()
        self.init_network()

    def init_network(self):
        self.sess.run(self.init_op)
        self.sess.run(self.target_init_op)

    def step(self, obs, explore=False, test_info=False):
        if (not test_info) and (self.args.buffer.steps_counter < self.args.warmup):
            return np.random.uniform(-1, 1, size=self.args.acts_dims)
        if self.args.goal_based: obs = goal_based_process(obs)

        # eps-greedy exploration
        if explore and np.random.uniform() <= self.args.eps_act:
            return np.random.uniform(-1, 1, size=self.args.acts_dims)

        feed_dict = {
            self.raw_obs_ph: [obs]
        }
        action, info = self.sess.run([self.pi, self.step_info], feed_dict)
        action = action[0]

        # uncorrelated gaussian explorarion
        if explore: action += np.random.normal(0, self.args.std_act, size=self.args.acts_dims)
        action = np.clip(action, -1, 1)

        if test_info: return action, info
        return action

    def step_batch(self, obs):
        actions = self.sess.run(self.pi, {self.raw_obs_ph: obs})
        return actions

    def feed_dict(self, batch):
        return {
            self.raw_obs_ph: batch['obs'],
            self.raw_obs_next_ph: batch['obs_next'],
            self.acts_ph: batch['acts'],
            self.rews_ph: batch['rews']
        }

    def train(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _, _ = self.sess.run([self.train_info, self.pi_train_op, self.q_train_op], feed_dict)
        return info

    def train_pi(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _ = self.sess.run([self.train_info_pi, self.pi_train_op], feed_dict)
        return info

    def train_q(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _ = self.sess.run([self.train_info_q, self.q_train_op], feed_dict)
        return info

    def normalizer_update(self, batch):
        self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))

    def target_update(self):
        self.sess.run(self.target_update_op)

    def save(self, filename, global_step=None):  # we just use the
        self.saver.save(self.sess, filename, global_step=global_step)
