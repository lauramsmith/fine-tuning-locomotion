"""For resetting the robot to a ready pose."""

import inspect
import os

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from motion_imitation.envs.env_wrappers import reset_task
from motion_imitation.robots import robot_config
from sac_dev.learning import sac_agent
import sac_dev.sac_configs
import tensorflow as tf


class RewardPlateau(object):
  """Terminal condition that returns True when the reward stops changing."""

  def __init__(self, n=5, delta=.001, lowbar=0.8):
    self.n = n
    self.delta = delta
    self.last_reward = float("inf")
    self.count_same = 0
    self.lowbar = lowbar

  def __call__(self, env):
    reward = env._task.reward(env)
    if reward > self.lowbar and abs(reward - self.last_reward) < self.delta:
      self.count_same += 1
    else:
      self.count_same = 0
    self.last_reward = reward
    return self.count_same >= self.n


class GetupResetter(object):
  """Single-policy resetter that first rolls over, then stands up."""

  def __init__(self, env, checkpoint_path):
    self._env = env
    timeout = lambda env: env.env_step_counter > 150
    upright = lambda env: env.task.reward(env) > .94
    # Here real_robot just means no starting pose randomization.
    self._reset_task = reset_task.ResetTask(
        terminal_conditions=(upright, timeout, RewardPlateau()),
        real_robot=True)
    old_task = self._env.task
    self._env.set_task(self._reset_task)

    self._graph = tf.Graph()
    self._sess = tf.Session(graph=self._graph)

    agent_configs = sac_dev.sac_configs.SAC_CONFIGS["A1-Motion-Imitation-Vanilla-SAC-Pretrain"]

    self._reset_model = sac_agent.SACAgent(
        env=self._env, sess=self._sess, **agent_configs)
    self._reset_model.load_model(checkpoint_path)
    self._env.set_task(old_task)

  def _run_single_episode(self, task, policy):
    self._env.set_task(task)
    obs = self._env.reset()
    done = False
    self._env.robot.running_reset_policy = True
    while not done:
      action = policy(obs)
      obs, _, done, _ = self._env.step(action)
    self._env.robot.running_reset_policy = False

  def __call__(self):
    for i in range(1, 6):
      print("Reset attempt {}/5".format(i))
      try:
        self._run_single_episode(
            self._reset_task,
            lambda x: self._reset_model.sample_action(x, True)[0])
        break
      except robot_config.SafetyError:
        continue
    self._env.robot.HoldCurrentPose()
