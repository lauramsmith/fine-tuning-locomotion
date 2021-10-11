# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
from motion_imitation.envs import locomotion_gym_config
from motion_imitation.envs import locomotion_gym_env
from motion_imitation.envs.env_wrappers import imitation_task
from motion_imitation.envs.env_wrappers import imitation_wrapper_env
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper
from motion_imitation.envs.env_wrappers import reset_task
from motion_imitation.envs.env_wrappers import simple_openloop
from motion_imitation.envs.env_wrappers import trajectory_generator_wrapper_env
from motion_imitation.envs.sensors import environment_sensors
from motion_imitation.envs.sensors import robot_sensors
from motion_imitation.envs.sensors import sensor_wrappers
from motion_imitation.envs.utilities import controllable_env_randomizer_from_config
from motion_imitation.robots import a1
from motion_imitation.robots import a1_robot
from motion_imitation.robots import robot_config


def build_env(task,
              motion_files=None,
              num_parallel_envs=0,
              mode="train",
              enable_randomizer=True,
              enable_rendering=False,
              reset_at_current_position=False,
              use_real_robot=False,
              realistic_sim=False):
  assert len(motion_files) > 0

  if task == "reset":
    curriculum_episode_length_start = curriculum_episode_length_end = 150
  else:
    curriculum_episode_length_start = 50
    curriculum_episode_length_end = 600

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.allow_knee_contact = True
  sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION
  sim_params.reset_at_current_position = reset_at_current_position
  sim_params.num_action_repeat = 33

  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  robot_kwargs = {"self_collision_enabled": True}
  ref_state_init_prob = 0.0

  if use_real_robot:
    robot_class = a1_robot.A1Robot
  else:
    robot_class = a1.A1

  if use_real_robot or realistic_sim:
    robot_kwargs["reset_func_name"] = "_SafeJointsReset"
    robot_kwargs["velocity_source"] = a1.VelocitySource.IMU_FOOT_CONTACT
  else:
    robot_kwargs["reset_func_name"] = "_PybulletReset"
  num_motors = a1.NUM_MOTORS
  traj_gen = simple_openloop.A1PoseOffsetGenerator(
      action_limit=np.array([0.802851455917, 4.18879020479, -0.916297857297] *
                            4) - np.array([0, 0.9, -1.8] * 4))

  sensors = [
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=num_motors), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=environment_sensors.LastActionSensor(num_actions=num_motors), num_history=3)
  ]

  if task == "reset":
    task = reset_task.ResetTask()
  else:
    task = imitation_task.ImitationTask(
        ref_motion_filenames=motion_files,
        real_robot=use_real_robot,
        enable_cycle_sync=True,
        tar_frame_steps=[1, 2, 10, 30],
        ref_state_init_prob=ref_state_init_prob,
        enable_rand_init_time=enable_randomizer,
        warmup_time=.3)

  randomizers = []
  if enable_randomizer:
    randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
    randomizers.append(randomizer)

  env = locomotion_gym_env.LocomotionGymEnv(
      gym_config=gym_config,
      robot_class=robot_class,
      robot_kwargs=robot_kwargs,
      env_randomizers=randomizers,
      robot_sensors=sensors,
      task=task)

  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env, trajectory_generator=traj_gen)

  if mode == "test":
    curriculum_episode_length_start = curriculum_episode_length_end

  env = imitation_wrapper_env.ImitationWrapperEnv(env,
                                                  episode_length_start=curriculum_episode_length_start,
                                                  episode_length_end=curriculum_episode_length_end,
                                                  curriculum_steps=2000000,
                                                  num_parallel_envs=num_parallel_envs)
  return env
