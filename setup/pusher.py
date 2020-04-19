# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Pusher domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Internal dependencies.

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards

import numpy as np

SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 20e10
_BIG_TARGET = .01
_SMALL_TARGET = .01


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('pusher.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns pusher with sparse reward with 5e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Pusher(target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns pusher with sparse reward with 1e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Pusher(target_size=_SMALL_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Pusher domain."""

  def object_to_target(self):
      """Returns the vector from object to goal region"""
      return (self.named.data.geom_xpos['target', :2] -
              self.named.data.geom_xpos['goal_object', :2])

  def object_to_target_dist(self):
      """Returns the distance between the object and the goal region"""
      return np.linalg.norm(self.object_to_target())

class Stoch_Physics(mujoco.Physics):
  """Physics simulation with additional features for the Pusher domain."""

  def object_to_target(self):
      """Returns the vector from object to goal region"""
      return (self.named.data.geom_xpos['target', :2] -
              self.named.data.geom_xpos['goal_object', :2])

  def object_to_target_dist(self):
      """Returns the distance between the object and the goal region"""
      return np.linalg.norm(self.object_to_target())

  def stoch_step(self):
      return self.step


class Pusher(base.Task):
  """A pushing `Task` to reach the target."""

  def __init__(self, target_size, random=None):
    """Initialize an instance of `Pusher`.
    Args:
      target_size: A `float`, tolerance to determine whether finger reached the
          target.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._target_size = target_size
    super(Pusher, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    physics.named.model.geom_size['target', 0] = self._target_size
    #randomizers.randomize_limited_and_rotational_joints(physics, self.random)

    # randomize target position
    angle = self.random.uniform(0, 2 * np.pi)
    radius = self.random.uniform(.05, .20)
    physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
    physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)

  def get_observation(self, physics):
    """Returns an observation of the state and the target position."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['to_target'] = physics.object_to_target()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    radii = physics.named.model.geom_size[['target', 'goal_object'], 0].sum()
    return rewards.tolerance(physics.object_to_target_dist(), (0, radii))
