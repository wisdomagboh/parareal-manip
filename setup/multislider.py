"""multislider Domain"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# multislider.py

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards

import numpy as np

_DEFAULT_TIME_LIMIT = 1
_CONTROL_TIMESTEP = .002

SUITE = containers.TaggedTasks()

def get_model_and_assets():
	"""Returns a tuple containing the model XML string and a dict of assets."""
	return common.read_model('multislider.xml'), common.ASSETS

@SUITE.add()
def collide(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Returns the Collide task."""
	physics = Physics.from_xml_string(*get_model_and_assets())
	task = Simple()
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)

class Physics(mujoco.Physics):
	"""Physics with additional features for the Simple domain."""

class Simple(base.Task):

	def __init__(self, random=None):
		"""Initializes an instance of Simple."""
		super(Simple, self).__init__(random=random)

	def initialize_episode(self, physics):
		"""Sets the state of the environment at the start of each episode."""
		pass
  
	def get_observation(self, physics):
		"""Returns either the pure state or a set of egocentric features."""
		obs = collections.OrderedDict()
		return obs

	def get_reward(self, physics):
		"""Returns a reward to the agent."""
		return 0





