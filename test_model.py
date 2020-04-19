from __future__ import absolute_import, division, print_function, unicode_literals

from dm_control import mujoco
from dm_control import suite
from dm_control import viewer
from dm_control.suite import common

import numpy as np
import math as m
from math import pi, sqrt, sin ,cos
import pandas as pd
import random as r
import os
import sys
import subprocess
import matplotlib.pyplot as plt
import pyquaternion as pq
from timeit import default_timer as timer
from PIL import Image
import xml.etree.ElementTree as ET
import glob
import time

import tensorflow as tf
from tensorflow.keras.models import load_model as load
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from IPython import embed

suite_dir = "".join(suite.__path__)

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

global L

test_data_loc = 'random_ms'

# Either load a compiled model or just the weights of a model
# (if the model was trained with a custom loss, only able to use weights)
load_weights_only = True

load_model_loc = 'model_.hdf5'
load_weights_loc = 'model_weights.h5'

_DOF = 2
_MAX_SLIDERS = 4		# number of sliders in the rendering xml file (per state)
_N_SLIDERS = 4			# number of sliders the network takes/outputs
_VARS = 6
_CTRL_DUR = 1

N_QPOS = _DOF + 7*_MAX_SLIDERS
N_QVEL = _DOF + 6*_MAX_SLIDERS

cols_in = _DOF + _N_SLIDERS*_VARS
cols_out = _N_SLIDERS*_VARS

use_customloss = True
use_weighted_loss = True

max_penetration = 5e-4
penalizing_factor = 2

rs = 0.05115
rp = 0.0145

render = True
show_error = True

sample_size = 1000
batch_size = 1

#############################################################################################################

def data_import(test_data_loc):
	''' This function imports csv datafiles for testing, and crops their size accordingly.

	Args:
		test_data_loc: 	Prefix to name of test data files.

	Returns:
		x: 				Test input data.
		y:				Test output labels.
	'''

	# Load in all testing data
	test_inputs = pd.read_csv('%s_initial.csv' % test_data_loc, sep=',', header=None)
	x = np.array(test_inputs)
	test_labels = pd.read_csv('%s_final.csv' % test_data_loc, sep=',', header=None)
	y = np.array(test_labels)

	x = x[:,:cols_in]
	y = y[:,:cols_out]

	L = np.shape(x)[0]

	ind = r.sample(range(L), sample_size)

	x = x[ind,:]
	y = y[ind,:]

	return x, y

def reset(state, physics):
	''' This function resets the physics of an environment and implements a new state.

	Args:
		state:		New physics state to set.
		physics:	Physics object to reset.

	'''

	with physics.reset_context():
		physics.set_state(state)
		physics.step()

def physics_from_xml(xml_location):
	''' This function takes an xml file and creates a physics instance.

	Args:
		xml_location:	Location of the xml environment file to use.

	Returns:
		root:			xml root that can be edited if necessary.
		physics:		Physics object created form xml.
	'''

	# Load in .xml file and get root
	tree = ET.parse(xml_location)
	root = tree.getroot()
	
	# Create string from updated .xml file and create physics environment
	xml_string = ET.tostring(root, encoding='unicode')
	physics = mujoco.Physics.from_xml_string(xml_string)

	return root, physics

def renderer(indices, folder_prefix='', empty_folder=True):
	''' This function renders the predicted state (red) against the correct state (green) for comparison.

	Args:
		indices:			Indices of the test dataset to render from.
		folder_prefix:		Prefix of the folder to save renders to.
		empty_folder:		Decide whether or not to empty the folder.

	'''

	if render:

		# If folder exists, and we would like to empty it ...
		if os.path.isdir('%stest_model_renders' % folder_prefix) and empty_folder:
			
			# ...remove directory...
			subprocess.call(["rm", "-r", '%stest_model_renders' % folder_prefix])
			
			# ...and create empty directory
			os.mkdir('%stest_model_renders' % folder_prefix)

		# If it doesn't exist...
		elif not os.path.isdir('%stest_model_renders' % folder_prefix):
			
			# ...create empty directory
			os.mkdir('%stest_model_renders' % folder_prefix)

		# Load in .xml file and get physics object
		_, physics = physics_from_xml('%s/multislider_render.xml' % suite_dir)

		for i in indices:

			state_im = '%stest_model_renders/#%d.png' % (folder_prefix, int(i+1))

			# Set input state and render
			length = 164
			state = np.zeros((length,))

			state[0:_DOF] = np.array([0, 0])						# initial state pusher pos
			state[N_QPOS:N_QPOS+_DOF] = np.array([x[i,0],x[i,1]])	# final state pusher pos
		
			for m in range(_MAX_SLIDERS):

				if m == 0:	
					state[_DOF:_DOF+7] = np.array([x[i,_DOF], 0, 0.00005, 1, 0, 0, 0])
					state[N_QPOS+_DOF:N_QPOS+_DOF+7] = np.array([x[i,_DOF]+y[i,0], y[i,1], 0.00015, 1, 0, 0, 0])
					state[2*N_QPOS:2*N_QPOS+7] = np.array([x[i,_DOF]+p[i,0], p[i,1], 0.00025, 1, 0, 0, 0])
				elif m < _N_SLIDERS:
					state[_DOF+7*m:_DOF+7*(m+1)] = np.array([x[i,_DOF+_VARS*m-1], x[i,_DOF+_VARS*m], 0.00005, 1, 0, 0, 0])
					state[N_QPOS+_DOF+7*m:N_QPOS+_DOF+7*(m+1)] = np.array([x[i,_DOF+_VARS*m-1]+y[i,_VARS*m], x[i,_DOF+_VARS*m]+y[i,_VARS*m+1], 0.00015, 1, 0, 0, 0])
					state[2*N_QPOS+7*m:2*N_QPOS+7*(m+1)] = np.array([x[i,_DOF+_VARS*m-1]+p[i,_VARS*m], x[i,_DOF+_VARS*m]+p[i,_VARS*m+1], 0.00025, 1, 0, 0, 0])
				else:
					state[_DOF+7*m:_DOF+7*(m+1)] = np.array([m, m, 0.00005, 1, 0, 0, 0])
					state[N_QPOS+_DOF+7*m:N_QPOS+_DOF+7*(m+1)] = np.array([m+1, m+1, 0.00015, 1, 0, 0, 0])
					state[2*N_QPOS+7*m:2*N_QPOS+7*(m+1)] = np.array([m+2, m+2, 0.00025, 1, 0, 0, 0])

			reset(state, physics)

			image_array = physics.render(height=480, width=600, camera_id='fixed')
			img = Image.fromarray(image_array, 'RGB')
			img.save(state_im)

def get_model(model_location, load_weights_only=True):
	''' This function gets a trained, compiled model either from a model file or a weights file.

	Args:
		model_location: 	Location of the model or weights to load.
		load_weights_only:	Decide whether to load compiled model or just weights.

	Returns:
		model:				Compiled model.
	'''

	if load_weights_only:

		# Network architecture - must be the exact same as the model's which you are loading in
		inputs = layers.Input(shape=(cols_in,))
		layer1 = layers.Dense(512, use_bias=True, activation='relu')(inputs)
		layer2 = layers.Dense(256, use_bias=True, activation='relu')(layer1)
		layer3 = layers.Dense(128, use_bias=True, activation='relu')(layer2)
		layer4 = layers.Dense(64, use_bias=True, activation='relu')(layer3)
		outputs = layers.Dense(cols_out, activation='linear')(layer4)
		model = keras.Model(inputs=inputs, outputs = outputs)
		
		# Set loss weights
		if use_weighted_loss:
			''' Weighs them depending on the rms average of the labels'''

			rmsl = np.zeros((cols_out,))
			weights = []

			for c in range(cols_out):
				rmsl[c] = sqrt(np.mean(np.square(y[:,c])))

			max_rmsl = max(rmsl)

			for c in range(cols_out):
				weights.append(max_rmsl/rmsl[c])

			weights = [weights]
		else:
			''' Sets all loss weights to 1 '''
			
			weights = []

			for c in range(cols_out):
				weights.append(1.)

			weights = [weights]

		# Set optimizer
		optimizer = keras.optimizers.Nadam(lr=0.0001, decay=0)
		
		# Compile
		if use_customloss:
			model.compile(optimizer=optimizer, loss=customloss_wrapper(inputs, weights), metrics=['accuracy'])
		else:
			model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

		# Load in weights of a trained model
		model.load_weights(model_location)

	else:

		# Load in trained & compiled model
		model = load(model_location)

	# Initialize predict function to reduce computation time
	start = timer()
	_ = model.predict(np.zeros((2,cols_in)))
	init_t = 1000*( timer() - start )

	return model, init_t

def get_predictions(model, x, y):
	''' This function gets predictions from a model and a test dataset and computes error.

	Args:
		model:		Model to test.
		x:			Input test data.
		y:			Output test labels.

	Returns:
		p:			Predictions from model.
		e:			Prediction error (p-y)
	'''

	# Number of batches in test data
	n_batches = int(sample_size/batch_size)

	# Initialize arrays with zeros
	nn_t = np.zeros((n_batches,))
	p = np.zeros((sample_size,cols_out))
	e = np.zeros((sample_size,cols_out))

	# Make predictions and time
	for batch in range(n_batches):

		i = batch*batch_size

		start = timer()

		p[i:i+batch_size,:] = model.predict(x[i:i+batch_size,:])

		if show_error:
			e[i:i+batch_size,:] = p[i:i+batch_size,:]-y[i:i+batch_size,:]

		nn_t[batch] = 1000*(timer()-start)

		if batch==0:
			print(' ')
			print('Sample size: 		%d' % sample_size)
			print('Batch size:		%d' % batch_size)
			print(' ')
			print('			Time (ms)')
			print('Initialization:		%.3f' % init_t)
			print('First batch: 		%.3f' % nn_t[0])


	if n_batches > 1:

		avg = nn_t[1:].mean()
		print('Average thereafter: 	%.3f' % avg)

	return p, e

def plot_error(e, y):
	''' This function uses the error and labels data to plot average error across the different variables.

	Args:
		e:		Prediction error array.
		y:		Output labels.

	'''
	
	if show_error:

		boolean_ind = np.zeros((sample_size,cols_out), dtype=bool)

		for i in range(sample_size):

			boolean_ind[i,0:_VARS] = 1

			for m in range(1, _N_SLIDERS):
				if not x[i,_DOF+_VARS*m-1].is_integer():
					boolean_ind[i,m*_VARS:(m+1)*_VARS] = 1
		
		useful_e = e[boolean_ind]
		useful_y = y[boolean_ind]
		
		useful_e = np.reshape(useful_e, (int(len(useful_e)/_VARS),_VARS))
		useful_y = np.reshape(useful_y, (int(len(useful_y)/_VARS),_VARS))

		rms_error = np.zeros((_VARS,))
		rms_label = np.zeros((_VARS,))

		for c in range(_VARS):
			rms_error[c] = sqrt( np.mean(useful_e[:,c]**2) )
			rms_label[c] = sqrt( np.mean(useful_y[:,c]**2) )

		print()
		print('RMSE')
		print('dx: %f 	(vs. %f)' % (rms_error[0], rms_label[0]))
		print('dy: %f 	(vs. %f)' % (rms_error[1], rms_label[1]))
		print('d0: %f 	(vs. %f)' % (rms_error[2], rms_label[2]))
		print('vx: %f 	(vs. %f)' % (rms_error[3], rms_label[3]))
		print('vy: %f 	(vs. %f)' % (rms_error[4], rms_label[4]))
		print('v0: %f 	(vs. %f)' % (rms_error[5], rms_label[5]))	

		i = np.zeros_like(useful_e)

		for c in range(_VARS):

			i[:,c] += c

			plt.scatter(i[:,c],useful_e[:,c],s=5)

		
		plt.title('RMS Error per variable')
		plt.legend(['X', 'Y', 'theta', 'vx', 'vy', 'vtheta'], ncol=2, loc='best')
		plt.show(block=False)
		plt.pause(15)
		plt.close()

def customloss_wrapper(inputs, loss_weights):

	def customloss(y_true, y_pred):

		losses = tf.reduce_mean(tf.math.squared_difference(y_true,y_pred))

		loss = tf.reduce_mean(tf.multiply(loss_weights,losses))/tf.math.reduce_sum(loss_weights)

		for m in range(_N_SLIDERS):

			if m == 0:
				pusher_penetration = tf.math.sqrt( tf.math.square((inputs[:,_DOF]+y_pred[:,0])-inputs[:,0]*_CTRL_DUR) + tf.math.square(y_pred[:,1]-inputs[:,1]*_CTRL_DUR) ) - (rs + rp)
			else:
				pusher_penetration = tf.math.sqrt( tf.math.square((inputs[:,_DOF+_VARS*m-1]+y_pred[:,_VARS*m])-inputs[:,0]*_CTRL_DUR) + tf.math.square(y_pred[:,_VARS*m+1]-inputs[:,1]*_CTRL_DUR) ) - (rs + rp)

			zero = tf.zeros_like(pusher_penetration)

			pusher_penetration = tf.where(pusher_penetration<(zero-max_penetration), pusher_penetration, zero)

			loss += tf.math.square(tf.reduce_mean(pusher_penetration)*penalizing_factor)

			
			slider_penetration = 0

			for i in range(m+1, _N_SLIDERS):
				
				if m == 0:
					penetration = tf.math.sqrt( tf.math.square( (inputs[:,_DOF]+y_pred[:,0]) - (inputs[:,_DOF+_VARS*i-1]+y_pred[:,_VARS*i]) ) + tf.math.square( y_pred[:,1] - (inputs[:,_DOF+_VARS*i]+y_pred[:,_VARS*i+1]) ) ) - 2*rs
				else:
					penetration = tf.math.sqrt( tf.math.square( (inputs[:,_DOF+_VARS*m-1]+y_pred[:,_VARS*m]) - (inputs[:,_DOF+_VARS*i-1]+y_pred[:,_VARS*i]) ) + tf.math.square( (inputs[:,_DOF+_VARS*m]+y_pred[:,_VARS*m+1]) - (inputs[:,_DOF+_VARS*i]+y_pred[:,_VARS*i+1]) ) ) - 2*rs

				zero = tf.zeros_like(penetration)

				penetration = tf.where(penetration<(zero-max_penetration), penetration, zero)

				slider_penetration += tf.math.square(tf.reduce_mean(penetration)*penalizing_factor)

			slider_penetration = K.print_tensor(slider_penetration)

			loss += slider_penetration

		return loss

	return customloss

#####################################################################################################

if __name__ == '__main__':

	x, y = data_import(test_data_loc)

	L = np.shape(x)[0]

	if load_weights_only:
		model_loc = load_weights_loc
	else:
		model_loc = load_model_loc

	model, init_t = get_model(model_loc)

	p, e = get_predictions(model, x, y)

	# Check for problematic and perfect x & y position error and get their indices
	problematic = []
	perfect = []

	for l in range(sample_size):

		for m in range(_N_SLIDERS):
			
			if abs(e[l,0]) < .1*rs and abs(e[l,1]) < .1*rs:
				if abs(e[l,6]) < .1*rs and abs(e[l,7]) < .1*rs:
					if abs(e[l,12]) < .1*rs and abs(e[l,13]) < .1*rs:
						if abs(e[l,18]) < .1*rs and abs(e[l,19]) < .1*rs:
							perfect.append(l)

			if abs(e[l,_VARS*m]) > .6*rs:
				problematic.append(l)

			elif abs(e[l,_VARS*m+1]) > .6*rs:
				problematic.append(l)


	# Create renders of initial and final states 
	#renderer(problematic, folder_prefix='bad_')
	#renderer(perfect, folder_prefix='good_')
	#renderer(r.sample(range(sample_size), 100))

	# Error plots and calculations
	plot_error(e,y)




