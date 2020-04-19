import warnings
warnings.filterwarnings("ignore")

import subprocess
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math as m
import xml.etree.ElementTree as ET
import os
import timeit
import multiprocessing as mp
import time
import IPython
import tensorflow as tf
from numpy import mean, sqrt, square, arange
from math import pi, sqrt, sin, cos, atan2

from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import backend as K

from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
from pyquaternion import Quaternion
from PIL import Image

from dm_control import mujoco
from dm_control import suite
from dm_control.suite import pusher

from config import *

# sample call python3.5 parareal_manipulation.py --learned_and_analytical True --n_actions 4 --chosen_objs 1 --number_of_samples 10 --num_cores 4

# Load environment for robotic pushing
env = suite.load(domain_name="pusher", task_name="easy")
physics = env.physics

# Set physics simulator parameters
physics.model.opt.timestep = SIM_TIMESTEP
physics.model.opt.integrator = SIM_INTEGRATOR

parser = argparse.ArgumentParser(description='Parareal for Robotic Manipulation')
parser.add_argument('--learned_and_analytical', action='store', default=True)
parser.add_argument('--n_actions', action="store", default=4)
parser.add_argument('--num_cores', action="store", default=4)
parser.add_argument('--chosen_objs', action="store", default=1)
parser.add_argument('--number_of_samples', action="store", default=10)

args = parser.parse_args()

if args.learned_and_analytical == "True":		  	# To use both learned and analytical during experiments (valid only for 1 slider)
	learned_and_analytical = True
else:
	learned_and_analytical = False

number_of_samples = int(args.number_of_samples)   	# Number of random samples for experiment
chosen_objs = [int(args.chosen_objs)]				# Number of objects in a scene 1 - 4
n_actions = int(args.n_actions)						# Number of actions in a control sequence (tested for 4 and 8)
num_cores = int(args.num_cores)						# Number of cores for parallel processing

num_time_slices = n_actions
max_parareal_iter = n_actions + 1

subprocess.call(["mkdir", "-p", "exp_dataset"])
main_data_path = "./exp_dataset/{}slider_{}/".format(chosen_objs[0], n_actions)

# Q holds the parareal update for each iteration and time slice
Q = np.zeros((max_parareal_iter+1, num_time_slices+1, N_QPOS+N_QVEL))
control_sequence = np.zeros((n_actions,2)) 			# Initialization of control sequence

def initializer():
	global physics

def coarse_int_analytical(x_0, u):
	"""Predicts the next state with the analytical model for a single slider.

	Args:
		x_0: 	Initial state.
		u:   	An action.

	Returns:
		x_next: The resulting state
	"""

	def get_slider_ang_vel(v_pusher, moment_arm):

		denominator = np.linalg.norm(v_pusher)*np.linalg.norm(moment_arm)

		try:
			theta_inv = np.dot(v_pusher,moment_arm)/denominator
		except:
			# denominator is zero
			theta_inv = 0.

		if abs(theta_inv) > 1.0:
			theta_inv = 1.0*np.sign(theta_inv)

		theta = m.acos(theta_inv)

		v_tangential = v_pusher*m.sin(theta)

		if abs(moment_arm[1]) > 0:
			solu_0 = -v_tangential[0]/moment_arm[1]
		else:
			solu_0 = 0.
		if abs(moment_arm[0]) > 0:
			solu_1 = v_tangential[1]/moment_arm[0]
		else:
			solu_1 = 0.

		solutions = np.array([solu_0, solu_1])
		max_index = np.argmax(abs(solutions))
		ang_vel_1 = -solutions[max_index]
		ang_vel = ang_vel_1*ANG_VEL_MULTIPLIER

		if abs(ang_vel) > ANG_VEL_THRESH:
			ang_vel = ANG_VEL_THRESH*np.sign(ang_vel)

		return ang_vel

	x_next = x_0.copy()
	slider_addr = DOF

	# Update pusher position
	x_next[0:DOF] = x_0[0:DOF] + u[0:DOF]*CTRL_DUR

	# Update pusher velocity
	x_next[N_QPOS:N_QPOS+DOF] = u[0:DOF].copy()

	# Update slider's linear position
	v_pusher = u[0:DOF].copy()
	percent_contact, moment_arm = get_pusher_slider_contact_vars(x_0, v_pusher)
	x_next[slider_addr:slider_addr+2] = x_0[slider_addr:slider_addr+2] + v_pusher[0:2]*CTRL_DUR*percent_contact
	x_next[slider_addr+2] = OBJECT_Z

	# Update the slider's linear velocity
	x_next[N_QPOS+slider_addr:N_QPOS+slider_addr+2] = v_pusher.copy()

	# Update slider's angular velocity
	ang_vel = get_slider_ang_vel(v_pusher, moment_arm)
	x_next[N_QPOS+slider_addr+5] = ang_vel

	# Update slider quat
	current_quat = Quaternion(x_0[slider_addr+3:slider_addr+7])
	additional_angle = ang_vel*CTRL_DUR*percent_contact*ANGLE_MULTIPLIER
	new_quat = current_quat*Quaternion(axis=[0,0,1], angle=additional_angle)
	x_next[slider_addr+3:slider_addr+7] = new_quat.elements

	return x_next

def get_pusher_slider_contact_vars(x_0, v_pusher):
	"""Calculates how much contact the pusher has with the slider and the corresponding
		moment arm from the contact point.

	Args:
		x_0:			Initial state
		v_pusher:		Pusher velocity

	Returns:
		percent_contact:	Percentage of pusher's total motion after first contact with the slider (a.k.a p_c)
		vec_moment_arm:		Moment arm calculated from first contact point to slider's center (a.k.a r_c)
	"""

	with physics.reset_context():
		physics.set_state(x_0)
		try:
			physics.step()
		except:
			#print ('Invalid state')
			percent_contact = 0.
			vec_moment_arm = np.array([1e50, 1e50])
			return percent_contact, vec_moment_arm

	slider_pos = physics.named.data.geom_xpos['goal_object'][0:2]
	init_pusher_pos = physics.named.data.geom_xpos['tool'][0:2]
	final_pusher_pos = init_pusher_pos + v_pusher[0:2]*CTRL_DUR

	pusher_vec = final_pusher_pos - init_pusher_pos
	total_pusher_dist = np.linalg.norm(pusher_vec)

	try:
		pusher_unit_vec = (pusher_vec)/total_pusher_dist
	except:
		# total pusher distance is zero
		pusher_unit_vec = np.zeros(DOF)

	# Define line followed by pusher
	delta_pos = 0.
	# Stretch the line a bit on both sides to accomodate errors (avoid faslse negatives)
	pusher_radius = PUSHER_RADIUS # Radius of the cylindrical pusher

	# Define slider's position
	pusher_line = LineString([init_pusher_pos, final_pusher_pos])
	contains_flag = False # Set to true if the pusher is inside the slider at the initial state
	# Create the pusher point
	p_pusher = Point(init_pusher_pos[0], init_pusher_pos[1])

	# Slider is a cylinder
	slider_radius = SLIDER_RADIUS + pusher_radius
	dist_between_centers = np.linalg.norm(slider_pos - init_pusher_pos)
	p_slider = Point(slider_pos[0], slider_pos[1])
	# Create the circular object
	cylindrical_slider = p_slider.buffer(slider_radius)

	# Check contains with the actual cylinder (not inflated!)
	cylindrical_slider_actual = p_slider.buffer(SLIDER_RADIUS)
	cylindrical_pusher = p_pusher.buffer(pusher_radius)

	# Check if the start state of the pusher is inside the slider
	if cylindrical_slider_actual.contains(cylindrical_pusher):
		intersection = init_pusher_pos.copy()
		contains_flag = True
	else:
		# Find intersection between circle and pusher line
		intersection = cylindrical_slider.boundary.intersection(pusher_line)
		intersection_inner = cylindrical_slider_actual.boundary.intersection(pusher_line)


	percent_contact = 0

	if contains_flag:
		percent_contact = 1.
		vec_moment_arm = slider_pos - init_pusher_pos
	elif intersection.geom_type == 'GeometryCollection' and intersection_inner.geom_type != 'GeometryCollection':
		percent_contact = 1.
		vec_moment_arm = slider_pos - init_pusher_pos
	elif intersection.geom_type == 'GeometryCollection' and intersection_inner.geom_type == 'GeometryCollection':
		# No intersection
		percent_contact = 0.
		vec_moment_arm = np.array([1e50, 1e50]) # Large numbers (infinity)
	elif intersection.geom_type == 'Point':
		# A single intersection point
		intersection_point = intersection.coords[0]
		contact_point = intersection_point
		dist_before_contact = init_pusher_pos - contact_point
		percent_contact = 1 - (np.linalg.norm(dist_before_contact) / total_pusher_dist)

		# Calculate the moment arm vector
		vec_to_center = slider_pos - intersection_point
		unit_vec_to_center = vec_to_center/np.linalg.norm(vec_to_center)
		contact_point_on_slider = intersection_point + pusher_radius*unit_vec_to_center
		vec_moment_arm =  slider_pos - contact_point_on_slider

	else:
		# Multiple intersection points - Assume only two!
		intersection_point_1 = intersection.geoms[0].coords[0]
		intersection_point_2 = intersection.geoms[1].coords[0]

		contact_point_1 = intersection_point_1
		contact_point_2 = intersection_point_2

		dist_between_intersections = np.linalg.norm(np.array(contact_point_1) - np.array(contact_point_2))
		#print ('distance_between_contact_points', dist_between_intersections)
		if dist_between_intersections < 1e-1:
			percent_contact = 0.
			vec_moment_arm = np.array([1e50, 1e50])

		else:
			dist_before_contact_1 = np.linalg.norm(init_pusher_pos - contact_point_1)
			dist_before_contact_2 = np.linalg.norm(init_pusher_pos - contact_point_2)
			distances = np.array([dist_before_contact_1, dist_before_contact_2])
			min_dist_before_contact = np.min(distances)
			percent_contact = 1 - (np.linalg.norm(min_dist_before_contact) / total_pusher_dist)
			#print ('percent contact', percent_contact)
			argmin = np.argmin(distances)
			if argmin == 0:
				intersection_point = intersection_point_1
			else:
				intersection_point = intersection_point_2

			# Calculate the moment arm vector
			vec_to_center = slider_pos - intersection_point
			unit_vec_to_center = vec_to_center/np.linalg.norm(vec_to_center)
			contact_point_on_slider = intersection_point + pusher_radius*unit_vec_to_center
			vec_moment_arm = slider_pos - contact_point_on_slider

	return percent_contact, vec_moment_arm

def coarse_int_learned(x_0, u):
	""" Uses a trained model to predict next state.

	Args:
		x_0: 	Initial state.
		u:   	An action.

	Returns:
		x_next: The resulting state
	"""

	x_next = x_0.copy()

	# Update pusher position
	x_next[0:DOF] = x_0[0:DOF] + u[0:DOF]*CTRL_DUR

	# Update pusher velocity
	x_next[N_QPOS:N_QPOS+DOF] = u[0:DOF]

	consult_nn = False

	# Check if the pusher ever comes into contact with a slider
	#   - if not, leave slider positions & velocities unchanged without consulting network
	for step in range(0,STEPS+1):

		if consult_nn == True:
			break

		for m in range(N_S):
			x_sep = x_0[DOF+7*m]-(x_0[0] + step*u[0]*CTRL_DUR/STEPS)
			y_sep = x_0[DOF+7*m+1]-(x_0[1] + step*u[1]*CTRL_DUR/STEPS)

			new_sep = sqrt(x_sep**2 + y_sep**2)

			if new_sep-(rs+rp) < -PENETRATION_EPS:
				consult_nn = True
				break

	if consult_nn:
		''' Uses an adjusted coordinate system where the new positive x-axis
		is defined from the centre of the pusher in the direction of the
		centre of the first slider (initial state). '''

		# Set input array for prediction with zeros dummy row
		inputs = np.zeros((2,DOF+VARS*NUM_OBJS-1))

		# Angle shift from world coords to adjusted coords
		da = atan2(x_0[3]-x_0[1], x_0[2]-x_0[0])

		if da < 0:
			da += 2*pi 	# now always in range 0 < da < 2pi

		Oz_0 = np.zeros((NUM_OBJS,))		# for initial slider orientations relative to world
		yaw_0 = np.zeros((NUM_OBJS,))		# for initial slider orientations in adjusted coords

		d = np.zeros((NUM_OBJS,))			# for distances between centre of sliders and pusher
		t = np.zeros((NUM_OBJS,))			# for angle between centreline, d, and adjusted coords

		for m in range(NUM_OBJS):

			# Initial orientation
			Oz_0[m], _, _ = Quaternion(x_0[DOF+7*m+3:DOF+7*m+7]).yaw_pitch_roll

			# Convert to adjusted coords
			yaw_0[m] = Oz_0[m] - da

			if yaw_0[m] < 0:
				yaw_0[m] += 2*pi

			d[m] = sqrt( (x_0[DOF+7*m]-x_0[0])**2 + (x_0[DOF+7*m+1]-x_0[1])**2 )
			t[m] = atan2(x_0[DOF+7*m+1]-x_0[1], x_0[DOF+7*m]-x_0[0]) - da

			if m == 0:		# first slider is always active
				# x,theta positions (initial ypos of slider m=0 is always 0)
				inputs[1,DOF]   = d[m]
				inputs[1,DOF+1] = yaw_0[0]
				# Velocities
				inputs[1,DOF+2] = x_0[N_QPOS+DOF]*cos(da) + x_0[N_QPOS+DOF+1]*sin(da)
				inputs[1,DOF+3] = -x_0[N_QPOS+DOF]*sin(da) + x_0[N_QPOS+DOF+1]*cos(da)
				inputs[1,DOF+4] = x_0[N_QPOS+DOF+5]

			elif m < N_S:	# check if slider m was active...
				# x,y,theta positions
				inputs[1,DOF+VARS*m-1] = d[m]*cos(t[m])
				inputs[1,DOF+VARS*m]   = d[m]*sin(t[m])
				inputs[1,DOF+VARS*m+1] = yaw_0[m]
				# Velocities
				inputs[1,DOF+VARS*m+2] = x_0[N_QPOS+DOF+6*m]*cos(da) + x_0[N_QPOS+DOF+6*m+1]*sin(da)
				inputs[1,DOF+VARS*m+3] = -x_0[N_QPOS+DOF+6*m]*sin(da) + x_0[N_QPOS+DOF+6*m+1]*cos(da)
				inputs[1,DOF+VARS*m+4] = x_0[N_QPOS+DOF+6*m+5]

			else:			# ...if not, then set the values which the network was trained with, leaving others as zero
				inputs[1,DOF+VARS*m-1]  = m+1
				inputs[1,DOF+VARS*m]    = m+1

			inputs[1,0] = u[0]*cos(da) + u[1]*sin(da)
			inputs[1,1] = -u[0]*sin(da) + u[1]*cos(da)

		# Make prediction
		p = model.predict(inputs)[1,:]

		Oz_1 = np.zeros((NUM_OBJS,))	# for new orientation of sliders in world coords

		for m in range(NUM_OBJS):

			# Distance moved by slider m
			r = sqrt(p[VARS*m]**2 + p[VARS*m+1]**2)
			# Direction it moved in relative to world x-axis
			alpha = da + atan2(p[VARS*m+1],p[VARS*m])

			# New orientation
			Oz_1[m] = Oz_0[m] + p[VARS*m+2]

			if m < N_S:
				# New slider position
				x_next[DOF+7*m] = x_0[DOF+7*m] + r*cos(alpha)
				x_next[DOF+7*m+1] = x_0[DOF+7*m+1] + r*sin(alpha)
				x_next[DOF+7*m+3:DOF+7*m+7] = Quaternion(axis=[0,0,1], angle=Oz_1[m]).elements

				# x,y,theta velocities in adjusted coords
				x_next[N_QPOS+6*m+DOF] = p[VARS*m+3]*cos(-da) + p[VARS*m+4]*sin(-da)
				x_next[N_QPOS+6*m+DOF+1] = -p[VARS*m+3]*sin(-da) + p[VARS*m+4]*cos(-da)
				x_next[N_QPOS+6*m+DOF+5] = p[VARS*m+5]
			else:
				# Ensure inactive slider velocities are zero
				x_next[N_QPOS+DOF+6*m:N_QPOS+DOF+6*m+6] = 0
	else:
		# Set all new velocities to zero if no collision occurred
		x_next[N_QPOS:] = 0

	return x_next

def get_trained_model(load_weights_loc):
	# Network architecture - must be the exact same as the model's which you are loading in
	inputs = layers.Input(shape=(cols_in,))
	layer1 = layers.Dense(512, use_bias=True, activation='relu')(inputs)
	layer2 = layers.Dense(256, use_bias=True, activation='relu')(layer1)
	layer3 = layers.Dense(128, use_bias=True, activation='relu')(layer2)
	layer4 = layers.Dense(64, use_bias=True, activation='relu')(layer3)
	outputs = layers.Dense(cols_out, activation='linear')(layer4)
	model = keras.Model(inputs=inputs, outputs = outputs)

	# Set loss weights
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
	model.load_weights(load_weights_loc)

	# Warm up predict function
	_ = model.predict(np.zeros((2,DOF+VARS*NUM_OBJS-1)))

	return model

def customloss_wrapper(inputs, loss_weights):

	def customloss(y_true, y_pred):

		losses = tf.reduce_mean(tf.math.squared_difference(y_true,y_pred))

		loss = tf.reduce_mean(tf.multiply(loss_weights,losses))/tf.math.reduce_sum(loss_weights)

		for m in range(NUM_OBJS):

			if m == 0:
				pusher_penetration = tf.math.sqrt( tf.math.square((inputs[:,DOF]+y_pred[:,0])-inputs[:,0]*CTRL_DUR) + tf.math.square(y_pred[:,1]-inputs[:,1]*CTRL_DUR) ) - (rs + rp)
			else:
				pusher_penetration = tf.math.sqrt( tf.math.square((inputs[:,DOF+VARS*m-1]+y_pred[:,VARS*m])-inputs[:,0]*CTRL_DUR) + tf.math.square(y_pred[:,VARS*m+1]-inputs[:,1]*CTRL_DUR) ) - (rs + rp)

			zero = tf.zeros_like(pusher_penetration)

			pusher_penetration = tf.where(pusher_penetration<(zero-max_penetration), pusher_penetration, zero)

			loss += tf.math.square(tf.reduce_mean(pusher_penetration)*penalizing_factor)


			slider_penetration = 0

			for i in range(m+1, NUM_OBJS):

				if m == 0:
					penetration = tf.math.sqrt( tf.math.square( (inputs[:,DOF]+y_pred[:,0]) - (inputs[:,DOF+VARS*i-1]+y_pred[:,VARS*i]) ) + tf.math.square( y_pred[:,1] - (inputs[:,DOF+VARS*i]+y_pred[:,VARS*i+1]) ) ) - 2*rs
				else:
					penetration = tf.math.sqrt( tf.math.square( (inputs[:,DOF+VARS*m-1]+y_pred[:,VARS*m]) - (inputs[:,DOF+VARS*i-1]+y_pred[:,VARS*i]) ) + tf.math.square( (inputs[:,DOF+VARS*m]+y_pred[:,VARS*m+1]) - (inputs[:,DOF+VARS*i]+y_pred[:,VARS*i+1]) ) ) - 2*rs

				zero = tf.zeros_like(penetration)

				penetration = tf.where(penetration<(zero-max_penetration), penetration, zero)

				slider_penetration += tf.math.square(tf.reduce_mean(penetration)*penalizing_factor)

			loss += slider_penetration

		return loss

	return customloss

def fine_int(x_0, u):
	"""Uses the full physics simulator Mujoco to predict the next state.

	Args:
		x_0: Initial state.
		u:   An action.

	Returns:
		x_next: The resulting state
	"""

	with physics.reset_context():
		physics.set_state(x_0)
		try:
			physics.step()
		except:
			# Unstable
			pass

	for k in range(num_substeps):
		physics.data.ctrl[:] = u[0:DOF]
		try:
			physics.step()
		except:
			# Unstable
			pass

	x_next = physics.get_state()

	return x_next

def parareal_engine(x_0, U, num_iter, coarse_model="learned"):
	"""Rolls out a trajectory using Parareal

	Args:
		x_0:			Initial state
		U: 				Control sequence
		num_iter:		Number of Parareal iterations to use for prediction
		coarse_model:	Learned/Analytical coarse model
	Returns:
		X: 			The corresponding state sequence

	"""

	global Q
	num_time_slices = U.shape[0]
	Q = np.zeros((num_iter+1, num_time_slices+1, N_QPOS+N_QVEL))

	for k in range(num_iter+1):
		Q[k,0] = x_0.copy()

	# Find the initial coarse predictions across time slices
	for p in range(num_time_slices):
		if coarse_model != "learned":
			Q[0,p+1] = coarse_int_analytical(Q[0,p], U[p])
		else:
			Q[0,p+1] = coarse_int_learned(Q[0,p], U[p])

	# Parareal iterations
	for k in range(1,num_iter+1):
		pool_input = []
		for p in range (num_time_slices):
			pool_input.append([Q[k-1,p], U[p]])
		pool_input = np.array(pool_input)
		fine_predictions = pool.map(parallel_fine, pool_input)
		for p in range(num_time_slices):
			if coarse_model != "learned":
				Q[k, p+1] = coarse_int_analytical(Q[k,p], U[p])- coarse_int_analytical(Q[k-1,p], U[p]) + fine_predictions[p]
			else:
				# Learned coarse model
				Q[k, p+1] = coarse_int_learned(Q[k,p], U[p])- coarse_int_learned(Q[k-1,p], U[p]) + fine_predictions[p]

	X = Q[num_iter].copy()

	return X

def mujoco_engine(x_0, U):
	"""Rolls out a trajectory using the full physics simulator Mujoco

	Args:
		x_0: 	Initial state
		U: 		Control sequence

	Returns:
		X: 		The corresponding state sequence
	"""

	num_ctrl_steps = U.shape[0]
	state_size = x_0.shape[0]
	X = np.zeros(( num_ctrl_steps+1, state_size))

	X[0] = x_0.copy()

	for p in range(num_ctrl_steps):
		X[p+1] = fine_int(X[p], U[p])
	return X

def parallel_fine(input):
	return fine_int(input[0], input[1])

def calculate_theoretical_speedup(K, P):
	"""Computes the theoretical speedup provided by Parareal

	Args:
		K:			Number of iterations
		P:			Number of timeslices

	Returns:
		speedup:	The theoretical speedup w.r.t to using only the fine model.

	"""

	x_0 = init_state.copy()

	start_time_coarse = timeit.default_timer()
	coarse_int_learned(x_0, control_sequence[0])
	c_c = timeit.default_timer() - start_time_coarse

	start_time_fine = timeit.default_timer()
	fine_int(x_0, control_sequence[0])
	c_f = timeit.default_timer() - start_time_fine

	speedup = 1. / ((1 + K)*(c_c/c_f) + (K/P))

	# print ('coarse time is', c_c)
	# print ('fine time is', c_f)

	#print('coarse model is %.3f times faster than the fine model' % (c_f/c_c))

	# print ('Theoretical speedup is', speedup)

	return speedup

def capture_state_frames(Q, run_mode):
	"""Creates a new directory and captures states in Q.
	"""

	if run_mode == 0:
		run_folder = 'learned_model'
	elif run_mode == 1:
		run_folder = 'analytical_model'

	if learned_and_analytical:

		if sequence == 1 and run_mode == 0:
			subprocess.call(["rm", "-rf", main_data_path+"parareal_frames"])
			subprocess.call(["mkdir", "-p", main_data_path+"parareal_frames"])
			subprocess.call(["mkdir", "-p", main_data_path+"parareal_frames/%s" % run_folder])

		elif sequence == 1 and run_mode == 1:
			subprocess.call(["mkdir", "-p", main_data_path+'parareal_frames/%s' % run_folder])

		subprocess.call(["mkdir", "-p", main_data_path+'parareal_frames/%s/sequence_%d' % (run_folder, sequence)])
	else:
		if sequence == 1 and clear_frames:
			subprocess.call(["rm", "-rf", main_data_path+"parareal_frames"])
			subprocess.call(["mkdir", "-p", main_data_path+'parareal_frames'])

		subprocess.call(["mkdir", "-p", main_data_path+'parareal_frames/sequence_%d' % sequence])

	for iter in range(max_parareal_iter):

		if learned_and_analytical:
			subprocess.call(["mkdir", "-p", main_data_path+'parareal_frames/%s/sequence_%d/iteration_%d' % (run_folder, sequence, iter)])
		else:
			subprocess.call(["mkdir", "-p", main_data_path+'parareal_frames/sequence_%d/iteration_%d' % (sequence, iter)])

		for p in range(num_time_slices+1):
			with physics.reset_context():
				physics.set_state(Q[iter,p])
				try:
					physics.step()
				except:
					pass
			image_data = physics.render(height=480, width=640, camera_id=0)
			img = Image.fromarray(image_data, 'RGB')
			if learned_and_analytical:
				img.save(main_data_path+"parareal_frames/{}/sequence_{}/iteration_{}/frame-{}.png".format(run_folder,sequence,iter,p))
			else:
				img.save(main_data_path+"parareal_frames/sequence_{}/iteration_{}/frame-{}.png".format(sequence,iter,p))

def save_data(parareal_time_array, mujoco_time_array, expected_parareal_time_array, run_mode):
	"""Creates a new directory and saves experimental data
	"""
	if run_mode == 0:
		# Learned model
		run_mode = 'learned_model'
	elif run_mode == 1:
		# Analytical model
		run_mode = 'analytical_model'

	if sequence == 1 and run_mode == 'learned_model':
		if os.path.isdir("parareal_data"):
			subprocess.call(["rm", "-rf", main_data_path+"parareal_data"])
			subprocess.call(["mkdir", "-p", main_data_path+"parareal_data"])
		else:
			subprocess.call(["mkdir", "-p", main_data_path+"parareal_data"])

		subprocess.call(["mkdir", "-p", main_data_path+"parareal_data/%s" % run_mode])

	elif sequence == 1 and run_mode == 'analytical_model':
		subprocess.call(["mkdir", "-p", main_data_path+"parareal_data/%s" % run_mode])

	subprocess.call(["mkdir", "-p", main_data_path+"parareal_data/%s/sequence_%d" % (run_mode,sequence)])
	np.save(main_data_path+'parareal_data/%s/sequence_%d/Q' % (run_mode, sequence), Q)
	np.save(main_data_path+'parareal_data/%s/sequence_%d/init_state' % (run_mode, sequence), init_state)
	np.save(main_data_path+'parareal_data/%s/sequence_%d/control_sequence' % (run_mode, sequence), control_sequence)
	np.save(main_data_path+'parareal_data/%s/sequence_%d/parareal_time_array' % (run_mode, sequence), parareal_time_array)
	np.save(main_data_path+'parareal_data/%s/sequence_%d/expected_parareal_time_array' % (run_mode, sequence), expected_parareal_time_array)
	np.save(main_data_path+'parareal_data/%s/sequence_%d/mujoco_time_array' % (run_mode, sequence), mujoco_time_array)

def get_random_feasible_state(N_S, init_state):
	"""Performs rejection sampling to find a feasible state (no object penetration)

	Args:
		N_S:				Number of (active) sliders
		init_state: 		Initial state

	Returns:
		feasible_state: 	A random feasible state
	"""

	feasible_state = init_state.copy()

	for M in range(NUM_OBJS):

		if M == 0:
			# First slider can have any starting position in this range
			xpos = np.random.uniform(init_state[0]+rs+rp, init_state[0]+rp+rs+0.1)
			ypos = np.random.uniform(init_state[1]-rp-rs, init_state[1]+rp+rs)

			feasible_state[2:4]    = np.array([xpos,ypos])

		elif M < N_S:
			penetration = True

			# for other sliders, ensure there is no penetration between any two objects
			while penetration:
				xpos = np.random.uniform(init_state[0]-rp-2*rs, init_state[0]+rp+(M+1.2)*rs)
				ypos = np.random.uniform(init_state[1]-rp-2*rs, init_state[1]+rp+2*rs)

				pusher_penetration = sqrt( (xpos-init_state[0])**2 + (ypos-init_state[1])**2 ) - (rs + rp)

				if pusher_penetration < -PENETRATION_EPS:
					continue

				for i in range(M):

					slider_penetration = sqrt( (xpos-feasible_state[DOF+7*i])**2 + (ypos-feasible_state[DOF+7*i+1])**2 ) - 2*rs

					if slider_penetration < -PENETRATION_EPS:
						break

				if slider_penetration > -PENETRATION_EPS and pusher_penetration > -PENETRATION_EPS:
					penetration = False

			feasible_state[DOF+7*M:DOF+7*M+2] = np.array([xpos,ypos])

		else:
			# Set other, inactive sliders out of range
			feasible_state[DOF+7*M:DOF+7*M+2] = np.array([M+3,M+1])

	return feasible_state

def run_experiment(run_mode):

	x_0 = init_state.copy()
	U = control_sequence

	sq = sequence - value*number_of_samples

	for k in range(n_actions+1):
		# Calculate parareal time
		start_time_parareal = timeit.default_timer()
		if run_mode == 1:
			parareal_state_sequence = parareal_engine(x_0, U, k, coarse_model="analytical")
		elif run_mode == 0:
			parareal_state_sequence = parareal_engine(x_0, U, k, coarse_model="learned")

		parareal_time = timeit.default_timer() - start_time_parareal
		parareal_time_array[k,sq-1] = parareal_time

	#print()

	for k in range(n_actions+1):
		# Calculate MUJOCO time
		start_time_mujoco = timeit.default_timer()
		mujoco_state_sequence = mujoco_engine(x_0, U)
		mujoco_time = timeit.default_timer() - start_time_mujoco
		mujoco_time_array[k,sq-1] = mujoco_time

		# Calculate expected PARAREAL time
		expected_parareal_time = mujoco_time/calculate_theoretical_speedup(k, n_actions)
		expected_parareal_time_array[k,sq-1] = expected_parareal_time

	#print()

	if run_mode == 0 and sq == number_of_samples:
		mean_expected_parareal_time_array = np.mean(expected_parareal_time_array, axis=1)
		std_expected_parareal_time_array = 1.96*np.std(expected_parareal_time_array, axis=1)
		mean_mujoco_time_array = np.mean(mujoco_time_array, axis=1)
		std_mujoco_time_array = 1.96*np.std(mujoco_time_array, axis=1)
		mean_parareal_time_array = np.mean(parareal_time_array, axis=1)
		std_parareal_time_array = 1.96*np.std(parareal_time_array, axis=1)

	capture_state_frames(Q, run_mode)
	save_data(parareal_time_array, mujoco_time_array, expected_parareal_time_array, run_mode)

	return parareal_state_sequence, mujoco_state_sequence

if __name__ == "__main__":
	mp.set_start_method('spawn')
	pool = mp.Pool(num_cores, initializer, ())

	model = get_trained_model(load_weights_loc)

	init_state = np.array([
		1.72    , 0.025     ,
		1.8     , 0.025        , 0.4262    , 1.    , 0.    , 0.    , 0.    ,
		10      , 36       , 0.4262    , 1.    , 0.    , 0.    , 0.    ,
		4       , -10      , 0.4262    , 1.    , 0.    , 0.    , 0.    ,
		-8      , 12     , 0.4262    , 1.    , 0.    , 0.    , 0.    ,
		0.      , 0.        ,
		0.      , 0.        , 0.        , 0.    , 0.    , 0.    ,
		0.      , 0.        , 0.        , 0.    , 0.    , 0.    ,
		0.      , 0.        , 0.        , 0.    , 0.    , 0.    ,
		0.      , 0.        , 0.        , 0.    , 0.    , 0.
		])

	sequence = 1

	print ()

	print ("== Experiment started for {} object(s) and {} actions in a sequence ===".format(chosen_objs[0], n_actions))

	print ()

	for value in range(len(chosen_objs)):  # choose number of active sliders in the environment

		N_S = chosen_objs[value]

		parareal_time_array = np.zeros((n_actions+1,number_of_samples))
		mujoco_time_array = np.zeros((n_actions+1,number_of_samples))
		expected_parareal_time_array = np.zeros((n_actions+1,number_of_samples))

		while sequence - number_of_samples*value <= number_of_samples:

			print ('sample', sequence)

			init_state = get_random_feasible_state(N_S, init_state)

			contact_count = 0
			while contact_count < 1:
				for n in range(n_actions):
					control_sequence[n,0] = random.uniform(-ACTION_MAGNITUDE, ACTION_MAGNITUDE)
					control_sequence[n,1] = random.uniform(-m.sqrt((0.01)-control_sequence[n,0]**2), m.sqrt((0.01)-control_sequence[n,0]**2))

				state_sequence = mujoco_engine(init_state, control_sequence)

				for a in range(n_actions):

						for step in range(0,STEPS+1):

							for s in range(N_S):
								x_sep = state_sequence[a,DOF+7*s]-(state_sequence[a,0] + step*control_sequence[a,0]*CTRL_DUR/STEPS)
								y_sep = state_sequence[a,DOF+7*s+1]-(state_sequence[a,1] + step*control_sequence[a,1]*CTRL_DUR/STEPS)

								new_sep = sqrt(x_sep**2 + y_sep**2)

								if new_sep-(rs+rp) < -5e-4:
									contact_count += 1
									break

							if contact_count >= 1:
								break

			Q = np.zeros((max_parareal_iter+1, num_time_slices+1, N_QPOS+N_QVEL))

			for k in range(max_parareal_iter+1):
				Q[k,0] = init_state.copy()

			run_experiment(0) # Use learned coarse model

			if learned_and_analytical:
				run_experiment(1)	# Use analytical model as well

			sequence += 1

	print ("Great! Experiment done. Data and frames saved.")
	print ()
	pool.close()
