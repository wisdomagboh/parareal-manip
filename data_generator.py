''' Uses MuJoCo physics to produce state data, which is written to .csv to be used to train a model. '''

import dm_control
from dm_control import mujoco
from dm_control import suite
from dm_control import viewer
from dm_control.suite import common

import numpy as np 
import math as m
from math import pi, cos, sin, tan, atan2, sqrt
import random as r
from PIL import Image
import subprocess
import sys
import glob
import os
import xml.etree.ElementTree as ET
import csv
import pyquaternion as pq
from pyquaternion import Quaternion
from timeit import default_timer as timer
import datetime
from IPython import embed

_SUITE_DIR = "".join(suite.__path__)

_MAX_SLIDERS = 4		# number of sliders the xml file contains
_CAP_SLIDERS = 4		# limit the number of sliders to generate data for (otherwise should be same as _MAX_SLIDERS)
_DOF = 2				# robot dof
_VARS = 6				# variables per slider (x, y, 0, vx, vy, v0)

N_QPOS = _DOF + 7*_MAX_SLIDERS
N_QVEL = _DOF + 6*_MAX_SLIDERS

SLIDER_CYLINDER = True
SLIDER_BOX = False

# Max penetration in state 0 before it is skipped
allowable_penetration = 5e-4

# Set timing conditions
dt = 0.002
_CTRL_DUR = 1
steps = int(_CTRL_DUR/dt)

# Filenames csv
disc_files = 'discrete_ms'
rand_files = 'random_ms_2'

# Physics xml filename
physics_xml = 'multislider.xml'

# Rendering xml filename
render_xml = 'multislider_render.xml'

############################################################################################################

render = True
render_frames = False
clear = False			# Clear csv files or not

# Generation mode
_discrete = False
_random = True

# Iterations to complete in random mode (will be split between the numbers of active sliders in the env)
repeats = 100

############################################################################################################

def check_contact(physics, max_penetration):
	''' This function checks for contact between any two geoms not including the plane surface, above a maximum allowable penetration.

	Args:
		physics:			The physics object to check.
		max_penetration:	The maximum allowed before qualifying as a significant 'contact'	

	Returns:
		_contact:			Whether or not there was a contact between two geoms, not including the plane.
	'''

	_contact = False

	# Get geom ID for plane and check for contact
	plane_geom_id = physics.named.model.geom_bodyid['table_0']

	# Get arrays of ids of geoms in contact
	geom_1_contact_array = physics.data.contact['geom1']
	geom_2_contact_array = physics.data.contact['geom2']
	penetration_array = physics.data.contact['dist']

	for k in range(geom_1_contact_array.shape[0]):
		# Get ids of geoms to check for conact
		geom_1_id = physics.model.geom_bodyid[geom_1_contact_array[k]]
		geom_2_id = physics.model.geom_bodyid[geom_2_contact_array[k]]
		penetration = penetration_array[k]

		# Check for a collision not involving table (i.e. between any two objects)
		if np.any(plane_geom_id != geom_1_id) and np.any(plane_geom_id != geom_2_id):
			if abs(penetration) > max_penetration:	
				# Significant contact between two objects
				_contact = True

	return _contact

def reset(state, physics, reset_acc=False):
	''' This function resets the physics of an environment and implements a new state.

	Args:
		state:		New physics state to set.
		physics:	Physics object to reset.
		reset_acc:	Whether or not to reset the accelerations in the environment.

	'''
	with physics.reset_context():
		physics.set_state(state)
		if reset_acc:
			physics.data.qacc_warmstart[:] = 0 
			physics.data.qacc[:] = 0 
		physics.step()

def clear_csv_files(clear, filename):
	''' This function clears the csv files passed to it, depending on the clear variable. Also requires input to confirm.

	Args:
		clear:		Whether or not to clear the files.
		filename:	Prefix of the filenames to clear.
	'''

	filename_in = '%s_initial.csv' % filename
	filename_out = '%s_final.csv' % filename

	# Clear data files, or don't
	if clear:

		sure = input("Are you sure you wish to erase the '%s' data? (Y/n)    " % filename)

		if sure == 'y' or sure == 'Y' or not sure:

			with open(filename_in, mode='w') as file:
				file.close()
			with open(filename_out, mode='w') as file:
				file.close()

			print()
			print('Files cleared.')
			print()

		elif sure == 'n' or sure == 'N':

			raise SystemExit("Execution terminated - amend value of 'clear' variable to continue.")

		else:

			raise ValueError("Invalid input - use y/n/enter")

def renderer(X, Y, folder_suffix="", empty_folder=True):
	''' This function renders the initial state alongside the final state for comparison/debugging.

	Args:
		X:					Input state data.
		Y:					Final state data.
		folder_suffix:		Suffix of the folder to save renders to.
		empty_folder:		Decide whether or not to empty the folder.

	'''

	if render:

		if iterations == 0:

			if os.path.isdir('renders_%s' % folder_suffix) and empty_folder:
				
				subprocess.call(["rm", "-r", 'renders_%s' % folder_suffix])
				
				# Create empty directory
				os.mkdir('renders_%s' % folder_suffix)

			elif not os.path.isdir('renders_%s' % folder_suffix):
				
				# Create empty directory
				os.mkdir('renders_%s' % folder_suffix)


		# Load in .xml file and get root
		xml_loc = '%s/%s' % (_SUITE_DIR, render_xml)
		_, physics = physics_from_xml(xml_loc)

		image_loc = 'renders_%s/#%d.png' % (folder_suffix, iterations+contact_counts)

		# Set input state and render
		new_state = np.zeros((3*(N_QPOS+N_QVEL)-2*_DOF,))

		# Set new position of pusher
		new_state[N_QPOS : N_QPOS+2] = np.array([ X[0]*_CTRL_DUR, X[1]*_CTRL_DUR ])

		# Set slider positions (velocities are left as zero, as we are only interested in rendering an image here)
		for m in range(_MAX_SLIDERS):

			new_state[_DOF+7*m:_DOF+7*(m+1)] 			      =	np.array([ X[_DOF+_VARS*m], 				X[_DOF+_VARS*m+1], 			0, 1, 0, 0, 0 ])
			new_state[N_QPOS+_DOF+7*m:N_QPOS+_DOF+7*(m+1)]    =	np.array([ X[_DOF+_VARS*m] + Y[_VARS*m], 	X[_DOF+_VARS*m+1] + Y[_VARS*m+1], 0, 1, 0, 0, 0 ])

		# There is an extra state in the rendering xml which is not required here, so set to a long distance away
		new_state[[2*N_QPOS+1]+[2*N_QPOS+8]+[2*N_QPOS+15]+[2*N_QPOS+22]] = 10

		reset(new_state, physics)

		image_array = physics.render(height=480, width=600, camera_id='fixed')
		img = Image.fromarray(image_array, 'RGB')
		img.save(image_loc)

def frame_renderer(physics, foldername, step_count, steps_per_frame):
	''' This function renders the frames at each physics step during the execution of an action.

	Args:
		physics:			Physics object to render.
		foldername:			Folder to save renders to.
		step_count:			Number of physics steps executed so far (minus 1).
		steps_per_frame:	Number of steps to take between frames.
	'''

	if render_frames:

		if step_count == 0: 

			if iterations == 0:
				subprocess.call(["rm", "-r", foldername])
				os.mkdir(foldername)
			
			os.mkdir('%s/#%d' % (foldername, iterations+contact_counts))

		if step_count % steps_per_frame == 0:
	
				image_array = physics.render(height=360, width=480, camera_id='fixed')
				img = Image.fromarray(image_array, 'RGB')
				img.save('%s/#%d/frame#%d.png' % (foldername, iterations+contact_counts, step_count/steps_per_frame))

def write_csv_row(filename,row):
	'''This function appends a row to a csv file.

	Args:
		filename:	Name of the csv file to write to.
		row:		1D array containing the data to write to csv.

	'''

	with open(filename, mode='a') as file:
		f = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		f.writerow(row)

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

def progress():
	''' This function prints the progress of the execution to the terminal every 1%. '''

	global last, goal

	if _discrete:

		# On first run, calculate how many combinations will be attempted (goal)
		if iterations == 0:

			goal = 0

			for N_S in range(1, _CAP_SLIDERS+1):

				product = (a_gp[0]-2*N_S)*(a_gp[1]-2*N_S)*(a_gp[2]-N_S)*(a_gp[3]-N_S)

				for m in range(N_S):
					product *= (x_gp+m-2)*y_gp

				goal += product

			print()
			print('Total iterations: %d' % goal)
			print()

		# Update number of completed combinations
		completed = contact_counts + iterations

	elif _random:

		completed = iterations
		goal = repeats

	n = int(100*completed/goal)

	total_seconds = timer() - start

	mins = total_seconds / 60

	hours = mins // 60
	seconds = 60*(mins % 60)

	mins = seconds // 60
	seconds = seconds % 60

	if n == 0 and last != 0:

		last = 0

		print('			Time elapsed:		Successful iterations:')
		print('0%s complete.		%02d:%02d:%06.3f		%d/%d' % ('%', hours, mins, seconds, iterations,(iterations+contact_counts)))
	
	elif  last != n:

		last = n

		print('%d%s complete.		%02d:%02d:%06.3f		%d/%d' % (n,'%', hours, mins, seconds, iterations,(iterations+contact_counts)))

def recursive_runner(N, k):
	''' This function creates nested for loops and recursively calls 'runner'

	Args:
		N:		1D array of integers, each value determines the number of iterations in a single loop.
		k:		Number of nested for loops to create. On the first call, k should equal len(N).

	'''

	if k > 0:
		for i in range(int(N[k-1])):
			n[k-1] = i
			recursive_runner(N, k-1)
	else:
		runner(_N_SLIDERS, n, files)
		progress()
	
def runner(_N_SLIDERS, n, files):
	''' This function uses a physics object and excutes actions on an environment, using a pusher.
		The first action is not recorded, and is taken to create realistic velocities to use as an input state.
		The second action is recorded. 
		Initial positions and velocities of the sliders and the pusher action is recorded in one file ('_initial')
		Slider displacements and final velocities are recorded in another file ('_final').

	Args:
		_N_SLIDERS:		Number of active sliders in the environment.
		n:				1D array which is mapped to certain variable inputs.
		files:			Prefix of files to save data to.

	'''

	global iterations, contact_counts, collisions

	filename_in = '%s_initial.csv' % files
	filename_out = '%s_final.csv' % files

	x_0 = np.zeros((N_QPOS + N_QVEL,))

	# Set velocities for first action
	d0 = n[-4]*2*pi/(3*(N[-4]-1)) - pi/3
	m0 = (n[-3]+1)*v_max/N[-3]				# (doesn't produce zero magnitude for action 1)

	V = np.zeros((2,))
	V[0] = m0*cos(d0)
	V[1] = m0*sin(d0)	

	_CTRL_DUR_0 = 0.1/m0
	
	steps_0 = int(_CTRL_DUR_0/dt)

	# Boundaries for slider positions
	lower_xlim = rp + rs
	upper_xlims = np.zeros((_N_SLIDERS,))		# defined later for each slider
	ylims = (1.2*_N_SLIDERS-1)*Rs if _N_SLIDERS > 2 else 1.5*rs

	xpos = np.zeros((_N_SLIDERS,))
	ypos = np.zeros((_N_SLIDERS,))

	# Set initial positions and orientations of sliders
	for m in range(_MAX_SLIDERS):

		# Give all sliders random starting orientation
		theta0 = r.uniform(0, 2*pi)
		x_0[_DOF + 7*m + 3:_DOF + 7*m + 7] = Quaternion(axis=[0,0,1], angle=theta0).elements
		
		# Slider z-position
		x_0[_DOF + 7*m + 2] = hs

		# Set x & y positions within limits depending on slider number
		if m < _N_SLIDERS:
			upper_xlims[m] = rp + (m+1)*Rs + m0*_CTRL_DUR_0

			xpos[m] = n[2*m]*(upper_xlims[m]-lower_xlim)/(N[2*m]-1) + lower_xlim
			ypos[m] = n[2*m+1]*2*ylims/(N[2*m+1]-1) - ylims

			x_0[_DOF+7*m:_DOF+7*m+2] = np.array([xpos[m], ypos[m]])
		else:
			# Unused sliders positioned out of range
			x_0[_DOF+7*m:_DOF+7*m+2] = np.array([m+1, m+1])

	# Reset physics and set state zero
	reset(x_0, physics, reset_acc=True)

	_contact = check_contact(physics, allowable_penetration)

	# If contact occurred, increment contact count and return --> next loop iteration
	if _contact:
		contact_counts += 1
		return

	# Iterate over step count
	for step_count in range(steps_0):
		physics.data.ctrl[0:2] = V
		physics.data.ctrl[2:4] = V*step_count*dt
		physics.step()

		frame_renderer(physics, 'frames_A1', step_count, 50)

	# Get new state data
	x_1 = physics.get_state()

	inputs = np.zeros((_DOF+_MAX_SLIDERS*_VARS,))

	yaw1 = np.zeros((_N_SLIDERS,))

	# Get remaining position and velocity data for sliders
	for m in range(_MAX_SLIDERS):

		if m < _N_SLIDERS:

			# Get orientation
			yaw1[m], _, _ = Quaternion(x_1[_DOF+7*m+3:_DOF+7*m+7]).yaw_pitch_roll

			if yaw1[m] < 0:
				yaw1[m] += 2*pi

			# Position relative to new axes
			inputs[_DOF+_VARS*m]	= x_1[_DOF+7*m] - x_1[0]
			inputs[_DOF+_VARS*m+1]	= x_1[_DOF+7*m+1] - x_1[1]
			
			# Orientation relative to new axes
			inputs[_DOF+_VARS*m+2]	= yaw1[m]

			# Velocities relative to new axes
			inputs[_DOF+_VARS*m+3] 	= x_1[N_QPOS+_DOF+6*m]
			inputs[_DOF+_VARS*m+4] 	= x_1[N_QPOS+_DOF+6*m+1]
			inputs[_DOF+_VARS*m+5]	= x_1[N_QPOS+_DOF+6*m+5]

		else:
			# Set distant position for unused sliders, leave orientation and velocities as zero
			inputs[_DOF+_VARS*m]	= 1+m
			inputs[_DOF+_VARS*m+1]	= 1+m		


	# # Choose second action direction relative to centre of slider 1 (forward-focused)
	
	x = 0.2	# Approximate proportion of velocities aimed backwards (where pi/2 < d1 < 3pi/2) 
	
	# Choose a velocity direction
	if n[-2] <= 0.5*(1-x)*N[-2]:

		d1 = n[-2]*pi/((1-x)*N[-2])

	elif 0.5*(1-x)*N[-2] < n[-2] < 0.5*(1+x)*N[-2]:
		
		d1 = n[-2]*pi/(x*N[-2]) + (1 - 1/(2*x))*pi

	elif n[-2] >= 0.5*(1+x)*N[-2]:
		
		d1 = n[-2]*pi/((1-x)*N[-2]) + (1 - x/(1-x))*pi

	# Second action magnitude
	m1 = n[-1]*v_max/(N[-1]-1)

	# Second action for action array
	U = np.zeros((2,))
	U[0] = m1*cos(d1)
	U[1] = m1*sin(d1)

	reset(x_1, physics)

	# Iterate over step count
	for step_count in range(steps):
		physics.data.ctrl[0:2] = U
		physics.data.ctrl[2:4] = x_1[0:2] + U*step_count*dt
		physics.step()

		frame_renderer(physics, 'frames_A2', step_count, 50)
		
	# Get next state data
	x_2 = physics.get_state()

	# Action - calculated using average motion rather than applied velocities (as these would not be achieved perfectly)
	inputs[0] = (x_2[0]-x_1[0])/_CTRL_DUR
	inputs[1] = (x_2[1]-x_1[1])/_CTRL_DUR

	# Output '_VARS' values per slider
	outputs = np.zeros((_MAX_SLIDERS*_VARS,))

	yaw2 = np.zeros((_N_SLIDERS,))

	# Get new position and velocity data for sliders
	for m in range(_N_SLIDERS):

		# Get new orientation of sliders, in range 0 to 2pi
		yaw2[m], _, _ = Quaternion(x_2[_DOF+7*m+3:_DOF+7*m+7]).yaw_pitch_roll
		
		if yaw2[m] < 0:
			yaw2[m] += 2*pi
		
		# Calculate x, y & angular displacements of sliders (angular always -pi to pi)
		outputs[_VARS*m] = x_2[_DOF+7*m]-x_1[_DOF+7*m]
		outputs[_VARS*m+1] = x_2[_DOF+7*m+1]-x_1[_DOF+7*m+1]

		if abs(yaw2[m]-yaw1[m]) <= pi:
			outputs[_VARS*m+2] = yaw2[m]-yaw1[m]
		elif (yaw2[m]-yaw1[m]) > pi:
			outputs[_VARS*m+2] = yaw2[m]-yaw1[m]-2*pi
		elif (yaw2[m]-yaw1[m]) < -pi:
			outputs[_VARS*m+2] = yaw2[m]-yaw1[m]+2*pi		

		# Final x, y & angular velocities of sliders
		outputs[_VARS*m+3] = x_2[N_QPOS+_DOF+6*m]
		outputs[_VARS*m+4] = x_2[N_QPOS+_DOF+6*m+1]
		outputs[_VARS*m+5] = x_2[N_QPOS+_DOF+6*m+5]

	# Get parameters in order to render image of first action
	s0 = np.zeros((_DOF+_MAX_SLIDERS*_VARS-1,))
	s1 = np.zeros((_MAX_SLIDERS*_VARS,))

	s0[0:_DOF] = x_1[0:_DOF]/_CTRL_DUR

	for m in range(_MAX_SLIDERS):

		if m < _N_SLIDERS:
			s0[_DOF+_VARS*m:_DOF+_VARS*m+2] = x_0[_DOF+7*m:_DOF+7*m+2]
			s1[_VARS*m:_VARS*m+2] = x_1[_DOF+7*m:_DOF+7*m+2] - x_0[_DOF+7*m:_DOF+7*m+2]
		else:
			s0[_DOF+_VARS*m:_DOF+_VARS*m+2]	= 1+m

	# Write initial state to file
	write_csv_row(filename_in, inputs)

	# Write displacements and final velocities to file
	write_csv_row(filename_out, outputs)

	# Render image of action 2
	renderer(inputs, outputs, folder_suffix='A2')

	# Render image of action 1
	renderer(s0, s1, folder_suffix='A1')

	# If any displacements are somewhat significant, then successful collision (probably) occurred
	for m in range(_MAX_SLIDERS):

		if np.any(outputs[_VARS*m:_VARS*m+2] > 10**-4):
			collisions += 1

		# Angular displacement typically an order of magnitude above x-y displacement
		elif outputs[_VARS*m+2] > 10**-3:
			collisions += 1

	iterations += 1

###########################################################################################################

if __name__ == '__main__':

	print()
	print('			%s' % datetime.datetime.now().time() )
	print()

	start = timer()

	last = 1

	# Get physics object and xml root from xml file
	xml_location = '%s/%s' % (_SUITE_DIR, physics_xml)
	root, physics = physics_from_xml(xml_location)

	rp = float(root[4][0][0].get('size').split(' ')[0])	# radius of pusher

	# rs is minimum radius of slider, Rs is maximum

	if SLIDER_CYLINDER:

		rs = Rs = float(root[4][1][0].get('size').split(' ')[0])	# radius of sliders
		hs = float(root[4][1][0].get('size').split(' ')[1])	# half-height of sliders

	elif SLIDER_BOX:

		xs = float(root[4][2][0].get('size').split(' ')[0])	# x length of slider
		ys = float(root[4][2][0].get('size').split(' ')[1])	# y length of slider
		hs = float(root[4][2][0].get('size').split(' ')[2])	# half-height of sliders

		rs = min(xy, ys)
		Rs = sqrt(xs**2 + ys**2)

	v_max = 0.1		# maximum pushing velocity

	# Choose number of grid points per variable
	x_gp = 4
	y_gp = 5
	a_gp = np.array([12,11,10,9])  # A1 direction & magintude, followed by A2

	contact_counts = 0
	iterations = 0
	collisions = 0

	''' 
	N[i] is the number of values to be used for variable i in the discrete mode,
	n[i] is a value from 0 to N[i] that will be used to calculate the value of the variable i in 'runner'.
	In discrete mode n[i] is always an integer, in random mode n[i] is a float.
	The equations in 'runner' map each n[i]~N[i] pair to a value in the desired range of values for that variable.

	'''

	if _random and _discrete: 

		raise ValueError('Both run modes selected - please select exactly one.')

	elif not _random and not _discrete:

		raise ValueError('Neither run mode selected - please select exactly one.')

	elif _random:

		files = rand_files

		# Clear data files depending on 'clear' variable
		clear_csv_files(clear, files)

		progress()

		for _N_SLIDERS in range(1, _CAP_SLIDERS+1):

			N = np.zeros((2*_N_SLIDERS+4,))
			n = np.zeros((2*_N_SLIDERS+4,))

			# Input state
			for m in range(_N_SLIDERS):
				N[2*m] = x_gp+m-1			# x_0 xpos slider m
				N[2*m+1] = y_gp				# x_0 ypos slider m

			# First action
			N[-4] = a_gp[0]					# A1 direction
			N[-3] = a_gp[1]					# A1 magnitude

			# Second action
			N[-2] = a_gp[2]					# A2 direction
			N[-1] = a_gp[3]					# A2 magnitude

			# These variables loop and have the same value at n=0 and n=N, i.e. A2 direction hich has a range of 2pi radians
			cyclic_ind = np.array([-2])

			# Generate equal amounts of data for each number of sliders
			while iterations - repeats*(_N_SLIDERS-1)/_CAP_SLIDERS < repeats/_CAP_SLIDERS:

				for i in range(len(N)):
					
					if np.any(cyclic_ind == i or cyclic_ind == i-len(N)):
						n[i] = r.uniform(0, N[i])
					else:
						n[i] = r.uniform(0, N[i]-1)		

				runner(_N_SLIDERS, n, files)

				progress()

		print()
		print('NUMBER OF CONTACT INSTANCES = %d' % contact_counts)
		print('DATA POINTS RECORDED = %d' % repeats)
		print('COLLISION DATA POINTS = %d' % collisions)
		print()

	elif _discrete:

		files = disc_files

		# Clear data files depending on 'clear' variable
		clear_csv_files(clear, files)

		progress()

		''' Manipulates the number of gridpoints defined above depending on the number of sliders being used
		(if these gridpoint functions are changed, then the progress function will need updating to calculate the total) '''
		
		for _N_SLIDERS in range(1, _CAP_SLIDERS+1):

			N = np.zeros((2*_N_SLIDERS+4,))
			n = np.zeros((2*_N_SLIDERS+4,))

			# Input state
			for m in range(_N_SLIDERS):
				N[2*m] = x_gp+m-2				# x_0 xpos slider m
				N[2*m+1] = y_gp					# x_0 ypos slider m

			# First action
			N[-4] = a_gp[0]-2*_N_SLIDERS		# A1 direction
			N[-3] = a_gp[1]-2*_N_SLIDERS		# A1 magnitude

			# Second action
			N[-2] = a_gp[2]-_N_SLIDERS			# A2 direction
			N[-1] = a_gp[3]-_N_SLIDERS			# A2 magnitude

			recursive_runner(N, len(N))

		print()
		print('NUMBER OF CONTACT INSTANCES = %d' % contact_counts)
		print('DATA POINTS RECORDED = %d' % (goal-contact_counts))
		print('COLLISION DATA POINTS = %d' % collisions)
		print()
