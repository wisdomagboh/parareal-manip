import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import IPython
import subprocess
import matplotlib.pyplot as plt
import numpy as np

from pyquaternion import Quaternion
from PIL import Image
from numpy import mean, sqrt, square, arange
from math import pi, sqrt, sin, cos, atan2

from config import *

# sample call python3.5 plot_generation.py --learned_and_analytical True --n_actions 4 --chosen_objs 1 --number_of_samples 10 --error_type xy --combine_error False

parser = argparse.ArgumentParser(description='Parareal for Robotic Manipulation')
parser.add_argument('--learned_and_analytical', action='store', default="True")
parser.add_argument('--n_actions', action="store", default=4)
parser.add_argument('--chosen_objs', action="store", default=1)
parser.add_argument('--number_of_samples', action="store", default=10)
parser.add_argument('--error_type', action="store", default='xy')
parser.add_argument('--combine_error', action="store", default="True")

args = parser.parse_args()

number_of_samples = int(args.number_of_samples)   	# Number of random samples for experiment
chosen_objs = [int(args.chosen_objs)]				# Number of objects in a scene 1 - 4
n_actions = int(args.n_actions)						# Number of actions in a control sequence (tested for 4 and 8)
error_type = args.error_type						# Error options for plots -- other options are xytheta and all

if args.learned_and_analytical == "True":		  	# To use both learned and analytical during experiments (valid only for 1 slider)
	learned_and_analytical = True
else:
	learned_and_analytical = False

if args.combine_error == "True":					# For the multi-object case combine errors from all objects.
	combine_error = True
else:
	combine_error = False

num_time_slices = n_actions
max_parareal_iter = n_actions + 1

dataset_path = "/home/oliver/Documents/final_datasets/"

clear_plots = False 								# Clear folders containing plots

def load_data(run_mode):

	if run_mode == 0:
		run_folder = 'new_model'
	elif run_mode == 1:
		run_folder = 'old_model'

	data_path = dataset_path+'{}slider_{}/parareal_data/{}/sequence_{}/'.format(N_S, n_actions, run_folder,sequence)

	Q = np.load(data_path+'Q.npy')
	mujoco_state_sequence = Q[-1].copy()

	mujoco_time_array = np.load(data_path+'mujoco_time_array.npy')
	parareal_time_array = np.load(data_path+'expected_parareal_time_array.npy')

	return Q, mujoco_state_sequence, parareal_time_array, mujoco_time_array

def get_displacment_results(run_mode):

	if run_mode == 0:
		run_folder = 'new_model'
	elif run_mode == 1:
		run_folder = 'old_model'

	N_S = chosen_objs[0]

	total_disp = np.zeros((number_of_samples, chosen_objs[0]))

	for sample in range (1, number_of_samples+1):
		data_path = dataset_path+'{}slider_{}/parareal_data/{}/sequence_{}/'.format(N_S, n_actions, run_folder,sample)

		Q = np.load(data_path+'Q.npy')
		mujoco_state_sequence = Q[-1].copy()

		for m in range(chosen_objs[0]):
			r_disp = 0
			for p in range(num_time_slices):
				# Calculate position displacements
				y_disp = mujoco_state_sequence[p+1][DOF+7*m+1] - mujoco_state_sequence[p][DOF+7*m+1]
				x_disp = mujoco_state_sequence[p+1][DOF+7*m] - mujoco_state_sequence[p][DOF+7*m]
				r_disp += sqrt(x_disp**2 + y_disp**2)

			total_disp[sample-1,m] = r_disp

	mean_disp = np.mean(total_disp, axis=0)
	std_disp = np.std(total_disp, axis=0)

	# Write results to file
	file1 = open(dataset_path+"{}slider_{}/results.txt".format(N_S, n_actions),"w+")
	file1.write("+++++++++++++++ Total Object Dispacement Results (m) +++++++++++ \n")
	file1.write("Mean total object displacement: {} \n".format(np.mean(mean_disp)))
	file1.write("Standard deviation of object displacement: {} \n".format(np.mean(std_disp)))
	file1.close()

	return None

def get_results(run_mode):

	Q, mujoco_state_sequence, parareal_time_array, mujoco_time_array = load_data(run_mode)

	sq = sequence - value*number_of_samples

	if run_mode == 0 and sq == number_of_samples:
		# Timing results for Learned Model
		mean_mujoco_time_array = np.mean(mujoco_time_array, axis=1)
		mean_parareal_time_array = np.mean(parareal_time_array, axis=1)

		time_plots(mean_parareal_time_array, mean_mujoco_time_array, run_mode)

	if run_mode == 1 and sq == number_of_samples:
		# Timing results for Analytical Model
		mean_mujoco_time_array = np.mean(mujoco_time_array, axis=1)
		mean_parareal_time_array = np.mean(parareal_time_array, axis=1)

		time_plots(mean_parareal_time_array, mean_mujoco_time_array, run_mode)

	state_error_plots(Q, mujoco_state_sequence, run_mode)

	return None

def state_error_plots(Q, Q_actual, run_mode):
	PLT_OFFSET = 1
	plot_path = dataset_path+'{}slider_{}/'.format(N_S, n_actions)
	font_size = 13.5
	line_width = 2.5
	line_style = '-.'
	legend_loc = "lower left"
	plt.rcParams.update({'font.size': font_size})
	plt.rcParams.update({'figure.figsize': [6.4, 4.8]})

	if error_type == "xy":
		plot_ylabel = "RMS error (m)"
	else:
		plot_ylabel = "RMS error"

	y_min = 1e-4
	y_max = 0.04
	x_min = 0
	PLOT_MAX = num_time_slices+1
	x_max = PLOT_MAX-1-PLT_OFFSET

	obj_error_x = np.zeros((max_parareal_iter,num_time_slices+1,N_S))
	obj_error_y = np.zeros((max_parareal_iter,num_time_slices+1,N_S))
	obj_error_theta = np.zeros((max_parareal_iter,num_time_slices+1,N_S))
	#rob_error_x = np.zeros((max_parareal_iter,num_time_slices+1))
	#rob_error_y = np.zeros((max_parareal_iter,num_time_slices+1))

	rms_obj_y = np.zeros((max_parareal_iter,N_S))
	rms_obj_x = np.zeros((max_parareal_iter,N_S))
	rms_obj_theta = np.zeros((max_parareal_iter,N_S))
	#rms_rob_y = np.zeros(max_parareal_iter)
	#rms_rob_x = np.zeros(max_parareal_iter)

	obj_error_vx = np.zeros((max_parareal_iter,num_time_slices+1,N_S))
	obj_error_vy = np.zeros((max_parareal_iter,num_time_slices+1,N_S))
	obj_error_vtheta = np.zeros((max_parareal_iter,num_time_slices+1,N_S))

	rms_obj_vy = np.zeros((max_parareal_iter,N_S))
	rms_obj_vx = np.zeros((max_parareal_iter,N_S))
	rms_obj_vtheta = np.zeros((max_parareal_iter,N_S))

	ROBOT_INDEX = 0

	for m in range(N_S):

		for p in range(num_time_slices+1):
			for k in range(max_parareal_iter):
				quat_actual = Q_actual[p][DOF+7*m+3:DOF+7*m+7]
				quat = Q[k][p][DOF+7*m+3:DOF+7*m+7]
				quat_error = Quaternion.absolute_distance(Quaternion(quat_actual).unit,Quaternion(quat).unit)
				obj_error_theta[k,p,m]= quat_error
				# Calculate position errors
				obj_error_y[k,p,m]=Q_actual[p][DOF+7*m+1]- Q[k][p][DOF+7*m+1]
				obj_error_x[k,p,m]=Q_actual[p][DOF+7*m] - Q[k][p][DOF+7*m]
				#rob_error_y[k,p]=Q_actual[p][ROBOT_INDEX+1] - Q[k][p][ROBOT_INDEX+1]
				#rob_error_x[k,p]=Q_actual[p][ROBOT_INDEX] - Q[k][p][ROBOT_INDEX]
		else:
			# Calculate the RMS error along the whole trajectory
			for k in range(max_parareal_iter):
				rms_obj_theta[k,m]= sqrt(mean(square(obj_error_theta[k,:,m])))
				rms_obj_y[k,m]= sqrt(mean(square(obj_error_y[k,:,m])))
				rms_obj_x[k,m]= sqrt(mean(square(obj_error_x[k,:,m])))
				#rms_rob_y[k]= sqrt(mean(square(rob_error_y[k,:])))
				#rms_rob_x[k]= sqrt(mean(square(rob_error_x[k,:])))


		for p in range(num_time_slices+1):
			for k in range(max_parareal_iter):
				# Calculate velocity errors
				obj_error_vtheta[k,p,m] = Q_actual[p][N_QPOS+DOF+6*m+5]- Q[k][p][N_QPOS+DOF+6*m+5]
				obj_error_vy[k,p,m] = Q_actual[p][N_QPOS+DOF+6*m+1]- Q[k][p][N_QPOS+DOF+6*m+1]
				obj_error_vx[k,p,m] = Q_actual[p][N_QPOS+DOF+6*m] - Q[k][p][N_QPOS+DOF+6*m]

		# Calculate the RMS error along the whole trajectory
		for k in range(max_parareal_iter):
			rms_obj_vtheta[k,m]= sqrt(mean(square(obj_error_vtheta[k,:,m])))
			rms_obj_vy[k,m]= sqrt(mean(square(obj_error_vy[k,:,m])))
			rms_obj_vx[k,m]= sqrt(mean(square(obj_error_vx[k,:,m])))

	if sequence == 1 and clear_plots:
		subprocess.call(["rm", "-r", plot_path+"parareal_plots"])
		subprocess.call(["mkdir", plot_path+"parareal_plots"])

	x_axis = np.arange(PLOT_MAX)

	if error_type == "xy":
		rms_obj_overall[:,:,sequence-number_of_samples*value-1,run_mode] = (rms_obj_y[0:PLOT_MAX,:]+rms_obj_x[0:PLOT_MAX,:])/2

	elif error_type == "xytheta":
		rms_obj_overall[:,:,sequence-number_of_samples*value-1,run_mode] = (rms_obj_y[0:PLOT_MAX,:]+rms_obj_x[0:PLOT_MAX,:]+rms_obj_theta[0:PLOT_MAX,:])/3

	elif error_type == "all":
		rms_obj_overall[:,:,sequence-number_of_samples*value-1,run_mode] = (rms_obj_y[0:PLOT_MAX,:]+rms_obj_x[0:PLOT_MAX,:]+rms_obj_theta[0:PLOT_MAX,:]+rms_obj_vx[0:PLOT_MAX,:]+rms_obj_vy[0:PLOT_MAX,:]+rms_obj_vtheta[0:PLOT_MAX,:])/6

	if sequence == (value+1)*number_of_samples:

		# Decide plot filenames
		if combine_error:
			plotname = plot_path+'parareal_plots/{}S_rms_{}_error_combined.png'.format(N_S, error_type)
			log_plotname = plot_path+'parareal_plots/{}S_rms_log_{}_error_combined.png'.format(N_S, error_type)
		else:
			plotname = plot_path+'parareal_plots/{}S_rms_{}_error.png'.format(N_S, error_type)
			log_plotname = plot_path+'parareal_plots/{}S_rms_log_{}_error.png'.format(N_S, error_type)

		# STANDARD plots

		# Either plot single line, or one for each slider
		if combine_error:

			std = np.std(rms_obj_overall, axis=(1,2))
			err = 1.96*np.sqrt( std/rms_obj_overall[0,:,:].size )

			plt.plot(x_axis[0:PLOT_MAX-PLT_OFFSET], np.mean(np.mean(rms_obj_overall, axis=1), axis=1)[0:PLOT_MAX-PLT_OFFSET], linewidth=line_width)

		else:

			for m in range(N_S):

				if learned_and_analytical:
					plt.plot(x_axis[0:PLOT_MAX], np.mean(rms_obj_overall[:,m,:,0], axis=1), label='Learned model', linewidth=line_width)
					plt.plot(x_axis[0:PLOT_MAX], np.mean(rms_obj_overall[:,m,:,1], axis=1), label='Analytical model', linewidth=line_width)

				else:
					plt.plot(x_axis[0:PLOT_MAX-PLT_OFFSET], np.mean(rms_obj_overall[:,m,:,0], axis=1)[0:PLOT_MAX-PLT_OFFSET], label='$Slider_{}$'.format(m+1), linewidth=line_width)


		if not combine_error:
			plt.legend(loc="best", frameon=True)

		plt.xlim(x_min, x_max)
		plt.margins(x=0)
		plt.xticks(np.arange(0, x_max+1, step=1))
		plt.grid(True)
		plt.xlabel('Number of iterations', fontsize=font_size)
		plt.ylabel('{}'.format(plot_ylabel),fontsize=font_size)
		plt.title('{} ({})  vs. Number of iterations'.format(plot_ylabel, error_type))
		plt.savefig(plotname)
		plt.cla()


		# LOG plots

		# Either plot single line, or one for each slider
		if combine_error:
			print ('True')
			plt.semilogy(x_axis[0:PLOT_MAX-PLT_OFFSET], np.mean(np.mean(rms_obj_overall, axis=1), axis=1)[0:PLOT_MAX-PLT_OFFSET], linewidth=line_width)

		else:

			for m in range(N_S):

				if learned_and_analytical:
					plt.semilogy(x_axis[0:PLOT_MAX-PLT_OFFSET], np.mean(rms_obj_overall[:,m,:,0], axis=1)[0:PLOT_MAX-PLT_OFFSET], label='Learned model', linewidth=line_width)
					plt.semilogy(x_axis[0:PLOT_MAX-PLT_OFFSET], np.mean(rms_obj_overall[:,m,:,1], axis=1)[0:PLOT_MAX-PLT_OFFSET], label='Analytical model', linewidth=line_width)
				else:
					plt.semilogy(x_axis[0:PLOT_MAX-PLT_OFFSET], np.mean(rms_obj_overall[:,m,:,0], axis=1)[0:PLOT_MAX-PLT_OFFSET], label='$Slider_{}$'.format(m+1), linewidth=line_width)

		if not combine_error:
			plt.legend(loc=legend_loc, frameon=True)
		plt.ylim(y_min, y_max)
		plt.xlim(x_min, x_max)
		plt.margins(x=0)
		plt.xticks(np.arange(0, x_max+1, step=1))
		plt.grid(True)
		axes = plt.axes()
		axes.spines['right'].set_visible(True)
		axes.spines['top'].set_visible(True)
		plt.xlabel('Number of iterations', fontsize=font_size)
		plt.ylabel('{}'.format(plot_ylabel), fontsize=font_size)

		#plt.title('RMS log error vs. Number of iterations')
		plt.savefig(log_plotname)

		plt.cla()

def time_plots(par_time_arr, mujoco_time_arr, run_mode):
	font_size = 13.5
	line_width = 2.5
	line_style = '-.'
	plot_path = dataset_path+'{}slider_{}/'.format(N_S, n_actions)
	plt.rcParams.update({'font.size': font_size})
	plt.rcParams.update({'figure.figsize': [6.4, 4.8]})
	y_min = 0
	y_max = np.mean(mujoco_time_arr)*1.2
	x_axis_range = mujoco_time_arr.shape[0]
	x_min = 0
	x_max = x_axis_range - 1
	x_axis = np.arange(x_axis_range)

	if run_mode == 0:
		run_folder = 'new_model'
	elif run_mode == 1:
		run_folder = 'old_model'

	if sequence == 1 and clear_plots:
		subprocess.call(["rm", "-r", "time_plots"])
		subprocess.call(["mkdir", "time_plots"])

	#mujoco_plot_array = np.mean(mujoco_time_arr)*np.ones(mujoco_time_arr.shape)
	mujoco_plot_array = mujoco_time_arr.copy()
	plt.plot(x_axis, mujoco_plot_array, linewidth=line_width, marker='o')
	plt.plot(x_axis, par_time_arr, linewidth=line_width, marker='o')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.margins(0)
	plt.xticks(np.arange(0, len(x_axis), step=1))

	plt.grid(True)
	plt.legend(['Physics engine','Parareal'], loc='best')
	plt.xlabel('Number of iterations', fontsize=font_size)
	plt.ylabel('Physics simulation time (s)',fontsize=font_size)
	plt.savefig(plot_path+'time_plots/%dS_time_plot' % N_S)
	plt.cla()

	#print ("mujoco total time for {} is {}".format(run_mode,mujoco_time_arr))
	#print ("Parareal total time for {} is {}".format(run_mode, par_time_arr))


	# Write results to file
	file1 = open(dataset_path+"{}slider_{}/results.txt".format(N_S, n_actions),"a")

	if run_mode == 1:
		# Analytical model
		file1.write("Parareal time per iteration (Analytical): {} \n".format(par_time_arr))
	else:
		file1.write("\n")
		file1.write("+++++++++++++++ Physics Simulation Timining Results (s) +++++++++++ \n")
		file1.write("Mean Mujoco total time: {} \n".format(np.mean(mujoco_time_arr)))
		file1.write("Parareal time per iteration (Learned): {} \n".format(par_time_arr))


	file1.close()

if __name__ == "__main__":
	sequence = 1

	get_displacment_results(0)

	for value in range(len(chosen_objs)):  # choose number of active sliders in the environment
		N_S = chosen_objs[value]
		rms_obj_overall = np.zeros((max_parareal_iter, N_S, number_of_samples,2))

		parareal_time_array = np.zeros((n_actions+1,number_of_samples))
		mujoco_time_array = np.zeros((n_actions+1,number_of_samples))
		expected_parareal_time_array = np.zeros((n_actions+1,number_of_samples))

		while sequence - number_of_samples*value <= number_of_samples:
			#print ('seququence', sequence)
			get_results(0)
			if learned_and_analytical:
				get_results(1)

			sequence += 1

	print ('\n Done. Plots and image frames generated. Results written to file. \n')
