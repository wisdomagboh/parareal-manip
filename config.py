
STEPS = 8                       # Number of coarse steps
ACTION_MAGNITUDE = 0.1          # MAX pusher speed
PENETRATION_EPS = 5e-4          # Penetration tolerance during rejection sampling

# State description parameters
NUM_OBJS = 4       				# Number of dynamic objects in the scene (incl. goal obj.)
DOF = 2            				# Robot's degrees of freedom
N_QPOS = DOF + NUM_OBJS*7 		# Number of position variables
N_QVEL = DOF + NUM_OBJS*6		# Number of velocity variables
CTRL_DUR = 1 					# Control duration for each action is 1 second
SLIDER_CYLINDER = True			# Using only cylindrical sliders
PUSHER_RADIUS = 0.0145
SLIDER_RADIUS =  0.05115

# Parameters for the fine model (mujoco)
SIM_TIMESTEP = 1e-3				# Simulation time-step
SIM_INTEGRATOR = 1 				# 4th Order Runge-Kutta (0 for Semi-Implicit Euler)
num_substeps = int (CTRL_DUR/SIM_TIMESTEP)

# Parameters for analytical coarse model
OBJECT_Z = 0.4262				# Table height
ANG_VEL_MULTIPLIER = 0.1
ANGLE_MULTIPLIER = 1.0
MAX_PENETRATION_DEPTH = 1e-5
ANG_VEL_THRESH = 0.05

# Paraemters for learned coarse model
use_customloss = True
max_penetration = 5e-4
penalizing_factor = 2
load_weights_loc = 'model_weights.h5'	# Location of model weight file
VARS = 6           						# Variables per slider (xpos, ypos, vx, vy)
cols_in = DOF + NUM_OBJS*VARS - 1    	# number of input columns to network
cols_out = NUM_OBJS*VARS             	# number of output columns of network
rs = SLIDER_RADIUS
rp = PUSHER_RADIUS

clear_frames = True 				# Clear folders containing frames
