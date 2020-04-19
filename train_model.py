from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import math as m
from math import pi, sqrt, sin, cos
import random
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from IPython.display import clear_output
from IPython import embed

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model as load
from tensorflow.keras import backend as K

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

#MODE = 'new'			# create and train new model
#MODE = 'load model'	# load and train a compiled model (cannot be used with custom loss)
MODE = 'load weights'	# load and train from model weights only (for use with custom loss)

# Normalize input dataset to values between 0 and 1, using min and max values of each input
NORMALIZE = False

# Location of training data
training_data = 'discrete_ms'
test_data = 'random_ms'

# Location of model/weights
load_model_from = 'model.hdf5'
load_weights_from = 'model_weights.h5'

# Whether or not to save the model and weights after each epoch
save_model = False
save_weights = True

# Location to save model/weights to
save_model_to = 'model_x.hdf5'
save_weights_to = 'model_weights_x.h5'

# Folder name to save metrics plots to after each epoch
loss_acc_plots_dir = 'loss_acc_plots'

new_lr = 0.0005		# learning rate for model
new_decay = 6e-5		# learning rate decay
n_epochs = 100
batch_size = 1024

_DOF = 2
_N_SLIDERS = 4		# maximum number of sliders in the dataset
_VARS = 6			# variables monitored per slider
_CTRL_DUR = 1

cols_in = _DOF + _N_SLIDERS*_VARS 	    # number of input columns to nn 
cols_out = _N_SLIDERS*_VARS				# number of output columns of nn

use_customloss = True 		# if False, uses mean squared error
use_weighted_loss = True 	# uses average values with aim of balancing percentage error across outputs	

max_penetration = 5e-4		# maximum penetration before penalizing loss
penalizing_factor = 2		# factor to multiply penetration value by (before squaring)

rs = 0.05115		# radius of sliders
rp = 0.0145			# radius of pusher

#################################################################################################

def normalizer(train_dataset, test_dataset):
	''' This function normalizes the input datasets to values between 0 and 1.

	Args:
		train_dataset: Training input dataset to be normalized.
		test_dataset: Test input dataset to be normalized. Normalized using max/min value from train dataset.

	Returns:
		train_dataset: Normalized training dataset, or unchanged train_dataset.
		test_dataset: Normalized test dataset, or unchanged test_dataset.
	'''

	if NORMALIZE:

		high = np.zeros((cols_in,))
		low = np.zeros((cols_in,))

		for c in range(C):
			high[c] = max(train_dataset[:,c])
			low[c] = min(train_dataset[:,c])
			train_dataset[:,c] = (train_dataset[:,c]-low[c])/(high[c]-low[c])
			test_dataset[:,c] = (test_dataset[:,c]-low[c])/(high[c]-low[c])

		print()
		print('high =	[%f, %f, %f, %f, %f, %f, %f]' % (high[0], high[1], high[2], high[3], high[4], high[5], high[6]))
		print('low =	[%f, %f, %f, %f, %f, %f, %f]' % (low[0], low[1], low[2], low[3], low[4], low[5], low[6]))
		print()

	return train_dataset, test_dataset

def data_import(files_train, files_test):
	''' This function imports csv datafiles for training and testing, and crops their size accordingly.

	Args:
		files_train: 	Prefix to name of training data files.
		files_test: 	Prefix to name of test data files.

	Returns:
		dataset: 		Training input data.
		labels:			Training output labels.
		test_dataset:	Testing input data.
		test_labels:	Testing output labels.
	'''


	dataset = np.array(pd.read_csv('%s_initial.csv' % files_train, sep=',', header=None))
	labels = np.array(pd.read_csv('%s_final.csv' % files_train, sep=',', header=None))

	test_dataset = np.array(pd.read_csv('%s_initial.csv' % files_test, sep=',', header=None))
	test_labels = np.array(pd.read_csv('%s_final.csv' % files_test, sep=',', header=None))		

	dataset = dataset[:,:cols_in]
	labels = labels[:,:cols_out]

	test_dataset = test_dataset[:1000*_N_SLIDERS,:cols_in]
	test_labels = test_labels[:1000*_N_SLIDERS,:cols_out]

	return dataset, labels, test_dataset, test_labels

def shuffle(dataset, labels):
	''' This function shuffles a dataset and labels pair into a random order equally,

	Args:
		dataset:	Input dataset to be shuffled.
		labels:		Output labels to be shuffled.

	Returns:
		dataset:	Shuffled input dataset.
		labels:		Shuffled output labels with same shuffled order as dataset.
	'''

	# Shuffle dataset into a random permutation
	indices = np.random.permutation(L)

	dataset = dataset[indices]
	labels = labels[indices]

	return dataset, labels

def get_model(MODE):
	''' This function loads or compiles a network model depending on the run mode.

	Args:
		MODE:	Run mode - choose whether to create new model, load a compiled model, or load weights.

	Returns:
		model:	A compiled model.
	'''

	if use_weighted_loss:
		''' Weighs losses depending on the mean of the output columns, 
		aiming to balance percentage error rather than absolute error'''
		
		rms_labels = np.zeros((cols_out,))
		weights = []

		# Calculate rms average of each column in labels
		for c in range(cols_out):
			rms_labels[c] = sqrt(np.mean(np.square(labels[:,c])))

		# Find the highest rms value
		max_rms_labels = max(rms_labels)

		# Weigh each loss by the maximum rms divided by its rms
		for c in range(cols_out):
			weights.append(max_rms_labels/rms_labels[c])

		weights = [weights]

	else:
		''' Sets all loss weights to 1 '''

		weights = []

		for c in range(cols_out):
			weights.append(1.)

		weights = [weights]

	if MODE == 'new':

		# Network architecture
		inputs = layers.Input(shape=(cols_in,))
		layer1 = layers.Dense(1024, use_bias=True, activation='relu')(inputs)
		layer2 = layers.Dense(512, use_bias=True, activation='relu')(layer1)
		layer3 = layers.Dense(256, use_bias=True, activation='relu')(layer2)
		layer4 = layers.Dense(128, use_bias=True, activation='relu')(layer3)
		layer5 = layers.Dense(64, use_bias=True, activation='relu')(layer4)
		outputs = layers.Dense(cols_out, activation='linear')(layer5)
		model = keras.Model(inputs=inputs, outputs = outputs)
		
		# Compile with optimizer and loss function
		optimizer = keras.optimizers.Nadam(lr=new_lr, decay=new_decay)
		
		if use_customloss:
			model.compile(optimizer=optimizer, loss=customloss_wrapper(inputs, weights), metrics=['accuracy'])
		else:
			model.compile(optimizer=optimizer, loss='mse', loss_weights=weights, metrics=['accuracy'])

	elif MODE == 'load model':

		# Load an aleady trained and compiled model 
		model = load(load_model_from)

		# Change training parameters
		K.set_value(model.optimizer.lr, new_lr)
		K.set_value(model.optimizer.decay, new_decay)

	elif MODE == 'load weights':

		# Network architecture - must be the same as the model's which you are loading in
		inputs = layers.Input(shape=(cols_in,))
		layer1 = layers.Dense(1024, use_bias=True, activation='relu')(inputs)
		layer2 = layers.Dense(512, use_bias=True, activation='relu')(layer1)
		layer3 = layers.Dense(256, use_bias=True, activation='relu')(layer2)
		layer4 = layers.Dense(128, use_bias=True, activation='relu')(layer3)
		layer5 = layers.Dense(64, use_bias=True, activation='relu')(layer4)
		outputs = layers.Dense(cols_out, activation='linear')(layer5)
		model = keras.Model(inputs=inputs, outputs = outputs)

		optimizer = keras.optimizers.Nadam(lr=new_lr, decay=new_decay)
		
		if use_customloss:
			model.compile(optimizer=optimizer, loss=customloss_wrapper(inputs, weights), metrics=['accuracy'])
		else:
			model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
		
		model.load_weights(load_weights_from)

	else:
		
		raise ValueError("'MODE' variable invalid.")

	return model

def customloss_wrapper(inputs, loss_weights):

	def customloss(y_true, y_pred):

		losses = tf.reduce_mean(tf.math.squared_difference(y_true,y_pred), axis=0)

		loss = tf.reduce_mean(tf.multiply(loss_weights,losses))/tf.reduce_mean(loss_weights)

		for m in range(_N_SLIDERS):

			# Compute penetration between pusher and sliders for all samples in batch
			if m == 0:
				pusher_penetration = tf.math.sqrt( tf.math.square((inputs[:,_DOF]+y_pred[:,0])-inputs[:,0]*_CTRL_DUR) + tf.math.square(y_pred[:,1]-inputs[:,1]*_CTRL_DUR) ) - (rs + rp)
			else:
				pusher_penetration = tf.math.sqrt( tf.math.square((inputs[:,_DOF+_VARS*m-1]+y_pred[:,_VARS*m])-inputs[:,0]*_CTRL_DUR) + tf.math.square(y_pred[:,_VARS*m+1]-inputs[:,1]*_CTRL_DUR) ) - (rs + rp)

			# Where penetration is insignificant or zero, set to zero
			zeros = tf.zeros_like(pusher_penetration)
			pusher_penetration = tf.where(pusher_penetration<(zeros-max_penetration), pusher_penetration, zeros)

			# Compute the mean of the new penetration array, multiply by factor and square --> add to loss
			penalize = tf.math.square(tf.reduce_mean(pusher_penetration)*penalizing_factor)
			loss += penalize
			
			# Initialize variable
			slider_penetration = 0

			# For each slider-slider pair, compute penetration in the same way and add to total slider penetration
			for i in range(m+1, _N_SLIDERS):
			
				if m == 0:
					penetration = tf.math.sqrt( tf.math.square( (inputs[:,_DOF]+y_pred[:,0]) - (inputs[:,_DOF+_VARS*i-1]+y_pred[:,_VARS*i]) ) + tf.math.square( y_pred[:,1] - (inputs[:,_DOF+_VARS*i]+y_pred[:,_VARS*i+1]) ) ) - 2*rs
				else:
					penetration = tf.math.sqrt( tf.math.square( (inputs[:,_DOF+_VARS*m-1]+y_pred[:,_VARS*m]) - (inputs[:,_DOF+_VARS*i-1]+y_pred[:,_VARS*i]) ) + tf.math.square( (inputs[:,_DOF+_VARS*m]+y_pred[:,_VARS*m+1]) - (inputs[:,_DOF+_VARS*i]+y_pred[:,_VARS*i+1]) ) ) - 2*rs

				zeros = tf.zeros_like(penetration)
				penetration = tf.where(penetration<(zeros-max_penetration), penetration, zeros)

				slider_penetration += tf.math.square(tf.reduce_mean(penetration)*penalizing_factor)

			# Add overall slider penetration factor to loss
			loss += slider_penetration

		return loss

	return customloss

class PlotMetrics(keras.callbacks.Callback):
	''' Callback to plot losses+accuracy after each epoch'''

	def on_train_begin(self, logs={}):

		folder_check = os.path.isdir(loss_acc_plots_dir)

		if not folder_check:
			os.mkdir(loss_acc_plots_dir)

		self.losses = []
		self.accuracies = []
		self.val_losses = []
		self.val_accuracies = []
		self.xaxis = []

		# Choose name for loss and accuracy plots, avoiding overwriting
		num = int(0)
		check = True

		while check:
			num += int(1)
			self.plotname = '%s/#%d.png' % (loss_acc_plots_dir, int(num))
			check = os.path.isfile(self.plotname)


	def on_epoch_end(self, epoch, logs={}):

		self.xaxis.append(epoch)
		self.losses.append(logs.get('loss'))
		self.accuracies.append(logs.get('accuracy'))
		self.val_losses.append(logs.get('val_loss'))
		self.val_accuracies.append(logs.get('val_accuracy'))

		fig = plt.figure(figsize=(16,7))

		plt.subplot(1,2,1)
		plt.plot(self.xaxis, self.accuracies, 'b')
		plt.plot(self.xaxis, self.val_accuracies, 'r')
		plt.title('MODEL ACCURACY', fontsize=19, fontweight='bold')
		plt.ylabel('Accuracy', fontsize=14)
		plt.xlabel('Epochs', fontsize=14)
		plt.xticks(fontsize=10)
		plt.yticks(fontsize=10)
		plt.grid(b=True)
		plt.legend(['training acc', 'validation acc'], loc='lower right')

		plt.subplot(1,2,2)
		plt.plot(self.xaxis, self.losses, 'b')
		plt.plot(self.xaxis, self.val_losses, 'r')
		plt.title('MODEL LOSS', fontsize=19, fontweight='bold')
		plt.ylabel('Loss', fontsize=14)
		plt.xlabel('Epochs', fontsize=14)
		plt.xticks(fontsize=10)
		plt.yticks(fontsize=10)
		plt.grid(b=True)
		plt.legend(['training loss', 'validation loss'], loc='upper right')
		
		plt.savefig(self.plotname)
		plt.close()

class ModelCheckpoint(keras.callbacks.Callback):
	"""Save the model after every epoch."""

	def __init__(self, filepath, SAVE, monitor='val_loss', verbose=0,
				 save_best_only=False, save_weights_only=False,
				 mode='auto', period=1,):
		super(ModelCheckpoint, self).__init__()
		self.monitor = monitor
		self.verbose = verbose
		self.filepath = filepath
		self.save_best_only = save_best_only
		self.save_weights_only = save_weights_only
		self.period = period
		self.epochs_since_last_save = 0
		self.SAVE = SAVE

		if mode not in ['auto', 'min', 'max']:
			warnings.warn('ModelCheckpoint mode %s is unknown, '
						  'fallback to auto mode.' % (mode),
						  RuntimeWarning)
			mode = 'auto'

		if mode == 'min':
			self.monitor_op = np.less
			self.best = np.Inf
		elif mode == 'max':
			self.monitor_op = np.greater
			self.best = -np.Inf
		else:
			if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
				self.monitor_op = np.greater
				self.best = -np.Inf
			else:
				self.monitor_op = np.less
				self.best = np.Inf

	def on_epoch_end(self, epoch, logs=None):
		if self.SAVE:	
			logs = logs or {}
			self.epochs_since_last_save += 1
			if self.epochs_since_last_save >= self.period:
				self.epochs_since_last_save = 0
				filepath = self.filepath.format(epoch=epoch + 1, **logs)
				if self.save_best_only:
					current = logs.get(self.monitor)
					if current is None:
						warnings.warn('Can save best model only with %s available, '
						'skipping.' % (self.monitor), RuntimeWarning)
					else:
						if self.monitor_op(current, self.best):
							if self.verbose > 0:
								print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
									  ' saving model to %s'
									  % (epoch + 1, self.monitor, self.best,
										 current, filepath))
							self.best = current
							if self.save_weights_only:
								self.model.save_weights(filepath, overwrite=True)
							else:
								self.model.save(filepath, overwrite=True)
						else:
							if self.verbose > 0:
								print('\nEpoch %05d: %s did not improve from %0.5f' %
									  (epoch + 1, self.monitor, self.best))
				else:
					if self.verbose > 0:
						print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
					if self.save_weights_only:
						self.model.save_weights(filepath, overwrite=True)
					else:
						self.model.save(filepath, overwrite=True)

#################################################################################################

if __name__ == '__main__':

	# Import all datasets from csv
	dataset, labels, test_dataset, test_labels = data_import(training_data, test_data)

	L = np.shape(dataset)[0]
	L1 = np.shape(test_dataset)[0]

	# Normalize
	dataset, test_dataset = normalizer(dataset, test_dataset)

	# Shuffle train and test data into a random order (equally)
	train_dataset, train_labels = shuffle(dataset, labels)

	model = get_model(MODE)

	# Callback to save (and overwrite) after every epoch
	saver_model = ModelCheckpoint(save_model_to, save_model)
	saver_weights = ModelCheckpoint(save_weights_to, save_weights, save_weights_only=True)

	# Callback to plot loss and accuracy after every epoch
	plotmetrics = PlotMetrics()

	# Train the model, with callbacks
	Model = model.fit(train_dataset, train_labels, epochs=n_epochs, batch_size=batch_size, validation_split=0.2, callbacks=[saver_weights, plotmetrics], shuffle=True)

	# Evaluate on test data
	print('Test 1/1')
	model.evaluate(test_dataset, test_labels)

	print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')